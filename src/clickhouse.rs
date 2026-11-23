//! ClickHouse in Docker over its HTTP interface, with MergeTree tables or views over the Parquet parts.
//!
//! The dataset folder is mounted read-only and set as the server's `user_files_path`, so the server reads
//! the parts itself through the `file` table function. Each query runs with `max_threads` set to `--threads`,
//! `join_use_nulls` on, the standard SQL semantics for outer joins, and the query condition and file count caches
//! off, so hot runs recompute their answers. Loaded tables merge their parts with `OPTIMIZE TABLE … FINAL`.
//!
//! `--suite documents --source stored` runs no container: it starts the `--server` binary natively over the server
//! folder a load left, the first `--data-dir`, inheriting this process's cores and memory, with its logs and config in
//! a temporary folder and that load's disks and users kept. Each partition is the table `<collection>_p<p>`
//! of `key FixedString(16), doc String`, read through `JSONExtract`, or with `--document-column json` the table
//! `<collection>_json_p<p>` of a native `JSON` column, which `--source loaded` creates once from the first, with
//! `INSERT … SELECT` and `OPTIMIZE … FINAL`. Further `--data-dir` folders, one per drive, select the first partitions;
//! without them, every partition runs. All partitions query at once, each with `--threads` divided by the
//! partitions as `max_threads`: a keyed gather sends its keys as an external table for `WHERE key IN keys`, a sample
//! orders by `rand()` under a `LIMIT`, and a scan selects every row, all `FORMAT Arrow`.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features clickhouse-backend --bin tabl-eval-clickhouse -- \
//!     --dataset data/clickbench-100m --source loaded,parquet
//! ```

use std::{
    alloc::System,
    collections::{hash_map::Entry, BTreeMap, HashMap},
    fmt::Write,
    io::Cursor,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use arrow::{array::RecordBatch, ipc::reader::FileReader};
use clap::Parser;
use serde_json::{json, Value};

use tableval::{
    docker::{ContainerHandle, Startup},
    parse_duration_flag, run,
    suite::{partition_of, DocumentField, Draw, FieldType, Workload},
    unsupported, Answer, Backend, CommonArgs, Configuration, Executed, Query, Result, Source, Suite, Table,
};

const IMAGE: &str = "clickhouse/clickhouse-server:25.8";
/// Where the dataset folder appears inside the container, set as the server's `user_files_path`.
const DATASET: &str = "/dataset";

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    /// Time limit for container start and readiness.
    #[arg(long, env = "TABLEVAL_STARTUP_TIME_LIMIT", default_value = "120s", value_parser = parse_duration_flag)]
    startup_time_limit: Duration,
    /// The ClickHouse binary a stored source runs natively.
    #[arg(long, env = "TABLEVAL_SERVER", default_value = "clickhouse")]
    server: PathBuf,
    /// string reads documents as `doc String` through JSONExtract, json from a native `JSON` column.
    #[arg(long, env = "TABLEVAL_DOCUMENT_COLUMN", value_enum, default_value = "string")]
    document_column: DocumentColumn,
}

/// How a documents table holds its documents.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
enum DocumentColumn {
    String,
    Json,
}

const JSON_TABLES: &str = include_str!("../sql/documents/clickhouse-json-tables.sql");

struct ClickhouseBackend {
    container: ContainerHandle,
    agent: ureq::Agent,
    threads: usize,
    version: String,
    limits: Value,
}

impl ClickhouseBackend {
    fn open(configuration: &Configuration, path: &Path, dataset: &Path, startup_time_limit: Duration) -> Result<Self> {
        let startup = Startup {
            data: Some((dataset, DATASET)),
            time_limit: startup_time_limit,
        };
        let container = ContainerHandle::start_with(
            IMAGE,
            8123,
            path,
            "/var/lib/clickhouse",
            &[("CLICKHOUSE_SKIP_USER_SETUP", "1")],
            &["--", "--user_files_path=/dataset/"],
            &startup,
        )?;
        let config = ureq::Agent::config_builder().http_status_as_error(false).build();
        let mut backend = Self {
            container,
            agent: ureq::Agent::new_with_config(config),
            threads: configuration.threads.get(),
            version: String::new(),
            limits: Value::Null,
        };
        backend.container.ready(|| backend.post("SELECT 1").map(drop))?;
        backend.version = backend.post("SELECT version()")?.trim().to_owned();
        backend.limits = backend.container.limits()?;
        Ok(backend)
    }

    /// Sends one statement and returns the response body, or the server's error text.
    fn post(&self, sql: &str) -> Result<String> {
        let url = format!(
            "http://127.0.0.1:{}/?max_threads={}&join_use_nulls=1&use_query_condition_cache=0&use_cache_for_count_from_files=0",
            self.container.port, self.threads
        );
        let mut response = self.agent.post(&url).send(sql).map_err(|e| e.to_string())?;
        let status = response.status();
        let body = response
            .body_mut()
            .with_config()
            .limit(u64::MAX)
            .read_to_string()
            .map_err(|e| e.to_string())?;
        match status.is_success() {
            true => Ok(body),
            false if body.contains("NOT_IMPLEMENTED") || body.contains("UNSUPPORTED_METHOD") => {
                Err(unsupported(body.trim()))
            }
            false => Err(body.trim().to_owned()),
        }
    }
}

/// `{trips.part_000.parquet,trips.part_001.parquet}`, the `file` function's path of the parts.
fn part_glob(table: &Table) -> String {
    let names: Vec<String> = table
        .parts
        .iter()
        .map(|part| part.file_name().unwrap_or_default().to_string_lossy().into_owned())
        .collect();
    format!("{{{}}}", names.join(","))
}

impl Backend for ClickhouseBackend {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("backend".into(), json!("clickhouse")),
            ("library_version".into(), json!(self.version)),
            ("image".into(), json!(IMAGE)),
            ("threads".into(), json!(self.threads)),
            ("join_use_nulls".into(), json!(true)),
            ("use_query_condition_cache".into(), json!(false)),
            ("use_cache_for_count_from_files".into(), json!(false)),
            ("container_limits".into(), self.limits.clone()),
        ])
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        for table in tables {
            let scan = format!("SELECT * FROM file('{}', Parquet)", part_glob(table));
            let statement = match source {
                Source::Loaded => format!(
                    "CREATE TABLE {} ENGINE = MergeTree ORDER BY tuple() AS {scan}",
                    table.name
                ),
                Source::Parquet => format!("CREATE VIEW {} AS {scan}", table.name),
                Source::Stored => return Err(unsupported("a container holds no stored tables")),
            };
            self.post(&statement).map_err(|e| format!("{}: {e}", table.name))?;
            if source == Source::Loaded {
                self.post(&format!("OPTIMIZE TABLE {} FINAL", table.name))
                    .map_err(|e| format!("{}: {e}", table.name))?;
            }
        }
        Ok(())
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        let start = Instant::now();
        let body = self.post(&format!("{} FORMAT JSONCompact", query.sql))?;
        let executed = Executed::since(start);
        let response: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
        for row in response["data"].as_array().ok_or("no data in the response")? {
            for cell in row.as_array().ok_or("a row is not an array")? {
                match cell {
                    Value::Null => answer.push_null(),
                    Value::Number(number) => answer.push_f64(number.as_f64().unwrap_or(f64::NAN)),
                    Value::String(text) => answer.push_str(text),
                    other => answer.push_str(&other.to_string()),
                }
            }
            answer.end_row();
        }
        Ok(executed)
    }

    fn disk_bytes(&self) -> Result<u64> {
        self.container.disk_bytes()
    }

    fn server_usage(&self) -> Result<Option<Value>> {
        self.container.stats().map(Some)
    }

    fn server_cgroup(&self) -> Option<std::path::PathBuf> {
        self.container.cgroup()
    }
}

/// The ClickHouse type a field's cells convert into, nullable, since a document may lack the field.
fn nullable_type(field: &DocumentField) -> &'static str {
    match field.field_type {
        FieldType::Text => "Nullable(String)",
        FieldType::Integer => "Nullable(Int64)",
        FieldType::Real => "Nullable(Float64)",
        FieldType::Boolean => "Nullable(Bool)",
    }
}

/// `key` and every field of `fields` as select expressions over `doc`.
fn select_list(column: DocumentColumn, fields: &[DocumentField]) -> String {
    let mut list = vec!["key".to_owned()];
    list.extend(fields.iter().map(|field| match column {
        DocumentColumn::String => format!(
            "JSONExtract(doc, '{0}', '{1}') AS {0}",
            field.name,
            nullable_type(field)
        ),
        DocumentColumn::Json => format!("CAST(doc.{0}, '{1}') AS {0}", field.name, nullable_type(field)),
    }));
    list.join(", ")
}

/// The statement one partition's table answers a documents query with; a keyed gather reads the external table
/// `keys`.
fn documents_sql(
    workload: Workload,
    column: DocumentColumn,
    table: &str,
    fields: &[DocumentField],
    wanted: usize,
) -> String {
    let select = format!("SELECT {} FROM {table}", select_list(column, fields));
    match workload {
        Workload::KeyedGather => format!("{select} WHERE key IN keys FORMAT Arrow"),
        Workload::SampleGather => format!("{select} ORDER BY rand() LIMIT {wanted} FORMAT Arrow"),
        Workload::Scan => format!("{select} FORMAT Arrow"),
    }
}

/// `text` as one URL query value.
fn percent_encoded(text: &str) -> String {
    let mut encoded = String::with_capacity(text.len() * 3);
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => encoded.push(byte as char),
            _ => {
                let _ = write!(encoded, "%{byte:02X}");
            }
        }
    }
    encoded
}

/// The text between the first `<tag>` and its closing tag in `text`, both included.
fn element<'text>(text: &'text str, tag: &str) -> Option<&'text str> {
    let start = text.find(&format!("<{tag}>"))?;
    let closing = format!("</{tag}>");
    let end = start + text[start..].find(&closing)? + closing.len();
    Some(&text[start..end])
}

/// A ClickHouse server run from its binary over a stored server folder, stopped when dropped.
struct NativeServer {
    process: Child,
    port: u16,
    /// The temporary folder of the config and logs, which goes with the server.
    scratch: tempfile::TempDir,
}

impl NativeServer {
    /// Starts `binary` over `root`, the folder a load's server ran in, keeping its disks and users.
    fn start(binary: &Path, root: &Path) -> Result<Self> {
        let loaded =
            std::fs::read_to_string(root.join("config.xml")).map_err(|e| format!("{}: {e}", root.display()))?;
        let storage = element(&loaded, "storage_configuration").unwrap_or_default();
        let port = TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .map_err(|e| e.to_string())?
            .port();
        let scratch = tempfile::Builder::new()
            .prefix("tableval-clickhouse-")
            .tempdir()
            .map_err(|e| e.to_string())?;
        let (root, logs) = (root.display(), scratch.path().display());
        let config = format!(
            "<clickhouse>\n  <logger><level>warning</level><log>{logs}/server.log</log><errorlog>{logs}/error.log</errorlog></logger>\n  \
             <http_port>{port}</http_port><listen_host>127.0.0.1</listen_host>\n  <path>{root}/data/</path>\n  \
             <user_files_path>{logs}/</user_files_path><users_config>{root}/users.xml</users_config>\n  {storage}\n</clickhouse>\n"
        );
        let config_path = scratch.path().join("config.xml");
        std::fs::write(&config_path, config).map_err(|e| e.to_string())?;
        let process = Command::new(binary)
            .arg("server")
            .arg("--config-file")
            .arg(&config_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("{}: {e}", binary.display()))?;
        Ok(Self { process, port, scratch })
    }

    /// The last lines of the server's error log.
    fn errors(&self) -> String {
        let log = std::fs::read_to_string(self.scratch.path().join("error.log")).unwrap_or_default();
        let lines: Vec<&str> = log.lines().collect();
        lines[lines.len().saturating_sub(10)..].join("\n")
    }
}

impl Drop for NativeServer {
    fn drop(&mut self) {
        // SAFETY: the process is this server's own child, not yet reaped.
        unsafe { libc::kill(self.process.id() as libc::pid_t, libc::SIGTERM) };
        if let Err(error) = self.process.wait() {
            eprintln!("ClickHouse server shutdown: {error}");
        }
    }
}

/// The documents suite over a stored ClickHouse server folder, its server run natively.
struct StoredDocuments {
    server: Option<NativeServer>,
    agent: ureq::Agent,
    binary: PathBuf,
    root: PathBuf,
    /// Partitions to query: as many as further data directories, or every one without them.
    selected: usize,
    tables: Vec<Table>,
    column: DocumentColumn,
    threads: usize,
    startup_time_limit: Duration,
    version: String,
    draws: HashMap<PathBuf, Draw>,
}

impl StoredDocuments {
    fn port(&self) -> Result<u16> {
        self.server
            .as_ref()
            .map(|server| server.port)
            .ok_or_else(|| "the ClickHouse server is not running".into())
    }

    /// Threads each partition's query may take.
    fn partition_threads(&self) -> usize {
        (self.threads / self.tables.len().max(1)).max(1)
    }

    /// Sends `body` to the server under the URL parameters `parameters` and answers the response body.
    fn send(agent: &ureq::Agent, port: u16, parameters: &str, content_type: &str, body: &[u8]) -> Result<Vec<u8>> {
        let url = format!(
            "http://127.0.0.1:{port}/?use_query_condition_cache=0&output_format_arrow_string_as_string=1\
             &output_format_arrow_compression_method=none{parameters}"
        );
        let mut response = agent
            .post(&url)
            .header("Content-Type", content_type)
            .send(body)
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let bytes = response
            .body_mut()
            .with_config()
            .limit(u64::MAX)
            .read_to_vec()
            .map_err(|e| e.to_string())?;
        match status.is_success() {
            true => Ok(bytes),
            false => Err(String::from_utf8_lossy(&bytes).trim().to_owned()),
        }
    }

    fn post(&self, sql: &str) -> Result<String> {
        let bytes = Self::send(&self.agent, self.port()?, "", "text/plain", sql.as_bytes())?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Starts the server and waits until it answers.
    fn start(&mut self) -> Result<()> {
        self.server = None;
        let server = NativeServer::start(&self.binary, &self.root)?;
        let deadline = Instant::now() + self.startup_time_limit;
        self.server = Some(server);
        loop {
            match self.post("SELECT version()") {
                Ok(version) => {
                    self.version = version.trim().to_owned();
                    return Ok(());
                }
                Err(error) => {
                    let server = self.server.as_mut().ok_or("the ClickHouse server is not running")?;
                    let exited = server.process.try_wait().map_err(|e| e.to_string())?.is_some();
                    if exited || Instant::now() >= deadline {
                        return Err(format!("ClickHouse did not become ready: {error}\n{}", server.errors()));
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }

    /// Creates the native `JSON` table of every partition from its `doc String` table, all partitions at once.
    fn create_json_tables(&self) -> Result<()> {
        let threads = self.partition_threads().to_string();
        std::thread::scope(|scope| {
            #[expect(clippy::needless_collect, reason = "every partition starts before the first join")]
            let partitions: Vec<_> = self
                .tables
                .iter()
                .map(|table| {
                    let threads = &threads;
                    scope.spawn(move || -> Result<()> {
                        let json_table = self.table_name(&table.name)?;
                        let policy = self.post(&format!(
                            "SELECT storage_policy FROM system.tables WHERE database = currentDatabase() AND name = '{}'",
                            table.name
                        ))?;
                        for statement in JSON_TABLES.split(';').map(str::trim).filter(|text| !text.is_empty()) {
                            let statement = statement
                                .replace("{json_table}", &json_table)
                                .replace("{table}", &table.name)
                                .replace("{policy}", policy.trim())
                                .replace("{threads}", threads);
                            self.post(&statement).map_err(|e| format!("{json_table}: {e}"))?;
                        }
                        Ok(())
                    })
                })
                .collect();
            partitions.into_iter().try_for_each(|partition| {
                partition
                    .join()
                    .map_err(|_| "a partition's setup panicked".to_owned())?
            })
        })
    }

    /// The table one partition's documents lie in, `<collection>_p<p>` or `<collection>_json_p<p>`.
    fn table_name(&self, table: &str) -> Result<String> {
        let (collection, partition) = partition_of(table).ok_or("a documents table names no partition")?;
        Ok(match self.column {
            DocumentColumn::String => table.to_owned(),
            DocumentColumn::Json => format!("{collection}_json_p{partition}"),
        })
    }
}

impl Backend for StoredDocuments {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("backend".into(), json!("clickhouse-native")),
            ("library_version".into(), json!(self.version)),
            ("server".into(), json!(self.binary)),
            ("server_folder".into(), json!(self.root)),
            ("document_column".into(), json!(self.column)),
            ("threads".into(), json!(self.threads)),
            ("partition_max_threads".into(), json!(self.partition_threads())),
            ("use_query_condition_cache".into(), json!(false)),
            ("output_format_arrow_string_as_string".into(), json!(true)),
            ("output_format_arrow_compression_method".into(), json!("none")),
        ])
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        let count = match self.selected {
            0 => tables.len(),
            selected => selected.min(tables.len()),
        };
        self.tables = tables[..count].to_vec();
        match (source, self.column) {
            (Source::Stored, _) => self.start(),
            (Source::Loaded, DocumentColumn::Json) => {
                self.start()?;
                self.create_json_tables()
            }
            _ => Err(unsupported(
                "the documents suite reads stored tables, and --source loaded creates only the JSON ones",
            )),
        }
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        let documents = query
            .documents
            .as_ref()
            .ok_or_else(|| unsupported("not a documents query"))?;
        if let Entry::Vacant(vacant) = self.draws.entry(documents.keys.clone()) {
            vacant.insert(Draw::read(&documents.keys)?);
        }
        let draw = &self.draws[&documents.keys];
        let port = self.port()?;
        let threads = self.partition_threads();
        let mut requests = Vec::with_capacity(self.tables.len());
        for table in &self.tables {
            let (_, partition) = partition_of(&table.name).ok_or("a documents table names no partition")?;
            let keys = &draw.keys[partition];
            let sql = documents_sql(
                documents.workload,
                self.column,
                &self.table_name(&table.name)?,
                documents.fields,
                keys.len(),
            );
            let wanted = match documents.workload {
                Workload::KeyedGather => keys.len() as u64,
                Workload::SampleGather => (keys.len() as u64).min(table.rows),
                Workload::Scan => table.rows,
            };
            let mut parameters = format!("&max_threads={threads}&query={}", percent_encoded(&sql));
            let (content_type, body) = match documents.workload {
                Workload::KeyedGather => {
                    parameters.push_str("&keys_structure=key%20FixedString(16)&keys_format=RowBinary");
                    let boundary = "tableval-keys";
                    let mut body = format!(
                        "--{boundary}\r\nContent-Disposition: form-data; name=\"keys\"; filename=\"keys\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                    )
                    .into_bytes();
                    body.extend(keys.iter().flatten());
                    body.extend(format!("\r\n--{boundary}--\r\n").bytes());
                    (format!("multipart/form-data; boundary={boundary}"), body)
                }
                _ => ("text/plain".to_owned(), Vec::new()),
            };
            requests.push((partition, wanted, parameters, content_type, body));
        }
        let agent = &self.agent;
        let start = Instant::now();
        let answered = std::thread::scope(|scope| {
            #[expect(clippy::needless_collect, reason = "every partition starts before the first join")]
            let partitions: Vec<_> = requests
                .iter()
                .map(|(_, _, parameters, content_type, body)| {
                    scope.spawn(move || -> Result<Vec<RecordBatch>> {
                        let bytes = Self::send(agent, port, parameters, content_type, body)?;
                        let reader = FileReader::try_new(Cursor::new(bytes), None).map_err(|e| e.to_string())?;
                        reader.map(|batch| batch.map_err(|e| e.to_string())).collect()
                    })
                })
                .collect();
            partitions
                .into_iter()
                .map(|partition| {
                    partition
                        .join()
                        .map_err(|_| "a partition's query panicked".to_owned())?
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let executed = Executed::since(start);
        for ((partition, wanted, ..), batches) in requests.iter().zip(&answered) {
            let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
            if rows as u64 != *wanted {
                return Err(format!("partition {partition} answered {rows} of {wanted} rows"));
            }
            answer.push_batches(batches)?;
        }
        Ok(executed)
    }

    fn disk_bytes(&self) -> Result<u64> {
        let tables = self
            .tables
            .iter()
            .map(|table| self.table_name(&table.name).map(|name| format!("'{name}'")))
            .collect::<Result<Vec<_>>>()?;
        let bytes = self.post(&format!(
            "SELECT sum(bytes_on_disk) FROM system.parts WHERE active AND database = currentDatabase() AND table IN ({})",
            tables.join(", ")
        ))?;
        bytes.trim().parse().map_err(|e| format!("disk bytes: {e}"))
    }

    fn multiple_data_dirs(&self) -> bool {
        true
    }

    fn reopen(&mut self) -> Result<()> {
        match self.server.is_some() {
            true => self.start(),
            false => Ok(()),
        }
    }

    fn server_cgroup(&self) -> Option<PathBuf> {
        let server = self.server.as_ref()?;
        tableval::measure::cgroup_of(&server.process.id().to_string())
    }
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    if cli.common.suite == Some(Suite::Documents) {
        let settings = [
            ("Startup time limit", tableval::spell_duration(cli.startup_time_limit)),
            ("Server", cli.server.display().to_string()),
            ("Document column", tableval::spell_value(&cli.document_column)),
        ];
        let options = json!({"server": cli.server, "document_column": cli.document_column});
        // The JSON setup writes into the stored server folder, so both sources open the first --data-dir itself
        let root = cli.common.data_dir[0].clone();
        let selected = cli.common.further_data_dirs(&root).len();
        return run(cli.common, options, &settings, |configuration, _| {
            let config = ureq::Agent::config_builder().http_status_as_error(false).build();
            Ok(Box::new_in(
                StoredDocuments {
                    server: None,
                    agent: ureq::Agent::new_with_config(config),
                    binary: cli.server.clone(),
                    root: root.clone(),
                    selected,
                    tables: Vec::new(),
                    column: cli.document_column,
                    threads: configuration.threads.get(),
                    startup_time_limit: cli.startup_time_limit,
                    version: String::new(),
                    draws: HashMap::new(),
                },
                System,
            ))
        });
    }
    let settings = [("Startup time limit", tableval::spell_duration(cli.startup_time_limit))];
    let dataset = cli.common.dataset.clone();
    run(cli.common, json!({"image": IMAGE}), &settings, |configuration, path| {
        Ok(Box::new_in(
            ClickhouseBackend::open(configuration, path, &dataset, cli.startup_time_limit)?,
            System,
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(
        configuration: &tableval::Configuration,
        path: &std::path::Path,
    ) -> tableval::Result<Box<dyn tableval::Backend, System>> {
        // The fixture keeps each source's storage beside its dataset, as `<dataset>-<source>`.
        let name = path.file_name().unwrap().to_string_lossy();
        let dataset = path.with_file_name(name.split('-').next().unwrap());
        Ok(Box::new_in(
            ClickhouseBackend::open(configuration, path, &dataset, Duration::from_secs(120))?,
            System,
        ))
    }

    #[test]
    fn documents_statements_extract_nullable_fields() {
        let fields = tableval::suite::scalar_fields("alphafold").unwrap();
        assert_eq!(
            documents_sql(
                Workload::KeyedGather,
                DocumentColumn::String,
                "alphafold_p0",
                &fields[..2],
                7
            ),
            "SELECT key, JSONExtract(doc, 'uniprot_accession', 'Nullable(String)') AS uniprot_accession, \
             JSONExtract(doc, 'taxonomy_id', 'Nullable(Int64)') AS taxonomy_id FROM alphafold_p0 WHERE key IN keys \
             FORMAT Arrow"
        );
        assert_eq!(
            documents_sql(
                Workload::SampleGather,
                DocumentColumn::Json,
                "alphafold_json_p0",
                &fields[8..9],
                7
            ),
            "SELECT key, CAST(doc.reviewed, 'Nullable(Bool)') AS reviewed FROM alphafold_json_p0 ORDER BY rand() \
             LIMIT 7 FORMAT Arrow"
        );
        assert_eq!(percent_encoded("key IN (1)"), "key%20IN%20%281%29");
    }

    /// Needs a running Docker daemon; run with `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn taxi_fixture() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::assert_taxi_answers(root.path(), open);
    }

    /// Needs a running Docker daemon; run with `cargo test -- --ignored`.
    #[cfg(feature = "generate")]
    #[test]
    #[ignore]
    fn tpch_answers() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::assert_tpch_answers(root.path(), open);
    }
}
