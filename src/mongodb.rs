//! MongoDB over the stores a load left, for the documents suite: one native `mongod` per partition.
//!
//! Each `--data-dir` is one partition's database folder, the first partitions in order, served by the `--server`
//! binary in normal mode on a free port, inheriting this process's cores and memory, with its log in a temporary
//! folder. WiredTiger's cache is the one setting off stock: every server takes half of its share of the memory
//! limit less 1 GB, the stock formula over one partition's share, since every server would otherwise size itself
//! from the whole limit. Documents lie in the `molecules` database under the collection's name, keyed by `_id`
//! as 16 binary bytes.
//!
//! All partitions run at once, each with `--threads` divided by the partitions as client lanes. A keyed gather splits
//! the draw's keys into one `find` with `$in` and a projection per lane, a sample runs `$sample` with the same
//! projection, and a scan runs one `find` over the whole collection. Raw BSON documents turn into Arrow columns
//! inside the timed region.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features mongodb-backend --bin tabl-eval-mongodb -- \
//!     --suite documents --source stored --dataset data/keys/alphafold --data-dir /mnt/nvme1/mongodb
//! ```

use std::{
    alloc::System,
    collections::{hash_map::Entry, BTreeMap, HashMap},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use arrow::array::{ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, RecordBatch, StringBuilder};
use clap::Parser;
use mongodb::{
    bson::{doc, spec::BinarySubtype, Binary, Bson, Document, RawBsonRef, RawDocumentBuf},
    sync::{Client, Collection},
};
use serde_json::{json, Value};

use tableval::{
    output, parse_duration_flag, run,
    suite::{fields_schema, keyed_batch, partition_of, DocumentField, Draw, FieldType, Workload},
    unsupported, Answer, Backend, Configuration, Executed, Query, Result, Source, Table,
};

/// Documents per Arrow batch and keys per `$in`.
const BATCH_ROWS: usize = 64 * 1024;
const DATABASE: &str = "molecules";

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: tableval::CommonArgs,
    /// Time limit for every server's start, which replays its journal after an unclean shutdown.
    #[arg(long, env = "TABLEVAL_STARTUP_TIME_LIMIT", default_value = "600s", value_parser = parse_duration_flag)]
    startup_time_limit: Duration,
    /// The `mongod` binary each partition runs.
    #[arg(long, env = "TABLEVAL_SERVER", default_value = "mongod")]
    server: PathBuf,
}

/// One partition's `mongod`, shut down cleanly when dropped.
struct Server {
    process: Child,
    client: Client,
    /// The temporary folder of the server's log, which goes with the server.
    scratch: tempfile::TempDir,
}

impl Server {
    fn start(binary: &Path, folder: &Path, cache_gigabytes: f64, lanes: usize, time_limit: Duration) -> Result<Self> {
        let port = TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .map_err(|e| e.to_string())?
            .port();
        let scratch = tempfile::Builder::new()
            .prefix("tableval-mongodb-")
            .tempdir()
            .map_err(|e| e.to_string())?;
        let process = Command::new(binary)
            .arg("--dbpath")
            .arg(folder)
            .args(["--port", &port.to_string(), "--bind_ip", "127.0.0.1", "--logpath"])
            .arg(scratch.path().join("mongod.log"))
            .args(["--wiredTigerCacheSizeGB", &format!("{cache_gigabytes:.2}")])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("{}: {e}", binary.display()))?;
        let uri = format!("mongodb://127.0.0.1:{port}/?directConnection=true&maxPoolSize={lanes}");
        let client = Client::with_uri_str(uri).map_err(|e| e.to_string())?;
        let mut server = Self {
            process,
            client,
            scratch,
        };
        let deadline = Instant::now() + time_limit;
        loop {
            let pinged = server.client.database("admin").run_command(doc! {"ping": 1}).run();
            match pinged {
                Ok(_) => return Ok(server),
                Err(error) => {
                    let exited = server.process.try_wait().map_err(|e| e.to_string())?.is_some();
                    if exited || Instant::now() >= deadline {
                        let log = std::fs::read_to_string(server.scratch.path().join("mongod.log")).unwrap_or_default();
                        let tail: Vec<&str> = log.lines().rev().take(5).collect();
                        return Err(format!(
                            "{} did not become ready: {error}\n{}",
                            folder.display(),
                            tail.join("\n")
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // SAFETY: the process is this server's own child, not yet reaped; SIGTERM is mongod's clean shutdown.
        unsafe { libc::kill(self.process.id() as libc::pid_t, libc::SIGTERM) };
        if let Err(error) = self.process.wait() {
            eprintln!("mongod shutdown: {error}");
        }
    }
}

/// Builds the Arrow batches of raw documents: the `_id` bytes, then every field as its type, null where a document
/// lacks the field or holds another type there.
struct Columns {
    fields: &'static [DocumentField],
    keys: Vec<u8>,
    cells: Vec<Cells>,
    rows: usize,
    batches: Vec<RecordBatch>,
}

enum Cells {
    Text(StringBuilder),
    Integer(Int64Builder),
    Real(Float64Builder),
    Boolean(BooleanBuilder),
}

impl Columns {
    fn new(fields: &'static [DocumentField]) -> Self {
        Self {
            fields,
            keys: Vec::new(),
            cells: fields
                .iter()
                .map(|field| match field.field_type {
                    FieldType::Text => Cells::Text(StringBuilder::new()),
                    FieldType::Integer => Cells::Integer(Int64Builder::new()),
                    FieldType::Real => Cells::Real(Float64Builder::new()),
                    FieldType::Boolean => Cells::Boolean(BooleanBuilder::new()),
                })
                .collect(),
            rows: 0,
            batches: Vec::new(),
        }
    }

    fn push(&mut self, document: &RawDocumentBuf) -> Result<()> {
        match document.get("_id").map_err(|e| e.to_string())? {
            Some(RawBsonRef::Binary(binary)) if binary.bytes.len() == 16 => self.keys.extend_from_slice(binary.bytes),
            _ => return Err("a document's _id is not 16 binary bytes".into()),
        }
        for (field, cells) in self.fields.iter().zip(&mut self.cells) {
            let value = document.get(field.name).map_err(|e| e.to_string())?;
            match (cells, value) {
                (Cells::Text(cells), Some(RawBsonRef::String(text))) => cells.append_value(text),
                (Cells::Text(cells), _) => cells.append_null(),
                (Cells::Integer(cells), Some(RawBsonRef::Int32(number))) => cells.append_value(number.into()),
                (Cells::Integer(cells), Some(RawBsonRef::Int64(number))) => cells.append_value(number),
                (Cells::Integer(cells), _) => cells.append_null(),
                (Cells::Real(cells), Some(RawBsonRef::Double(number))) => cells.append_value(number),
                (Cells::Real(cells), Some(RawBsonRef::Int32(number))) => cells.append_value(number.into()),
                (Cells::Real(cells), Some(RawBsonRef::Int64(number))) => cells.append_value(number as f64),
                (Cells::Real(cells), _) => cells.append_null(),
                (Cells::Boolean(cells), Some(RawBsonRef::Boolean(flag))) => cells.append_value(flag),
                (Cells::Boolean(cells), _) => cells.append_null(),
            }
        }
        self.rows += 1;
        match self.rows == BATCH_ROWS {
            true => self.flush(),
            false => Ok(()),
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.rows == 0 {
            return Ok(());
        }
        let columns: Vec<ArrayRef> = self
            .cells
            .iter_mut()
            .map(|cells| -> ArrayRef {
                match cells {
                    Cells::Text(cells) => Arc::new(cells.finish()),
                    Cells::Integer(cells) => Arc::new(cells.finish()),
                    Cells::Real(cells) => Arc::new(cells.finish()),
                    Cells::Boolean(cells) => Arc::new(cells.finish()),
                }
            })
            .collect();
        let cells = RecordBatch::try_new(fields_schema(self.fields), columns).map_err(|e| e.to_string())?;
        self.batches.push(keyed_batch(std::mem::take(&mut self.keys), &cells)?);
        self.rows = 0;
        Ok(())
    }

    /// Every document `cursor` yields, as batches.
    fn collect(
        fields: &'static [DocumentField],
        cursor: impl Iterator<Item = mongodb::error::Result<RawDocumentBuf>>,
    ) -> Result<Vec<RecordBatch>> {
        let mut columns = Self::new(fields);
        for document in cursor {
            columns.push(&document.map_err(|e| e.to_string())?)?;
        }
        columns.flush()?;
        Ok(columns.batches)
    }
}

/// The projection of `fields` and `_id`.
fn projection(fields: &[DocumentField]) -> Document {
    let mut projection = Document::new();
    for field in fields {
        projection.insert(field.name, 1);
    }
    projection
}

/// Finds `keys` in `collection` on `lanes` threads, one `$in` window per lane.
fn gather(
    collection: &Collection<RawDocumentBuf>,
    keys: &[[u8; 16]],
    fields: &'static [DocumentField],
    lanes: usize,
) -> Result<Vec<RecordBatch>> {
    let window = keys.len().div_ceil(lanes).clamp(1, BATCH_ROWS);
    let windows: Vec<&[[u8; 16]]> = keys.chunks(window).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..lanes)
            .map(|_| {
                let (windows, next) = (&windows, &next);
                scope.spawn(move || -> Result<Vec<RecordBatch>> {
                    let mut batches = Vec::new();
                    while let Some(keys) = windows.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let identifiers: Vec<Bson> = keys
                            .iter()
                            .map(|key| {
                                Bson::Binary(Binary {
                                    subtype: BinarySubtype::Generic,
                                    bytes: key.to_vec(),
                                })
                            })
                            .collect();
                        let cursor = collection
                            .find(doc! {"_id": {"$in": identifiers}})
                            .projection(projection(fields))
                            .run()
                            .map_err(|e| e.to_string())?;
                        batches.extend(Columns::collect(fields, cursor)?);
                    }
                    Ok(batches)
                })
            })
            .collect();
        let mut batches = Vec::new();
        for thread in threads {
            batches.extend(thread.join().map_err(|_| "a gather lane panicked".to_owned())??);
        }
        Ok(batches)
    })
}

/// The documents suite over stored MongoDB partitions.
struct StoredDocuments {
    servers: Vec<Server>,
    binary: PathBuf,
    folders: Vec<PathBuf>,
    tables: Vec<Table>,
    threads: usize,
    startup_time_limit: Duration,
    version: String,
    draws: HashMap<PathBuf, Draw>,
}

impl StoredDocuments {
    fn lanes(&self) -> usize {
        (self.threads / self.folders.len()).max(1)
    }

    /// WiredTiger's stock cache, half of the memory less 1 GB, over one partition's share of the memory limit.
    fn cache_gigabytes(&self) -> f64 {
        let share = output::budget().memory_limit as f64 / self.folders.len() as f64 / f64::from(1u32 << 30);
        ((share - 1.0) / 2.0).max(0.25)
    }

    /// Starts every partition's server, one after another, so journal replays never contend.
    fn start(&mut self) -> Result<()> {
        self.servers.clear();
        if self.folders.len() > self.tables.len() {
            return Err(format!(
                "{} data directories for {} partitions",
                self.folders.len(),
                self.tables.len()
            ));
        }
        for folder in &self.folders {
            let server = Server::start(
                &self.binary,
                folder,
                self.cache_gigabytes(),
                self.lanes(),
                self.startup_time_limit,
            )?;
            self.servers.push(server);
        }
        let build = self.servers[0]
            .client
            .database("admin")
            .run_command(doc! {"buildInfo": 1})
            .run()
            .map_err(|e| e.to_string())?;
        self.version = build.get_str("version").unwrap_or_default().to_owned();
        Ok(())
    }
}

impl Backend for StoredDocuments {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("backend".into(), json!("mongodb-native")),
            ("library_version".into(), json!(self.version)),
            ("server".into(), json!(self.binary)),
            ("stores".into(), json!(self.folders)),
            ("threads".into(), json!(self.threads)),
            ("lanes_per_partition".into(), json!(self.lanes())),
            ("wired_tiger_cache_gigabytes".into(), json!(self.cache_gigabytes())),
            (
                "deviations_from_stock".into(),
                json!(["--wiredTigerCacheSizeGB from one partition's share of the memory limit"]),
            ),
        ])
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        if source != Source::Stored {
            return Err(unsupported(
                "MongoDB serves the documents suite from the stores a load left, with --source stored",
            ));
        }
        self.tables = tables.to_vec();
        self.start()
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
        let mut partitions = Vec::with_capacity(self.servers.len());
        for (server, table) in self.servers.iter().zip(&self.tables) {
            let (name, partition) = partition_of(&table.name).ok_or("a documents table names no partition")?;
            let collection = server.client.database(DATABASE).collection::<RawDocumentBuf>(name);
            partitions.push((partition, table.rows, collection));
        }
        let (fields, lanes) = (documents.fields, self.lanes());
        let start = Instant::now();
        let answered = std::thread::scope(|scope| {
            let threads: Vec<_> = partitions
                .iter()
                .map(|(partition, _, collection)| {
                    let keys = &draw.keys[*partition];
                    scope.spawn(move || -> Result<Vec<RecordBatch>> {
                        match documents.workload {
                            Workload::KeyedGather => gather(collection, keys, fields, lanes),
                            Workload::SampleGather => {
                                let pipeline = [
                                    doc! {"$sample": {"size": keys.len() as i64}},
                                    doc! {"$project": projection(fields)},
                                ];
                                let cursor = collection
                                    .aggregate(pipeline)
                                    .with_type::<RawDocumentBuf>()
                                    .run()
                                    .map_err(|e| e.to_string())?;
                                Columns::collect(fields, cursor)
                            }
                            Workload::Scan => {
                                let cursor = collection
                                    .find(doc! {})
                                    .projection(projection(fields))
                                    .run()
                                    .map_err(|e| e.to_string())?;
                                Columns::collect(fields, cursor)
                            }
                        }
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().map_err(|_| "a partition panicked".to_owned())?)
                .collect::<Result<Vec<_>>>()
        })?;
        let executed = Executed::since(start);
        for ((partition, rows, _), batches) in partitions.iter().zip(&answered) {
            let keys = draw.keys[*partition].len() as u64;
            let wanted = match documents.workload {
                Workload::KeyedGather => keys,
                Workload::SampleGather => keys.min(*rows),
                Workload::Scan => *rows,
            };
            let answered: usize = batches.iter().map(RecordBatch::num_rows).sum();
            if answered as u64 != wanted {
                return Err(format!("partition {partition} answered {answered} of {wanted} rows"));
            }
            answer.push_batches(batches)?;
        }
        Ok(executed)
    }

    fn disk_bytes(&self) -> Result<u64> {
        self.folders
            .iter()
            .map(|folder| tableval::backend::directory_bytes(folder))
            .sum()
    }

    fn multiple_data_dirs(&self) -> bool {
        true
    }

    fn reopen(&mut self) -> Result<()> {
        match self.servers.is_empty() {
            true => Ok(()),
            false => self.start(),
        }
    }

    fn server_cgroup(&self) -> Option<PathBuf> {
        let server = self.servers.first()?;
        tableval::measure::cgroup_of(&server.process.id().to_string())
    }
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    let settings = [
        ("Startup time limit", tableval::spell_duration(cli.startup_time_limit)),
        ("Server", cli.server.display().to_string()),
    ];
    let common = cli.common.clone();
    run(
        cli.common,
        json!({"server": cli.server}),
        &settings,
        |configuration: &Configuration, path| {
            Ok(Box::new_in(
                StoredDocuments {
                    servers: Vec::new(),
                    binary: cli.server.clone(),
                    folders: std::iter::once(path.to_path_buf())
                        .chain(common.further_data_dirs(path))
                        .collect(),
                    tables: Vec::new(),
                    threads: configuration.threads.get(),
                    startup_time_limit: cli.startup_time_limit,
                    version: String::new(),
                    draws: HashMap::new(),
                },
                System,
            ))
        },
    )
}
