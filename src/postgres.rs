//! PostgreSQL in Docker: the row-store server baseline, loaded through `COPY` from the Parquet parts.
//!
//! `--threads` sets the parallel workers a single query may use. Dynamic shared memory lives in `mmap`ed
//! files rather than the container's 64 MB `/dev/shm`, which parallel hash joins would otherwise outgrow.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features postgres-backend --bin tabl-eval-postgres -- \
//!     --dataset data/tpch-sf1 --threads 1,8
//! ```

use std::{
    alloc::System,
    collections::BTreeMap,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

use clap::Parser;
use postgres::{Client, NoTls, SimpleQueryMessage};
use serde_json::{json, Value};

use tableval::{
    docker::{ContainerHandle, Startup},
    parse_duration_flag, rows, run, unsupported, Answer, Backend, CommonArgs, Configuration, Executed, Query, Result,
    Source, Table,
};

const IMAGE: &str = "postgres:18.6";

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    /// Shared buffer pool, in PostgreSQL's spelling like 4GB.
    #[arg(long, env = "TABLEVAL_SHARED_BUFFERS", default_value = "4GB")]
    shared_buffers: String,
    /// Time limit for container start and readiness.
    #[arg(long, env = "TABLEVAL_STARTUP_TIME_LIMIT", default_value = "120s", value_parser = parse_duration_flag)]
    startup_time_limit: Duration,
}

struct PostgresBackend {
    container: ContainerHandle,
    client: Client,
    threads: usize,
    shared_buffers: String,
    version: String,
}

impl PostgresBackend {
    fn open(
        configuration: &Configuration,
        path: &Path,
        shared_buffers: &str,
        startup_time_limit: Duration,
    ) -> Result<Self> {
        let threads = configuration.threads.get();
        let settings = [
            format!("shared_buffers={shared_buffers}"),
            format!("max_worker_processes={}", threads + 8),
            format!("max_parallel_workers={threads}"),
            format!("max_parallel_workers_per_gather={}", threads - 1),
            "dynamic_shared_memory_type=mmap".to_owned(),
            "fsync=off".to_owned(),
            "synchronous_commit=off".to_owned(),
            "full_page_writes=off".to_owned(),
        ];
        let mut command = vec!["postgres"];
        for setting in &settings {
            command.extend(["-c", setting.as_str()]);
        }
        let startup = Startup {
            data: None,
            time_limit: startup_time_limit,
        };
        let container = ContainerHandle::start_with(
            IMAGE,
            5432,
            path,
            "/var/lib/postgresql",
            &[("POSTGRES_PASSWORD", "tableval"), ("POSTGRES_DB", "tableval")],
            &command,
            &startup,
        )?;
        let url = format!(
            "host=127.0.0.1 port={} user=postgres password=tableval dbname=tableval",
            container.port
        );
        container.ready(|| Client::connect(&url, NoTls).map(drop).map_err(|e| format!("{e:?}")))?;
        let mut client = Client::connect(&url, NoTls).map_err(|e| format!("{e:?}"))?;
        let version = client
            .query_one("SHOW server_version", &[])
            .map_err(|e| e.to_string())?
            .get::<_, String>(0);
        Ok(Self {
            container,
            client,
            threads,
            shared_buffers: shared_buffers.to_owned(),
            version,
        })
    }
}

/// Appends `text` in `COPY`'s text format, where backslash, tab and line breaks are escaped.
fn push_escaped(line: &mut Vec<u8>, text: &str) {
    for byte in text.bytes() {
        match byte {
            b'\\' => line.extend_from_slice(b"\\\\"),
            b'\t' => line.extend_from_slice(b"\\t"),
            b'\n' => line.extend_from_slice(b"\\n"),
            b'\r' => line.extend_from_slice(b"\\r"),
            byte => line.push(byte),
        }
    }
}

impl Backend for PostgresBackend {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("backend".into(), json!("postgres")),
            ("library_version".into(), json!(self.version)),
            ("image".into(), json!(IMAGE)),
            ("threads".into(), json!(self.threads)),
            ("max_parallel_workers_per_gather".into(), json!(self.threads - 1)),
            ("shared_buffers".into(), json!(self.shared_buffers)),
        ])
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        if source == Source::Parquet {
            return Err(unsupported("PostgreSQL reads only its own tables"));
        }
        for table in tables {
            let declarations: Vec<String> = rows::columns(table)?
                .iter()
                .map(|(name, kind)| {
                    let declared = match kind {
                        rows::Kind::Integer => "BIGINT",
                        rows::Kind::Real => "DOUBLE PRECISION",
                        rows::Kind::Text => "TEXT",
                        rows::Kind::Date => "DATE",
                        rows::Kind::Timestamp => "TIMESTAMP",
                    };
                    format!("\"{}\" {declared}", name.to_lowercase())
                })
                .collect();
            self.client
                .batch_execute(&format!(
                    "CREATE UNLOGGED TABLE {} ({})",
                    table.name,
                    declarations.join(", ")
                ))
                .map_err(|e| e.to_string())?;
            let mut writer = self
                .client
                .copy_in(&format!("COPY {} FROM STDIN", table.name))
                .map_err(|e| e.to_string())?;
            let mut line = Vec::new();
            rows::for_each_row(table, |cells| {
                line.clear();
                for (index, cell) in cells.iter().enumerate() {
                    if index > 0 {
                        line.push(b'\t');
                    }
                    match *cell {
                        rows::Cell::Null => line.extend_from_slice(b"\\N"),
                        rows::Cell::Integer(value) => line.extend_from_slice(value.to_string().as_bytes()),
                        rows::Cell::Real(value) => line.extend_from_slice(format!("{value:?}").as_bytes()),
                        rows::Cell::Text(value) => push_escaped(&mut line, value),
                    }
                }
                line.push(b'\n');
                writer.write_all(&line).map_err(|e| e.to_string())
            })?;
            writer.finish().map_err(|e| format!("{}: {e}", table.name))?;
        }
        self.client.batch_execute("VACUUM ANALYZE").map_err(|e| e.to_string())
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        let start = Instant::now();
        let messages = self
            .client
            .simple_query(&query.sql)
            .map_err(|e| match e.as_db_error() {
                Some(error) if error.code().code() == "42883" => unsupported(error.message()),
                _ => e.to_string(),
            })?;
        let executed = Executed::since(start);
        for message in &messages {
            if let SimpleQueryMessage::Row(row) = message {
                for column in 0..row.len() {
                    match row.get(column) {
                        Some(text) => answer.push_str(text),
                        None => answer.push_null(),
                    }
                }
                answer.end_row();
            }
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

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    let settings = [
        ("Shared buffers", cli.shared_buffers.clone()),
        ("Startup time limit", tableval::spell_duration(cli.startup_time_limit)),
    ];
    run(
        cli.common,
        json!({"image": IMAGE, "shared_buffers": cli.shared_buffers}),
        &settings,
        |configuration, path| {
            Ok(Box::new_in(
                PostgresBackend::open(configuration, path, &cli.shared_buffers, cli.startup_time_limit)?,
                System,
            ))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(
        configuration: &tableval::Configuration,
        path: &std::path::Path,
    ) -> tableval::Result<Box<dyn tableval::Backend, System>> {
        Ok(Box::new_in(
            PostgresBackend::open(configuration, path, "256MB", Duration::from_secs(120))?,
            System,
        ))
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
