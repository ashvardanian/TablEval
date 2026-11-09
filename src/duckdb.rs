//! DuckDB through `duckdb-rs`, bundled, with its native database file or views over the Parquet parts.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --bin tabl-eval-duckdb -- \
//!     --dataset data/taxi-1b --rows 100M,1G --source loaded,parquet
//! ```

use std::{alloc::System, collections::BTreeMap, path::Path, time::Instant};

use clap::Parser;
use duckdb::Connection;
use serde_json::{json, Value};

use tableval::{output, run, Answer, Backend, CommonArgs, Configuration, Executed, Query, Result, Source, Table};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

struct DuckdbBackend {
    connection: Connection,
    version: String,
    threads: usize,
    path: std::path::PathBuf,
}

impl DuckdbBackend {
    fn open(configuration: &Configuration, path: &Path) -> Result<Self> {
        let threads = configuration.threads.get();
        let connection = connect(path, threads)?;
        let version = connection
            .query_row("SELECT version()", [], |row| row.get::<_, String>(0))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            connection,
            version,
            threads,
            path: path.to_path_buf(),
        })
    }

    /// A setting as DuckDB applied it, like `409.5 MiB`.
    fn setting(&self, name: &str) -> Option<String> {
        self.connection
            .query_row(&format!("SELECT current_setting('{name}')::VARCHAR"), [], |row| {
                row.get(0)
            })
            .ok()
    }
}

/// A connection to the database file on `threads` threads, with DuckDB's own 80% rule applied to the budget, which
/// `--memory-limit` may set below the cgroup DuckDB reads by itself.
fn connect(path: &Path, threads: usize) -> Result<Connection> {
    let connection = Connection::open(path.join("tableval.duckdb")).map_err(|e| e.to_string())?;
    let memory_limit = output::budget().memory_limit * 8 / 10;
    connection
        .execute_batch(&format!(
            "SET threads = {threads}; SET memory_limit = '{memory_limit} bytes';"
        ))
        .map_err(|e| e.to_string())?;
    Ok(connection)
}

/// `['a.parquet', 'b.parquet']`, DuckDB's list literal of the parts.
fn part_list(table: &Table) -> String {
    let quoted: Vec<String> = table
        .parts
        .iter()
        .map(|part| format!("'{}'", part.display().to_string().replace('\'', "''")))
        .collect();
    format!("[{}]", quoted.join(", "))
}

impl Backend for DuckdbBackend {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("backend".into(), json!("duckdb")),
            ("library_version".into(), json!(self.version)),
            ("threads".into(), json!(self.setting("threads"))),
            ("memory_limit".into(), json!(self.setting("memory_limit"))),
        ])
    }

    fn reopen(&mut self) -> Result<()> {
        let placeholder = Connection::open_in_memory().map_err(|e| e.to_string())?;
        std::mem::replace(&mut self.connection, placeholder)
            .close()
            .map_err(|(_, e)| e.to_string())?;
        self.connection = connect(&self.path, self.threads)?;
        Ok(())
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        let kind = match source {
            Source::Loaded => "TABLE",
            Source::Parquet => "VIEW",
        };
        for table in tables {
            self.connection
                .execute_batch(&format!(
                    "CREATE {kind} {} AS SELECT * FROM read_parquet({})",
                    table.name,
                    part_list(table)
                ))
                .map_err(|e| format!("{}: {e}", table.name))?;
        }
        self.connection.execute_batch("CHECKPOINT").map_err(|e| e.to_string())
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        let start = Instant::now();
        let mut statement = self.connection.prepare(&query.sql).map_err(|e| e.to_string())?;
        let batches: Vec<_> = statement.query_arrow([]).map_err(|e| e.to_string())?.collect();
        let executed = Executed::since(start);
        answer.push_batches(&batches)?;
        Ok(executed)
    }

    fn disk_bytes(&self) -> Result<u64> {
        tableval::backend::directory_bytes(&self.path)
    }
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    run(cli.common, json!({}), &[], |configuration, path| {
        Ok(Box::new_in(DuckdbBackend::open(configuration, path)?, System))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(
        configuration: &tableval::Configuration,
        path: &std::path::Path,
    ) -> tableval::Result<Box<dyn tableval::Backend, System>> {
        Ok(Box::new_in(DuckdbBackend::open(configuration, path)?, System))
    }

    #[test]
    #[ignore = "spawns systemd-run --user"]
    fn budget_reaches_duckdb() {
        if std::env::var_os("TABLEVAL_BUDGET_SCOPE").is_none() {
            let output = std::process::Command::new("systemd-run")
                .args(["--user", "--scope", "--quiet", "--collect"])
                .arg(format!("--unit=tableval-budget-{}", std::process::id()))
                .args(["-p", "MemoryMax=512M", "-p", "MemorySwapMax=0", "taskset", "-c", "0-1"])
                .arg(std::env::current_exe().unwrap())
                .args(["--exact", "tests::budget_reaches_duckdb", "--include-ignored"])
                .env("TABLEVAL_BUDGET_SCOPE", "1")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                stdout.contains("1 passed"),
                "{stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let budget = output::budget();
        assert_eq!((budget.cores, budget.memory_limit), (2, 512 << 20));
        let directory = tempfile::tempdir().unwrap();
        let configuration = Configuration {
            rows: None,
            threads: tableval::Threads(std::num::NonZeroUsize::new(budget.cores).unwrap()),
            source: Source::Loaded,
        };
        let mut backend = DuckdbBackend::open(&configuration, directory.path()).unwrap();
        backend.reopen().unwrap();
        let expected: String = backend
            .connection
            .query_row(
                &format!("SELECT format_bytes({})", (512u64 << 20) * 8 / 10),
                [],
                |row| row.get(0),
            )
            .unwrap();
        let metadata = backend.metadata();
        assert_eq!(metadata["memory_limit"], json!(expected));
        assert_eq!(metadata["threads"], json!("2"));
    }

    #[test]
    fn taxi_fixture() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::assert_taxi_answers(root.path(), open);
    }

    #[cfg(feature = "generate")]
    #[test]
    fn tpch_answers() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::assert_tpch_answers(root.path(), open);
    }
}
