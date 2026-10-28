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

use tableval::{run, Answer, Backend, CommonArgs, Configuration, Executed, Query, Result, Source, Table};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    /// Memory DuckDB may use before it spills, as a share like 80% or a size like 64GB.
    #[arg(long, env = "TABLEVAL_MEMORY_LIMIT", default_value = "80%")]
    memory_limit: String,
}

struct DuckdbBackend {
    connection: Connection,
    version: String,
    threads: usize,
    memory_limit: String,
    path: std::path::PathBuf,
}

impl DuckdbBackend {
    fn open(configuration: &Configuration, path: &Path, memory_limit: &str) -> Result<Self> {
        let connection = Connection::open(path.join("tableval.duckdb")).map_err(|e| e.to_string())?;
        let threads = configuration.threads.get();
        connection
            .execute_batch(&format!(
                "SET threads = {threads}; SET memory_limit = '{memory_limit}'; SET preserve_insertion_order = false;"
            ))
            .map_err(|e| e.to_string())?;
        let version = connection
            .query_row("SELECT version()", [], |row| row.get::<_, String>(0))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            connection,
            version,
            threads,
            memory_limit: memory_limit.to_owned(),
            path: path.to_path_buf(),
        })
    }
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
            ("threads".into(), json!(self.threads)),
            ("memory_limit".into(), json!(self.memory_limit)),
        ])
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
    let settings = [("Memory limit", cli.memory_limit.clone())];
    run(
        cli.common,
        json!({"memory_limit": cli.memory_limit}),
        &settings,
        |configuration, path| {
            Ok(Box::new_in(
                DuckdbBackend::open(configuration, path, &cli.memory_limit)?,
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
        Ok(Box::new_in(DuckdbBackend::open(configuration, path, "1GB")?, System))
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
