//! ClickHouse in Docker over its HTTP interface, with MergeTree tables or views over the Parquet parts.
//!
//! The dataset folder is mounted read-only and set as the server's `user_files_path`, so the server reads
//! the parts itself through the `file` table function. Each query runs with `max_threads` set to `--threads` and
//! `join_use_nulls` on, the standard SQL semantics for outer joins.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features clickhouse-backend --bin tabl-eval-clickhouse -- \
//!     --dataset data/clickbench-100m --source loaded,parquet
//! ```

use std::{
    alloc::System,
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant},
};

use clap::Parser;
use serde_json::{json, Value};

use tableval::{
    docker::{ContainerHandle, Startup},
    parse_duration_flag, run, unsupported, Answer, Backend, CommonArgs, Configuration, Executed, Query, Result, Source,
    Table,
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
}

struct ClickhouseBackend {
    container: ContainerHandle,
    agent: ureq::Agent,
    threads: usize,
    version: String,
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
        };
        backend.container.ready(|| backend.post("SELECT 1").map(drop))?;
        backend.version = backend.post("SELECT version()")?.trim().to_owned();
        Ok(backend)
    }

    /// Sends one statement and returns the response body, or the server's error text.
    fn post(&self, sql: &str) -> Result<String> {
        let url = format!(
            "http://127.0.0.1:{}/?max_threads={}&join_use_nulls=1",
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
            };
            self.post(&statement).map_err(|e| format!("{}: {e}", table.name))?;
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

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
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
