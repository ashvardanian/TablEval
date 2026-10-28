//! The contract every engine binary implements: load the tables once, then run named queries.

use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use serde_json::Value;

use crate::{answer::Answer, suite::Query};

pub type Result<T> = std::result::Result<T, String>;

/// Prefix of an error that marks a query the engine cannot express, recorded as `unsupported`.
pub const UNSUPPORTED: &str = "unsupported";

/// Where the queries read their rows from.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    /// Ingest the Parquet parts into the engine's own storage first.
    Loaded,
    /// Query the Parquet parts in place.
    Parquet,
}

/// One table of the dataset: its Parquet parts in natural order and their totals from the footers.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Table {
    pub name: String,
    pub parts: Vec<PathBuf>,
    pub rows: u64,
    pub bytes: u64,
}

/// Engine time of one query run, measured around execution and materialization, before the answer is filled.
pub struct Executed {
    pub seconds: f64,
}

impl Executed {
    pub fn since(start: Instant) -> Self {
        Self {
            seconds: start.elapsed().as_secs_f64(),
        }
    }
}

pub trait Backend {
    /// Engine name, version and the settings it applied, merged into the report's `config`.
    fn metadata(&self) -> BTreeMap<String, Value>;
    /// Ingests the tables for `Source::Loaded`, or registers their Parquet parts for `Source::Parquet`.
    fn load(&mut self, tables: &[Table], source: Source) -> Result<()>;
    /// Runs one query to completion and writes its rows into `answer` after the engine has materialized them.
    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed>;
    /// Bytes the engine's own storage occupies after loading.
    fn disk_bytes(&self) -> Result<u64> {
        Ok(0)
    }
    /// A snapshot of the server container's resources; `None` for embedded engines.
    fn server_usage(&self) -> Result<Option<Value>> {
        Ok(None)
    }
    /// The server container's cgroup, sampled alongside the client while each phase runs; `None` for
    /// embedded engines and hosts that do not expose it.
    fn server_cgroup(&self) -> Option<PathBuf> {
        None
    }
}

/// The error a backend returns for a query it cannot express.
pub fn unsupported(reason: &str) -> String {
    format!("{UNSUPPORTED}: {reason}")
}

/// Total size of the files under `path`, for engines that keep their storage in the run's data directory.
pub fn directory_bytes(path: &std::path::Path) -> Result<u64> {
    let mut total = 0;
    for entry in std::fs::read_dir(path).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let metadata = entry.metadata().map_err(|e| e.to_string())?;
        total += match metadata.is_dir() {
            true => directory_bytes(&entry.path())?,
            false => metadata.len(),
        };
    }
    Ok(total)
}
