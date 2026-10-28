//! Polars through its SQL context, over collected `DataFrame`s or lazy Parquet scans.
//!
//! Polars sizes one global thread pool from `POLARS_MAX_THREADS` when it first runs, so this binary takes
//! a single `--threads` value; sweep threads with one process per value.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features polars-backend --bin tabl-eval-polars -- \
//!     --dataset data/taxi-1b --rows 100M --threads 16 --source loaded,parquet
//! ```

use std::{alloc::System, collections::BTreeMap, time::Instant};

use clap::Parser;
use polars::{
    prelude::{AnyValue, DataFrame, IntoLazy, LazyFrame, PlRefPath, ScanArgsParquet},
    sql::SQLContext,
};
use serde_json::{json, Value};

use tableval::{run, unsupported, Answer, Backend, CommonArgs, Executed, Query, Result, Source, Table};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

struct PolarsBackend {
    context: SQLContext,
    threads: usize,
}

impl Backend for PolarsBackend {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("backend".into(), json!("polars")),
            ("library_version".into(), json!(polars::VERSION)),
            ("threads".into(), json!(self.threads)),
        ])
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        for table in tables {
            let parts = table
                .parts
                .iter()
                .map(|part| PlRefPath::try_from_path(part))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| e.to_string())?;
            let scan = LazyFrame::scan_parquet_files(parts, ScanArgsParquet::default())
                .map_err(|e| format!("{}: {e}", table.name))?;
            let frame = match source {
                Source::Parquet => scan,
                Source::Loaded => scan.collect().map_err(|e| format!("{}: {e}", table.name))?.lazy(),
            };
            self.context.register(&table.name, frame);
        }
        Ok(())
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        let start = Instant::now();
        let frame = self
            .context
            .execute(&query.sql)
            .map_err(|e| unsupported(&e.to_string()))?
            .collect()
            .map_err(|e| e.to_string())?;
        let executed = Executed::since(start);
        push_frame(&frame, answer)?;
        Ok(executed)
    }
}

fn push_frame(frame: &DataFrame, answer: &mut Answer) -> Result<()> {
    for row in 0..frame.height() {
        for column in frame.columns() {
            let value = column.get(row).map_err(|e| e.to_string())?;
            match value {
                AnyValue::Null => answer.push_null(),
                AnyValue::String(text) => answer.push_str(text),
                AnyValue::StringOwned(ref text) => answer.push_str(text),
                ref value if value.dtype().is_primitive_numeric() || value.dtype().is_decimal() => {
                    answer.push_f64(value.extract::<f64>().ok_or("unreadable number")?)
                }
                AnyValue::Boolean(flag) => answer.push_i64(i64::from(flag)),
                value => answer.push_str(&value.to_string()),
            }
        }
        answer.end_row();
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    let [threads] = cli.common.threads[..] else {
        return Err("Polars sizes one global pool per process, so pass one --threads value per run".into());
    };
    // SAFETY: no other thread exists yet, and Polars reads the variable only when its pool starts.
    unsafe { std::env::set_var("POLARS_MAX_THREADS", threads.to_string()) };
    run(cli.common, json!({}), &[], |configuration, _path| {
        Ok(Box::new_in(
            PolarsBackend {
                context: SQLContext::new(),
                threads: configuration.threads.get(),
            },
            System,
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(
        configuration: &tableval::Configuration,
        _path: &std::path::Path,
    ) -> tableval::Result<Box<dyn tableval::Backend, System>> {
        Ok(Box::new_in(
            PolarsBackend {
                context: SQLContext::new(),
                threads: configuration.threads.get(),
            },
            System,
        ))
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
