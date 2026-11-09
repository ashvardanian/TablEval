//! A query-engine-free floor: the four Taxi queries and TPC-H Q1 and Q6, hand-written over arrow-rs.
//!
//! The parts are decoded into Arrow batches once, during the load, and the queries run the folds of `folds.rs`
//! over them. Every other query reports `unsupported`.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features arrow-backend --bin tabl-eval-arrow -- \
//!     --dataset data/taxi-1b --rows 100M --threads 0
//! ```

use std::{alloc::System, collections::BTreeMap, fs::File};

use arrow::{
    array::RecordBatch,
    compute::cast,
    datatypes::{DataType, Decimal128Type},
};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{json, Value};

use tableval::{
    folds::Folds, run, unsupported, Answer, Backend, CommonArgs, Configuration, Executed, Query, Result, Source, Table,
};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

struct ArrowBackend {
    folds: Folds,
}

impl ArrowBackend {
    fn open(configuration: &Configuration) -> Result<Self> {
        Ok(Self {
            folds: Folds::open(configuration.threads.get())?,
        })
    }
}

impl Backend for ArrowBackend {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("backend".into(), json!("arrow")),
            ("library_version".into(), json!(env!("CARGO_PKG_VERSION"))),
            ("threads".into(), json!(self.folds.threads_count())),
        ])
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        if source == Source::Parquet {
            return Err(unsupported(
                "the Arrow baseline decodes the parts once, during the load",
            ));
        }
        for table in tables {
            let mut batches = Vec::new();
            for part in &table.parts {
                let file = File::open(part).map_err(|e| format!("{}: {e}", part.display()))?;
                let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                    .and_then(|builder| builder.build())
                    .map_err(|e| format!("{}: {e}", part.display()))?;
                for batch in reader {
                    batches.push(utf8_strings(batch.map_err(|e| e.to_string())?)?);
                }
            }
            self.folds.tables.insert(table.name.clone(), batches);
        }
        Ok(())
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        self.folds.run::<Decimal128Type>(query, answer)
    }
}

/// Casts view and large strings to plain `Utf8`, so the queries read one string layout.
fn utf8_strings(batch: RecordBatch) -> Result<RecordBatch> {
    let needs_cast = |data_type: &DataType| matches!(data_type, DataType::Utf8View | DataType::LargeUtf8);
    if !batch
        .schema()
        .fields()
        .iter()
        .any(|field| needs_cast(field.data_type()))
    {
        return Ok(batch);
    }
    let columns = batch
        .columns()
        .iter()
        .map(|column| match needs_cast(column.data_type()) {
            true => cast(column, &DataType::Utf8),
            false => Ok(column.clone()),
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let fields: Vec<_> = batch
        .schema()
        .fields()
        .iter()
        .zip(&columns)
        .map(|(field, column)| field.as_ref().clone().with_data_type(column.data_type().clone()))
        .collect();
    RecordBatch::try_new(std::sync::Arc::new(arrow::datatypes::Schema::new(fields)), columns).map_err(|e| e.to_string())
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    run(cli.common, json!({}), &[], |configuration, _path| {
        Ok(Box::new_in(ArrowBackend::open(configuration)?, System))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(
        configuration: &tableval::Configuration,
        _path: &std::path::Path,
    ) -> tableval::Result<Box<dyn tableval::Backend, System>> {
        Ok(Box::new_in(ArrowBackend::open(configuration)?, System))
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
