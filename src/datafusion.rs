//! Apache DataFusion through its SQL frontend, over in-memory tables or the Parquet parts in place.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features datafusion-backend --bin tabl-eval-datafusion -- \
//!     --dataset data/tpch-sf10 --source loaded,parquet
//! ```

use std::{alloc::System, collections::BTreeMap, sync::Arc, time::Instant};

use arrow::array::{AsArray, RecordBatch};
use clap::Parser;
use datafusion::{
    catalog::MemTable,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use serde_json::{json, Value};
use tokio::runtime::Runtime;

use tableval::{run, Answer, Backend, CommonArgs, Configuration, Executed, Query, Result, Source, Table};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

struct DatafusionBackend {
    context: SessionContext,
    runtime: Runtime,
    threads: usize,
}

impl DatafusionBackend {
    fn open(configuration: &Configuration) -> Result<Self> {
        let threads = configuration.threads.get();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        // ClickBench spells columns like `CounterID` unquoted, so identifiers keep their case.
        let config = SessionConfig::new()
            .with_target_partitions(threads)
            .set_bool("datafusion.sql_parser.enable_ident_normalization", false);
        Ok(Self {
            context: SessionContext::new_with_config(config),
            runtime,
            threads,
        })
    }
}

/// Moves a DataFusion batch, built on DataFusion's Arrow major, into this crate's Arrow, zero-copy.
fn import(batch: datafusion::arrow::array::RecordBatch) -> Result<RecordBatch> {
    let data = datafusion_arrow::array::Array::into_data(datafusion_arrow::array::StructArray::from(batch));
    let (array, schema) = datafusion_arrow::ffi::to_ffi(&data).map_err(|e| e.to_string())?;
    // SAFETY: both pairs are the Arrow C Data Interface's `#[repr(C)]` structs, laid out alike in every major.
    let imported = unsafe {
        let array = std::mem::transmute::<datafusion_arrow::ffi::FFI_ArrowArray, arrow::ffi::FFI_ArrowArray>(array);
        let schema = std::mem::transmute::<datafusion_arrow::ffi::FFI_ArrowSchema, arrow::ffi::FFI_ArrowSchema>(schema);
        tableval::answer::import_array(array, &schema)?
    };
    Ok(RecordBatch::from(imported.as_struct()))
}

impl Backend for DatafusionBackend {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("backend".into(), json!("datafusion")),
            ("library_version".into(), json!(datafusion::DATAFUSION_VERSION)),
            ("threads".into(), json!(self.threads)),
            ("target_partitions".into(), json!(self.threads)),
        ])
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        for table in tables {
            let parts: Vec<String> = table.parts.iter().map(|part| part.display().to_string()).collect();
            let context = &self.context;
            let provider = self.runtime.block_on(async {
                let frame = context.read_parquet(parts, ParquetReadOptions::default()).await?;
                match source {
                    Source::Parquet => Ok(frame.into_view()),
                    Source::Loaded => {
                        let schema = Arc::new(frame.schema().as_arrow().clone());
                        let partitions = frame.collect_partitioned().await?;
                        Ok(Arc::new(MemTable::try_new(schema, partitions)?) as _)
                    }
                }
            });
            let provider = provider.map_err(|e: datafusion::error::DataFusionError| format!("{}: {e}", table.name))?;
            self.context
                .register_table(table.name.as_str(), provider)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        let start = Instant::now();
        let context = &self.context;
        let batches = self
            .runtime
            .block_on(async { context.sql(&query.sql).await?.collect().await })
            .map_err(|e| e.to_string())?;
        let executed = Executed::since(start);
        let batches = batches.into_iter().map(import).collect::<Result<Vec<_>>>()?;
        answer.push_batches(&batches)?;
        Ok(executed)
    }
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    run(cli.common, json!({}), &[], |configuration, _path| {
        Ok(Box::new_in(DatafusionBackend::open(configuration)?, System))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(
        configuration: &tableval::Configuration,
        _path: &std::path::Path,
    ) -> tableval::Result<Box<dyn tableval::Backend, System>> {
        Ok(Box::new_in(DatafusionBackend::open(configuration)?, System))
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
