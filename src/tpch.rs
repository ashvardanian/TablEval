//! TPC-H tables written as Parquet parts through `tpchgen-arrow`, which matches `dbgen` row for row.

use arrow::{
    array::{RecordBatch, RecordBatchReader},
    compute::cast,
    datatypes::{DataType, Field, Schema},
};
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use tpchgen::generators::{
    CustomerGenerator, LineItemGenerator, NationGenerator, OrderGenerator, PartGenerator, PartSuppGenerator,
    RegionGenerator, SupplierGenerator,
};
use tpchgen_arrow::{
    CustomerArrow, LineItemArrow, NationArrow, OrderArrow, PartArrow, PartSuppArrow, RegionArrow, SupplierArrow,
};

use crate::{suite::TPCH_TABLES, Result};

fn batches(table: &str, scale_factor: f64, part: i32, parts: i32) -> Box<dyn RecordBatchReader + Send> {
    match table {
        "region" => Box::new(RegionArrow::new(RegionGenerator::new(scale_factor, part, parts))),
        "nation" => Box::new(NationArrow::new(NationGenerator::new(scale_factor, part, parts))),
        "supplier" => Box::new(SupplierArrow::new(SupplierGenerator::new(scale_factor, part, parts))),
        "customer" => Box::new(CustomerArrow::new(CustomerGenerator::new(scale_factor, part, parts))),
        "part" => Box::new(PartArrow::new(PartGenerator::new(scale_factor, part, parts))),
        "partsupp" => Box::new(PartSuppArrow::new(PartSuppGenerator::new(scale_factor, part, parts))),
        "orders" => Box::new(OrderArrow::new(OrderGenerator::new(scale_factor, part, parts))),
        _ => Box::new(LineItemArrow::new(LineItemGenerator::new(scale_factor, part, parts))),
    }
}

/// Casts string views to plain `Utf8`.
fn plain_strings(batch: &RecordBatch, schema: &std::sync::Arc<Schema>) -> Result<RecordBatch> {
    let columns = batch
        .columns()
        .iter()
        .zip(schema.fields())
        .map(|(column, field)| cast(column, field.data_type()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    RecordBatch::try_new(schema.clone(), columns).map_err(|e| e.to_string())
}

/// Writes part `part` of `parts`, counted from 1, of `table` as `<table>.part_NNN.parquet`, and returns its rows.
pub fn write_part(folder: &std::path::Path, table: &str, scale_factor: f64, part: i32, parts: i32) -> Result<u64> {
    let reader = batches(table, scale_factor, part, parts);
    let fields: Vec<Field> = reader
        .schema()
        .fields()
        .iter()
        .map(|field| match field.data_type() {
            DataType::Utf8View => field.as_ref().clone().with_data_type(DataType::Utf8),
            _ => field.as_ref().clone(),
        })
        .collect();
    let schema = std::sync::Arc::new(Schema::new(fields));
    let path = folder.join(format!("{table}.part_{:03}.parquet", part - 1));
    let file = std::fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let properties = WriterProperties::builder().set_compression(Compression::SNAPPY).build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(properties)).map_err(|e| e.to_string())?;
    let mut rows = 0;
    for batch in reader {
        let batch = plain_strings(&batch.map_err(|e| e.to_string())?, &schema)?;
        rows += batch.num_rows() as u64;
        writer.write(&batch).map_err(|e| e.to_string())?;
    }
    writer.close().map_err(|e| e.to_string())?;
    Ok(rows)
}

/// Writes every table at `scale_factor` into `folder`, one part each, as tests need.
pub fn write_tables(folder: &std::path::Path, scale_factor: f64) -> Result<()> {
    std::fs::create_dir_all(folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    for table in TPCH_TABLES {
        write_part(folder, table, scale_factor, 1, 1)?;
    }
    Ok(())
}
