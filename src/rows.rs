//! Parquet parts read back as typed rows, for the row stores that ingest through SQL rather than Arrow.

use std::fs::File;

use arrow::{
    array::{Array, AsArray, RecordBatch},
    compute::cast,
    datatypes::{DataType, Float64Type, Int64Type},
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::{backend::Table, Result};

/// Rows decoded per batch while streaming a part.
const BATCH_ROWS: usize = 64 * 1024;

/// The column types a row store declares; dates and timestamps travel as ISO text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Integer,
    Real,
    Text,
    Date,
    Timestamp,
}

impl Kind {
    fn of(data_type: &DataType) -> Self {
        match data_type {
            DataType::Date32 | DataType::Date64 => Self::Date,
            DataType::Timestamp(_, _) => Self::Timestamp,
            DataType::Boolean => Self::Integer,
            data_type if data_type.is_integer() => Self::Integer,
            data_type if data_type.is_numeric() => Self::Real,
            _ => Self::Text,
        }
    }
}

/// One cell of a decoded row.
#[derive(Clone, Copy, Debug)]
pub enum Cell<'a> {
    Null,
    Integer(i64),
    Real(f64),
    Text(&'a str),
}

/// The table's column names and kinds, from its first part's schema.
pub fn columns(table: &Table) -> Result<Vec<(String, Kind)>> {
    let part = table
        .parts
        .first()
        .ok_or_else(|| format!("{} has no parts", table.name))?;
    let file = File::open(part).map_err(|e| format!("{}: {e}", part.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    Ok(builder
        .schema()
        .fields()
        .iter()
        .map(|field| (field.name().clone(), Kind::of(field.data_type())))
        .collect())
}

/// Calls `each` once per row of every part, in order, with cells in column order.
pub fn for_each_row(table: &Table, mut each: impl FnMut(&[Cell<'_>]) -> Result<()>) -> Result<()> {
    for part in &table.parts {
        let file = File::open(part).map_err(|e| format!("{}: {e}", part.display()))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .and_then(|builder| builder.with_batch_size(BATCH_ROWS).build())
            .map_err(|e| format!("{}: {e}", part.display()))?;
        for batch in reader {
            let batch = batch.map_err(|e| e.to_string())?;
            let columns = normalize(&batch)?;
            let mut cells = Vec::with_capacity(columns.len());
            for row in 0..batch.num_rows() {
                cells.clear();
                for (kind, column) in &columns {
                    cells.push(match (kind, column.is_null(row)) {
                        (_, true) => Cell::Null,
                        (Kind::Integer, _) => Cell::Integer(column.as_primitive::<Int64Type>().value(row)),
                        (Kind::Real, _) => Cell::Real(column.as_primitive::<Float64Type>().value(row)),
                        _ => Cell::Text(column.as_string::<i32>().value(row)),
                    });
                }
                each(&cells)?;
            }
        }
    }
    Ok(())
}

/// Casts every column to `i64`, `f64` or text, with timestamps spelled `YYYY-MM-DD HH:MM:SS`.
fn normalize(batch: &RecordBatch) -> Result<Vec<(Kind, arrow::array::ArrayRef)>> {
    batch
        .columns()
        .iter()
        .map(|column| {
            let kind = Kind::of(column.data_type());
            let target = match kind {
                Kind::Integer => DataType::Int64,
                Kind::Real => DataType::Float64,
                _ => DataType::Utf8,
            };
            let mut array = cast(column, &target).map_err(|e| e.to_string())?;
            if kind == Kind::Timestamp {
                // Arrow spells timestamps with a `T`; row stores parse the space-separated form everywhere.
                let text: arrow::array::StringArray = array
                    .as_string::<i32>()
                    .iter()
                    .map(|value| value.map(|value| value.replacen('T', " ", 1)))
                    .collect();
                array = std::sync::Arc::new(text);
            }
            Ok((kind, array))
        })
        .collect()
}
