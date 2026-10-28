//! Canonical query answers: every engine's rows reduced to one digest that other engines can match.
//!
//! Numbers of every type round to eight significant digits and print as integers when integral, so a
//! `COUNT(*)` returned as `BIGINT`, `DOUBLE` or `DECIMAL`, and a sum accumulated in a different order,
//! all land on the same text. Strings that read as numbers take the same path, and timestamps print with
//! a space between date and time and no trailing fractional zeros. Row order is not part of the digest,
//! so queries without a total `ORDER BY` still compare.

use arrow::{
    array::{Array, RecordBatch},
    compute::cast,
    datatypes::DataType,
    util::display::{ArrayFormatter, FormatOptions},
};

use crate::Result;

/// Significant digits kept from every number.
const SIGNIFICANT_DIGITS: usize = 8;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(hash, |hash, &byte| (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME))
}

/// Result rows reduced to one hash per row; the cells of the row being written are kept until `end_row`.
#[derive(Default)]
pub struct Answer {
    row: Vec<u8>,
    row_hashes: Vec<u64>,
    number: String,
}

impl Answer {
    pub fn clear(&mut self) {
        self.row.clear();
        self.row_hashes.clear();
    }

    pub fn rows(&self) -> u64 {
        self.row_hashes.len() as u64
    }

    pub fn push_i64(&mut self, value: i64) {
        self.push_f64(value as f64);
    }

    pub fn push_f64(&mut self, value: f64) {
        use std::fmt::Write as _;

        self.number.clear();
        let _ = write!(self.number, "{value:.precision$e}", precision = SIGNIFICANT_DIGITS - 1);
        let rounded: f64 = self.number.parse().unwrap_or(value);
        if rounded.is_finite() && rounded.fract() == 0.0 && rounded.abs() < 1e15 {
            self.number.clear();
            let _ = write!(self.number, "{}", rounded as i64);
        }
        self.row.extend_from_slice(self.number.as_bytes());
        self.row.push(0x1f);
    }

    pub fn push_str(&mut self, value: &str) {
        if let Ok(number) = value.trim().parse::<f64>() {
            return self.push_f64(number);
        }
        let bytes = value.as_bytes();
        let datetime = bytes.len() >= 19 && bytes[4] == b'-' && matches!(bytes[10], b'T' | b' ') && bytes[13] == b':';
        if !datetime {
            self.row.extend_from_slice(bytes);
            self.row.push(0x1f);
            return;
        }
        let mut text = value;
        if text.contains('.') {
            text = text.trim_end_matches('0').trim_end_matches('.');
        }
        self.row.extend_from_slice(&text.as_bytes()[..10]);
        self.row.push(b' ');
        self.row.extend_from_slice(&text.as_bytes()[11..]);
        self.row.push(0x1f);
    }

    pub fn push_null(&mut self) {
        self.row.extend_from_slice(b"\0null");
        self.row.push(0x1f);
    }

    pub fn end_row(&mut self) {
        self.row_hashes.push(fnv(FNV_OFFSET, &self.row));
        self.row.clear();
    }

    /// The order-independent digest; with `ties`, only the row count, since tied rows under a `LIMIT` may
    /// legally differ between engines.
    pub fn digest(&mut self, ties: bool) -> u64 {
        if ties {
            return fnv(FNV_OFFSET, &self.rows().to_le_bytes());
        }
        self.row_hashes.sort_unstable();
        self.row_hashes
            .iter()
            .fold(FNV_OFFSET, |hash, row| fnv(hash, &row.to_le_bytes()))
    }

    /// Reads an answer file: a header line, then one row per line with cells split by `separator`.
    pub fn from_text(text: &str, separator: char) -> Self {
        let mut answer = Self::default();
        for line in text.lines().skip(1).filter(|line| !line.is_empty()) {
            for cell in line.split(separator) {
                match cell {
                    "NULL" => answer.push_null(),
                    _ => answer.push_str(cell),
                }
            }
            answer.end_row();
        }
        answer
    }

    /// Appends Arrow result batches; numeric columns are cast to `f64`, others go through Arrow's formatter.
    pub fn push_batches(&mut self, batches: &[RecordBatch]) -> Result<()> {
        let options = FormatOptions::default();
        for batch in batches {
            let mut columns = Vec::with_capacity(batch.num_columns());
            for column in batch.columns() {
                let data_type = column.data_type();
                let numeric = data_type.is_numeric() || *data_type == DataType::Boolean;
                columns.push(match numeric {
                    true => Column::Number(cast(column, &DataType::Float64).map_err(|e| e.to_string())?),
                    false => Column::Text(column.clone()),
                });
            }
            let formatters = columns
                .iter()
                .map(|column| match column {
                    Column::Number(_) => Ok(None),
                    Column::Text(array) => ArrayFormatter::try_new(array.as_ref(), &options).map(Some),
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            for row in 0..batch.num_rows() {
                for (column, formatter) in columns.iter().zip(&formatters) {
                    match column {
                        Column::Number(array) if array.is_null(row) => self.push_null(),
                        Column::Number(array) => self.push_f64(
                            array
                                .as_any()
                                .downcast_ref::<arrow::array::Float64Array>()
                                .unwrap()
                                .value(row),
                        ),
                        Column::Text(array) if array.is_null(row) => self.push_null(),
                        Column::Text(_) => self.push_str(&formatter.as_ref().unwrap().value(row).to_string()),
                    }
                }
                self.end_row();
            }
        }
        Ok(())
    }
}

enum Column {
    Number(arrow::array::ArrayRef),
    Text(arrow::array::ArrayRef),
}

/// Moves an array from another arrow-rs major into this crate's through the Arrow C Data Interface, zero-copy.
///
/// # Safety
///
/// `array` and `schema` must be a valid exported pair, such as `to_ffi` of any arrow-rs version returns.
pub unsafe fn import_array(
    array: arrow::ffi::FFI_ArrowArray,
    schema: &arrow::ffi::FFI_ArrowSchema,
) -> Result<arrow::array::ArrayRef> {
    let data = unsafe { arrow::ffi::from_ffi(array, schema) }.map_err(|e| e.to_string())?;
    Ok(arrow::array::make_array(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_match_across_types_and_rounding() {
        let mut integer = Answer::default();
        integer.push_i64(1478493);
        integer.push_str("13");
        integer.push_f64(56586554400.73);
        integer.end_row();
        let mut float = Answer::default();
        float.push_f64(1478493.0000000002);
        float.push_i64(13);
        float.push_str("56586554400.7301");
        float.end_row();
        assert_eq!(integer.digest(false), float.digest(false));
    }

    #[test]
    fn timestamps_and_row_order_do_not_matter() {
        let mut first = Answer::default();
        first.push_str("2013-07-15T12:34:00");
        first.end_row();
        first.push_str("b");
        first.end_row();
        let mut second = Answer::default();
        second.push_str("b");
        second.end_row();
        second.push_str("2013-07-15 12:34:00.000");
        second.end_row();
        assert_eq!(first.digest(false), second.digest(false));
    }

    #[test]
    fn strings_and_nulls_differ() {
        let mut text = Answer::default();
        text.push_str("null");
        text.end_row();
        let mut null = Answer::default();
        null.push_null();
        null.end_row();
        assert_ne!(text.digest(false), null.digest(false));
    }
}
