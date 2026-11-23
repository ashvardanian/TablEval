//! Canonical query answers: every engine's rows reduced to one digest that other engines can match.
//!
//! Numbers of every type round to eight significant digits and print as integers when integral, so a
//! `COUNT(*)` returned as `BIGINT`, `DOUBLE` or `DECIMAL`, and a sum accumulated in a different order,
//! all land on the same text. Strings that read as numbers take the same path, and timestamps print with
//! a space between date and time and no trailing fractional zeros. Row order is not part of the digest,
//! so queries without a total `ORDER BY` still compare.

use arrow::{
    array::{Array, ArrayAccessor, ArrayRef, AsArray, Float64Array, RecordBatch},
    compute::cast,
    datatypes::{DataType, Float64Type},
    error::ArrowError,
    util::display::{ArrayFormatter, FormatOptions},
};

use crate::Result;

/// Significant digits kept from every number.
const SIGNIFICANT_DIGITS: usize = 8;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// The two lowercase hex digits of every byte, as Arrow's formatter prints binary cells.
const HEX_PAIRS: [[u8; 2]; 256] = {
    let digits = b"0123456789abcdef";
    let mut pairs = [[0; 2]; 256];
    let mut byte = 0;
    while byte < 256 {
        pairs[byte] = [digits[byte >> 4], digits[byte & 0xf]];
        byte += 1;
    }
    pairs
};

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

    /// Appends Arrow result batches; numeric columns are cast to `f64`, strings and binary hex are pushed as
    /// Arrow's formatter would print them, and other columns go through that formatter.
    pub fn push_batches(&mut self, batches: &[RecordBatch]) -> Result<()> {
        use std::fmt::Write as _;

        let options = FormatOptions::default();
        let mut text = String::new();
        for batch in batches {
            let casts = batch
                .columns()
                .iter()
                .map(|column| {
                    let data_type = column.data_type();
                    match data_type.is_numeric() || *data_type == DataType::Boolean {
                        true => cast(column, &DataType::Float64).map(Some),
                        false => Ok(None),
                    }
                })
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            let sources: Vec<&ArrayRef> = batch
                .columns()
                .iter()
                .zip(&casts)
                .map(|(column, cast)| cast.as_ref().unwrap_or(column))
                .collect();
            let columns = sources
                .iter()
                .map(|source| Column::new(source, &options))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            for row in 0..batch.num_rows() {
                for (source, column) in sources.iter().zip(&columns) {
                    if source.is_null(row) {
                        self.push_null();
                        continue;
                    }
                    match column {
                        Column::Number(array) => self.push_f64(array.value(row)),
                        Column::Text(array) => self.push_str(array.value(row)),
                        Column::Bytes(array) => {
                            text.clear();
                            for &byte in array.value(row) {
                                text.extend(HEX_PAIRS[usize::from(byte)].map(char::from));
                            }
                            self.push_str(&text);
                        }
                        Column::Formatted(formatter) => {
                            text.clear();
                            let _ = write!(text, "{}", formatter.value(row));
                            self.push_str(&text);
                        }
                    }
                }
                self.end_row();
            }
        }
        Ok(())
    }
}

/// A batch column typed once, so that strings and binary cells skip Arrow's per-cell formatter.
enum Column<'a> {
    Number(&'a Float64Array),
    Text(Box<dyn ArrayAccessor<Item = &'a str> + 'a>),
    Bytes(Box<dyn ArrayAccessor<Item = &'a [u8]> + 'a>),
    Formatted(ArrayFormatter<'a>),
}

impl<'a> Column<'a> {
    /// Numeric columns arrive here already cast to `Float64`, so no other column type takes that arm.
    fn new(array: &'a ArrayRef, options: &FormatOptions<'a>) -> std::result::Result<Self, ArrowError> {
        Ok(match array.data_type() {
            DataType::Float64 => Self::Number(array.as_primitive::<Float64Type>()),
            DataType::Utf8 => Self::Text(Box::new(array.as_string::<i32>())),
            DataType::LargeUtf8 => Self::Text(Box::new(array.as_string::<i64>())),
            DataType::Utf8View => Self::Text(Box::new(array.as_string_view())),
            DataType::Binary => Self::Bytes(Box::new(array.as_binary::<i32>())),
            DataType::LargeBinary => Self::Bytes(Box::new(array.as_binary::<i64>())),
            DataType::BinaryView => Self::Bytes(Box::new(array.as_binary_view())),
            DataType::FixedSizeBinary(_) => Self::Bytes(Box::new(array.as_fixed_size_binary())),
            _ => Self::Formatted(ArrayFormatter::try_new(array.as_ref(), options)?),
        })
    }
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
    use std::sync::Arc;

    use arrow::{
        array::{
            BinaryArray, BinaryViewArray, Date32Array, DictionaryArray, FixedSizeBinaryArray, Int64Array,
            LargeBinaryArray, LargeStringArray, StringArray, StringViewArray, TimestampMillisecondArray,
        },
        datatypes::Int32Type,
    };

    use super::*;

    /// The formatter path `push_batches` replaced, kept as the reference its row hashes must match.
    fn push_batches_through_formatter(answer: &mut Answer, batches: &[RecordBatch]) {
        let options = FormatOptions::default();
        for batch in batches {
            let columns: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .map(|column| {
                    let data_type = column.data_type();
                    match data_type.is_numeric() || *data_type == DataType::Boolean {
                        true => cast(column, &DataType::Float64).unwrap(),
                        false => column.clone(),
                    }
                })
                .collect();
            let formatters: Vec<ArrayFormatter> = columns
                .iter()
                .map(|column| ArrayFormatter::try_new(column.as_ref(), &options).unwrap())
                .collect();
            for row in 0..batch.num_rows() {
                for (column, formatter) in columns.iter().zip(&formatters) {
                    match column.as_primitive_opt::<Float64Type>() {
                        _ if column.is_null(row) => answer.push_null(),
                        Some(numbers) => answer.push_f64(numbers.value(row)),
                        None => answer.push_str(&formatter.value(row).to_string()),
                    }
                }
                answer.end_row();
            }
        }
    }

    /// A xorshift stream, so the random arrays repeat from run to run.
    struct Draws(u64);

    impl Draws {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: u64) -> usize {
            (self.next() % bound) as usize
        }
    }

    /// Text that `push_str` reads as a number, as a timestamp, or as plain bytes.
    fn random_text(draws: &mut Draws) -> String {
        const PIECES: [&str; 10] = ["0", "7", "e", "-", ".", " ", "T", ":", "\u{e9}", "x"];
        match draws.below(4) {
            0 => format!("{}", (draws.next() >> 20) as f64 / 1000.0),
            1 => format!(
                "2013-07-{:02}T12:34:{:02}.{}",
                draws.below(28) + 1,
                draws.below(60),
                draws.below(1000)
            ),
            _ => (0..draws.below(24)).map(|_| PIECES[draws.below(10)]).collect(),
        }
    }

    /// Bytes whose hex is sometimes all decimal digits, which `push_str` then reads as a number.
    fn random_bytes(draws: &mut Draws, length: usize) -> Vec<u8> {
        let decimal = draws.below(4) == 0;
        (0..length)
            .map(|_| match decimal {
                true => (draws.below(10) << 4 | draws.below(10)) as u8,
                false => draws.next() as u8,
            })
            .collect()
    }

    fn random_batch(draws: &mut Draws, rows: usize) -> RecordBatch {
        let texts: Vec<Option<String>> = (0..rows)
            .map(|_| (draws.below(8) != 0).then(|| random_text(draws)))
            .collect();
        let blobs: Vec<Option<Vec<u8>>> = (0..rows)
            .map(|_| {
                let length = draws.below(20);
                (draws.below(8) != 0).then(|| random_bytes(draws, length))
            })
            .collect();
        let keys: Vec<Option<Vec<u8>>> = (0..rows)
            .map(|_| (draws.below(8) != 0).then(|| random_bytes(draws, 16)))
            .collect();
        let milliseconds: Vec<Option<i64>> = (0..rows)
            .map(|_| (draws.below(8) != 0).then(|| (draws.next() >> 24) as i64 - (1 << 39)))
            .collect();
        let days: Vec<Option<i32>> = milliseconds
            .iter()
            .map(|value| value.map(|value| (value / 86_400_000) as i32))
            .collect();
        let texts: Vec<Option<&str>> = texts.iter().map(Option::as_deref).collect();
        let blobs: Vec<Option<&[u8]>> = blobs.iter().map(Option::as_deref).collect();
        let columns: [(&str, ArrayRef); 11] = [
            ("utf8", Arc::new(StringArray::from(texts.clone()))),
            ("large_utf8", Arc::new(LargeStringArray::from(texts.clone()))),
            ("utf8_view", Arc::new(StringViewArray::from(texts.clone()))),
            (
                "dictionary",
                Arc::new(texts.into_iter().collect::<DictionaryArray<Int32Type>>()),
            ),
            ("binary", Arc::new(BinaryArray::from(blobs.clone()))),
            ("large_binary", Arc::new(LargeBinaryArray::from(blobs.clone()))),
            ("binary_view", Arc::new(BinaryViewArray::from(blobs))),
            (
                "fixed_size_binary",
                Arc::new(FixedSizeBinaryArray::try_from_sparse_iter_with_size(keys.into_iter(), 16).unwrap()),
            ),
            ("integer", Arc::new(Int64Array::from(milliseconds.clone()))),
            ("timestamp", Arc::new(TimestampMillisecondArray::from(milliseconds))),
            ("date", Arc::new(Date32Array::from(days))),
        ];
        RecordBatch::try_from_iter(columns).unwrap()
    }

    #[test]
    fn typed_columns_hash_as_the_formatter_did() {
        let mut draws = Draws(0x9e37_79b9_7f4a_7c15);
        for _ in 0..8 {
            let batches = [random_batch(&mut draws, 500), random_batch(&mut draws, 300)];
            let mut typed = Answer::default();
            typed.push_batches(&batches).unwrap();
            let mut formatted = Answer::default();
            push_batches_through_formatter(&mut formatted, &batches);
            assert_eq!(typed.row_hashes, formatted.row_hashes);
        }
    }

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
