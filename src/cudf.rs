//! NVIDIA cuDF through the `cudf` crate's bindings to libcudf: the four Taxi queries and TPC-H Q1 and Q6
//! on one GPU.
//!
//! The parts are read straight into device memory by libcudf's Parquet reader and concatenated, so only
//! `--source loaded` applies. Decimal columns are cast to `Float64` during the load, as SQLite and
//! PostgreSQL store them, because the bindings' binary operations cannot carry a decimal scale. Dates are
//! filtered by their year and day of the year, since libcudf neither compares dates with integers nor
//! takes date scalars through the bindings. Each query's result is copied back to the host inside the
//! timed region. libcudf's group-by leaves out rows whose keys are null, where SQL keeps them
//! as one more group, so a dataset with null keys reports a mismatch here. Linux with CUDA only.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features cudf-backend --bin tabl-eval-cudf -- \
//!     --dataset data/taxi-1b --rows 100M
//! ```

use std::{
    alloc::System,
    collections::{BTreeMap, HashMap},
    time::Instant,
};

use clap::Parser;
use cudf::{
    binaryop::binary_op,
    concatenate::concatenate_tables,
    io::parquet::read_parquet,
    types::{DataType, TypeId},
    unary::UnaryOp,
    AggregationKind, BinaryOp, Column, CudfType, GroupBy, ReduceOp, Scalar,
};
use serde_json::{json, Value};

use tableval::{rows, run, unsupported, Answer, Backend, CommonArgs, Executed, Query, Result, Source, Table};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

/// One result column copied to the host.
enum HostColumn {
    Numbers(Vec<Option<f64>>),
    Strings(Vec<Option<String>>),
}

#[derive(Default)]
struct CudfBackend {
    tables: HashMap<String, (cudf::Table, Vec<String>)>,
}

fn gpu(error: cudf::CudfError) -> String {
    error.to_string()
}

impl CudfBackend {
    fn column(&self, table: &str, name: &str) -> Result<Column> {
        let (table, names) = self
            .tables
            .get(table)
            .ok_or_else(|| unsupported(&format!("no `{table}` table")))?;
        let index = names
            .iter()
            .position(|column| column == name)
            .ok_or_else(|| format!("missing column `{name}`"))?;
        table.column(index).map_err(gpu)
    }

    /// The rows of `lineitem` that pass `mask`, as the named columns in order.
    fn lineitem_where(&self, mask: &Column, names: &[&str]) -> Result<Vec<Column>> {
        let columns = names
            .iter()
            .map(|name| self.column("lineitem", name))
            .collect::<Result<Vec<_>>>()?;
        cudf::Table::new(columns)
            .and_then(|table| table.apply_boolean_mask(mask))
            .and_then(cudf::Table::into_columns)
            .map_err(gpu)
    }

    /// TPC-H Q1: lines shipped by 1998-09-02, the 245th day of 1998, summarized per return flag and status.
    fn pricing_summary(&self) -> Result<Vec<HostColumn>> {
        let shipped = self.column("lineitem", "l_shipdate")?;
        let (year, day) = (
            shipped.extract_year().map_err(gpu)?,
            shipped.extract_day_of_year().map_err(gpu)?,
        );
        let in_1998_by_september = combine(
            &compare(&year, BinaryOp::Equal, 1998i16)?,
            BinaryOp::LogicalAnd,
            &compare(&day, BinaryOp::LessEqual, 245i16)?,
        )?;
        let mask = combine(
            &compare(&year, BinaryOp::Less, 1998i16)?,
            BinaryOp::LogicalOr,
            &in_1998_by_september,
        )?;
        let [flag, status, quantity, price, discount, tax]: [Column; 6] = self
            .lineitem_where(
                &mask,
                &[
                    "l_returnflag",
                    "l_linestatus",
                    "l_quantity",
                    "l_extendedprice",
                    "l_discount",
                    "l_tax",
                ],
            )?
            .try_into()
            .map_err(|_| "lineitem lost a column")?;
        let discounted = arithmetic(&price, BinaryOp::Mul, &from_constant(1.0, BinaryOp::Sub, &discount)?)?;
        let charge = arithmetic(&discounted, BinaryOp::Mul, &from_constant(1.0, BinaryOp::Add, &tax)?)?;
        let keys = cudf::Table::new(vec![flag, status]).map_err(gpu)?;
        let copy = |column: &Column| column.cast(column.data_type()).map_err(gpu);
        let values = cudf::Table::new(vec![
            copy(&quantity)?,
            copy(&price)?,
            discounted,
            charge,
            quantity,
            price,
            discount,
        ])
        .map_err(gpu)?;
        to_host(
            GroupBy::new(&keys)
                .agg(0, AggregationKind::Sum)
                .agg(1, AggregationKind::Sum)
                .agg(2, AggregationKind::Sum)
                .agg(3, AggregationKind::Sum)
                .agg(4, AggregationKind::Mean)
                .agg(5, AggregationKind::Mean)
                .agg(6, AggregationKind::Mean)
                .agg(6, AggregationKind::Count)
                .execute(&values)
                .map_err(gpu)?,
        )
    }

    /// TPC-H Q6: revenue from lines shipped in 1994 at a 5-7% discount and under 24 units.
    fn forecast_revenue(&self) -> Result<Vec<HostColumn>> {
        let year = self.column("lineitem", "l_shipdate")?.extract_year().map_err(gpu)?;
        let discount = self.column("lineitem", "l_discount")?;
        let quantity = self.column("lineitem", "l_quantity")?;
        // Both columns hold two decimal places, so midpoints keep the bounds exact after the cast to floats.
        let mask = [
            compare(&discount, BinaryOp::GreaterEqual, 0.045)?,
            compare(&discount, BinaryOp::LessEqual, 0.075)?,
            compare(&quantity, BinaryOp::Less, 23.995)?,
        ]
        .iter()
        .try_fold(compare(&year, BinaryOp::Equal, 1994i16)?, |mask, next| {
            combine(&mask, BinaryOp::LogicalAnd, next)
        })?;
        let [price, discount]: [Column; 2] = self
            .lineitem_where(&mask, &["l_extendedprice", "l_discount"])?
            .try_into()
            .map_err(|_| "lineitem lost a column")?;
        let revenue = arithmetic(&price, BinaryOp::Mul, &discount)?
            .reduce(ReduceOp::Sum, DataType::new(TypeId::Float64))
            .map_err(gpu)?;
        let revenue = match revenue.is_valid() {
            true => Some(revenue.value::<f64>().map_err(gpu)?),
            false => None,
        };
        Ok(vec![HostColumn::Numbers(vec![revenue])])
    }

    /// Counts rows per distinct key, the plan of every `count-by-*` query.
    fn count_by(&self, keys: Vec<Column>) -> Result<Vec<HostColumn>> {
        let keys = cudf::Table::new(keys).map_err(gpu)?;
        let counts = GroupBy::new(&keys)
            .agg(0, AggregationKind::Count)
            .execute(&keys)
            .map_err(gpu)?;
        to_host(counts)
    }
}

/// Casts every decimal column to `Float64`, leaving the rest untouched.
fn decimals_as_floats(table: cudf::Table) -> Result<cudf::Table> {
    let columns = table
        .into_columns()
        .map_err(gpu)?
        .into_iter()
        .map(|column| match column.data_type().id() {
            TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128 => {
                column.cast(DataType::new(TypeId::Float64)).map_err(gpu)
            }
            _ => Ok(column),
        })
        .collect::<Result<Vec<_>>>()?;
    cudf::Table::new(columns).map_err(gpu)
}

/// `column <op> value` as a boolean mask; `value` must have the column's type.
fn compare(column: &Column, op: BinaryOp, value: impl CudfType) -> Result<Column> {
    let value = Scalar::new(value).map_err(gpu)?;
    column
        .binary_op_scalar(&value, op, DataType::new(TypeId::Bool8))
        .map_err(gpu)
}

fn combine(left: &Column, op: BinaryOp, right: &Column) -> Result<Column> {
    left.binary_op(right, op, DataType::new(TypeId::Bool8)).map_err(gpu)
}

fn arithmetic(left: &Column, op: BinaryOp, right: &Column) -> Result<Column> {
    left.binary_op(right, op, DataType::new(TypeId::Float64)).map_err(gpu)
}

/// `value <op> column` in `Float64`, for `1 - discount` and `1 + tax`.
fn from_constant(value: f64, op: BinaryOp, column: &Column) -> Result<Column> {
    let value = Scalar::new(value).map_err(gpu)?;
    binary_op(&value, column, op, DataType::new(TypeId::Float64)).map_err(gpu)
}

fn to_host(table: cudf::Table) -> Result<Vec<HostColumn>> {
    table
        .into_columns()
        .map_err(gpu)?
        .into_iter()
        .map(|column| match column.data_type().id() {
            TypeId::String => column.to_optional_strings().map(HostColumn::Strings).map_err(gpu),
            _ => column
                .cast(DataType::new(TypeId::Float64))
                .and_then(|numbers| numbers.to_optional_vec::<f64>())
                .map(HostColumn::Numbers)
                .map_err(gpu),
        })
        .collect()
}

fn push_rows(columns: &[HostColumn], answer: &mut Answer) {
    let rows = match columns.first() {
        Some(HostColumn::Numbers(values)) => values.len(),
        Some(HostColumn::Strings(values)) => values.len(),
        None => 0,
    };
    for row in 0..rows {
        for column in columns {
            match column {
                HostColumn::Numbers(values) => match values[row] {
                    Some(value) => answer.push_f64(value),
                    None => answer.push_null(),
                },
                HostColumn::Strings(values) => match &values[row] {
                    Some(value) => answer.push_str(value),
                    None => answer.push_null(),
                },
            }
        }
        answer.end_row();
    }
}

impl Backend for CudfBackend {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([("backend".into(), json!("cudf"))])
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        if source == Source::Parquet {
            return Err(unsupported("cuDF reads the parts into device memory during the load"));
        }
        for table in tables {
            let names = rows::columns(table)?.into_iter().map(|(name, _)| name).collect();
            let parts = table
                .parts
                .iter()
                .map(|part| read_parquet(part.display().to_string()).map_err(gpu))
                .collect::<Result<Vec<_>>>()?;
            let combined = match parts.len() {
                1 => parts.into_iter().next().unwrap(),
                _ => concatenate_tables(&parts.iter().collect::<Vec<_>>()).map_err(gpu)?,
            };
            self.tables
                .insert(table.name.clone(), (decimals_as_floats(combined)?, names));
        }
        Ok(())
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        let start = Instant::now();
        let columns = match query.name.as_str() {
            "count-by-vendor" => self.count_by(vec![self.column("trips", "vendor_id")?])?,
            "average-amount-by-passengers" => {
                let keys = cudf::Table::new(vec![self.column("trips", "passenger_count")?]).map_err(gpu)?;
                let amounts = cudf::Table::new(vec![self.column("trips", "total_amount")?]).map_err(gpu)?;
                to_host(
                    GroupBy::new(&keys)
                        .agg(0, AggregationKind::Mean)
                        .execute(&amounts)
                        .map_err(gpu)?,
                )?
            }
            "count-by-passengers-and-year" => self.count_by(vec![
                self.column("trips", "passenger_count")?,
                self.column("trips", "pickup_at")?.extract_year().map_err(gpu)?,
            ])?,
            "count-by-passengers-year-and-distance" => self.count_by(vec![
                self.column("trips", "passenger_count")?,
                self.column("trips", "pickup_at")?.extract_year().map_err(gpu)?,
                self.column("trips", "trip_distance")?
                    .unary_op(UnaryOp::Floor)
                    .map_err(gpu)?,
            ])?,
            "q01" if self.tables.contains_key("lineitem") => self.pricing_summary()?,
            "q06" if self.tables.contains_key("lineitem") => self.forecast_revenue()?,
            _ => return Err(unsupported("not written against libcudf")),
        };
        let executed = Executed::since(start);
        push_rows(&columns, answer);
        Ok(executed)
    }
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    run(cli.common, json!({}), &[], |_configuration, _path| {
        Ok(Box::new_in(CudfBackend::default(), System))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(
        _configuration: &tableval::Configuration,
        _path: &std::path::Path,
    ) -> tableval::Result<Box<dyn tableval::Backend, System>> {
        Ok(Box::new_in(CudfBackend::default(), System))
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
