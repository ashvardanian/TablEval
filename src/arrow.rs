//! A query-engine-free floor: the four Taxi queries and TPC-H Q1 and Q6, hand-written over arrow-rs.
//!
//! The parts are decoded into Arrow batches once, during the load. Each query splits the batches evenly
//! across a ForkUnion pool, folds every slice into a per-thread partial, and merges the partials on the
//! calling thread. Every other query reports `unsupported`.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features arrow-backend --bin tabl-eval-arrow -- \
//!     --dataset data/taxi-1b --rows 100M --threads 0
//! ```

use std::{
    alloc::System,
    collections::{BTreeMap, HashMap},
    fs::File,
    time::Instant,
};

use arrow::{
    array::{Array, AsArray, RecordBatch},
    compute::{cast, kernels::temporal::date_part, DatePart},
    datatypes::{DataType, Date32Type, Decimal128Type, Float64Type, Int16Type, Int32Type},
};
use clap::Parser;
use forkunion::{self as fu, IndexedSplit, ThreadPool, Topology};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{json, Value};

use tableval::{run, unsupported, Answer, Backend, CommonArgs, Configuration, Executed, Query, Result, Source, Table};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

/// `1998-09-02`, `1994-01-01` and `1995-01-01` as days since the Unix epoch, the bounds of TPC-H Q1 and Q6.
const Q1_SHIPPED_BY: i32 = 10471;
const Q6_SHIPPED_FROM: i32 = 8766;
const Q6_SHIPPED_BEFORE: i32 = 9131;

struct ArrowBackend {
    pool: ThreadPool,
    tables: HashMap<String, Vec<RecordBatch>>,
}

impl ArrowBackend {
    fn open(configuration: &Configuration) -> Result<Self> {
        let pool = Topology::new()
            .and_then(|topology| ThreadPool::spawn(&topology, configuration.threads.get()))
            .map_err(|e| format!("ForkUnion: {e:?}"))?;
        Ok(Self {
            pool,
            tables: HashMap::new(),
        })
    }

    /// Folds every batch of `table` into one partial per thread, and returns the partials.
    fn fold<'a, T: Default + Send>(
        &'a mut self,
        table: &str,
        fold: impl Fn(&mut T, &'a RecordBatch) + Sync,
    ) -> Result<Vec<T>> {
        let batches = self
            .tables
            .get(table)
            .ok_or_else(|| unsupported(&format!("no `{table}` table")))?;
        let threads = self.pool.threads_count();
        let split = IndexedSplit::new(batches.len(), threads);
        let mut partials: Vec<T> = (0..threads).map(|_| T::default()).collect();
        // One partial per thread, so the pool's own split hands each thread exactly its own.
        self.pool
            .for_slices_mut(&mut partials, |partial, at: fu::ThreadInDomain| {
                debug_assert_eq!(partial.len(), 1);
                for batch in &batches[split.get(at.thread)] {
                    fold(&mut partial[0], batch);
                }
            })
            .map_err(|e| format!("ForkUnion: {e:?}"))?;
        Ok(partials)
    }
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> &'a dyn Array {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("missing column `{name}`"))
        .as_ref()
}

fn years(batch: &RecordBatch, name: &str) -> arrow::array::Int32Array {
    let years = date_part(column(batch, name), DatePart::Year).expect("a timestamp or date column");
    years.as_primitive::<Int32Type>().clone()
}

/// Sums and counts per group, merged by key; averages merge as these pairs, never as averages.
#[derive(Default, Clone, Copy)]
struct Moments {
    sum: f64,
    count: u64,
}

impl Backend for ArrowBackend {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("backend".into(), json!("arrow")),
            ("library_version".into(), json!(env!("CARGO_PKG_VERSION"))),
            ("threads".into(), json!(self.pool.threads_count())),
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
            self.tables.insert(table.name.clone(), batches);
        }
        Ok(())
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        match query.name.as_str() {
            "count-by-vendor" => self.count_by_vendor(answer),
            "average-amount-by-passengers" => self.average_amount_by_passengers(answer),
            "count-by-passengers-and-year" => self.count_by_passengers_and_year(answer),
            "count-by-passengers-year-and-distance" => self.count_by_passengers_year_and_distance(answer),
            "q01" if self.tables.contains_key("lineitem") => self.pricing_summary(answer),
            "q06" if self.tables.contains_key("lineitem") => self.forecast_revenue(answer),
            _ => Err(unsupported("not hand-written for the Arrow baseline")),
        }
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

fn push_optional(answer: &mut Answer, value: Option<i64>) {
    match value {
        Some(value) => answer.push_i64(value),
        None => answer.push_null(),
    }
}

impl ArrowBackend {
    fn count_by_vendor(&mut self, answer: &mut Answer) -> Result<Executed> {
        let start = Instant::now();
        let partials = self.fold("trips", |counts: &mut HashMap<Option<&str>, u64>, batch| {
            for vendor in column(batch, "vendor_id").as_string::<i32>() {
                *counts.entry(vendor).or_default() += 1;
            }
        })?;
        let mut counts = HashMap::new();
        for partial in partials {
            for (vendor, count) in partial {
                *counts.entry(vendor).or_insert(0) += count;
            }
        }
        let executed = Executed::since(start);
        for (vendor, count) in counts {
            match vendor {
                Some(vendor) => answer.push_str(vendor),
                None => answer.push_null(),
            }
            answer.push_i64(count as i64);
            answer.end_row();
        }
        Ok(executed)
    }

    fn average_amount_by_passengers(&mut self, answer: &mut Answer) -> Result<Executed> {
        let start = Instant::now();
        let partials = self.fold("trips", |groups: &mut HashMap<Option<i16>, Moments>, batch| {
            let passengers = column(batch, "passenger_count").as_primitive::<Int16Type>();
            let amounts = column(batch, "total_amount").as_primitive::<Float64Type>();
            for (passengers, amount) in passengers.iter().zip(amounts.iter()) {
                let group = groups.entry(passengers).or_default();
                if let Some(amount) = amount {
                    group.sum += amount;
                    group.count += 1;
                }
            }
        })?;
        let mut groups: HashMap<Option<i16>, Moments> = HashMap::new();
        for partial in partials {
            for (passengers, moments) in partial {
                let group = groups.entry(passengers).or_default();
                group.sum += moments.sum;
                group.count += moments.count;
            }
        }
        let executed = Executed::since(start);
        for (passengers, moments) in groups {
            push_optional(answer, passengers.map(i64::from));
            match moments.count {
                0 => answer.push_null(),
                count => answer.push_f64(moments.sum / count as f64),
            }
            answer.end_row();
        }
        Ok(executed)
    }

    fn count_by_passengers_and_year(&mut self, answer: &mut Answer) -> Result<Executed> {
        let start = Instant::now();
        let partials = self.fold(
            "trips",
            |counts: &mut HashMap<(Option<i16>, Option<i32>), u64>, batch| {
                let passengers = column(batch, "passenger_count").as_primitive::<Int16Type>();
                let years = years(batch, "pickup_at");
                for key in passengers.iter().zip(years.iter()) {
                    *counts.entry(key).or_default() += 1;
                }
            },
        )?;
        let mut counts = HashMap::new();
        for partial in partials {
            for (key, count) in partial {
                *counts.entry(key).or_insert(0) += count;
            }
        }
        let executed = Executed::since(start);
        for ((passengers, year), count) in counts {
            push_optional(answer, passengers.map(i64::from));
            push_optional(answer, year.map(i64::from));
            answer.push_i64(count as i64);
            answer.end_row();
        }
        Ok(executed)
    }

    fn count_by_passengers_year_and_distance(&mut self, answer: &mut Answer) -> Result<Executed> {
        type Key = (Option<i16>, Option<i32>, Option<i64>);
        let start = Instant::now();
        let partials = self.fold("trips", |counts: &mut HashMap<Key, u64>, batch| {
            let passengers = column(batch, "passenger_count").as_primitive::<Int16Type>();
            let distances = column(batch, "trip_distance").as_primitive::<Float64Type>();
            let years = years(batch, "pickup_at");
            for ((passengers, year), distance) in passengers.iter().zip(years.iter()).zip(distances.iter()) {
                let distance = distance.map(|distance| distance.floor() as i64);
                *counts.entry((passengers, year, distance)).or_default() += 1;
            }
        })?;
        let mut counts: HashMap<Key, u64> = HashMap::new();
        for partial in partials {
            for (key, count) in partial {
                *counts.entry(key).or_insert(0) += count;
            }
        }
        let mut rows: Vec<(Key, u64)> = counts.into_iter().collect();
        rows.sort_unstable_by(|(left, left_count), (right, right_count)| {
            (left.1, std::cmp::Reverse(*left_count), left.0, left.2).cmp(&(
                right.1,
                std::cmp::Reverse(*right_count),
                right.0,
                right.2,
            ))
        });
        let executed = Executed::since(start);
        for ((passengers, year, distance), count) in rows {
            push_optional(answer, passengers.map(i64::from));
            push_optional(answer, year.map(i64::from));
            push_optional(answer, distance);
            answer.push_i64(count as i64);
            answer.end_row();
        }
        Ok(executed)
    }

    /// TPC-H Q1 in exact decimal arithmetic: prices and quantities at scale 2, so the discounted price is at
    /// scale 4 and the charge at scale 6.
    fn pricing_summary(&mut self, answer: &mut Answer) -> Result<Executed> {
        #[derive(Default, Clone, Copy)]
        struct Sums {
            quantity: i128,
            price: i128,
            discounted: i128,
            charge: i128,
            discount: i128,
            count: u64,
        }
        let start = Instant::now();
        let partials = self.fold("lineitem", |groups: &mut HashMap<(&str, &str), Sums>, batch| {
            let shipped = column(batch, "l_shipdate").as_primitive::<Date32Type>();
            let flags = column(batch, "l_returnflag").as_string::<i32>();
            let statuses = column(batch, "l_linestatus").as_string::<i32>();
            let quantities = column(batch, "l_quantity").as_primitive::<Decimal128Type>();
            let prices = column(batch, "l_extendedprice").as_primitive::<Decimal128Type>();
            let discounts = column(batch, "l_discount").as_primitive::<Decimal128Type>();
            let taxes = column(batch, "l_tax").as_primitive::<Decimal128Type>();
            for row in 0..batch.num_rows() {
                if shipped.value(row) > Q1_SHIPPED_BY {
                    continue;
                }
                let sums = groups.entry((flags.value(row), statuses.value(row))).or_default();
                let (price, discount) = (prices.value(row), discounts.value(row));
                let discounted = price * (100 - discount);
                sums.quantity += quantities.value(row);
                sums.price += price;
                sums.discounted += discounted;
                sums.charge += discounted * (100 + taxes.value(row));
                sums.discount += discount;
                sums.count += 1;
            }
        })?;
        let mut groups: HashMap<(&str, &str), Sums> = HashMap::new();
        for partial in partials {
            for (key, part) in partial {
                let sums = groups.entry(key).or_default();
                sums.quantity += part.quantity;
                sums.price += part.price;
                sums.discounted += part.discounted;
                sums.charge += part.charge;
                sums.discount += part.discount;
                sums.count += part.count;
            }
        }
        let executed = Executed::since(start);
        for ((flag, status), sums) in groups {
            let count = sums.count as f64;
            answer.push_str(flag);
            answer.push_str(status);
            answer.push_f64(sums.quantity as f64 / 1e2);
            answer.push_f64(sums.price as f64 / 1e2);
            answer.push_f64(sums.discounted as f64 / 1e4);
            answer.push_f64(sums.charge as f64 / 1e6);
            answer.push_f64(sums.quantity as f64 / 1e2 / count);
            answer.push_f64(sums.price as f64 / 1e2 / count);
            answer.push_f64(sums.discount as f64 / 1e2 / count);
            answer.push_i64(sums.count as i64);
            answer.end_row();
        }
        Ok(executed)
    }

    /// TPC-H Q6: revenue at scale 4 from lines shipped in 1994 at a 5-7% discount and under 24 units.
    fn forecast_revenue(&mut self, answer: &mut Answer) -> Result<Executed> {
        let start = Instant::now();
        let partials = self.fold("lineitem", |revenue: &mut i128, batch| {
            let shipped = column(batch, "l_shipdate").as_primitive::<Date32Type>();
            let quantities = column(batch, "l_quantity").as_primitive::<Decimal128Type>();
            let prices = column(batch, "l_extendedprice").as_primitive::<Decimal128Type>();
            let discounts = column(batch, "l_discount").as_primitive::<Decimal128Type>();
            for row in 0..batch.num_rows() {
                let (date, discount) = (shipped.value(row), discounts.value(row));
                if (Q6_SHIPPED_FROM..Q6_SHIPPED_BEFORE).contains(&date)
                    && (5..=7).contains(&discount)
                    && quantities.value(row) < 2400
                {
                    *revenue += prices.value(row) * discount;
                }
            }
        })?;
        let revenue: i128 = partials.into_iter().sum();
        let executed = Executed::since(start);
        answer.push_f64(revenue as f64 / 1e4);
        answer.end_row();
        Ok(executed)
    }
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
