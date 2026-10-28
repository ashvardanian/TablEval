//! Fetches NYC Taxi or ClickBench and rewrites it as `<table>.part_NNN.parquet` with one fixed schema.
//!
//! __Taxi__ reads the TLC's monthly yellow-cab files, whose schema drifts over the years: vendors are
//! names until 2010 and integer codes after, pickup times are strings until 2010, and passenger counts
//! turn into doubles in 2020. Every month is projected to five columns, `vendor_id` as text (codes 1 and
//! 2 become `CMT` and `VTS`, the names the early files use), `pickup_at` as a microsecond timestamp,
//! `passenger_count` as `Int16`, and `trip_distance` and `total_amount` as `Float64`, then split into
//! parts of `--part-rows` rows.
//!
//! __ClickBench__ reads the 100 `hits_N.parquet` files of the partitioned set, one part each, turning
//! binary columns into text, `EventDate` into a date and the three Unix-second times into timestamps.
//! The 43 queries are fetched beside them as `queries.sql`, from a pinned ClickBench commit.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features download --bin tabl-download -- \
//!     --dataset clickbench --output data/clickbench-100m
//! ```

use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, AsArray, Int64Array, RecordBatch, StringArray},
    compute::{cast, cast_with_options, CastOptions},
    datatypes::{DataType, Field, Int64Type, Schema, TimeUnit},
};
use clap::{Parser, ValueEnum};
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter},
    basic::Compression,
    file::properties::WriterProperties,
};

use tableval::{fixture::taxi_schema, parse_count_flag, Result};

const TAXI_URL: &str = "https://d37ci6vzurychx.cloudfront.net/trip-data";
const CLICKBENCH_URL: &str = "https://datasets.clickhouse.com/hits_compatible/athena_partitioned";
/// The ClickBench commit whose PostgreSQL-dialect queries every engine runs.
const CLICKBENCH_COMMIT: &str = "08b0c98a4f493a216029b1bf54c5e4717b8401cc";
const CLICKBENCH_PARTS: usize = 100;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Taxi,
    Clickbench,
}

#[derive(Parser)]
struct Cli {
    /// Which dataset to fetch.
    #[arg(long, env = "TABLEVAL_DOWNLOAD_DATASET", value_enum)]
    dataset: Source,
    /// Destination folder; `data/taxi` or `data/clickbench-100m` when unset.
    #[arg(long, env = "TABLEVAL_OUTPUT")]
    output: Option<PathBuf>,
    /// First Taxi month, as YYYY-MM.
    #[arg(long, env = "TABLEVAL_FROM", default_value = "2009-01")]
    from: String,
    /// Last Taxi month, as YYYY-MM.
    #[arg(long, env = "TABLEVAL_TO", default_value = "2024-12")]
    to: String,
    /// Rows per Taxi part, with decimal K, M or G.
    #[arg(long, env = "TABLEVAL_PART_ROWS", default_value = "10M", value_parser = parse_count_flag)]
    part_rows: u64,
    /// Keep the original files under `raw/` instead of deleting each once it is rewritten.
    #[arg(long, env = "TABLEVAL_KEEP_RAW", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new())]
    keep_raw: bool,
}

/// Downloads `url` into `path` unless a complete copy is there, through a `.partial` file renamed at the end.
fn fetch(agent: &ureq::Agent, url: &str, path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    let partial = path.with_extension("partial");
    let mut response = agent.get(url).call().map_err(|e| format!("{url}: {e}"))?;
    let mut file = File::create(&partial).map_err(|e| format!("{}: {e}", partial.display()))?;
    let mut body = response.body_mut().with_config().limit(u64::MAX).reader();
    std::io::copy(&mut body, &mut file).map_err(|e| format!("{url}: {e}"))?;
    file.sync_all().map_err(|e| e.to_string())?;
    std::fs::rename(&partial, path).map_err(|e| e.to_string())
}

fn writer(path: &Path, schema: Arc<Schema>) -> Result<ArrowWriter<File>> {
    let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let properties = WriterProperties::builder().set_compression(Compression::SNAPPY).build();
    ArrowWriter::try_new(file, schema, Some(properties)).map_err(|e| e.to_string())
}

/// The first column whose name matches one of `names`, ignoring case.
fn find<'a>(batch: &'a RecordBatch, names: &[&str]) -> Result<&'a ArrayRef> {
    let schema = batch.schema();
    let index = schema
        .fields()
        .iter()
        .position(|field| names.iter().any(|name| field.name().eq_ignore_ascii_case(name)))
        .ok_or_else(|| format!("no column named any of {names:?}"))?;
    Ok(batch.column(index))
}

fn taxi_batch(batch: &RecordBatch) -> Result<RecordBatch> {
    let lossy = CastOptions {
        safe: true,
        ..Default::default()
    };
    let vendor = find(batch, &["vendor_name", "vendor_id", "vendorid"])?;
    let vendor: ArrayRef = match vendor.data_type() {
        data_type if data_type.is_integer() => {
            let codes = cast(vendor, &DataType::Int64).map_err(|e| e.to_string())?;
            let names: StringArray = codes
                .as_primitive::<Int64Type>()
                .iter()
                .map(|code| {
                    code.map(|code| match code {
                        1 => "CMT".to_owned(),
                        2 => "VTS".to_owned(),
                        code => code.to_string(),
                    })
                })
                .collect();
            Arc::new(names)
        }
        _ => cast(vendor, &DataType::Utf8).map_err(|e| e.to_string())?,
    };
    let pickup = find(
        batch,
        &["trip_pickup_datetime", "pickup_datetime", "tpep_pickup_datetime"],
    )?;
    let microseconds = DataType::Timestamp(TimeUnit::Microsecond, None);
    let pickup = cast_with_options(pickup, &microseconds, &lossy).map_err(|e| e.to_string())?;
    let numeric = |names: &[&str], data_type: &DataType| -> Result<ArrayRef> {
        cast_with_options(find(batch, names)?, data_type, &lossy).map_err(|e| e.to_string())
    };
    RecordBatch::try_new(
        taxi_schema(),
        vec![
            vendor,
            pickup,
            numeric(&["passenger_count"], &DataType::Int16)?,
            numeric(&["trip_distance"], &DataType::Float64)?,
            numeric(&["total_amount", "total_amt"], &DataType::Float64)?,
        ],
    )
    .map_err(|e| e.to_string())
}

/// Months from `from` to `to` inclusive, as `YYYY-MM`.
fn months(from: &str, to: &str) -> Result<Vec<String>> {
    let parse = |text: &str| -> Result<(u32, u32)> {
        let (year, month) = text.split_once('-').ok_or("expected YYYY-MM")?;
        let year = year.parse().map_err(|_| "expected YYYY-MM")?;
        let month = month.parse().map_err(|_| "expected YYYY-MM")?;
        Ok((year, month))
    };
    let (mut year, mut month) = parse(from).map_err(|e| format!("--from=\"{from}\" does not parse, {e}"))?;
    let end = parse(to).map_err(|e| format!("--to=\"{to}\" does not parse, {e}"))?;
    let mut months = Vec::new();
    while (year, month) <= end {
        months.push(format!("{year:04}-{month:02}"));
        (year, month) = if month == 12 { (year + 1, 1) } else { (year, month + 1) };
    }
    Ok(months)
}

fn download_taxi(cli: &Cli, agent: &ureq::Agent, folder: &Path) -> Result<()> {
    let raw = folder.join("raw");
    std::fs::create_dir_all(&raw).map_err(|e| e.to_string())?;
    let (mut part, mut part_rows, mut total) = (0, 0, 0u64);
    let mut output: Option<ArrowWriter<File>> = None;
    for month in months(&cli.from, &cli.to)? {
        let name = format!("yellow_tripdata_{month}.parquet");
        let path = raw.join(&name);
        fetch(agent, &format!("{TAXI_URL}/{name}"), &path)?;
        let file = File::open(&path).map_err(|e| e.to_string())?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .and_then(|builder| builder.build())
            .map_err(|e| format!("{name}: {e}"))?;
        for batch in reader {
            let mut batch = taxi_batch(&batch.map_err(|e| e.to_string())?)?;
            while batch.num_rows() > 0 {
                let writer = match &mut output {
                    Some(writer) => writer,
                    None => output.insert(writer(
                        &folder.join(format!("trips.part_{part:03}.parquet")),
                        taxi_schema(),
                    )?),
                };
                let take = (cli.part_rows - part_rows).min(batch.num_rows() as u64) as usize;
                writer.write(&batch.slice(0, take)).map_err(|e| e.to_string())?;
                batch = batch.slice(take, batch.num_rows() - take);
                (part_rows, total) = (part_rows + take as u64, total + take as u64);
                if part_rows == cli.part_rows {
                    output.take().unwrap().close().map_err(|e| e.to_string())?;
                    (part, part_rows) = (part + 1, 0);
                }
            }
        }
        eprintln!(
            "{month}: {total} rows so far in {} parts",
            part + usize::from(part_rows > 0)
        );
        if !cli.keep_raw {
            std::fs::remove_file(&path).map_err(|e| e.to_string())?;
        }
    }
    if let Some(writer) = output {
        writer.close().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Binary columns as text, `EventDate` days as a date, and Unix-second times as timestamps.
fn clickbench_batch(batch: &RecordBatch) -> Result<RecordBatch> {
    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        let name = field.name().as_str();
        let column: ArrayRef = match (name, column.data_type()) {
            (_, DataType::Binary | DataType::LargeBinary | DataType::BinaryView) => {
                let bytes = cast(column, &DataType::LargeBinary).map_err(|e| e.to_string())?;
                let text: StringArray = bytes
                    .as_binary::<i64>()
                    .iter()
                    .map(|value| value.map(|value| String::from_utf8_lossy(value).into_owned()))
                    .collect();
                Arc::new(text)
            }
            ("EventDate", data_type) if data_type.is_integer() => {
                let days = cast(column, &DataType::Int32).map_err(|e| e.to_string())?;
                cast(&days, &DataType::Date32).map_err(|e| e.to_string())?
            }
            ("EventTime" | "ClientEventTime" | "LocalEventTime", data_type) if data_type.is_integer() => {
                let seconds = cast(column, &DataType::Int64).map_err(|e| e.to_string())?;
                let micros: Int64Array = seconds
                    .as_primitive::<Int64Type>()
                    .iter()
                    .map(|value| value.map(|value| value * 1_000_000))
                    .collect();
                cast(&micros, &DataType::Timestamp(TimeUnit::Microsecond, None)).map_err(|e| e.to_string())?
            }
            _ => column.clone(),
        };
        fields.push(Field::new(name, column.data_type().clone(), field.is_nullable()));
        columns.push(column);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|e| e.to_string())
}

fn download_clickbench(cli: &Cli, agent: &ureq::Agent, folder: &Path) -> Result<()> {
    let queries =
        format!("https://raw.githubusercontent.com/ClickHouse/ClickBench/{CLICKBENCH_COMMIT}/postgresql/queries.sql");
    fetch(agent, &queries, &folder.join("queries.sql"))?;
    let raw = folder.join("raw");
    std::fs::create_dir_all(&raw).map_err(|e| e.to_string())?;
    for index in 0..CLICKBENCH_PARTS {
        let part = folder.join(format!("hits.part_{index:03}.parquet"));
        if part.exists() {
            continue;
        }
        let name = format!("hits_{index}.parquet");
        let path = raw.join(&name);
        fetch(agent, &format!("{CLICKBENCH_URL}/{name}"), &path)?;
        let file = File::open(&path).map_err(|e| e.to_string())?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .and_then(|builder| builder.build())
            .map_err(|e| format!("{name}: {e}"))?;
        let temporary = part.with_extension("partial");
        let mut output = None;
        let mut rows = 0;
        for batch in reader {
            let batch = clickbench_batch(&batch.map_err(|e| e.to_string())?)?;
            rows += batch.num_rows();
            let writer = match &mut output {
                Some(writer) => writer,
                None => output.insert(writer(&temporary, batch.schema())?),
            };
            writer.write(&batch).map_err(|e| e.to_string())?;
        }
        if let Some(writer) = output {
            writer.close().map_err(|e| e.to_string())?;
        }
        std::fs::rename(&temporary, &part).map_err(|e| e.to_string())?;
        eprintln!("{name}: {rows} rows");
        if !cli.keep_raw {
            std::fs::remove_file(&path).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    let folder = cli.output.clone().unwrap_or_else(|| {
        PathBuf::from(match cli.dataset {
            Source::Taxi => "data/taxi",
            Source::Clickbench => "data/clickbench-100m",
        })
    });
    std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    eprintln!("- Dataset: {}", tableval::spell_value(&cli.dataset));
    eprintln!("- Output: {}", folder.display());
    if cli.dataset == Source::Taxi {
        eprintln!("- Months: {} to {}", cli.from, cli.to);
        eprintln!("- Part rows: {}", tableval::spell_count(cli.part_rows));
    }
    eprintln!("- Keep raw: {}", cli.keep_raw);
    let agent = ureq::Agent::new_with_defaults();
    match cli.dataset {
        Source::Taxi => download_taxi(&cli, &agent, &folder)?,
        Source::Clickbench => download_clickbench(&cli, &agent, &folder)?,
    }
    if !cli.keep_raw {
        let _ = std::fs::remove_dir(folder.join("raw"));
    }
    std::io::stderr().flush().map_err(|e| e.to_string())
}
