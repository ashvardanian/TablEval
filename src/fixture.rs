//! Known answers every backend's tests check: a four-trip Taxi table, and TPC-H at scale 0.01.

use std::{alloc::System, path::Path, sync::Arc};

use arrow::{
    array::{Float64Array, Int16Array, RecordBatch, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
};
use parquet::arrow::ArrowWriter;

use crate::{backend::UNSUPPORTED, Answer, Backend, Configuration, Dataset, Query, Result, Source, Threads};

/// The normalized `trips` schema `tabl-download` writes.
pub fn taxi_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("vendor_id", DataType::Utf8, true),
        Field::new("pickup_at", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("passenger_count", DataType::Int16, true),
        Field::new("trip_distance", DataType::Float64, true),
        Field::new("total_amount", DataType::Float64, true),
    ]))
}

/// Writes `trips.part_000.parquet` with four trips into `folder`.
pub fn write_taxi_fixture(folder: &Path) -> Result<()> {
    let microseconds = |date: &str| -> i64 {
        let days = match date {
            "2020-01-23" => 18284,
            "2019-01-23" => 17919,
            "2018-01-23" => 17554,
            _ => 17553,
        };
        (days * 86_400 + 14 * 3_600 + 34 * 60 + 45) * 1_000_000
    };
    let batch = RecordBatch::try_new(
        taxi_schema(),
        vec![
            Arc::new(StringArray::from(vec!["Uber", "Lyft", "Uber", "Lyft"])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                microseconds("2020-01-23"),
                microseconds("2019-01-23"),
                microseconds("2018-01-23"),
                microseconds("2018-01-22"),
            ])),
            Arc::new(Int16Array::from(vec![3, 2, 4, 4])),
            Arc::new(Float64Array::from(vec![2.3, 2.5, 5.3, 5.3])),
            Arc::new(Float64Array::from(vec![23.0, 15.0, 18.0, 17.5])),
        ],
    )
    .map_err(|e| e.to_string())?;
    std::fs::create_dir_all(folder).map_err(|e| e.to_string())?;
    let file = std::fs::File::create(folder.join("trips.part_000.parquet")).map_err(|e| e.to_string())?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).map_err(|e| e.to_string())?;
    writer.write(&batch).map_err(|e| e.to_string())?;
    writer.close().map_err(|e| e.to_string())?;
    Ok(())
}

fn expected(query: &str) -> Answer {
    let rows: &[&[&str]] = match query {
        "count-by-vendor" => &[&["Uber", "2"], &["Lyft", "2"]],
        "average-amount-by-passengers" => &[&["2", "15"], &["3", "23"], &["4", "17.75"]],
        "count-by-passengers-and-year" => &[&["2", "2019", "1"], &["3", "2020", "1"], &["4", "2018", "2"]],
        _ => &[
            &["4", "2018", "5", "2"],
            &["2", "2019", "2", "1"],
            &["3", "2020", "2", "1"],
        ],
    };
    let mut answer = Answer::default();
    for row in rows {
        for cell in *row {
            answer.push_str(cell);
        }
        answer.end_row();
    }
    answer
}

/// Writes the fixture into `root/taxi`, loads it through `open` from every source the backend accepts, with
/// each source's storage in `root/taxi-<source>`, and checks the four answers.
pub fn assert_taxi_answers(root: &Path, open: impl Fn(&Configuration, &Path) -> Result<Box<dyn Backend, System>>) {
    let folder = root.join("taxi");
    write_taxi_fixture(&folder).unwrap();
    assert_answers(&folder, &open, |query| Some(expected(&query.name)), 4);
}

/// Generates TPC-H at scale 0.01 into `root/tpch`, loads it like `assert_taxi_answers`, and checks every
/// query the backend supports against the pinned answers.
#[cfg(feature = "generate")]
pub fn assert_tpch_answers(root: &Path, open: impl Fn(&Configuration, &Path) -> Result<Box<dyn Backend, System>>) {
    let folder = root.join("tpch");
    crate::tpch::write_tables(&folder, 0.01).unwrap();
    let dataset = Dataset::open(&folder).unwrap();
    assert_answers(&folder, &open, |query| dataset.expected(query), 1);
}

/// Runs every query of the dataset in `folder` from each accepted source and panics listing every query that
/// failed or returned other rows; `required` is the fewest answered queries a source may have.
fn assert_answers(
    folder: &Path,
    open: &impl Fn(&Configuration, &Path) -> Result<Box<dyn Backend, System>>,
    expected: impl Fn(&Query) -> Option<Answer>,
    required: usize,
) {
    let dataset = Dataset::open(folder).unwrap();
    let tables = dataset.tables(None).unwrap();
    let mut loaded_any = false;
    let mut problems = Vec::new();
    for source in [Source::Loaded, Source::Parquet] {
        let name = folder.file_name().unwrap_or_default().to_string_lossy();
        let storage = folder.with_file_name(format!("{name}-{source:?}"));
        std::fs::create_dir_all(&storage).unwrap();
        let configuration = Configuration {
            rows: None,
            threads: Threads(std::num::NonZeroUsize::new(2).unwrap()),
            source,
        };
        let mut backend = open(&configuration, &storage).unwrap();
        match backend.load(&tables, source) {
            Err(error) if error.starts_with(UNSUPPORTED) => continue,
            loaded => loaded.unwrap(),
        }
        loaded_any = true;
        let mut answered = 0;
        for query in &dataset.queries {
            let mut answer = Answer::default();
            match backend.run(query, &mut answer) {
                Err(error) if error.starts_with(UNSUPPORTED) => continue,
                Err(error) => problems.push(format!("{source:?} {}: {error}", query.name)),
                Ok(_) => {
                    let digest = answer.digest(query.ties);
                    let pinned = expected(query).map(|mut expected| (expected.digest(query.ties), expected.rows()));
                    match pinned {
                        Some((pinned, rows)) if pinned != digest => problems.push(format!(
                            "{source:?} {}: {} rows differ from the {rows} pinned rows",
                            query.name,
                            answer.rows()
                        )),
                        _ => answered += 1,
                    }
                }
            }
        }
        if answered < required {
            problems.push(format!(
                "{source:?}: answered {answered} queries, fewer than {required}"
            ));
        }
    }
    assert!(loaded_any, "the backend accepted no source");
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
