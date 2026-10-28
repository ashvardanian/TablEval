# Data

Datasets go here and stay out of version control; only this README is tracked.
Each dataset is a folder of Parquet parts named `<table>.part_NNN.parquet`, read in natural order, so `part_9` comes before `part_10`.
Its tables tell the suite apart: `trips` is NYC Taxi, the eight TPC-H tables are TPC-H, and `hits` with its `queries.sql` is ClickBench.
`--rows` cuts the fact table, `trips` or `hits`, to the shortest prefix of whole parts holding that many rows, so every engine, servers included, reads the same files.
The engines' own storage lands in the same folder by default, as `tabl-eval-<engine>-<hash>/`, one per configuration, replaced on every run.

## NYC Taxi

Every yellow-cab trip the NYC Taxi & Limousine Commission published for 2009 through 2024, from its monthly Parquet files.
The schema drifts over those years, so `tabl-download` keeps five columns under one type each:

| Column            | Type                | Taken from                                                    |
| :---------------- | :------------------ | :------------------------------------------------------------ |
| `vendor_id`       | `Utf8`              | `vendor_name`, `vendor_id` or `VendorID`; codes 1, 2 → CMT, VTS |
| `pickup_at`       | timestamp µs        | `Trip_Pickup_DateTime`, `pickup_datetime` or `tpep_pickup_datetime` |
| `passenger_count` | `Int16`             | `passenger_count`, an integer until 2019 and a double after   |
| `trip_distance`   | `Float64`           | `trip_distance`                                               |
| `total_amount`    | `Float64`           | `Total_Amt` or `total_amount`                                 |

Later vendor codes keep their numbers as text, and values that do not convert become nulls.
`--from` and `--to` pick other months, and `--part-rows` another part size.

<details>
<summary>1.78G rows, 31 GB downloaded, 10M rows per part</summary>

```sh
cargo run --release --no-default-features --features download --bin tabl-download -- \
    --dataset taxi --output data/taxi
cargo run --release --bin tabl-eval-duckdb -- \
    --dataset data/taxi --rows 100M,1G --source loaded,parquet
```

</details>

## TPC-H

The eight TPC-H tables, generated in-process by [`tpchgen-rs`](https://github.com/clflushopt/tpchgen-rs), which matches `dbgen` row for row, so nothing is downloaded.
The six scaling tables get one part per unit of scale, about 6M `lineitem` rows each.
Prices and quantities stay `Decimal128(15, 2)`, dates stay `Date32`, and strings are plain `Utf8`.
At scale factors 0.01 and 1 every query is checked against the pinned answers in [`sql/tpch/`](../sql/tpch/README.md).

<details>
<summary>Scale 1: 6M lineitem rows; scale 10 and 100 for 60M and 600M</summary>

```sh
cargo run --release --no-default-features --features generate --bin tabl-generate -- \
    --scale-factor 1 --output data/tpch-sf1
cargo run --release --no-default-features --features datafusion-backend --bin tabl-eval-datafusion -- \
    --dataset data/tpch-sf1 --source loaded,parquet
```

</details>

## ClickBench

The 100 parts of ClickBench's partitioned `hits` table, about 1M rows of anonymized web analytics each.
`tabl-download` turns the binary columns into text, `EventDate` into a date, and `EventTime`, `ClientEventTime` and `LocalEventTime` from Unix seconds into timestamps.
It also fetches the 43 queries in their PostgreSQL spelling from a pinned ClickBench commit into `queries.sql`, rather than copying them into this repository, since ClickBench licenses them for non-commercial use only.
Queries ending in a `LIMIT` over a non-unique order are verified by row count only.

<details>
<summary>99,997,497 rows, 14.7 GB downloaded, 100 parts</summary>

```sh
cargo run --release --no-default-features --features download --bin tabl-download -- \
    --dataset clickbench --output data/clickbench-100m
cargo run --release --no-default-features --features clickhouse-backend --bin tabl-eval-clickhouse -- \
    --dataset data/clickbench-100m --source loaded,parquet
```

</details>

Downloads go through `raw/` inside the dataset folder, through a `.partial` file renamed once complete, and each original is deleted after it is rewritten unless `--keep-raw` is set.
A ClickBench download resumes at the first missing part; a Taxi download restarts its parts but reuses the originals kept in `raw/`.
