# Source

`bench.rs` is the library root.
Each engine has one binary source file with its own typed command-line options and its `Backend` implementation.
The shared runner owns the sweep, the load and query phases, verification, measurement and reports.

## Responsibilities

The shared runner and its contracts live in:

- [`bench.rs`](bench.rs): `CommonArgs`, the sweep over rows, threads and sources, cold and hot runs, and count and duration parsing.
- [`backend.rs`](backend.rs): the `Backend` trait, `Table`, `Source`, `Executed` and the `unsupported` error convention.
- [`suite.rs`](suite.rs): the Taxi, TPC-H and ClickBench query catalogs, dataset discovery from Parquet footers, row prefixes and pinned TPC-H answers.
- [`answer.rs`](answer.rs): canonical result rows and their order-independent digest, plus the Arrow C Data Interface import other Arrow majors cross through.

Measurement and infrastructure are split into:

- [`measure.rs`](measure.rs): process CPU, memory and I/O, and a server container's through its cgroup, sampled every 100 ms.
- [`output.rs`](output.rs): reports, statuses, machine information, configuration hashes and atomic report writes.
- [`perf_counters.rs`](perf_counters.rs): optional system-wide Linux hardware counters per phase.
- [`docker.rs`](docker.rs): container lifecycle, the read-only dataset mount, readiness, the container's cgroup, and resource snapshots.
- [`rows.rs`](rows.rs): Parquet parts read back as typed rows for engines that ingest through SQL.
- [`fixture.rs`](fixture.rs): the four-trip Taxi table and TPC-H at scale 0.01, with the answers every backend's tests check.
- [`tpch.rs`](tpch.rs): TPC-H tables written as Parquet parts, behind the `generate` feature.

Each backend has its own binary:

- Embedded engines: [`duckdb.rs`](duckdb.rs), [`datafusion.rs`](datafusion.rs), [`polars.rs`](polars.rs) and [`sqlite.rs`](sqlite.rs).
- The hand-written baseline: [`arrow.rs`](arrow.rs).
- Database servers: [`clickhouse.rs`](clickhouse.rs) and [`postgres.rs`](postgres.rs).
- GPU: [`cudf.rs`](cudf.rs).

[`generate.rs`](generate.rs) builds `tabl-generate` for TPC-H, in parallel over `tpch.rs`, and [`../scripts/download.rs`](../scripts/download.rs) builds `tabl-download` for Taxi and ClickBench.

## Vocabulary & Contracts

A `Backend` is opened once per configuration, with the run's own folder under `--data-dir`.
`load` receives every table with its parts already cut to the configuration's rows, and either ingests them, for `Source::Loaded`, or registers them in place, for `Source::Parquet`.
`run` executes one query, materializes its result inside the timed region, and only then fills the `Answer`.
`Executed` carries that engine time; parsing the result into the `Answer` is never timed.

An error starting with `unsupported` marks a query or source the engine cannot express; the runner records it as `unsupported` and moves on.
Any other error is `failed`, and the run exits nonzero after finishing the remaining queries.
A query whose digest differs from the pinned answer, or whose hot runs disagree with its cold run, is `mismatch`.

`Answer` reduces rows to one hash each and the result to the hash of the sorted row hashes, so row order never matters.
Numbers of every type round to eight significant digits and print as integers when integral; strings that read as numbers take the same path, and timestamps drop the `T` and trailing fractional zeros.
That lets a `COUNT(*)` returned as `BIGINT`, `DOUBLE` or `DECIMAL`, and a sum accumulated in a different order, match.

DuckDB, the Arrow baseline and the shared code use Arrow 60; DataFusion pins Arrow 59.
Its batches move into Arrow 60 through the C Data Interface, zero-copy, so one Arrow reader fills every `Answer`.
Polars, SQLite, PostgreSQL, ClickHouse and cuDF fill it from their own value types.

## Tests

Each backend's `taxi_fixture` test writes the four-trip table, loads it from every source the backend accepts, and checks the four Taxi answers.
With the `generate` feature, its `tpch_answers` test generates TPC-H at scale 0.01 the same way and checks every query the backend supports against the pinned answers, listing all failures and mismatches at once.
The server tests are `#[ignore]`d, since they need Docker, and `scripts/check-servers.sh` runs them; the cuDF tests need a CUDA device.
`answer.rs` pins the canonicalization: numbers across types and rounding, timestamps, row order, and nulls apart from the text `null`.
`sqlite.rs` pins its dialect rewrites, and `suite.rs` parses a pinned TPC-H answer.
