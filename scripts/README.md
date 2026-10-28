# Scripts

`plot.py` reads the JSON reports the benchmark binaries write and draws four SVG charts:

- `load-throughput.svg`: fact-table rows loaded per second.
- `query-seconds.svg`: the fastest hot run of every query, one panel per dataset, row count and query.
- `speedup-over-duckdb.svg`: each query's speedup over the baseline backend against the fact-table rows, for sweeps over `--rows`.
- `memory.svg`: peak client RSS, plus the server container's peak memory.

They use RetriEval's and CrudEval's palette, cream marks on a `#C44536` field.
Labels name the backend, thread count and source.
Queries that returned rows other than the pinned or first answer are drawn hatched, so they never read as wins.
Unsupported and failed queries are left out.

```sh
uv run scripts/plot.py results/
uv run scripts/plot.py results/ --output plots/ --baseline datafusion
```

The script declares its Python and matplotlib dependencies inline for uv, and accepts a report folder or one JSON file.

## Downloads

`download.rs` builds `tabl-download`, which fetches NYC Taxi or ClickBench and rewrites it into `<table>.part_NNN.parquet` with one fixed schema; [`data/README.md`](../data/README.md) has the recipes.

## Checks

`check-servers.sh` builds the server binaries, runs their Taxi fixture tests against fresh containers, then loads TPC-H at scale 0.01 into ClickHouse and PostgreSQL at one and four threads.
Every query is checked against the pinned answers, and any failed or mismatched query fails the script.
It needs Docker and writes its reports into a fresh temporary folder.
