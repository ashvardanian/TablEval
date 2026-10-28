# Docker

These Compose files are standalone equivalents of the pinned server images the server binaries start.
The binaries manage their own containers through the Docker CLI and never read these files or attach to a server started by hand.
Use them to inspect a server's startup configuration or to query it outside a benchmark.

- [`clickhouse.yml`](clickhouse.yml): ClickHouse 25.8, with `data/tpch-sf1` mounted read-only as its `user_files_path`.
- [`postgres.yml`](postgres.yml): PostgreSQL 18.6.

```sh
docker compose -f docker/clickhouse.yml up -d
docker compose -f docker/clickhouse.yml port clickhouse 8123
docker compose -f docker/clickhouse.yml down
```

Ports bind to localhost on host ports Docker picks.
Both configurations trade durability for load speed, since every run reloads its tables from the Parquet parts.
`tabl-eval-postgres` also sets `max_worker_processes`, `max_parallel_workers` and `max_parallel_workers_per_gather` from `--threads`.
