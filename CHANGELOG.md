# Changelog

Runs of TablEval's backends, one dated section per run day, newest first.
Every run loads its tables once, times each query cold and then hot, and verifies answers or digests unless stated.
Times are seconds, cold then the best hot run.

## November 19, 2026

### Documents on Graviton4

The documents suite on AWS `i8ge.metal-24xl`: one Graviton4 socket of 96 Neoverse V2 cores, one NUMA domain, 84 threads, and eight local NVMe drives with one ext4 file system each.
Each collection is split into 8 partitions by a hash of the key, one store per partition and drive, and every query runs over all partitions at once.
UStore ran at `83a093a` and on a working tree ahead of it, which made each partition's executor on that partition's own cores and gathered each partition's keys in one call.
Runs began on November 18; 600 s capped a cold run, and a capped query skipped its hot runs.
Every keyed-gather and scan digest agrees across the systems that ran it.

AlphaFold, 220,322,986 documents:

| Query                  | UStore NVMe   | UStore RocksDB | ClickHouse String | ClickHouse JSON | MongoDB       |
| :--------------------- | ------------: | -------------: | ----------------: | --------------: | ------------: |
| `keyed-gather-1K-f1`   | 0.110 / 0.043 |  0.013 / 0.004 |       0.27 / 0.07 |     0.12 / 0.01 |   0.04 / 0.00 |
| `keyed-gather-100K-f1` | 0.444 / 0.103 |  0.082 / 0.022 |       3.72 / 1.62 |     0.35 / 0.14 |   0.52 / 0.02 |
| `keyed-gather-10M-f1`  |   6.81 / 6.61 |    5.63 / 1.49 |       5.12 / 2.31 |     0.82 / 0.60 |   8.55 / 2.57 |
| `keyed-gather-10M-all` |   9.90 / 9.19 |    6.92 / 3.26 |       5.18 / 2.90 |     4.40 / 2.46 |   8.22 / 2.24 |
| `sample-gather-1K-f3`  | 0.122 / 0.043 | 210.54 / 10.68 |       4.11 / 1.70 |     0.58 / 0.28 |   0.08 / 0.06 |
| `sample-gather-10M-f3` |   7.32 / 7.47 | 211.52 / 13.44 |      10.08 / 9.09 |     1.76 / 1.42 |  63.18 / 5.54 |
| `scan-f1`              | 43.27 / 42.75 |   32.33 / 9.85 |       5.17 / 2.89 |     1.15 / 1.07 | 51.84 / 36.18 |
| `scan-f3`              | 45.40 / 44.87 |  34.51 / 10.81 |       7.60 / 5.55 |     2.47 / 2.38 | 57.19 / 39.43 |

OAS unpaired, 2,146,235,273 documents with 25 flat fields:

| Query                  | UStore NVMe          | UStore RocksDB  | ClickHouse String |
| :--------------------- | -------------------: | --------------: | ----------------: |
| `keyed-gather-1K-f1`   |        0.021 / 0.001 |   0.014 / 0.004 |     0.201 / 0.068 |
| `keyed-gather-100K-f1` |        0.073 / 0.057 |   0.078 / 0.023 |       5.08 / 3.07 |
| `keyed-gather-10M-f1`  |          5.92 / 5.86 |     6.53 / 1.51 |     20.77 / 13.73 |
| `keyed-gather-10M-all` |         10.07 / 9.74 |     9.08 / 5.39 |     22.92 / 15.70 |
| `sample-gather-10M-f3` |          6.16 / 6.14 |         not run |     49.28 / 42.75 |
| `scan-f1`              | 862.15, capped       | 310.42 / 343.13 |     33.43 / 27.11 |
| `scan-f3`              | 878.16, capped       |         not run |     59.64 / 54.13 |

- At `83a093a` with the harness's first layout, ten one-thread executors per partition, UStore NVMe took 51.47 s cold for `keyed-gather-10M-f1` and hit the cap on `scan-f1`; their spinning workers oversubscribed the 84 cores.
- UStore NVMe scans walk cursor pages serially on the calling thread, so a parallel scan needs a values output on `ustore_docs_scan_t` or a multi-range cursor.
- In the 8-partition gather each drive serves about 180K reads/s, while one partition alone reaches its drive's 345K, so the remaining NVMe gather gap is not the device.
- UStore RocksDB samples walk every key, about 210 s cold; its block cache is sized from the memory budget.
- The 8 UStore NVMe stores over OAS take 90.8 s to open and hold about 153 GB resident, about 71 bytes per key; the times above exclude opening.
- ClickHouse JSON stores a native `JSON` column as typed subcolumns, its columnar best case; ClickHouse String is the like-for-like row against schemaless documents.
- MongoDB runs with `--wiredTigerCacheSizeGB` derived from the budget, and never loaded OAS.

These ran on an unpublished UStore tree, so none of them is a Results row.

## November 6, 2026

### UStore Backend Check

`tabl-eval-ustore` on the Razer, with an RTX 4060 Laptop GPU and one CA6-8D1024 NVMe drive, over TPC-H at scale 1: 6,001,215 `lineitem` rows stored as 2.0 GB of JSON documents.
Both gathers matched the pinned answers, with digests `fddc68cd` for `q01` and `2872c1fb` for `q06`.

| Gather | Load  | `q01`        | `q06`        |
| :----- | ----: | -----------: | -----------: |
| GPU    | 21.6  | 27.39, 27.27 | 27.00, 26.97 |
| CPU    | 19.6  | 35.71, 35.68 | 32.83, 34.64 |

- Reading the drive took about 23 of the 27 s: the engine read each document alone, 64 in flight, moving 26 GB for 2.0 GB of documents.
- Reading each window's keys 256 documents apart, rather than in key order, kept the 64 reads off the same flash pages and cut `q01` from 108.7 s to 27 s on the GPU.
- Hot equals cold, since a batched read of the NVMe engine caches nothing.

One query pair at scale 1 on a laptop is a backend check, not a Results row.
