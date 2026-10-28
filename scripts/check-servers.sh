#!/bin/sh
# Loads TPC-H at scale 0.01 into each managed server and checks every answer against the pinned ones.
# Needs Docker; reports land in a fresh temporary folder, and any failed or mismatched query fails the run.
set -eu
cd "$(dirname "$0")/.."
features=clickhouse-backend,postgres-backend,generate
cargo build --release --no-default-features --features "$features"
cargo test --release --no-default-features --features "$features" --bins -- --ignored
root=$(mktemp -d "${TMPDIR:-/tmp}/tableval-servers-XXXXXX")
echo "Validation reports: $root"
target/release/tabl-generate --scale-factor 0.01 --output "$root/tpch-sf0.01"
for server in clickhouse postgres; do
    target/release/tabl-eval-$server --dataset "$root/tpch-sf0.01" --threads 1,4 --repetitions 1 \
        --data-dir "$root/$server/data" --output "$root/$server/reports"
done
echo "Every server answered TPC-H at scale 0.01 as pinned. Reports: $root"
