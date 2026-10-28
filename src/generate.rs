//! Generates TPC-H tables as Parquet parts through `tpchgen-arrow`, which matches `dbgen` row for row.
//!
//! The six scaling tables are split into one part per unit of scale, about 6M `lineitem` rows each, and
//! the parts are generated in parallel. Strings are written as plain `Utf8`, so every reader sees one layout.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features generate --bin tabl-generate -- \
//!     --scale-factor 10 --output data/tpch-sf10
//! ```

use std::{path::PathBuf, sync::Mutex};

use clap::Parser;
use forkunion::{ThreadPool, Topology};

use tableval::{parse_threads_flag, suite::TPCH_TABLES, tpch::write_part, Result, Threads};

#[derive(Parser)]
struct Cli {
    /// TPC-H scale factor; 1 is about 1 GB of text, and 0.01 and 1 have pinned answers.
    #[arg(long, env = "TABLEVAL_SCALE_FACTOR")]
    scale_factor: f64,
    /// Destination folder; `data/tpch-sf<scale>` when unset.
    #[arg(long, env = "TABLEVAL_OUTPUT")]
    output: Option<PathBuf>,
    /// Generator threads, 0 for all cores.
    #[arg(long, env = "TABLEVAL_THREADS", default_value = "0", value_parser = parse_threads_flag)]
    threads: Threads,
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    if !cli.scale_factor.is_finite() || cli.scale_factor <= 0.0 {
        return Err(format!(
            "--scale-factor=\"{}\" does not parse, expected a positive scale",
            cli.scale_factor
        ));
    }
    let folder = cli
        .output
        .unwrap_or_else(|| PathBuf::from(format!("data/tpch-sf{}", cli.scale_factor)));
    std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    let parts = cli.scale_factor.ceil().max(1.0) as i32;
    let jobs: Vec<(&str, i32, i32)> = TPCH_TABLES
        .iter()
        .flat_map(|&table| {
            let count = match table {
                "region" | "nation" => 1,
                _ => parts,
            };
            (1..=count).map(move |part| (table, part, count))
        })
        .collect();
    eprintln!("- Scale factor: {}", cli.scale_factor);
    eprintln!("- Output: {}", folder.display());
    eprintln!("- Threads: {}", cli.threads);
    eprintln!("- Parts: {}", jobs.len());
    let mut pool = Topology::new()
        .and_then(|topology| ThreadPool::spawn(&topology, cli.threads.get()))
        .map_err(|e| format!("ForkUnion: {e:?}"))?;
    let failures = Mutex::new(Vec::new());
    pool.for_n_dynamic(jobs.len(), |job, _at| {
        let (table, part, count) = jobs[job];
        match write_part(&folder, table, cli.scale_factor, part, count) {
            Ok(rows) => eprintln!("{table} part {part}/{count}: {rows} rows"),
            Err(error) => failures.lock().unwrap().push(format!("{table} part {part}: {error}")),
        }
    })
    .map_err(|e| format!("ForkUnion: {e:?}"))?;
    let failures = failures.into_inner().unwrap();
    match failures.is_empty() {
        true => Ok(()),
        false => Err(failures.join("\n")),
    }
}
