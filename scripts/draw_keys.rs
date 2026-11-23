//! Draws the keys every backend of the documents suite gathers, once per collection and count, with a fixed seed.
//!
//! The inputs are the partition folders a load read, `<collection>/p<partition>` with key-sorted Arrow files holding a
//! `key` column. Each count lands as `<count>.arrow` in `--output`, the dataset folder `--suite documents` reads,
//! with every partition's document count in its metadata, so scans verify their rows against it.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --bin tabl-draw-keys -- \
//!     --inputs '/mnt/nvme*/docs-bench/input/alphafold/p*' --counts 1K,100K,10M --output data/keys/alphafold
//! ```

use std::path::PathBuf;

use clap::Parser;

use tableval::{parse_count_flag, spell_count, suite::Draw, Result};

#[derive(Parser)]
struct Cli {
    /// A glob of the partition folders, `<collection>/p<partition>` each, every partition named once; files it matches
    /// are skipped.
    #[arg(long, env = "TABLEVAL_INPUTS")]
    inputs: String,
    /// Keys drawn per file, with decimal K, M, G or T.
    #[arg(long, env = "TABLEVAL_COUNTS", value_delimiter = ',', required = true, value_parser = parse_count_flag)]
    counts: Vec<u64>,
    /// Folder the draws land in, one `<count>.arrow` each.
    #[arg(long, env = "TABLEVAL_OUTPUT")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    let mut partitions: Vec<(usize, PathBuf)> = Vec::new();
    for path in glob::glob(&cli.inputs).map_err(|e| format!("--inputs: {e}"))? {
        let path = path.map_err(|e| e.to_string())?;
        // Listings such as `p4.arrow.list` sit beside the partition folders and match the same glob
        if !path.is_dir() {
            continue;
        }
        let partition = path
            .file_name()
            .and_then(|name| name.to_str()?.strip_prefix('p')?.parse().ok())
            .ok_or_else(|| format!("{} is no p<partition> folder", path.display()))?;
        partitions.push((partition, path));
    }
    partitions.sort();
    if partitions
        .iter()
        .enumerate()
        .any(|(index, (partition, _))| index != *partition)
    {
        return Err("--inputs must name the partitions p0, p1, … once each".into());
    }
    let folders: Vec<PathBuf> = partitions.into_iter().map(|(_, folder)| folder).collect();
    std::fs::create_dir_all(&cli.output).map_err(|e| format!("{}: {e}", cli.output.display()))?;
    for (draw, &count) in Draw::sample(&folders, &cli.counts)?.iter().zip(&cli.counts) {
        let path = cli.output.join(format!("{}.arrow", spell_count(count)));
        draw.write(&path)?;
        let drawn: usize = draw.keys.iter().map(Vec::len).sum();
        let rows: u64 = draw.rows.iter().sum();
        eprintln!("{}: {drawn} of {rows} {} documents", path.display(), draw.collection);
    }
    Ok(())
}
