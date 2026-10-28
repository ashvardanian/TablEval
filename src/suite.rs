//! The three query suites and the dataset folders they run over.
//!
//! A dataset folder holds `<table>.part_NNN.parquet` files, read in natural order. Its tables tell the
//! suite apart: `trips` is NYC Taxi, `lineitem` with the other seven TPC-H tables is TPC-H, and `hits` with
//! the `queries.sql` that `tabl-download` fetched is ClickBench.

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::{answer::Answer, backend::Table, Result};

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Suite {
    Taxi,
    Tpch,
    Clickbench,
}

impl Suite {
    /// The table `--rows` cuts, and the one throughput is counted in.
    pub fn fact_table(self) -> &'static str {
        match self {
            Self::Taxi => "trips",
            Self::Tpch => "lineitem",
            Self::Clickbench => "hits",
        }
    }
}

/// One named query; `ties` marks a `LIMIT` over a non-unique order, where only the row count is verified.
#[derive(Clone, Debug)]
pub struct Query {
    pub name: String,
    pub sql: String,
    pub ties: bool,
}

const TAXI: [(&str, &str); 4] = [
    ("count-by-vendor", include_str!("../sql/taxi/count-by-vendor.sql")),
    (
        "average-amount-by-passengers",
        include_str!("../sql/taxi/average-amount-by-passengers.sql"),
    ),
    (
        "count-by-passengers-and-year",
        include_str!("../sql/taxi/count-by-passengers-and-year.sql"),
    ),
    (
        "count-by-passengers-year-and-distance",
        include_str!("../sql/taxi/count-by-passengers-year-and-distance.sql"),
    ),
];

const TPCH: [&str; 22] = [
    include_str!("../sql/tpch/q01.sql"),
    include_str!("../sql/tpch/q02.sql"),
    include_str!("../sql/tpch/q03.sql"),
    include_str!("../sql/tpch/q04.sql"),
    include_str!("../sql/tpch/q05.sql"),
    include_str!("../sql/tpch/q06.sql"),
    include_str!("../sql/tpch/q07.sql"),
    include_str!("../sql/tpch/q08.sql"),
    include_str!("../sql/tpch/q09.sql"),
    include_str!("../sql/tpch/q10.sql"),
    include_str!("../sql/tpch/q11.sql"),
    include_str!("../sql/tpch/q12.sql"),
    include_str!("../sql/tpch/q13.sql"),
    include_str!("../sql/tpch/q14.sql"),
    include_str!("../sql/tpch/q15.sql"),
    include_str!("../sql/tpch/q16.sql"),
    include_str!("../sql/tpch/q17.sql"),
    include_str!("../sql/tpch/q18.sql"),
    include_str!("../sql/tpch/q19.sql"),
    include_str!("../sql/tpch/q20.sql"),
    include_str!("../sql/tpch/q21.sql"),
    include_str!("../sql/tpch/q22.sql"),
];

const TPCH_ANSWERS_SF0_01: [&str; 22] = [
    include_str!("../sql/tpch/answers-sf0.01/q01.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q02.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q03.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q04.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q05.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q06.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q07.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q08.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q09.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q10.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q11.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q12.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q13.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q14.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q15.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q16.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q17.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q18.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q19.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q20.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q21.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q22.csv"),
];

const TPCH_ANSWERS_SF1: [&str; 22] = [
    include_str!("../sql/tpch/answers-sf1/q01.csv"),
    include_str!("../sql/tpch/answers-sf1/q02.csv"),
    include_str!("../sql/tpch/answers-sf1/q03.csv"),
    include_str!("../sql/tpch/answers-sf1/q04.csv"),
    include_str!("../sql/tpch/answers-sf1/q05.csv"),
    include_str!("../sql/tpch/answers-sf1/q06.csv"),
    include_str!("../sql/tpch/answers-sf1/q07.csv"),
    include_str!("../sql/tpch/answers-sf1/q08.csv"),
    include_str!("../sql/tpch/answers-sf1/q09.csv"),
    include_str!("../sql/tpch/answers-sf1/q10.csv"),
    include_str!("../sql/tpch/answers-sf1/q11.csv"),
    include_str!("../sql/tpch/answers-sf1/q12.csv"),
    include_str!("../sql/tpch/answers-sf1/q13.csv"),
    include_str!("../sql/tpch/answers-sf1/q14.csv"),
    include_str!("../sql/tpch/answers-sf1/q15.csv"),
    include_str!("../sql/tpch/answers-sf1/q16.csv"),
    include_str!("../sql/tpch/answers-sf1/q17.csv"),
    include_str!("../sql/tpch/answers-sf1/q18.csv"),
    include_str!("../sql/tpch/answers-sf1/q19.csv"),
    include_str!("../sql/tpch/answers-sf1/q20.csv"),
    include_str!("../sql/tpch/answers-sf1/q21.csv"),
    include_str!("../sql/tpch/answers-sf1/q22.csv"),
];

/// The eight TPC-H tables, in the order `tabl-generate` writes them.
pub const TPCH_TABLES: [&str; 8] = [
    "region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem",
];

/// One Parquet part and the totals its footer reports.
#[derive(Clone, Debug)]
struct Part {
    path: PathBuf,
    rows: u64,
    bytes: u64,
}

/// A dataset folder: its suite, its tables' parts and its queries.
pub struct Dataset {
    pub suite: Suite,
    pub path: PathBuf,
    tables: Vec<(String, Vec<Part>)>,
    pub queries: Vec<Query>,
}

impl Dataset {
    pub fn open(path: &Path) -> Result<Self> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.extension().is_some_and(|extension| extension == "parquet") {
                files.push(path);
            }
        }
        // Natural order within one table: `part_9` before `part_10`.
        files.sort_by_key(|path| {
            let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            (table_name(&name).to_owned(), name.len(), name)
        });
        let mut tables: Vec<(String, Vec<Part>)> = Vec::new();
        for file in files {
            let name = table_name(&file.file_name().unwrap_or_default().to_string_lossy()).to_owned();
            let part = read_footer(&file)?;
            match tables.last_mut() {
                Some((last, parts)) if *last == name => parts.push(part),
                _ => tables.push((name, vec![part])),
            }
        }
        let has = |name: &str| tables.iter().any(|(table, _)| table == name);
        let suite = if has("trips") {
            Suite::Taxi
        } else if TPCH_TABLES.iter().all(|table| has(table)) {
            Suite::Tpch
        } else if has("hits") {
            Suite::Clickbench
        } else {
            return Err(format!(
                "{} holds neither `trips`, the eight TPC-H tables, nor `hits` as <table>.part_NNN.parquet",
                path.display()
            ));
        };
        let queries = match suite {
            Suite::Taxi => TAXI.iter().map(|(name, sql)| query(name, sql, false)).collect(),
            Suite::Tpch => TPCH
                .iter()
                .enumerate()
                .map(|(index, sql)| query(&format!("q{:02}", index + 1), sql, false))
                .collect(),
            Suite::Clickbench => {
                let file = path.join("queries.sql");
                let text = std::fs::read_to_string(&file)
                    .map_err(|e| format!("{}: {e}; fetch it with tabl-download", file.display()))?;
                text.lines()
                    .filter(|line| !line.trim().is_empty())
                    .enumerate()
                    .map(|(index, sql)| query(&format!("q{index:02}"), sql, sql.contains("LIMIT")))
                    .collect()
            }
        };
        Ok(Self {
            suite,
            path: path.to_path_buf(),
            tables,
            queries,
        })
    }

    /// Every table, with the fact table cut to the shortest prefix of parts holding at least `rows`.
    pub fn tables(&self, rows: Option<u64>) -> Result<Vec<Table>> {
        let fact = self.suite.fact_table();
        let mut tables = Vec::with_capacity(self.tables.len());
        for (name, parts) in &self.tables {
            let mut count = parts.len();
            if let Some(rows) = rows.filter(|_| name == fact) {
                let available: u64 = parts.iter().map(|part| part.rows).sum();
                if rows > available {
                    return Err(format!("--rows {rows} exceeds the {available} rows of `{name}`"));
                }
                let mut total = 0;
                count = parts
                    .iter()
                    .position(|part| {
                        total += part.rows;
                        total >= rows
                    })
                    .map_or(parts.len(), |index| index + 1);
            }
            let parts = &parts[..count];
            tables.push(Table {
                name: name.clone(),
                parts: parts.iter().map(|part| part.path.clone()).collect(),
                rows: parts.iter().map(|part| part.rows).sum(),
                bytes: parts.iter().map(|part| part.bytes).sum(),
            });
        }
        Ok(tables)
    }

    /// The pinned answer for TPC-H at scale factors 0.01 and 1, judged by the 1.5M orders per unit of scale.
    pub fn expected(&self, query: &Query) -> Option<Answer> {
        if self.suite != Suite::Tpch {
            return None;
        }
        let orders: u64 = self
            .tables
            .iter()
            .find(|(name, _)| name == "orders")?
            .1
            .iter()
            .map(|part| part.rows)
            .sum();
        let index = query.name.strip_prefix('q')?.parse::<usize>().ok()?.checked_sub(1)?;
        let answers = match orders {
            15_000 => &TPCH_ANSWERS_SF0_01,
            1_500_000 => &TPCH_ANSWERS_SF1,
            _ => return None,
        };
        Some(Answer::from_text(answers.get(index)?, '|'))
    }
}

fn query(name: &str, sql: &str, ties: bool) -> Query {
    Query {
        name: name.to_owned(),
        sql: sql.trim().trim_end_matches(';').to_owned(),
        ties,
    }
}

/// `lineitem.part_007.parquet` → `lineitem`, `region.parquet` → `region`.
fn table_name(file: &str) -> &str {
    let stem = file.strip_suffix(".parquet").unwrap_or(file);
    stem.split_once(".part_").map_or(stem, |(name, _)| name)
}

fn read_footer(path: &Path) -> Result<Part> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let bytes = file.metadata().map_err(|e| e.to_string())?.len();
    let reader = SerializedFileReader::new(file).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Part {
        path: path.to_path_buf(),
        rows: reader.metadata().file_metadata().num_rows() as u64,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_names_drop_part_suffixes() {
        assert_eq!(table_name("lineitem.part_007.parquet"), "lineitem");
        assert_eq!(table_name("region.parquet"), "region");
    }

    #[test]
    fn tpch_answers_parse() {
        let answer = Answer::from_text(TPCH_ANSWERS_SF1[0], '|');
        assert_eq!(answer.rows(), 4);
    }
}
