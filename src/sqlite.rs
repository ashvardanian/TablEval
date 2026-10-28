//! SQLite through `rusqlite`, bundled: the single-threaded row-store baseline among embedded engines.
//!
//! SQLite has no date type, so dates and timestamps are stored as ISO text, and three spellings in the
//! shared SQL are rewritten before a query runs: `CAST('1998-09-02' AS date)` becomes the bare literal,
//! `EXTRACT(YEAR FROM x)` becomes `CAST(strftime('%Y', x) AS INTEGER)`, and `SUBSTRING(x FROM a FOR b)`
//! becomes `substr(x, a, b)`.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features sqlite-backend --bin tabl-eval-sqlite -- \
//!     --dataset data/tpch-sf1 --threads 1
//! ```

use std::{alloc::System, collections::BTreeMap, path::Path, time::Instant};

use clap::Parser;
use rusqlite::{functions::FunctionFlags, types::ValueRef, Connection};
use serde_json::{json, Value};

use tableval::{
    rows, run, unsupported, Answer, Backend, CommonArgs, Configuration, Executed, Query, Result, Source, Table,
};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    /// Page cache size.
    #[arg(long, env = "TABLEVAL_CACHE_SIZE", default_value = "1GB", value_parser = parse_cache_size)]
    cache_size: u64,
}

fn parse_cache_size(text: &str) -> Result<u64> {
    let split = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
    let level = ["", "kb", "mb", "gb", "tb"]
        .iter()
        .position(|unit| text[split..].eq_ignore_ascii_case(unit));
    level
        .and_then(|level| text[..split].parse::<u64>().ok()?.checked_mul(1u64 << (10 * level)))
        .filter(|&bytes| bytes != 0)
        .ok_or_else(|| "expected a size like 64MB or 1GB".into())
}

struct SqliteBackend {
    connection: Connection,
    cache_size: u64,
    path: std::path::PathBuf,
}

impl SqliteBackend {
    fn open(configuration: &Configuration, path: &Path, cache_size: u64) -> Result<Self> {
        if configuration.threads.get() != 1 {
            return Err("SQLite runs each query on one thread; pass --threads 1".into());
        }
        let connection = Connection::open(path.join("tableval.sqlite")).map_err(|e| e.to_string())?;
        connection
            .execute_batch(&format!(
                "PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF; PRAGMA cache_size = -{};",
                cache_size / 1024
            ))
            .map_err(|e| e.to_string())?;
        // The bundled build leaves out SQLite's math functions, and the Taxi queries floor distances.
        connection
            .create_scalar_function("floor", 1, FunctionFlags::SQLITE_DETERMINISTIC, |context| {
                Ok(context.get::<Option<f64>>(0)?.map(f64::floor))
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            connection,
            cache_size,
            path: path.to_path_buf(),
        })
    }
}

/// Finds `pattern` ignoring ASCII case, from byte `from`.
fn find_ignoring_case(text: &str, pattern: &str, from: usize) -> Option<usize> {
    let (text, pattern) = (text.as_bytes(), pattern.as_bytes());
    (from..text.len().saturating_sub(pattern.len()) + 1)
        .find(|&start| text[start..start + pattern.len()].eq_ignore_ascii_case(pattern))
}

/// Replaces each `opening … )` call, whose arguments hold no parentheses, with `rewrite(arguments)`.
fn rewrite_calls(sql: &str, opening: &str, rewrite: impl Fn(&str) -> Option<String>) -> String {
    let mut result = String::with_capacity(sql.len());
    let mut cursor = 0;
    while let Some(start) = find_ignoring_case(sql, opening, cursor) {
        let arguments = start + opening.len();
        let Some(end) = sql[arguments..].find(')').map(|offset| arguments + offset) else {
            break;
        };
        match rewrite(&sql[arguments..end]) {
            Some(replacement) => {
                result.push_str(&sql[cursor..start]);
                result.push_str(&replacement);
            }
            None => result.push_str(&sql[cursor..=end]),
        }
        cursor = end + 1;
    }
    result.push_str(&sql[cursor..]);
    result
}

/// The shared SQL in SQLite's spelling.
fn dialect(sql: &str) -> String {
    let split = |arguments: &str, separator: &str| {
        let position = find_ignoring_case(arguments, separator, 0)?;
        Some((
            arguments[..position].trim().to_owned(),
            arguments[position + separator.len()..].trim().to_owned(),
        ))
    };
    let sql = rewrite_calls(sql, "CAST(", |arguments| {
        let (value, target) = split(arguments, " AS ")?;
        target.eq_ignore_ascii_case("date").then_some(value)
    });
    let sql = rewrite_calls(&sql, "EXTRACT(", |arguments| {
        let (field, value) = split(arguments, " FROM ")?;
        field
            .eq_ignore_ascii_case("year")
            .then(|| format!("CAST(strftime('%Y', {value}) AS INTEGER)"))
    });
    rewrite_calls(&sql, "SUBSTRING(", |arguments| {
        let (value, range) = split(arguments, " FROM ")?;
        let (first, length) = split(&range, " FOR ")?;
        Some(format!("substr({value}, {first}, {length})"))
    })
}

impl Backend for SqliteBackend {
    fn metadata(&self) -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("backend".into(), json!("sqlite")),
            ("library_version".into(), json!(rusqlite::version())),
            ("threads".into(), json!(1)),
            ("cache_size".into(), json!(self.cache_size)),
        ])
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        if source == Source::Parquet {
            return Err(unsupported("SQLite reads only its own tables"));
        }
        for table in tables {
            let columns = rows::columns(table)?;
            let declarations: Vec<String> = columns
                .iter()
                .map(|(name, kind)| {
                    let declared = match kind {
                        rows::Kind::Integer => "INTEGER",
                        rows::Kind::Real => "REAL",
                        _ => "TEXT",
                    };
                    format!("\"{name}\" {declared}")
                })
                .collect();
            let placeholders = vec!["?"; columns.len()].join(", ");
            let transaction = self.connection.transaction().map_err(|e| e.to_string())?;
            transaction
                .execute_batch(&format!("CREATE TABLE {} ({})", table.name, declarations.join(", ")))
                .map_err(|e| e.to_string())?;
            {
                let mut insert = transaction
                    .prepare(&format!("INSERT INTO {} VALUES ({placeholders})", table.name))
                    .map_err(|e| e.to_string())?;
                rows::for_each_row(table, |cells| {
                    let values = cells.iter().map(|cell| match *cell {
                        rows::Cell::Null => rusqlite::types::Value::Null,
                        rows::Cell::Integer(value) => rusqlite::types::Value::Integer(value),
                        rows::Cell::Real(value) => rusqlite::types::Value::Real(value),
                        rows::Cell::Text(value) => rusqlite::types::Value::Text(value.to_owned()),
                    });
                    insert
                        .execute(rusqlite::params_from_iter(values))
                        .map(drop)
                        .map_err(|e| e.to_string())
                })?;
            }
            transaction.commit().map_err(|e| e.to_string())?;
        }
        self.connection.execute_batch("ANALYZE").map_err(|e| e.to_string())
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        let sql = dialect(&query.sql);
        let start = Instant::now();
        let mut statement = self.connection.prepare(&sql).map_err(|e| match e.to_string() {
            message if message.contains("no such function") => unsupported(&message),
            message => message,
        })?;
        let width = statement.column_count();
        let mut cells = Vec::new();
        let mut rows = statement.query([]).map_err(|e| e.to_string())?;
        while let Some(row) = rows.next().map_err(|e| e.to_string())? {
            for column in 0..width {
                cells.push(
                    row.get::<_, rusqlite::types::Value>(column)
                        .map_err(|e| e.to_string())?,
                );
            }
        }
        let executed = Executed::since(start);
        for row in cells.chunks(width.max(1)) {
            for value in row {
                match ValueRef::from(value) {
                    ValueRef::Null => answer.push_null(),
                    ValueRef::Integer(value) => answer.push_i64(value),
                    ValueRef::Real(value) => answer.push_f64(value),
                    ValueRef::Text(text) | ValueRef::Blob(text) => answer.push_str(&String::from_utf8_lossy(text)),
                }
            }
            answer.end_row();
        }
        Ok(executed)
    }

    fn disk_bytes(&self) -> Result<u64> {
        tableval::backend::directory_bytes(&self.path)
    }
}

fn main() -> Result<()> {
    let cli: Cli = tableval::parse_cli();
    let settings = [("Cache size", cli.cache_size.to_string())];
    run(
        cli.common,
        json!({"cache_size": cli.cache_size}),
        &settings,
        |configuration, path| {
            Ok(Box::new_in(
                SqliteBackend::open(configuration, path, cli.cache_size)?,
                System,
            ))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect_rewrites_dates_years_and_substrings() {
        assert_eq!(
            dialect("WHERE d <= CAST('1998-09-02' AS date) AND CAST(x AS INTEGER) > 0"),
            "WHERE d <= '1998-09-02' AND CAST(x AS INTEGER) > 0"
        );
        assert_eq!(
            dialect("SELECT extract(year FROM o_orderdate) AS y"),
            "SELECT CAST(strftime('%Y', o_orderdate) AS INTEGER) AS y"
        );
        assert_eq!(
            dialect("substring(c_phone FROM 1 FOR 2) IN ('13')"),
            "substr(c_phone, 1, 2) IN ('13')"
        );
    }

    fn open(
        configuration: &tableval::Configuration,
        path: &std::path::Path,
    ) -> tableval::Result<Box<dyn tableval::Backend, System>> {
        let configuration = Configuration {
            threads: tableval::Threads(std::num::NonZeroUsize::MIN),
            ..*configuration
        };
        Ok(Box::new_in(SqliteBackend::open(&configuration, path, 1 << 20)?, System))
    }

    #[test]
    fn taxi_fixture() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::assert_taxi_answers(root.path(), open);
    }

    #[cfg(feature = "generate")]
    #[test]
    fn tpch_answers() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::assert_tpch_answers(root.path(), open);
    }
}
