//! Shared benchmark runner: CLI sweeps, the load phase, cold and hot query runs, and verified answers.

pub mod answer;
pub mod backend;
#[cfg(feature = "tier2")]
pub mod docker;
pub mod fixture;
pub mod measure;
pub mod output;
pub mod perf_counters;
pub mod rows;
pub mod suite;
#[cfg(feature = "generate")]
pub mod tpch;

use std::{
    alloc::System,
    fmt,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use clap::Args;

pub use crate::{
    answer::Answer,
    backend::{unsupported, Backend, Executed, Result, Source, Table},
    suite::{Dataset, Query, Suite},
};
use crate::{
    backend::UNSUPPORTED,
    measure::ResourceSampler,
    output::{ConfigReport, DatasetInfo, PhaseReport, QueryReport, Status},
    perf_counters::PerfCounters,
};

/// Version of the report layout, bumped when a field changes meaning.
const REPORT_SCHEMA_VERSION: u32 = 1;

/// Whether result rows are checked.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verify {
    /// Pinned answers where they exist, and agreement between the cold and hot runs everywhere.
    Answers,
    /// No digests, to measure their cost separately.
    None,
}

#[derive(Args, Clone, Debug, serde::Serialize)]
pub struct CommonArgs {
    /// Dataset folder of `<table>.part_NNN.parquet` files; its tables pick the suite.
    #[arg(long, env = "TABLEVAL_DATASET")]
    pub dataset: PathBuf,
    /// Fact-table row prefixes in whole parts, with decimal K, M, G or T; all rows when unset; a sweep.
    #[arg(long, env = "TABLEVAL_ROWS", value_delimiter = ',', value_parser = parse_count_flag)]
    pub rows: Vec<u64>,
    /// Engine threads, 0 for all cores; a sweep.
    #[arg(long, env = "TABLEVAL_THREADS", value_delimiter = ',', default_value = "0", value_parser = parse_threads_flag)]
    pub threads: Vec<Threads>,
    /// loaded ingests the parts first, parquet queries them in place; a sweep.
    #[arg(
        long,
        env = "TABLEVAL_SOURCE",
        value_enum,
        value_delimiter = ',',
        default_value = "loaded"
    )]
    pub source: Vec<Source>,
    /// Query names in order; every query of the suite when unset.
    #[arg(long, env = "TABLEVAL_QUERIES", value_delimiter = ',')]
    pub queries: Vec<String>,
    /// Hot runs after each cold run.
    #[arg(long, env = "TABLEVAL_REPETITIONS", default_value = "3", value_parser = parse_repetitions_flag)]
    pub repetitions: usize,
    /// A cold run slower than this, like 60s, skips its hot runs; it never interrupts a running query.
    #[arg(long, env = "TABLEVAL_TIME_LIMIT", value_parser = parse_duration_flag)]
    pub time_limit: Option<Duration>,
    /// answers checks result digests, none skips them.
    #[arg(long, env = "TABLEVAL_VERIFY", value_enum, default_value = "answers")]
    pub verify: Verify,
    /// Parent directory for the engines' own storage, one marked folder per configuration.
    #[arg(long, env = "TABLEVAL_DATA_DIR", default_value = "data")]
    pub data_dir: PathBuf,
    /// Directory for JSON reports.
    #[arg(long, env = "TABLEVAL_OUTPUT", default_value = "results")]
    pub output: PathBuf,
    /// Record system-wide hardware counters per phase; requires Linux and the perf-counters feature.
    #[arg(long, env = "TABLEVAL_PERF_COUNTERS", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new())]
    pub perf_counters: bool,
    /// Sync and drop the host page cache before each query's cold run; Linux as root only.
    #[arg(long, env = "TABLEVAL_DROP_CACHES", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new())]
    pub drop_caches: bool,
}

/// One point of the sweep, as a backend sees it when it opens.
#[derive(Clone, Copy, Debug)]
pub struct Configuration {
    pub rows: Option<u64>,
    pub threads: Threads,
    pub source: Source,
}

/// A thread count; `0` on the command line resolves to every core, so it is never zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Threads(pub NonZeroUsize);

impl Threads {
    pub fn get(self) -> usize {
        self.0.get()
    }
}

impl fmt::Display for Threads {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Parses the command line; a bad value prints `--name="value" does not parse, expected ...` and exits with 1.
pub fn parse_cli<T: clap::Parser>() -> T {
    use clap::error::{ContextKind, ContextValue, ErrorKind};

    let error = match T::try_parse() {
        Ok(cli) => return cli,
        Err(error) if !error.use_stderr() => error.exit(),
        Err(error) => error,
    };
    let context = |kind| match error.get(kind) {
        Some(ContextValue::String(text)) => text.as_str(),
        _ => "",
    };
    let expected = match (
        error.kind(),
        std::error::Error::source(&error),
        error.get(ContextKind::ValidValue),
    ) {
        (ErrorKind::ValueValidation, Some(expected), _) => expected.to_string(),
        (ErrorKind::InvalidValue, _, Some(ContextValue::Strings(values))) if !values.is_empty() => {
            format!("expected one of {}", values.join(", "))
        }
        (ErrorKind::InvalidValue, _, _) => "expected a non-empty value".into(),
        _ => {
            eprint!("{error}");
            std::process::exit(1)
        }
    };
    let flag = context(ContextKind::InvalidArg).split(' ').next().unwrap_or_default();
    eprintln!(
        "{flag}=\"{}\" does not parse, {expected}",
        context(ContextKind::InvalidValue)
    );
    std::process::exit(1)
}

/// Parses a count like `10000` or `10M`: `K`, `M`, `G`, `T` are powers of 1000 in any case, and zero is `None`.
pub fn parse_count(text: &str) -> Option<u64> {
    let split = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
    let exponent = ["", "k", "m", "g", "t"]
        .iter()
        .position(|unit| text[split..].eq_ignore_ascii_case(unit))?;
    let count = text[..split]
        .parse::<u64>()
        .ok()?
        .checked_mul(1000u64.pow(exponent as u32))?;
    (count != 0).then_some(count)
}

/// Parses a duration like `200ms` or `10s`; a bare number, a fraction or zero is `None`.
pub fn parse_duration(text: &str) -> Option<Duration> {
    let digits = |text: &str| parse_count(text).filter(|_| text.bytes().all(|byte| byte.is_ascii_digit()));
    match text.strip_suffix("ms") {
        Some(count) => digits(count).map(Duration::from_millis),
        None => digits(text.strip_suffix('s')?).map(Duration::from_secs),
    }
}

/// Spells a duration the way `parse_duration` reads it: `1s`, `1500ms`.
pub fn spell_duration(duration: Duration) -> String {
    let milliseconds = duration.as_millis();
    match milliseconds % 1000 {
        0 => format!("{}s", milliseconds / 1000),
        _ => format!("{milliseconds}ms"),
    }
}

/// Spells a count the way `parse_count` reads it: `100M`, `1500`.
pub fn spell_count(count: u64) -> String {
    let (mut count, mut unit) = (count, "");
    for larger in ["K", "M", "G", "T"] {
        if count == 0 || count % 1000 != 0 {
            break;
        }
        count /= 1000;
        unit = larger;
    }
    format!("{count}{unit}")
}

/// `parse_count` for clap.
pub fn parse_count_flag(text: &str) -> Result<u64> {
    parse_count(text).ok_or_else(|| "expected a positive count like 1500 or 100M".into())
}

/// Parses a thread count for clap, with `0` resolved to every available core.
pub fn parse_threads_flag(text: &str) -> Result<Threads> {
    let all_cores = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    match text {
        "0" => Ok(Threads(all_cores)),
        _ => parse_count(text)
            .filter(|_| text.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|count| NonZeroUsize::new(count as usize))
            .map(Threads)
            .ok_or_else(|| "expected a count, 0 for all cores".into()),
    }
}

fn parse_repetitions_flag(text: &str) -> Result<usize> {
    match text {
        "0" => Ok(0),
        _ => parse_count(text)
            .filter(|_| text.bytes().all(|byte| byte.is_ascii_digit()))
            .map(|count| count as usize)
            .ok_or_else(|| "expected a count of hot runs, 0 for the cold run alone".into()),
    }
}

/// `parse_duration` for clap.
pub fn parse_duration_flag(text: &str) -> Result<Duration> {
    parse_duration(text).ok_or_else(|| "expected a duration like 200ms or 60s".into())
}

/// Spells a `clap::ValueEnum` value the way the command line takes it.
pub fn spell_value<T: clap::ValueEnum>(value: &T) -> String {
    value
        .to_possible_value()
        .map_or_else(String::new, |value| value.get_name().to_owned())
}

fn spell_list<T: fmt::Display>(values: &[T]) -> String {
    values.iter().map(T::to_string).collect::<Vec<_>>().join(",")
}

/// Runs every configuration; `backend_settings` are echoed after the common settings as "- Name: value".
pub fn run(
    args: CommonArgs,
    backend_options: impl serde::Serialize,
    backend_settings: &[(&str, String)],
    open: impl Fn(&Configuration, &Path) -> Result<Box<dyn Backend, System>>,
) -> Result<()> {
    if args.perf_counters && !cfg!(all(target_os = "linux", feature = "perf-counters")) {
        return Err("--perf-counters requires Linux and the perf-counters feature".into());
    }
    if args.drop_caches && !cfg!(target_os = "linux") {
        return Err("--drop-caches requires Linux".into());
    }
    let backend_options = serde_json::to_value(backend_options).map_err(|e| e.to_string())?;
    let dataset = Dataset::open(&args.dataset)?;
    if dataset.suite == Suite::Tpch && !args.rows.is_empty() {
        return Err("TPC-H scales by folder, such as data/tpch-sf10, so --rows does not apply".into());
    }
    let queries = match args.queries.is_empty() {
        true => dataset.queries.clone(),
        false => args
            .queries
            .iter()
            .map(|name| {
                dataset
                    .queries
                    .iter()
                    .find(|query| query.name == *name)
                    .cloned()
                    .ok_or_else(|| format!("--queries=\"{name}\" names no {:?} query", dataset.suite))
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let unset = || "all".to_owned();
    eprintln!(
        "- Dataset: {} ({})",
        args.dataset.display(),
        spell_value(&dataset.suite)
    );
    eprintln!(
        "- Rows: {}",
        match args.rows.is_empty() {
            true => unset(),
            false => args
                .rows
                .iter()
                .map(|&rows| spell_count(rows))
                .collect::<Vec<_>>()
                .join(","),
        }
    );
    eprintln!("- Threads: {}", spell_list(&args.threads));
    eprintln!(
        "- Source: {}",
        args.source.iter().map(spell_value).collect::<Vec<_>>().join(",")
    );
    eprintln!(
        "- Queries: {}",
        queries
            .iter()
            .map(|query| query.name.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    eprintln!("- Repetitions: {}", args.repetitions);
    eprintln!(
        "- Time limit: {}",
        args.time_limit.map_or_else(|| "none".to_owned(), spell_duration)
    );
    eprintln!("- Verify: {}", spell_value(&args.verify));
    eprintln!("- Data dir: {}", args.data_dir.display());
    eprintln!("- Output: {}", args.output.display());
    eprintln!("- Perf counters: {}", args.perf_counters);
    eprintln!("- Drop caches: {}", args.drop_caches);
    for (name, value) in backend_settings {
        eprintln!("- {name}: {value}");
    }
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let engine = executable.file_stem().and_then(|s| s.to_str()).unwrap_or("tableval");
    let rows_sweep: Vec<Option<u64>> = match args.rows.is_empty() {
        true => vec![None],
        false => args.rows.iter().copied().map(Some).collect(),
    };
    let mut counters = match args.perf_counters {
        true => Some(PerfCounters::new().map_err(|e| format!("--perf-counters: {e}"))?),
        false => None,
    };
    let mut failures = 0;
    let mut last_report = None;
    for &rows in &rows_sweep {
        let tables = dataset.tables(rows)?;
        let fact = tables
            .iter()
            .find(|table| table.name == dataset.suite.fact_table())
            .ok_or("the dataset lost its fact table")?;
        let (fact_rows, total_bytes) = (fact.rows, tables.iter().map(|table| table.bytes).sum::<u64>());
        for &threads in &args.threads {
            for &source in &args.source {
                let configuration = Configuration { rows, threads, source };
                let mut workload = args.clone();
                workload.rows = rows.into_iter().collect();
                workload.threads = vec![threads];
                workload.source = vec![source];
                let identity = output::config_hash(&(&args.dataset, rows, threads, source, &backend_options))?;
                let path = args.data_dir.join(format!("{engine}-{identity}"));
                if path.exists() {
                    std::fs::remove_dir_all(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                }
                std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
                let mut backend = open(&configuration, &path)?;
                let mut report = ConfigReport {
                    schema_version: REPORT_SCHEMA_VERSION,
                    started_unix_seconds: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    machine: output::collect_machine_info(),
                    dataset: DatasetInfo {
                        suite: dataset.suite,
                        path: args.dataset.clone(),
                        tables: tables.clone(),
                    },
                    workload,
                    config: backend.metadata(),
                    load: None,
                    queries: Vec::new(),
                };
                report.config.insert("backend_options".into(), backend_options.clone());
                if args.perf_counters {
                    report.config.insert(
                        "hardware_counters_scope".into(),
                        "every online CPU, other processes included".into(),
                    );
                }
                eprintln!(
                    "{engine} · {} rows · {threads} threads · {}",
                    spell_count(fact_rows),
                    spell_value(&source)
                );
                let load = measure_load(
                    backend.as_mut(),
                    &tables,
                    source,
                    counters.as_mut(),
                    fact_rows,
                    total_bytes,
                )?;
                eprintln!(
                    "{:<40} {:>12.3} {:>12} {:>10} {}",
                    "load",
                    load.elapsed_seconds,
                    "",
                    spell_count(load.rows),
                    spell_value_status(load.status)
                );
                let loaded = load.status == Status::Completed;
                failures += usize::from(load.status == Status::Failed);
                report.load = Some(load);
                last_report = Some(output::write_report(&args.output, &report)?);
                if !loaded {
                    continue;
                }
                eprintln!(
                    "{:<40} {:>12} {:>12} {:>10} status",
                    "query", "cold (s)", "hot min (s)", "rows"
                );
                for query in &queries {
                    let expected = match args.verify {
                        Verify::Answers => dataset.expected(query),
                        Verify::None => None,
                    };
                    let result = measure_query(backend.as_mut(), query, expected, &args, counters.as_mut())?;
                    let hot = result.hot_seconds.iter().copied().fold(f64::INFINITY, f64::min);
                    eprintln!(
                        "{:<40} {:>12} {:>12} {:>10} {}",
                        query.name,
                        result
                            .cold_seconds
                            .map_or_else(String::new, |seconds| format!("{seconds:.3}")),
                        match hot.is_finite() {
                            true => format!("{hot:.3}"),
                            false => String::new(),
                        },
                        result.rows,
                        spell_value_status(result.status)
                    );
                    if let Some(error) = &result.error {
                        if result.status != Status::Unsupported {
                            eprintln!("  {error}");
                        }
                    }
                    failures += usize::from(matches!(result.status, Status::Failed | Status::Mismatch));
                    report.queries.push(result);
                    last_report = Some(output::write_report(&args.output, &report)?);
                }
            }
        }
    }
    match (failures, last_report) {
        (0, _) => Ok(()),
        (failures, Some(path)) => Err(format!(
            "{failures} phases failed or mismatched; report: {}",
            path.display()
        )),
        (failures, None) => Err(format!("{failures} phases failed or mismatched")),
    }
}

/// Writes dirty pages back and empties the host page cache, so the next read comes from the device.
fn drop_page_cache() -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        unsafe {
            libc::sync();
        }
        std::fs::write("/proc/sys/vm/drop_caches", b"3\n").map_err(|e| format!("--drop-caches: {e}"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("--drop-caches requires Linux".into())
    }
}

fn spell_value_status(status: Status) -> String {
    serde_json::to_value(status)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn status_of(error: &str) -> Status {
    match error.starts_with(UNSUPPORTED) {
        true => Status::Unsupported,
        false => Status::Failed,
    }
}

fn start_counters(counters: Option<&mut PerfCounters>) -> Result<()> {
    match counters {
        Some(counters) => counters.reset_and_enable().map_err(|e| e.to_string()),
        None => Ok(()),
    }
}

fn read_counters(counters: Option<&mut PerfCounters>) -> Result<Option<perf_counters::CounterSample>> {
    counters
        .map(|counters| counters.disable_and_read().map_err(|e| e.to_string()))
        .transpose()
}

/// The server's usage sampled through its cgroup, or the backend's own snapshot where no cgroup is reachable.
fn server_usage(backend: &dyn Backend, sampled: Option<measure::ResourceUsage>) -> Result<Option<serde_json::Value>> {
    let Some(sampled) = sampled else {
        return backend.server_usage();
    };
    let mut usage = serde_json::to_value(sampled).map_err(|e| e.to_string())?;
    usage["scope"] = "cgroup, sampled every 100 ms".into();
    Ok(Some(usage))
}

fn measure_load(
    backend: &mut dyn Backend,
    tables: &[Table],
    source: Source,
    mut counters: Option<&mut PerfCounters>,
    rows: u64,
    bytes: u64,
) -> Result<PhaseReport> {
    let sampler = ResourceSampler::start(backend.server_cgroup());
    start_counters(counters.as_deref_mut())?;
    let start = Instant::now();
    let loaded = backend.load(tables, source);
    let elapsed_seconds = start.elapsed().as_secs_f64();
    let hardware_counters = read_counters(counters)?;
    let sampled = sampler.stop()?;
    let (status, error) = match loaded {
        Ok(()) => (Status::Completed, None),
        Err(error) => (status_of(&error), Some(error)),
    };
    Ok(PhaseReport {
        status,
        elapsed_seconds,
        rows,
        bytes,
        throughput: rows as f64 / elapsed_seconds.max(f64::MIN_POSITIVE),
        client_usage: sampled.client,
        server_usage: server_usage(backend, sampled.server)?,
        hardware_counters,
        disk_bytes: backend.disk_bytes()?,
        error,
    })
}

fn measure_query(
    backend: &mut dyn Backend,
    query: &Query,
    mut expected: Option<Answer>,
    args: &CommonArgs,
    mut counters: Option<&mut PerfCounters>,
) -> Result<QueryReport> {
    let verify = args.verify == Verify::Answers;
    let mut report = QueryReport {
        query: query.name.clone(),
        status: Status::Completed,
        cold_seconds: None,
        hot_seconds: Vec::with_capacity(args.repetitions),
        rows: 0,
        digest: None,
        ties: query.ties,
        verified: expected.is_some(),
        client_usage: Default::default(),
        server_usage: None,
        hardware_counters: None,
        error: None,
    };
    let mut answer = Answer::default();
    if args.drop_caches {
        drop_page_cache()?;
    }
    let sampler = ResourceSampler::start(backend.server_cgroup());
    start_counters(counters.as_deref_mut())?;
    let mut first_digest = None;
    for run in 0..=args.repetitions {
        answer.clear();
        let executed = match backend.run(query, &mut answer) {
            Ok(executed) => executed,
            Err(error) => {
                report.status = status_of(&error);
                report.error = Some(error);
                break;
            }
        };
        report.rows = answer.rows();
        if verify {
            let digest = answer.digest(query.ties);
            match first_digest {
                None => {
                    first_digest = Some(digest);
                    report.digest = Some(format!("{digest:016x}"));
                    if let Some(mut expected) = expected.take() {
                        if expected.digest(query.ties) != digest {
                            report.status = Status::Mismatch;
                            report.error = Some(format!(
                                "{} rows differ from the {} pinned rows",
                                answer.rows(),
                                expected.rows()
                            ));
                        }
                    }
                }
                Some(first) if first != digest => {
                    report.status = Status::Mismatch;
                    report.error = Some(format!("hot run {run} returned different rows than the cold run"));
                }
                Some(_) => {}
            }
        }
        match run {
            0 => report.cold_seconds = Some(executed.seconds),
            _ => report.hot_seconds.push(executed.seconds),
        }
        let over_limit = args
            .time_limit
            .is_some_and(|limit| run == 0 && executed.seconds > limit.as_secs_f64());
        if over_limit && report.status == Status::Completed {
            report.status = Status::TimedOut;
        }
        if over_limit || report.status == Status::Mismatch {
            break;
        }
    }
    report.hardware_counters = read_counters(counters)?;
    let sampled = sampler.stop()?;
    report.client_usage = sampled.client;
    report.server_usage = server_usage(backend, sampled.server)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_durations_round_trip() {
        assert_eq!(parse_count("100M"), Some(100_000_000));
        assert_eq!(parse_count("0"), None);
        assert_eq!(spell_count(1_000_000_000), "1G");
        assert_eq!(parse_duration("60s"), Some(Duration::from_secs(60)));
        assert_eq!(parse_duration("1.5s"), None);
        assert_eq!(spell_duration(Duration::from_millis(1500)), "1500ms");
    }
}
