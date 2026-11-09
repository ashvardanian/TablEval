//! Configuration reports, machine information, and atomic JSON output.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use serde::Serialize;
use serde_json::Value;

use crate::{backend::Table, measure::ResourceUsage, perf_counters::CounterSample, suite::Suite, CommonArgs, Result};

/// How a load or a query ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    Completed,
    /// The engine cannot express the query, or the source.
    Unsupported,
    Failed,
    /// The cold run exceeded `--time-limit`, so the hot runs were skipped.
    TimedOut,
    /// The rows differ from the pinned answer or between runs.
    Mismatch,
}

#[derive(Serialize)]
pub struct DatasetInfo {
    pub suite: Suite,
    pub path: PathBuf,
    pub tables: Vec<Table>,
}

#[derive(Serialize)]
pub struct PhaseReport {
    pub status: Status,
    pub elapsed_seconds: f64,
    pub rows: u64,
    pub bytes: u64,
    /// Fact-table rows per second.
    pub throughput: f64,
    pub client_usage: ResourceUsage,
    pub server_usage: Option<Value>,
    pub hardware_counters: Option<CounterSample>,
    pub disk_bytes: u64,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct QueryReport {
    pub query: String,
    pub status: Status,
    pub cold_seconds: Option<f64>,
    pub hot_seconds: Vec<f64>,
    pub rows: u64,
    /// Order-independent digest of the canonical rows; only the row count under `ties`.
    pub digest: Option<String>,
    pub ties: bool,
    /// Whether a pinned answer was checked, rather than only the agreement between runs.
    pub verified: bool,
    pub client_usage: ResourceUsage,
    pub server_usage: Option<Value>,
    pub hardware_counters: Option<CounterSample>,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct ConfigReport {
    pub schema_version: u32,
    pub started_unix_seconds: u64,
    pub machine: MachineInfo,
    pub dataset: DatasetInfo,
    pub workload: CommonArgs,
    #[serde(serialize_with = "serialize_config")]
    pub config: BTreeMap<String, Value>,
    pub load: Option<PhaseReport>,
    pub queries: Vec<QueryReport>,
}

#[derive(Serialize)]
pub struct MachineInfo {
    pub cpu_model: String,
    pub logical_cores: usize,
    pub physical_cores: Option<usize>,
    pub ram_bytes: u64,
    pub kernel: Option<String>,
    pub os: Option<String>,
    pub architecture: &'static str,
    pub disks: Vec<DiskInfo>,
    pub budget: Budget,
}

/// The cores and memory every engine gets: this process's affinity mask and cgroup limit, which containers inherit.
#[derive(Clone, Debug, Serialize)]
pub struct Budget {
    pub cores: usize,
    /// The affinity mask as a cpuset list, like `0-15`, for `docker create --cpuset-cpus`.
    pub cpus: String,
    /// Bytes the engine may use, page cache included.
    pub memory_limit: u64,
}

static BUDGET: OnceLock<Budget> = OnceLock::new();

impl Budget {
    /// The environment's budget, narrowed by `--memory-limit`, which may not exceed the cgroup's limit.
    pub fn resolve(memory_limit: Option<u64>) -> Result<Self> {
        let cores = std::thread::available_parallelism().map_or(1, |cores| cores.get());
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let cpus = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .map_or_else(|| format!("0-{}", cores - 1), |list| list.trim().to_owned());
        let environment = cgroup_memory_limit().unwrap_or_else(|| sysinfo::System::new_all().total_memory());
        let memory_limit = match memory_limit {
            Some(bytes) if bytes > environment => {
                return Err(format!(
                    "--memory-limit {bytes} exceeds the {environment} bytes this process may use"
                ))
            }
            Some(bytes) => bytes,
            None => environment,
        };
        Ok(Self {
            cores,
            cpus,
            memory_limit,
        })
    }
    /// Fixes the run's budget; later calls keep the first.
    pub fn install(self) -> &'static Self {
        BUDGET.get_or_init(|| self)
    }
}

/// The run's budget, or the environment's when no run installed one, as in tests.
pub fn budget() -> &'static Budget {
    BUDGET.get_or_init(|| Budget::resolve(None).expect("the environment's budget needs no flag"))
}

/// The tightest `memory.max` from this process's cgroup up to the root, or `None` when none is set.
fn cgroup_memory_limit() -> Option<u64> {
    let membership = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut path = PathBuf::from("/sys/fs/cgroup");
    path.push(
        membership
            .lines()
            .find_map(|line| line.strip_prefix("0::"))?
            .trim_start_matches('/'),
    );
    path.ancestors()
        .take_while(|directory| directory.starts_with("/sys/fs/cgroup"))
        .filter_map(|directory| {
            std::fs::read_to_string(directory.join("memory.max"))
                .ok()?
                .trim()
                .parse()
                .ok()
        })
        .min()
}

#[derive(Serialize)]
pub struct DiskInfo {
    pub name: String,
    pub mount_point: PathBuf,
    pub file_system: String,
    pub total_bytes: u64,
    pub available_bytes: u64,
}

pub fn collect_machine_info() -> MachineInfo {
    let system = sysinfo::System::new_all();
    let disks = sysinfo::Disks::new_with_refreshed_list();
    MachineInfo {
        cpu_model: system.cpus().first().map(|c| c.brand().to_owned()).unwrap_or_default(),
        logical_cores: system.cpus().len(),
        physical_cores: sysinfo::System::physical_core_count(),
        ram_bytes: system.total_memory(),
        kernel: sysinfo::System::kernel_version(),
        os: sysinfo::System::long_os_version(),
        architecture: std::env::consts::ARCH,
        disks: disks
            .iter()
            .map(|disk| DiskInfo {
                name: disk.name().to_string_lossy().into_owned(),
                mount_point: disk.mount_point().to_path_buf(),
                file_system: disk.file_system().to_string_lossy().into_owned(),
                total_bytes: disk.total_space(),
                available_bytes: disk.available_space(),
            })
            .collect(),
        budget: budget().clone(),
    }
}

pub fn config_hash(value: &impl Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
    }
    Ok(format!("{hash:016x}"))
}

pub fn write_report(directory: &Path, report: &ConfigReport) -> Result<PathBuf> {
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let backend = report
        .config
        .get("backend")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let hash = config_hash(&(&report.workload, &report.dataset.path, &report.config))?;
    let path = directory.join(format!("{backend}-{hash}.json"));
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let file = std::fs::File::create(&temporary).map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&file, report).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    std::fs::rename(&temporary, &path).map_err(|e| e.to_string())?;
    Ok(path)
}

fn serialize_config<S: serde::Serializer>(
    value: &BTreeMap<String, Value>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;

    let mut map = serializer.serialize_map(Some(value.len()))?;
    for (key, value) in value {
        map.serialize_entry(key, value)?;
    }
    map.end()
}
