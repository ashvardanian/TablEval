//! Process resource usage, sampled every 100 ms while a load or query runs, and the same for a server
//! container through its cgroup when the host exposes one.

use std::{
    path::{Path, PathBuf},
    sync::mpsc,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use serde::Serialize;
use sysinfo::{ProcessesToUpdate, System as ProcessSystem};

use crate::Result;

#[derive(Default, Serialize)]
pub struct ResourceUsage {
    pub cpu_seconds: f64,
    pub cpu_avg_percent: f64,
    pub cpu_max_percent: f32,
    pub rss_avg_bytes: u64,
    pub rss_max_bytes: u64,
    pub virtual_memory_avg_bytes: u64,
    pub virtual_memory_max_bytes: u64,
    pub read_bytes: u64,
    pub written_bytes: u64,
}

/// What a cgroup reports at one instant: resident memory, CPU time, and bytes read and written.
#[derive(Clone, Copy, Default)]
struct CgroupCounters {
    memory_bytes: u64,
    cpu_microseconds: u64,
    read_bytes: u64,
    written_bytes: u64,
}

/// The cgroup v2 folder of process `pid`, a number or `self`, from `/proc/<pid>/cgroup`; `None` without
/// cgroup v2 or memory accounting.
pub fn cgroup_of(pid: &str) -> Option<PathBuf> {
    let membership = std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?;
    let relative = membership.lines().find_map(|line| line.strip_prefix("0::"))?;
    let path = Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
    path.join("memory.current").exists().then_some(path)
}

/// Reads cgroup v2's `memory.current`, `cpu.stat` and `io.stat`; `None` once the cgroup is gone.
fn read_cgroup(path: &Path) -> Option<CgroupCounters> {
    let memory_bytes = std::fs::read_to_string(path.join("memory.current"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let cpu = std::fs::read_to_string(path.join("cpu.stat")).ok()?;
    let cpu_microseconds = cpu
        .lines()
        .find_map(|line| line.strip_prefix("usage_usec "))
        .and_then(|value| value.trim().parse().ok())?;
    let mut counters = CgroupCounters {
        memory_bytes,
        cpu_microseconds,
        ..Default::default()
    };
    for field in std::fs::read_to_string(path.join("io.stat"))
        .unwrap_or_default()
        .split_whitespace()
    {
        let parse = |value: &str| value.parse::<u64>().unwrap_or(0);
        if let Some(value) = field.strip_prefix("rbytes=") {
            counters.read_bytes += parse(value);
        } else if let Some(value) = field.strip_prefix("wbytes=") {
            counters.written_bytes += parse(value);
        }
    }
    Some(counters)
}

/// Usage of the benchmark process, and of the server's cgroup when one was given.
pub struct Sampled {
    pub client: ResourceUsage,
    pub server: Option<ResourceUsage>,
}

pub struct ResourceSampler {
    stop: mpsc::Sender<()>,
    handle: JoinHandle<Sampled>,
}

impl ResourceSampler {
    /// Starts sampling this process, and the cgroup at `server` alongside it.
    pub fn start(server: Option<PathBuf>) -> Self {
        let (stop, receiver) = mpsc::channel();
        let (ready, initialized) = mpsc::sync_channel(0);
        let handle = std::thread::spawn(move || {
            let pid = sysinfo::get_current_pid().unwrap();
            let mut system = ProcessSystem::new();
            system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
            let initial = system.process(pid).map(|p| (p.accumulated_cpu_time(), p.disk_usage()));
            let server_initial = server.as_deref().and_then(read_cgroup);
            let start = Instant::now();
            let _ = ready.send(());
            let mut usage = ResourceUsage::default();
            let mut server_usage = ResourceUsage::default();
            let (mut samples, mut server_samples) = (0, 0);
            let mut rss_sum = 0u128;
            let mut virtual_sum = 0u128;
            let mut server_memory_sum = 0u128;
            let mut server_previous = server_initial.map(|counters| (start, counters));
            loop {
                let stopped = !matches!(
                    receiver.recv_timeout(Duration::from_millis(100)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                );
                system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    samples += 1;
                    rss_sum += process.memory() as u128;
                    virtual_sum += process.virtual_memory() as u128;
                    usage.rss_max_bytes = usage.rss_max_bytes.max(process.memory());
                    usage.virtual_memory_max_bytes = usage.virtual_memory_max_bytes.max(process.virtual_memory());
                    usage.cpu_max_percent = usage.cpu_max_percent.max(process.cpu_usage());
                    if let Some((cpu, disk)) = &initial {
                        usage.cpu_seconds = process.accumulated_cpu_time().saturating_sub(*cpu) as f64 / 1000.0;
                        usage.read_bytes = process
                            .disk_usage()
                            .total_read_bytes
                            .saturating_sub(disk.total_read_bytes);
                        usage.written_bytes = process
                            .disk_usage()
                            .total_written_bytes
                            .saturating_sub(disk.total_written_bytes);
                    }
                }
                if let (Some(path), Some(first), Some((when, previous))) =
                    (server.as_deref(), server_initial, server_previous)
                {
                    if let Some(counters) = read_cgroup(path) {
                        let now = Instant::now();
                        server_samples += 1;
                        server_memory_sum += counters.memory_bytes as u128;
                        server_usage.rss_max_bytes = server_usage.rss_max_bytes.max(counters.memory_bytes);
                        let interval = now.duration_since(when).as_secs_f64().max(f64::MIN_POSITIVE);
                        let busy = counters.cpu_microseconds.saturating_sub(previous.cpu_microseconds) as f64 / 1e6;
                        server_usage.cpu_max_percent =
                            server_usage.cpu_max_percent.max((busy / interval * 100.0) as f32);
                        server_usage.cpu_seconds =
                            counters.cpu_microseconds.saturating_sub(first.cpu_microseconds) as f64 / 1e6;
                        server_usage.read_bytes = counters.read_bytes.saturating_sub(first.read_bytes);
                        server_usage.written_bytes = counters.written_bytes.saturating_sub(first.written_bytes);
                        server_previous = Some((now, counters));
                    }
                }
                if stopped {
                    break;
                }
            }
            let elapsed = start.elapsed().as_secs_f64();
            usage.cpu_avg_percent = usage.cpu_seconds / elapsed * 100.0;
            usage.rss_avg_bytes = (rss_sum / samples.max(1) as u128) as u64;
            usage.virtual_memory_avg_bytes = (virtual_sum / samples.max(1) as u128) as u64;
            let server = server_initial.map(|_| {
                server_usage.cpu_avg_percent = server_usage.cpu_seconds / elapsed * 100.0;
                server_usage.rss_avg_bytes = (server_memory_sum / server_samples.max(1) as u128) as u64;
                server_usage
            });
            Sampled { client: usage, server }
        });
        let _ = initialized.recv();
        Self { stop, handle }
    }

    pub fn stop(self) -> Result<Sampled> {
        let _ = self.stop.send(());
        self.handle.join().map_err(|_| "resource sampler panicked".into())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn reads_this_process_cgroup() {
        let Some(path) = cgroup_of("self") else {
            return;
        };
        let counters = read_cgroup(&path).unwrap();
        assert!(counters.memory_bytes > 0);
        assert!(counters.cpu_microseconds > 0);
    }
}
