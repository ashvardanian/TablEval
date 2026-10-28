//! System-wide Linux hardware counters around each load and query, read the way `perf stat -a` would.
//!
//! Engines run their own thread pools, spawned before any measurement starts, so counters scoped to the
//! calling thread or inherited by its children would miss most of the work. One counter group per online
//! CPU (`pid == -1`) captures every thread, other tenants' activity included.
//!
//! Needs `CAP_PERFMON` or `kernel.perf_event_paranoid` at 0 or below, and six file descriptors per CPU,
//! so hosts with more than about 170 CPUs need `ulimit -n` raised past the usual 1,024.

use std::io;

#[cfg(all(target_os = "linux", feature = "perf-counters"))]
use perf_event::events::Hardware;
#[cfg(all(target_os = "linux", feature = "perf-counters"))]
use perf_event::{Builder, Group};
use serde::Serialize;

/// A set of counters the kernel schedules together and reads atomically.
///
/// This is what Linux calls a *counter group* in `perf_event_open(2)`: one
/// file descriptor per member, all members enabled / disabled / read through
/// the leader fd so they cover identical time windows. Without grouping, the
/// kernel may time-multiplex your counters onto fewer physical PMU registers
/// and ratios like IPC drift.
#[cfg(all(target_os = "linux", feature = "perf-counters"))]
type KernelCounterGroup = Group;

/// A readable handle to one counter inside a [`KernelCounterGroup`].
#[cfg(all(target_os = "linux", feature = "perf-counters"))]
type KernelCounter = perf_event::Counter;

/// Counter readings summed across all online CPUs for one scoped region.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CounterSample {
    pub cycles: u64,
    pub instructions: u64,
    pub cache_references: u64,
    pub cache_misses: u64,
    pub branch_misses: u64,
}

/// System-wide hardware performance counters spanning every online CPU.
///
/// Construction opens one [`KernelCounterGroup`] per CPU, each holding five
/// counters (cycles, instructions, cache-refs, cache-misses, branch-misses)
/// read atomically via `PERF_FORMAT_GROUP`. The same [`PerfCounters`] is
/// reused across many `reset_and_enable` / `disable_and_read` pairs to avoid
/// per-step fd churn.
#[cfg(all(target_os = "linux", feature = "perf-counters"))]
pub struct PerfCounters {
    per_cpu: Vec<CpuCounters, std::alloc::System>,
}

#[cfg(all(target_os = "linux", feature = "perf-counters"))]
struct CpuCounters {
    kernel_group: KernelCounterGroup,
    cycles: KernelCounter,
    instructions: KernelCounter,
    cache_references: KernelCounter,
    cache_misses: KernelCounter,
    branch_misses: KernelCounter,
}

#[cfg(all(target_os = "linux", feature = "perf-counters"))]
impl PerfCounters {
    pub fn new() -> io::Result<Self> {
        let logical_cpus = std::thread::available_parallelism().map_or(1, usize::from);

        let mut per_cpu = Vec::with_capacity_in(logical_cpus, std::alloc::System);
        for cpu in 0..logical_cpus {
            match open_cpu_counters(cpu) {
                Ok(entry) => per_cpu.push(entry),
                // The first CPU failing is almost always a permission problem, not an offline core.
                Err(error) if per_cpu.is_empty() => return Err(error),
                // Later CPUs may be offline.
                Err(_) => continue,
            }
        }

        if per_cpu.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no online CPUs accepted a perf counter group",
            ));
        }
        Ok(Self { per_cpu })
    }

    pub fn reset_and_enable(&mut self) -> io::Result<()> {
        for entry in &mut self.per_cpu {
            entry.kernel_group.reset()?;
            entry.kernel_group.enable()?;
        }
        Ok(())
    }

    pub fn disable_and_read(&mut self) -> io::Result<CounterSample> {
        let mut sample = CounterSample::default();
        for entry in &mut self.per_cpu {
            entry.kernel_group.disable()?;
            let counts = entry.kernel_group.read()?;
            sample.cycles += counts[&entry.cycles];
            sample.instructions += counts[&entry.instructions];
            sample.cache_references += counts[&entry.cache_references];
            sample.cache_misses += counts[&entry.cache_misses];
            sample.branch_misses += counts[&entry.branch_misses];
        }
        Ok(sample)
    }
}

/// Open a five-counter group for `perf stat -a` equivalent capture on one CPU.
#[cfg(all(target_os = "linux", feature = "perf-counters"))]
fn open_cpu_counters(cpu: usize) -> io::Result<CpuCounters> {
    // A no-op software leader owns the group, so the five counters below read atomically through it.
    let mut leader = Group::builder();
    leader.any_pid().one_cpu(cpu).include_kernel();
    let mut kernel_group = leader.build_group()?;

    let add_member = |group: &mut Group, event| -> io::Result<KernelCounter> {
        let mut builder = Builder::new(event);
        builder.any_pid().one_cpu(cpu).include_kernel();
        group.add(&builder)
    };

    let cycles = add_member(&mut kernel_group, Hardware::CPU_CYCLES)?;
    let instructions = add_member(&mut kernel_group, Hardware::INSTRUCTIONS)?;
    let cache_references = add_member(&mut kernel_group, Hardware::CACHE_REFERENCES)?;
    let cache_misses = add_member(&mut kernel_group, Hardware::CACHE_MISSES)?;
    let branch_misses = add_member(&mut kernel_group, Hardware::BRANCH_MISSES)?;

    Ok(CpuCounters {
        kernel_group,
        cycles,
        instructions,
        cache_references,
        cache_misses,
        branch_misses,
    })
}

/// Non-Linux / feature-off stub. `new()` always fails so the struct is never
/// held — the other methods are unreachable in practice.
#[cfg(not(all(target_os = "linux", feature = "perf-counters")))]
pub struct PerfCounters;

#[cfg(not(all(target_os = "linux", feature = "perf-counters")))]
impl PerfCounters {
    pub fn new() -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "perf-counters feature disabled or target_os != linux",
        ))
    }

    pub fn reset_and_enable(&mut self) -> io::Result<()> {
        unreachable!("stub PerfCounters is never constructed")
    }

    pub fn disable_and_read(&mut self) -> io::Result<CounterSample> {
        unreachable!("stub PerfCounters is never constructed")
    }
}
