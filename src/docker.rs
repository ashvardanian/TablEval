//! Owned Docker containers with protocol readiness and cleanup.

use std::{
    alloc::System,
    ffi::OsStr,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
};

use crate::backend::Result;

/// What a container sees of the host and how long it may take to start.
pub struct Startup<'a> {
    /// A host folder mounted read-only at a container path, such as the dataset the server ingests.
    pub data: Option<(&'a Path, &'a str)>,
    /// Time limit for container start and readiness, counted from its creation.
    pub time_limit: Duration,
}

impl Default for Startup<'_> {
    fn default() -> Self {
        Self {
            data: None,
            time_limit: Duration::from_secs(90),
        }
    }
}

pub struct ContainerHandle {
    id: String,
    pub port: u16,
    storage: String,
    deadline: Instant,
}

fn docker<I, S>(args: I) -> Result<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut args = args.into_iter();
    let first = args.next().ok_or("missing Docker command")?;
    let output = Command::new("docker")
        .arg(first.as_ref())
        .args(args)
        .output()
        .map_err(|e| format!("Docker: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "docker {}: {}",
            first.as_ref().to_string_lossy(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

impl ContainerHandle {
    pub fn start(
        image: &str,
        port: u16,
        path: &Path,
        storage: &str,
        env: &[(&str, &str)],
        command: &[&str],
    ) -> Result<Self> {
        Self::start_with(image, port, path, storage, env, command, &Startup::default())
    }

    pub fn start_with(
        image: &str,
        port: u16,
        path: &Path,
        storage: &str,
        env: &[(&str, &str)],
        command: &[&str],
        startup: &Startup<'_>,
    ) -> Result<Self> {
        std::fs::create_dir_all(path).map_err(|e| e.to_string())?;
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        let publish = format!("127.0.0.1::{port}");
        let mount = format!("type=bind,src={},dst={storage}", path.display());
        let mut args = Vec::with_capacity_in(8 + env.len() * 2 + command.len(), System);
        args.extend([
            "create".to_owned(),
            "--publish".into(),
            publish,
            "--mount".into(),
            mount,
        ]);
        if let Some((data, target)) = startup.data {
            let data = data.canonicalize().map_err(|e| format!("{}: {e}", data.display()))?;
            args.extend([
                "--mount".into(),
                format!("type=bind,src={},dst={target},readonly", data.display()),
            ]);
        }
        for (key, value) in env {
            args.extend(["--env".into(), format!("{key}={value}")]);
        }
        args.push(image.into());
        args.extend(command.iter().map(|s| (*s).into()));
        // Create first so every failure after ownership is established has a cleanup guard.
        let id = docker(&args)?;
        let mut container = Self {
            id,
            port: 0,
            storage: storage.into(),
            deadline: Instant::now() + startup.time_limit,
        };
        docker(["start", &container.id])?;
        let mut published_port = 0;
        container.ready(|| {
            let published = docker(["port", &container.id, &format!("{port}/tcp")])?;
            published_port = published
                .rsplit(':')
                .next()
                .ok_or("Docker did not publish a port")?
                .parse()
                .map_err(|e| format!("Docker port: {e}"))?;
            Ok(())
        })?;
        container.port = published_port;
        Ok(container)
    }

    pub fn name(&self) -> Result<String> {
        docker(["inspect", "--format", "{{.Name}}", &self.id]).map(|name| name.trim_start_matches('/').to_owned())
    }
    pub fn exec(&self, args: &[&str]) -> Result<String> {
        docker(["exec", self.id.as_str()].into_iter().chain(args.iter().copied()))
    }

    pub fn ready(&self, mut probe: impl FnMut() -> Result<()>) -> Result<()> {
        loop {
            match probe() {
                Ok(()) => return Ok(()),
                Err(error) => {
                    let running = docker(["inspect", "--format", "{{.State.Running}}", &self.id])?;
                    if running != "true" || Instant::now() >= self.deadline {
                        let state = docker(["inspect", "--format", "{{json .State}}", &self.id]).unwrap_or_default();
                        let logs = docker(["logs", "--tail", "20", &self.id]).unwrap_or_default();
                        return Err(format!("Server did not become ready: {error}\n{state}\n{logs}"));
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }

    pub fn stats(&self) -> Result<serde_json::Value> {
        let output = docker(["stats", "--no-stream", "--format", "{{json .}}", &self.id])?;
        let mut value: serde_json::Value = serde_json::from_str(&output).map_err(|e| e.to_string())?;
        value["scope"] = serde_json::json!("container");
        Ok(value)
    }

    /// The container's cgroup v2 folder on the host, found through its main process; `None` where the
    /// daemon runs elsewhere, as Docker Desktop's virtual machine does.
    pub fn cgroup(&self) -> Option<PathBuf> {
        let pid = docker(["inspect", "--format", "{{.State.Pid}}", &self.id]).ok()?;
        crate::measure::cgroup_of(&pid)
    }

    pub fn disk_bytes(&self) -> Result<u64> {
        let output = docker(["exec", &self.id, "du", "-sk", &self.storage])?;
        output
            .split_whitespace()
            .next()
            .ok_or("Empty disk usage")?
            .parse::<u64>()
            .map(|n| n * 1024)
            .map_err(|e| e.to_string())
    }
}

impl Drop for ContainerHandle {
    fn drop(&mut self) {
        if let Err(error) = docker(["rm", "--force", "--volumes", &self.id]) {
            eprintln!("ContainerHandle cleanup: {error}");
        }
    }
}
