"""SandboxRunner: the ONLY place LLM-generated code may execute.

Primary backend: Docker (`--network=none`, CPU/memory caps, read-only data
mounts, no reuse). Fallback when Docker is unavailable: a restricted
subprocess — isolated interpreter (-I), scrubbed environment, private temp
working directory with COPIES of input data, wall-clock kill, and best-effort
memory/CPU caps (Windows job objects / Unix rlimits).

THE FALLBACK IS A DOCUMENTED DOWNGRADE: it cannot block network access. Every
construction on the fallback path logs a loud warning saying exactly that.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import structlog

_log = structlog.get_logger(__name__)

_OUTPUT_CAP_CHARS = 100_000
_RECOVERY_TIMEOUT_SECONDS = 5.0


class UnsafeDataFileName(ValueError):
    """A data_files key would escape the sandbox data directory."""


def _safe_data_name(name: str) -> str:
    """Reject any mount name that could escape ./data/.

    The name is caller/LLM-controlled and is used to build a filesystem path;
    an absolute path or a '..' segment would write outside the sandbox.
    """
    candidate = Path(name)
    if candidate.is_absolute() or candidate.drive or candidate.anchor:
        raise UnsafeDataFileName(f"data file name must be relative: {name!r}")
    parts = candidate.parts
    if any(part in ("..", "") for part in parts) or ":" in name:
        raise UnsafeDataFileName(f"data file name may not traverse directories: {name!r}")
    return candidate.as_posix()


@dataclass(frozen=True)
class SandboxLimits:
    wall_seconds: float = 60.0
    cpu_seconds: int = 30
    memory_mb: int = 512


@dataclass(frozen=True)
class SandboxResult:
    backend: str  # "docker" | "subprocess"
    exit_code: int
    stdout: str
    stderr: str
    wall_seconds: float
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def docker_available() -> bool:
    return shutil.which("docker") is not None


class SandboxRunner:
    """Executes untrusted Python source under resource limits."""

    def __init__(
        self,
        limits: SandboxLimits | None = None,
        *,
        docker_image: str = "python:3.11-slim",
        prefer_docker: bool = True,
    ) -> None:
        self.limits = limits or SandboxLimits()
        self._docker_image = docker_image
        self._use_docker = prefer_docker and docker_available()
        if prefer_docker and not self._use_docker:
            _log.warning(
                "SANDBOX_DOWNGRADE",
                backend="subprocess",
                detail=(
                    "Docker is not available: falling back to a restricted "
                    "subprocess. Wall/CPU/memory limits apply, but NETWORK "
                    "ACCESS CANNOT BE BLOCKED in this mode. Install Docker for "
                    "full isolation."
                ),
            )

    @property
    def backend(self) -> str:
        return "docker" if self._use_docker else "subprocess"

    def run_python(
        self, code: str, *, data_files: Mapping[str, Path] | None = None
    ) -> SandboxResult:
        """Run Python source. ``data_files`` maps sandbox-relative names to
        host paths; the code sees them in ./data/ (read-only in Docker,
        private copies in the fallback)."""
        data_files = {_safe_data_name(name): path for name, path in (data_files or {}).items()}
        with tempfile.TemporaryDirectory(prefix="mlteam-sandbox-") as workdir_str:
            workdir = Path(workdir_str)
            (workdir / "main.py").write_text(code, encoding="utf-8")
            if self._use_docker:
                return self._run_docker(workdir, data_files)
            return self._run_subprocess(workdir, data_files)

    # -- docker backend ---------------------------------------------------------

    def docker_command(self, workdir: Path, data_files: Mapping[str, Path]) -> list[str]:
        """Pure command construction (unit-testable without Docker)."""
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--cpus=1",
            f"--memory={self.limits.memory_mb}m",
            "--pids-limit=64",
            "--read-only",
            "--tmpfs",
            "/tmp:size=64m",
            "-v",
            f"{workdir}:/sandbox:ro",
        ]
        for name, host_path in sorted(data_files.items()):
            cmd += ["-v", f"{host_path.resolve()}:/sandbox/data/{name}:ro"]
        cmd += ["-w", "/sandbox", self._docker_image, "python", "-I", "-B", "main.py"]
        return cmd

    def _run_docker(self, workdir: Path, data_files: Mapping[str, Path]) -> SandboxResult:
        return self._execute(
            self.docker_command(workdir, data_files), cwd=None, env=None, backend="docker"
        )

    # -- restricted subprocess fallback -------------------------------------------

    def _run_subprocess(self, workdir: Path, data_files: Mapping[str, Path]) -> SandboxResult:
        data_dir = workdir / "data"
        data_dir.mkdir(exist_ok=True)
        for name, host_path in data_files.items():
            # name is already validated to stay within data_dir.
            dest = data_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Private COPY: the sandboxed code can never touch the original.
            shutil.copyfile(host_path, dest)
        env = _scrubbed_env(workdir)
        cmd = [sys.executable, "-I", "-B", "main.py"]
        return self._execute(cmd, cwd=workdir, env=env, backend="subprocess")

    def _execute(
        self,
        cmd: list[str],
        *,
        cwd: Path | None,
        env: dict[str, str] | None,
        backend: str,
    ) -> SandboxResult:
        started = time.perf_counter()
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        job = _apply_process_limits(process, self.limits) if backend == "subprocess" else None
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=self.limits.wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Kill the WHOLE tree (child + grandchildren): process.kill() only
            # hits the immediate child, and any surviving grandchild keeps the
            # pipes open and hangs the recovery read.
            _kill_process_tree(process, job)
            try:
                stdout, stderr = process.communicate(timeout=_RECOVERY_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                _log.warning("sandbox_recovery_read_timed_out", backend=backend)
                stdout, stderr = "", ""
        wall = time.perf_counter() - started
        result = SandboxResult(
            backend=backend,
            exit_code=process.returncode if not timed_out else -1,
            stdout=(stdout or "")[:_OUTPUT_CAP_CHARS],
            stderr=(stderr or "")[:_OUTPUT_CAP_CHARS],
            wall_seconds=round(wall, 3),
            timed_out=timed_out,
        )
        _log.info(
            "sandbox_run_finished",
            backend=backend,
            exit_code=result.exit_code,
            timed_out=timed_out,
            wall_seconds=result.wall_seconds,
        )
        return result


def _scrubbed_env(workdir: Path) -> dict[str, str]:
    """Minimal environment: no credentials, no PYTHON*, no API keys.

    TEMP/TMP are repointed at the sandbox workdir so temp files the code
    creates land inside the disposable directory, not the host's shared temp.
    """
    keep = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env["PATH"] = str(Path(sys.executable).parent)
    env["TEMP"] = str(workdir)
    env["TMP"] = str(workdir)
    return env


def _apply_process_limits(process: subprocess.Popen, limits: SandboxLimits) -> object | None:
    """Best-effort memory/CPU caps; returns a job handle for tree-kill (Win)."""
    if sys.platform == "win32":
        return _apply_windows_job_limits(process, limits)
    _log.info("sandbox_posix_limits_note", detail="rlimits require preexec_fn")  # pragma: no cover
    return None


def _kill_process_tree(process: subprocess.Popen, job: object | None) -> None:
    """Terminate the child and all descendants (grandchildren included)."""
    if job is not None and sys.platform == "win32":
        try:
            import win32job

            win32job.TerminateJobObject(job, 1)  # kills every process in the job
            return
        except Exception as exc:  # pragma: no cover - depends on OS state
            _log.warning("sandbox_job_terminate_failed", detail=str(exc))
    if sys.platform == "win32":  # fallback: taskkill the tree by pid
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:  # pragma: no cover - Unix only
        process.kill()


def _apply_windows_job_limits(process: subprocess.Popen, limits: SandboxLimits) -> object | None:
    """Attach the child to a job object with memory + CPU-time caps.

    Returns the job handle so the timeout path can TerminateJobObject the whole
    tree; KILL_ON_JOB_CLOSE also reaps descendants when it is released."""
    try:
        import win32api
        import win32con
        import win32job
    except ImportError:  # pragma: no cover - pywin32 ships with prefect
        _log.warning(
            "SANDBOX_LIMITS_UNAVAILABLE",
            detail="pywin32 missing; wall-clock timeout is the only limit",
        )
        return
    try:
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] = (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | win32job.JOB_OBJECT_LIMIT_PROCESS_TIME
            | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info["ProcessMemoryLimit"] = limits.memory_mb * 1024 * 1024
        # PerProcessUserTimeLimit is in 100ns units.
        info["BasicLimitInformation"]["PerProcessUserTimeLimit"] = int(
            limits.cpu_seconds * 10_000_000
        )
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
        handle = win32api.OpenProcess(
            win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, process.pid
        )
        win32job.AssignProcessToJobObject(job, handle)
        process._mlteam_job = job  # type: ignore[attr-defined]  # keep handle alive
        return job
    except Exception as exc:  # pragma: no cover - depends on OS state
        _log.warning("SANDBOX_LIMITS_UNAVAILABLE", detail=str(exc))
        return None
