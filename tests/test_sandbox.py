"""SandboxRunner tests (subprocess fallback path; Docker command is pure)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlteam.sandbox import SandboxLimits, SandboxRunner, UnsafeDataFileName


def _runner(**kwargs) -> SandboxRunner:
    return SandboxRunner(prefer_docker=False, **kwargs)


def test_runs_code_and_captures_stdout():
    result = _runner().run_python("print('hello from sandbox')")
    assert result.ok
    assert result.backend == "subprocess"
    assert "hello from sandbox" in result.stdout


def test_captures_stderr_and_exit_code():
    result = _runner().run_python("import sys; sys.stderr.write('boom'); sys.exit(3)")
    assert not result.ok
    assert result.exit_code == 3
    assert "boom" in result.stderr


def test_wall_timeout_kills_runaway_code():
    runner = _runner(limits=SandboxLimits(wall_seconds=2.0))
    result = runner.run_python("while True:\n    pass")
    assert result.timed_out
    assert not result.ok
    assert result.wall_seconds < 30


def test_data_files_are_private_copies(tmp_path: Path):
    original = tmp_path / "input.txt"
    original.write_text("pristine", encoding="utf-8")
    code = (
        "from pathlib import Path\n"
        "p = Path('data/input.txt')\n"
        "print('saw:', p.read_text())\n"
        "p.write_text('MUTATED')\n"  # writes only touch the sandbox copy
    )
    result = _runner().run_python(code, data_files={"input.txt": original})
    assert result.ok
    assert "saw: pristine" in result.stdout
    assert original.read_text(encoding="utf-8") == "pristine"


def test_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-leak")
    code = "import os; print('leak' if os.environ.get('ANTHROPIC_API_KEY') else 'clean')"
    result = _runner().run_python(code)
    assert result.ok
    assert "clean" in result.stdout
    assert "leak" not in result.stdout


def test_docker_command_construction(tmp_path: Path):
    runner = SandboxRunner(prefer_docker=False, limits=SandboxLimits(memory_mb=256))
    data = tmp_path / "d.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    cmd = runner.docker_command(tmp_path, {"d.csv": data})
    joined = " ".join(cmd)
    assert "--network=none" in joined
    assert "--memory=256m" in joined
    assert ":ro" in joined  # read-only mounts
    assert cmd[-1] == "main.py"


def test_fallback_backend_is_reported():
    assert _runner().backend == "subprocess"


@pytest.mark.parametrize(
    "bad_name",
    ["../escape.txt", "sub/../../escape.txt", "C:/Windows/evil.txt", "/etc/passwd"],
)
def test_data_file_name_traversal_rejected(tmp_path: Path, bad_name: str):
    """A crafted data_files key must never write outside the sandbox."""
    src = tmp_path / "src.txt"
    src.write_text("x", encoding="utf-8")
    with pytest.raises(UnsafeDataFileName):
        _runner().run_python("pass", data_files={bad_name: src})


def test_nested_data_file_name_allowed(tmp_path: Path):
    src = tmp_path / "src.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    code = "from pathlib import Path; print(Path('data/nested/src.csv').read_text())"
    result = _runner().run_python(code, data_files={"nested/src.csv": src})
    assert result.ok and "1,2" in result.stdout


def test_wall_timeout_kills_child_spawning_payload():
    """A payload that spawns a sleeping grandchild must not defeat the wall
    clock: the whole tree is killed and the call returns near the deadline."""
    runner = _runner(limits=SandboxLimits(wall_seconds=3.0))
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "time.sleep(120)\n"
    )
    result = runner.run_python(code)
    assert result.timed_out
    assert result.wall_seconds < 30  # not blocked on the 120s grandchild
