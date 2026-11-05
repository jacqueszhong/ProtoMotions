# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the opt-in startup retry loop + startup watchdog.

train_slurm.py wraps srun in run_with_startup_retry (produced by
_startup_retry_bash) when --startup-max-attempts > 1 or --startup-stall-min > 0.
The bash is exercised here with a fake command -- a small script that appends
chosen lines to $LOGFILE and exits with a chosen code -- and the watchdog's poll
interval / stall window are overridden to sub-second values, exactly the pattern
test_train_slurm_watchdog.py uses. The last case is the acceptance gate: with the
feature off (the default) generate_slurm_script emits a byte-identical script."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TRAIN_SLURM_PATH = Path(__file__).resolve().parents[1] / "train_slurm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_slurm_startup_under_test", TRAIN_SLURM_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ts = _load_module()

# The bash under test: kill_tree (a dependency) plus the retry loop.
RETRY_BASH = ts._kill_tree_bash() + "\n" + ts._startup_retry_bash()

pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or shutil.which("bash") is None
    or shutil.which("pgrep") is None,
    reason="needs bash + pgrep + GNU date/stat/wc (the Linux CI runner has them)",
)


def _bash(body, env=None, timeout=40):
    full = dict(os.environ)
    if env:
        full.update(env)
    return subprocess.run(
        ["bash", "-c", RETRY_BASH + "\n" + body],
        env=full,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _exit_code(stdout):
    for line in stdout.splitlines():
        if line.startswith("EXIT="):
            return int(line.split("=", 1)[1])
    raise AssertionError(f"no EXIT= line in:\n{stdout}")


def _fake_cmd(tmp_path, name, body):
    """A fake srun: bump the attempt counter, record a RUN line, then `body`."""
    p = tmp_path / name
    p.write_text(
        "#!/bin/bash\n"
        'n=0; [ -f "$COUNTER" ] && n=$(cat "$COUNTER"); n=$((n + 1)); echo "$n" > "$COUNTER"\n'
        'echo RUN >> "$RUNS"\n' + body + "\n"
    )
    p.chmod(0o755)
    return p


def _runs(tmp_path):
    r = tmp_path / "runs"
    return len(r.read_text().splitlines()) if r.exists() else 0


def _env(tmp_path, log, **over):
    e = {
        "LOGFILE": str(log),
        "COUNTER": str(tmp_path / "counter"),
        "RUNS": str(tmp_path / "runs"),
        "STARTUP_MAX_ATTEMPTS": "3",
        "STARTUP_STALL_SEC": "0",
        "STARTUP_BACKOFF_SEC": "0",
        "STARTUP_REPEAT_LIMIT": "3",
    }
    e.update({k: str(v) for k, v in over.items()})
    return e


def _run(tmp_path, cmd, env):
    return _bash(f'run_with_startup_retry "{cmd}"\necho "EXIT=$?"', env=env)


# --- retry accounting --------------------------------------------------------


def test_success_on_first_attempt_runs_once(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "ok.sh", "exit 0")
    r = _run(tmp_path, cmd, _env(tmp_path, log))
    assert _exit_code(r.stdout) == 0
    assert _runs(tmp_path) == 1


def test_passes_multiple_arguments_to_the_command(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    # Production calls `run_with_startup_retry stdbuf -oL -eL srun ...` -- many argv
    # elements. Run the whole argv, not just $1: bash -c '...' is 3 tokens.
    env = _env(tmp_path, log)
    r = _bash(
        "run_with_startup_retry bash -c 'echo RUN >> \"$RUNS\"; exit 0'\necho \"EXIT=$?\"",
        env=env,
    )
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr
    assert _runs(tmp_path) == 1


def test_two_failures_then_success(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "flaky.sh", 'case "$n" in 1|2) exit 1;; *) exit 0;; esac')
    r = _run(tmp_path, cmd, _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=5))
    assert _exit_code(r.stdout) == 0
    assert _runs(tmp_path) == 3


def test_exhausting_attempts_returns_last_exit_code(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "fail.sh", "exit 7")
    r = _run(tmp_path, cmd, _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=3))
    assert _exit_code(r.stdout) == 7
    assert _runs(tmp_path) == 3


def test_reached_training_then_failed_is_not_retried(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    # Emits the training marker, then dies -- like a walltime kill mid-run. The
    # array element resumes from checkpoint; retrying in place is wrong.
    cmd = _fake_cmd(tmp_path, "midrun.sh", 'echo "Epoch 0: loss 1.2" >> "$LOGFILE"\nexit 1')
    r = _run(tmp_path, cmd, _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=5))
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 1


# --- startup watchdog --------------------------------------------------------


def test_startup_watchdog_kills_silent_child_and_retries(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "hang.sh", 'case "$n" in 1) sleep 100;; *) exit 0;; esac')
    env = _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=3, STARTUP_STALL_SEC=2, STARTUP_POLL_SEC="0.2")
    r = _bash(f'run_with_startup_retry "{cmd}"\necho "EXIT=$?"', env=env, timeout=30)
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr
    assert _runs(tmp_path) == 2  # attempt 1 killed, attempt 2 succeeded


def test_startup_watchdog_leaves_an_active_child_alone(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(
        tmp_path, "busy.sh", 'for i in $(seq 1 20); do echo tick >> "$LOGFILE"; sleep 0.1; done\nexit 0'
    )
    env = _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=3, STARTUP_STALL_SEC=3, STARTUP_POLL_SEC="0.2")
    r = _bash(f'run_with_startup_retry "{cmd}"\necho "EXIT=$?"', env=env, timeout=30)
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr
    assert _runs(tmp_path) == 1  # never killed


def test_startup_watchdog_disarms_after_training_marker(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    # Reaches training, then goes silent LONGER than the stall window. Because the
    # watchdog disarmed at the marker, this must survive (the inner watchdog owns
    # liveness now) rather than be killed and retried.
    cmd = _fake_cmd(tmp_path, "disarm.sh", 'echo "Epoch 0: start" >> "$LOGFILE"\nsleep 2.5\nexit 0')
    env = _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=3, STARTUP_STALL_SEC=2, STARTUP_POLL_SEC="0.2")
    r = _bash(f'run_with_startup_retry "{cmd}"\necho "EXIT=$?"', env=env, timeout=30)
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr
    assert _runs(tmp_path) == 1


# --- the repeat check: retry flaky, refuse deterministic ---------------------


def test_repeated_identical_exception_stops_at_repeat_limit(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "det.sh", 'echo "AssertionError: boom" >> "$LOGFILE"\nexit 1')
    r = _run(
        tmp_path, cmd, _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=10, STARTUP_REPEAT_LIMIT=3)
    )
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 3  # stopped at the repeat limit, not 10
    assert "deterministic, not flaky" in r.stderr
    assert "AssertionError: boom" in r.stderr  # the traceback is surfaced


def test_native_abort_without_traceback_uses_full_budget(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    # exit 250, an abort message but NO Python exception line -> the repeat check
    # must not trip; this is the heap bug and it must keep retrying.
    cmd = _fake_cmd(tmp_path, "abort.sh", 'echo "double free or corruption (out)" >> "$LOGFILE"\nexit 250')
    r = _run(
        tmp_path, cmd, _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=4, STARTUP_REPEAT_LIMIT=3)
    )
    assert _exit_code(r.stdout) == 250
    assert _runs(tmp_path) == 4


def test_alternating_exceptions_do_not_trip_repeat_check(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(
        tmp_path,
        "alt.sh",
        'if [ $((n % 2)) -eq 1 ]; then echo "ValueError: odd" >> "$LOGFILE"; '
        'else echo "KeyError: even" >> "$LOGFILE"; fi\nexit 1',
    )
    r = _run(
        tmp_path, cmd, _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=5, STARTUP_REPEAT_LIMIT=3)
    )
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 5  # never two identical in a row -> full budget


def test_message_less_exception_stops_at_repeat_limit(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    # A bare `assert` prints just "AssertionError" -- no colon, no message. This is
    # a common deterministic startup failure and must still be refused.
    cmd = _fake_cmd(tmp_path, "bare.sh", 'echo "AssertionError" >> "$LOGFILE"\nexit 1')
    r = _run(
        tmp_path, cmd, _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=10, STARTUP_REPEAT_LIMIT=3)
    )
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 3
    assert "deterministic, not flaky" in r.stderr


def test_same_class_with_varying_message_stops_at_repeat_limit(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    # Same exception class, a different message each attempt (a device id) -- the
    # bug is the same, so the class-only comparison must still count it.
    cmd = _fake_cmd(
        tmp_path, "vary.sh", 'echo "RuntimeError: CUDA error on device $n" >> "$LOGFILE"\nexit 1'
    )
    r = _run(
        tmp_path, cmd, _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=10, STARTUP_REPEAT_LIMIT=3)
    )
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 3


def test_repeat_check_reads_only_the_current_attempt_slice(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    # Odd attempts emit the exception, even attempts emit nothing (a native abort).
    # Only if each attempt's exception is read from ITS OWN log slice does this stay
    # under the repeat limit and use the full budget -- reading the whole
    # accumulated log would see the previous attempt's line during a silent attempt
    # and stop early. Guards the _mark slicing (three sites) directly.
    cmd = _fake_cmd(
        tmp_path, "slice.sh", 'case "$n" in 1|3|5) echo "ValueError: boom" >> "$LOGFILE";; esac\nexit 1'
    )
    r = _run(
        tmp_path, cmd, _env(tmp_path, log, STARTUP_MAX_ATTEMPTS=5, STARTUP_REPEAT_LIMIT=2)
    )
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 5  # never two same-class in a row -> whole budget


# --- the acceptance gate: default flags emit a byte-identical script ---------

GOLDEN_DEFAULT_SCRIPT = """#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --account=acct
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=4:00:00
#SBATCH --output=/exp/exp-TS/slurm_output.log
#SBATCH --error=/exp/exp-TS/slurm_output.log
#SBATCH --array=0-5%1

# Job array enables automatic resume: if job times out, next array task continues

stdbuf -oL -eL srun --container-image=/img.sqsh --no-container-mount-home --container-mounts=/scratch:/scratch:rw /bin/bash -c 'JOB_CMD_PLACEHOLDER'
"""


def _slurm_args(**over):
    base = dict(
        account="acct",
        partition="gpu",
        nodes=1,
        ngpu=8,
        slurm_time="4:00:00",
        array_size=5,
        use_wandb=False,
        startup_max_attempts=1,
        startup_stall_min=0,
        startup_repeat_limit=3,
        startup_retry_backoff_sec=15,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_default_flags_emit_byte_identical_script():
    script, _ = ts.generate_slurm_script(
        _slurm_args(), "/exp/exp-TS", "JOB_CMD_PLACEHOLDER", "/img.sqsh", "jobname"
    )
    assert "run_with_startup_retry" not in script
    assert script == GOLDEN_DEFAULT_SCRIPT


def test_site_default_startup_knobs_keep_the_feature_off():
    # The byte-identical guarantee holds only if the RESOLVED default is off. The
    # golden test above passes hardcoded args; guard the SITE_DEFAULTS values too so
    # a change that flips the shipped default (e.g. startup_max_attempts: 2) -- which
    # would wrap srun on a plain submission -- is caught here.
    assert ts.SITE_DEFAULTS["startup_max_attempts"] == 1
    assert ts.SITE_DEFAULTS["startup_stall_min"] == 0


def test_feature_on_wraps_srun_and_exports_the_knobs():
    script, _ = ts.generate_slurm_script(
        _slurm_args(startup_max_attempts=8, startup_stall_min=12),
        "/exp/exp-TS",
        "JOB_CMD_PLACEHOLDER",
        "/img.sqsh",
        "jobname",
    )
    assert "run_with_startup_retry stdbuf -oL -eL srun" in script
    assert "STARTUP_MAX_ATTEMPTS=8" in script
    assert "STARTUP_STALL_SEC=720" in script  # 12 minutes -> seconds
    assert "kill_tree()" in script  # its dependency is emitted too
