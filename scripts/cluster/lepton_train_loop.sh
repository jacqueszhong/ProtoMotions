#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# In-job setup + self-resuming training loop for ProtoMotions on DGX Cloud Lepton.
#
# On SLURM, a 4h walltime forces a job array: each array task is a fresh srun that
# resumes from last.ckpt, and train_agent.py runs with --use-slurm so it stops
# itself at ~3.5h before the walltime kill. Lepton jobs have NO walltime, so none
# of that applies -- train_agent.py runs straight to completion, and this loop
# only restarts it after a *crash*.
#
# Resume is filesystem-based and needs no flags: train_agent.py finds
# results/<exp>/last.ckpt in the (unchanged) snapshot cwd and continues. It saves
# last.ckpt every K epochs (default 10) regardless of SLURM, which is what makes a
# plain re-invocation safe. So each loop iteration runs the identical command; the
# first run starts fresh, later runs auto-resume.
#
# Positional args (main path):
#   1  PYTHON      interpreter invocation, e.g. "/workspace/isaaclab/isaaclab.sh -p"
#   2  SIMULATOR   isaacgym|isaaclab|newton (selects install path + IsaacLab warmup)
#   3  EXP_NAME    experiment name; checkpoints live at results/<EXP_NAME>/last.ckpt
#   4  EXP_FOLDER  code snapshot on AMLFS; cwd for training, holds results/
#   5  MAX_RUNS    crash-restart cap (loop iterations)
#   6+ the train_agent.py argument tail, forwarded verbatim as "$@"
#
# Each positional is shlex-quoted individually by train_lepton.py, so the
# container shell splits the command back into exactly these arguments and the
# train_agent.py tail reaches "$@" without a value being word-split on a space.
#
# Env knobs (defaults in parentheses):
#   STALL_TIMEOUT_MIN (30)  kill training if the durable log stops growing for this
#                           many minutes; 0 disables. Passed by train_lepton.py.
#   NO_PROGRESS_LIMIT (3)   give up after this many consecutive runs that do not
#                           advance last.ckpt (deterministic failure).
# Test-only knobs (see protomotions/tests/test_lepton_train_loop.py):
#   LEPTON_LOOP_SOURCE_ONLY  when set, `source` this file to get the functions
#                            below (kill_tree/run_with_stall_watchdog/...) WITHOUT
#                            running any of the main install+train path.
#   LEPTON_LOOP_SKIP_SETUP   skip the per-container install + IsaacLab warmup, so
#                            the loop itself can be exercised with a fake python.
#   STALL_POLL_SEC (60)      watchdog poll interval; sub-second in tests.
#   STALL_QUIET_LIMIT        watchdog no-growth polls before killing (default:
#                            STALL_TIMEOUT_MIN, i.e. the user knob is in minutes).

set -uo pipefail

# ===========================================================================
# Functions. Defined first so LEPTON_LOOP_SOURCE_ONLY tests can source them.
# ===========================================================================

ckpt_stamp() {
    # last.ckpt mtime in epoch-seconds, or "none" while it does not exist yet.
    [ -f "$CKPT" ] && stat -c %Y "$CKPT" 2>/dev/null || echo "none"
}

kill_tree() {
    # Kill a process AND all its descendants, children first. The training python
    # and its Fabric worker ranks live several levels below the process we
    # background (for IsaacLab, the isaaclab.sh wrapper). Killing only the parent
    # reparents the grandchildren to init -- they keep the GPUs and MASTER_PORT,
    # and the immediate resume then collides. Recurse depth-first so a parent is
    # never killed before its children.
    local root=$1 child
    for child in $(pgrep -P "$root" 2>/dev/null); do
        kill_tree "$child"
    done
    kill -9 "$root" 2>/dev/null
}

run_with_stall_watchdog() {
    # Run "$@", and if $LOGFILE stops growing for STALL_QUIET_LIMIT consecutive
    # polls of STALL_POLL_SEC seconds, kill the whole process tree so the loop
    # restarts instead of hanging. A wedged rank -- one stuck at setup while the
    # others block at the distributed barrier -- emits no output and no exit code,
    # but a blocked collective silences the HEALTHY ranks too, so "the durable log
    # stopped growing" is a reliable whole-job-wedge signal. Defaults poll every
    # 60s with the limit in minutes (STALL_TIMEOUT_MIN); a limit <= 0 disables it.
    local poll="${STALL_POLL_SEC:-60}"
    local quiet_limit="${STALL_QUIET_LIMIT:-${STALL_TIMEOUT_MIN:-30}}"
    "$@" &
    local pid=$!
    if [ "${quiet_limit}" -le 0 ]; then
        wait "$pid"
        return $?
    fi
    (
        last=-1
        quiet=0
        while kill -0 "$pid" 2>/dev/null; do
            sleep "$poll"
            size=$(stat -c %s "$LOGFILE" 2>/dev/null || echo 0)
            if [ "$size" != "$last" ]; then
                last=$size
                quiet=0
            else
                quiet=$((quiet + 1))
            fi
            if [ "$quiet" -ge "${quiet_limit}" ]; then
                echo "[watchdog] ${LOGFILE} has not grown for ${quiet_limit} poll(s)" \
                    "of ${poll}s -- training looks wedged (a rank stuck at" \
                    "setup/barrier?). Killing pid ${pid} and its descendants so the" \
                    "job restarts instead of burning the walltime." >&2
                kill_tree "$pid"
                break
            fi
        done
    ) &
    local watchdog=$!
    wait "$pid"
    local status=$?
    kill "$watchdog" 2>/dev/null
    wait "$watchdog" 2>/dev/null
    return "$status"
}

run_training_loop() {
    # Restart train_agent.py after a crash; give up only when a run makes no
    # forward progress -- last.ckpt's mtime does not advance -- for
    # NO_PROGRESS_LIMIT runs in a row. Keying on progress (not merely "does a
    # checkpoint exist?") handles both failure modes the simpler test gets wrong: a
    # deterministic crash *after* the first checkpoint (otherwise it would loop the
    # identical failure for the whole budget, resuming from the same epoch and
    # re-crashing without ever saving), and a *transient* crash *before* the first
    # checkpoint (otherwise a single early flake fails the job). A genuinely
    # deterministic startup error still fails fast, after NO_PROGRESS_LIMIT quick
    # attempts. "$@" is the train_agent.py argument tail.
    local no_progress=0 r before after status
    for r in $(seq 1 "$MAX_RUNS"); do
        echo "[lepton-loop] ===== run ${r}/${MAX_RUNS} ====="
        before=$(ckpt_stamp)
        # Identical command every iteration: train_agent.py auto-resumes when
        # results/<exp>/last.ckpt exists. No --use-slurm (no walltime autoresume).
        # shellcheck disable=SC2086  # $PYTHON is "cmd -flag": word-split intended.
        # The train tail is "$@" -- quoted, so each argument is preserved.
        run_with_stall_watchdog $PYTHON -u protomotions/train_agent.py "$@"
        status=$?

        if [ "$status" -eq 0 ]; then
            echo "[lepton-loop] run ${r} exited 0 -- training complete."
            return 0
        fi

        after=$(ckpt_stamp)
        if [ "$after" != "$before" ]; then
            no_progress=0
            echo "[lepton-loop] run ${r} crashed (exit ${status}); checkpoint advanced -- resuming."
        else
            no_progress=$((no_progress + 1))
            echo "[lepton-loop] run ${r} crashed (exit ${status}); no checkpoint progress" \
                "(${no_progress}/${NO_PROGRESS_LIMIT} in a row)." >&2
            if [ "$no_progress" -ge "$NO_PROGRESS_LIMIT" ]; then
                echo "[lepton-loop] ${NO_PROGRESS_LIMIT} runs in a row made no progress; this is" >&2
                echo "[lepton-loop] almost certainly a deterministic failure. Giving up -- see the" >&2
                echo "[lepton-loop] traceback(s) above." >&2
                touch "$GAVE_UP"
                return "$status"
            fi
            echo "[lepton-loop] retrying in case the crash was transient."
        fi
    done

    echo "[lepton-loop] exhausted ${MAX_RUNS} run(s) without completing." >&2
    return 1
}

# When a test sources this file just for the functions above, stop here -- do not
# parse positionals, install, or train.
if [ -n "${LEPTON_LOOP_SOURCE_ONLY:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

# ===========================================================================
# main
# ===========================================================================

PYTHON="${1:?python invocation}"
SIMULATOR="${2:?simulator}"
EXP_NAME="${3:?experiment name}"
EXP_FOLDER="${4:?exp folder}"
MAX_RUNS="${5:?max runs}"
shift 5  # remaining "$@" is the train_agent.py argument tail

cd "$EXP_FOLDER" || { echo "[lepton-loop] cannot cd to $EXP_FOLDER" >&2; exit 1; }

# Durable log on AMLFS: lep's live logs are tail-only and vanish once the
# container exits, so mirror all output to a file under the snapshot too (the
# SLURM path has slurm_output.log). No background readers here, so tee gets EOF
# and exits cleanly with the script. Appended across loop iterations and across
# any preemption relaunch. It is also the growth signal the stall watchdog reads.
LOGFILE="${EXP_FOLDER}/lepton_train.log"
exec > >(tee -a "$LOGFILE") 2>&1

# A give-up sentinel from a PRIOR attempt (deterministic failure, no progress).
# Exit NONZERO so this relaunch is marked failed rather than falsely successful:
# exiting 0 here would let the job's final state read as "completed" even though
# training never succeeded, which is dangerous for any automation watching the job
# status. train_lepton.py clears this sentinel when you deliberately resume with
# --exp-folder; remove it by hand otherwise to allow retries.
GAVE_UP="${EXP_FOLDER}/.lepton_loop_gave_up"
if [ -f "$GAVE_UP" ]; then
    echo "[lepton-loop] a previous attempt gave up (deterministic failure; see the" \
        "log above). Exiting NONZERO so this job is marked failed, not falsely" \
        "successful. Remove ${GAVE_UP} (or resubmit via train_lepton.py --exp-folder," \
        "which clears it) to allow retries." >&2
    exit 1
fi

CKPT="results/${EXP_NAME}/last.ckpt"
NO_PROGRESS_LIMIT="${NO_PROGRESS_LIMIT:-3}"
STALL_TIMEOUT_MIN="${STALL_TIMEOUT_MIN:-30}"

echo "[lepton-loop] host=$(hostname) exp=${EXP_NAME}"
echo "[lepton-loop] folder=${EXP_FOLDER}"
echo "[lepton-loop] durable log: ${LOGFILE}"
echo "[lepton-loop] python=${PYTHON} simulator=${SIMULATOR} max_runs=${MAX_RUNS}" \
    "no_progress_limit=${NO_PROGRESS_LIMIT} stall_timeout_min=${STALL_TIMEOUT_MIN}"

# --- one-time per-container setup -------------------------------------------
# Import the staged snapshot as an editable install; runtime deps already live in
# the image (--no-deps). IsaacLab's bare python/pip are not runnable, so pip must
# go through the isaaclab.sh wrapper -- a plain `pip install` there exits 127.
# Skipped under LEPTON_LOOP_SKIP_SETUP so a test can drive the loop with a fake
# python that needs no install.
if [ -z "${LEPTON_LOOP_SKIP_SETUP:-}" ]; then
    export PYTHONUNBUFFERED=1
    echo "[lepton-loop] installing protomotions from the snapshot..."
    if [ "$SIMULATOR" = "isaaclab" ]; then
        /workspace/isaaclab/isaaclab.sh -p -m pip uninstall -y protomotions || true
        /workspace/isaaclab/isaaclab.sh -p -m pip install -e . --no-deps || {
            echo "[lepton-loop] protomotions install failed" >&2; exit 1; }
        # NB: do NOT `pip install --upgrade wandb` here. The image's wandb already
        # satisfies protomotions' pin (<0.24); upgrading pulls wandb 0.29 (violates the
        # pin) and drags click to 8.5, which conflicts with isaacsim-kernel's click==8.1.7.
    else
        pip uninstall -y protomotions 2>/dev/null || true
        pip install -e . --no-dependencies || {
            echo "[lepton-loop] protomotions install failed" >&2; exit 1; }
    fi

    # IsaacLab lazily bootstraps a shared pip env the first time AppLauncher starts.
    # Warm it once here so the Fabric-spawned worker ranks (multi-GPU) do not race on
    # it. A single process runs this loop per node, so no rank coordination is needed
    # -- unlike the SLURM path, where srun launches one task per GPU.
    if [ "$SIMULATOR" = "isaaclab" ]; then
        echo "[lepton-loop] warming IsaacLab AppLauncher..."
        /workspace/isaaclab/isaaclab.sh -p -c \
            "from isaaclab.app import AppLauncher; app = AppLauncher({'headless': True}).app; app.close()" || {
            echo "[lepton-loop] IsaacLab warmup failed" >&2; exit 1; }
    fi
else
    echo "[lepton-loop] LEPTON_LOOP_SKIP_SETUP set -- skipping install + warmup (test)."
    export PYTHONUNBUFFERED=1
fi

# --- distributed rendezvous (multi-node) ------------------------------------
# Lightning Fabric's environment reads MASTER_ADDR / MASTER_PORT / NODE_RANK; on
# Lepton those are not set for us, so derive them from the Lepton worker env when
# --nodes > 1 (one worker == one node). Single-node binds to loopback -- sibling
# ranks cannot resolve the container hostname on Lepton. NB: train_lepton.py
# currently REFUSES --nodes > 1, so in practice NNODES is 1 here; this branch is
# groundwork kept ready for when multi-node is qualified.
NNODES="${LEPTON_JOB_TOTAL_WORKERS:-1}"
export NODE_RANK="${LEPTON_JOB_WORKER_INDEX:-0}"
if [ "$NNODES" -gt 1 ]; then
    export MASTER_ADDR="${MASTER_ADDR:-${LEPTON_JOB_WORKER_HOSTNAME_PREFIX}-0.${LEPTON_SUBDOMAIN}}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
    # GPU Direct RDMA is off by default on these nodes; without it cross-node NCCL
    # runs ~9x slower. SYS engages GDR via dmabuf. NB: this only helps when both
    # workers land in the SAME network group -- a cross-group 2-node job still
    # hangs/degrades, which the job's node selection (not this env) must prevent.
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-SYS}"
else
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export MASTER_PORT="${MASTER_PORT:-29500}"
echo "[lepton-loop] rendezvous: NNODES=${NNODES} NODE_RANK=${NODE_RANK}" \
    "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

# --- run --------------------------------------------------------------------
run_training_loop "$@"
exit $?
