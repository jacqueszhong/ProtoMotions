# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the ProtoMotions cluster launchers.

`train_slurm.py` (SLURM) and `train_lepton.py` (DGX Cloud Lepton) submit to very
different systems -- sbatch + srun + a job array with a walltime, versus
`lep job create` with no walltime and an in-job restart loop. Almost nothing is
shared between them *except* the one thing that must stay identical: the inner
`protomotions/train_agent.py` command line. Keeping that here stops the two
submitters from silently drifting on training arguments.

What is deliberately NOT here is `--use-slurm`. That flag installs the SLURM
walltime autoresume callback (stop-and-requeue at ~3.5h), which only makes sense
where a job array hands off before a walltime kill. Lepton jobs have no walltime,
so `train_lepton.py` omits it and relies on filesystem-based resume instead
(train_agent.py finds results/<exp>/last.ckpt and continues). Each launcher adds
its own cluster-specific flags around this shared tail.
"""

import shlex


# The code snapshot both launchers rsync to the cluster. One list, because the
# two used to keep their own and drifted: train_lepton's copy was missing
# `.worktrees` and `.claude`, which on a working tree that keeps its git
# worktrees in-tree is tens of gigabytes of duplicate checkouts pushed over a
# Teleport tunnel on every submit.
#
# Everything here is either not code (caches, editor state, session data), a
# duplicate checkout, or a large blob that is pre-staged on the cluster and
# passed by absolute path instead of re-uploaded.
RSYNC_EXCLUDES = [
    ".git",
    ".idea",
    ".claude",
    ".cursor",
    ".vscode",
    ".ruff_cache",
    "**/__pycache__",
    "**/*.egg-info",
    # Local worktrees and in-tree virtualenvs are full duplicate checkouts /
    # gigabytes -- never ship them.
    ".worktrees",
    ".venv**",
    "outputs",
    "output/*",
    "results",
    "exps",
    "tmp",
    "**/*tmp*",
    "wandb",
    # Root-level scratch caches (e.g. token/plan caches).
    "**/_*_cache.pt",
    # Large data blobs, pre-staged on the cluster rather than re-uploaded.
    "data/smpl",
    "data/amass",
    "data/pretrained_models",
    "docs",
    "docs/*",
    # Media that never belongs in a code snapshot.
    "**/*.mp4",
    "**/*.avi",
    "**/*.gif",
    "**/*.jpg",
    "**/*.png",
    "**/*.onnx",
]


def build_train_agent_argv(args, passthrough=None) -> list:
    """Return the shared `train_agent.py` argument tail as a list of argv tokens.

    A list, not a string, so the caller controls quoting for its own context:
    train_lepton.py hands these to `lep`'s container command as individually
    shlex-quoted positionals (so a value containing a space survives the container
    shell AND the loop's `"$@"`), while train_slurm.py joins them into its sbatch
    line via `build_train_agent_args` below. Both launchers stay in lockstep on
    WHICH arguments train_agent.py receives.

    Excludes the python invocation and any cluster-specific flag (notably
    `--use-slurm`): the caller prepends the interpreter and appends whatever its
    scheduler needs. `args` is the parsed argparse namespace; both launchers
    expose the same option names for these fields. `passthrough` are CLI flags
    the launcher did not recognise (typically experiment-contributed, e.g.
    gpc/prior.py's --tracker-checkpoint) -- forwarded verbatim to train_agent.py,
    which does know them.
    """
    argv = [
        f"--robot-name={args.robot_name}",
        f"--simulator={args.simulator}",
        f"--motion-file={args.motion_file}",
        f"--ngpu={args.ngpu}",
        f"--nodes={args.nodes}",
        f"--experiment-name={args.experiment_name}",
        f"--experiment-path={args.experiment_path}",
        f"--num-envs={args.num_envs}",
        f"--batch-size={args.batch_size}",
    ]

    # Exactly one of these is set (argparse mutually-exclusive group), but guard
    # on the iterations value the same way train_agent.py's callers do.
    if args.training_max_iterations is not None:
        argv.append(f"--training-max-iterations={args.training_max_iterations}")
    else:
        argv.append(f"--training-max-steps={args.training_max_steps}")

    if args.scenes_file:
        argv.append(f"--scenes-file={args.scenes_file}")
    if args.use_wandb:
        argv.append("--use-wandb")
        argv.append(f"--wandb-project={args.wandb_project}")
    if args.checkpoint:
        argv.append(f"--checkpoint={args.checkpoint}")
    # Experiment-contributed / unknown flags, forwarded verbatim, each as its own
    # token. BEFORE --overrides, which is nargs="*" in train_agent and would
    # otherwise swallow them as override values.
    if passthrough:
        argv.extend(passthrough)
    if args.overrides:
        argv.append("--overrides")
        argv.extend(args.overrides)

    return argv


def build_train_agent_args(args, passthrough=None) -> str:
    """String form of `build_train_agent_argv` for a shell context (the SLURM
    sbatch line). `shlex.join` quotes each token only as needed, so this is
    identical to a plain space-join for the usual space-free tokens and merely
    SAFER when a value contains a space. Prefer `build_train_agent_argv` where the
    tokens can be passed as real arguments (train_lepton.py)."""
    return shlex.join(build_train_agent_argv(args, passthrough))
