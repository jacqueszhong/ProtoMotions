# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Implements

ProtoMotions3 (NVlabs) — GPU-accelerated RL framework for physically simulated humanoids and robots. Implements motion-tracking/imitation methods (DeepMimic-style tracking, AMP, ASE, MaskedMimic, ADD) on SMPL characters and Unitree G1/H1_2, plus AMASS→robot retargeting, sim2sim validation across 5 physics backends, and ONNX export for real-robot deployment. This repo is a personal fork (`jacqueszhong/ProtoMotions`, working branch `dev_lsi`).

## Hard Requirements

No custom CUDA/C++ extensions to compile — all GPU code comes from the simulator backends. But each backend pins its own toolchain, and **each needs its own virtualenv** (dependency conflicts are real):

| Backend | Python | Torch/CUDA | Notes |
|---|---|---|---|
| IsaacGym Preview 4 | **3.8 exactly** | torch ≥2.2 cu121 | Proprietary NVIDIA EULA; no torch.compile on py3.8 |
| IsaacLab 2.3.0 | **3.11** | torch==2.7.0 | Needs IsaacSim; EULA prompt on first run |
| Newton 1.0.0 | 3.10+ (3.11 rec.) | torch cu124 | Driver ≥545, compute capability ≥5.0 (Warp JIT) |
| MuJoCo 3.5 | 3.10+ | CPU torch | No GPU; `--num-envs 1` only; debug/validation |
| Genesis | 3.10 | — | Experimental, untested |

- Reference training scale: full AMASS in ~12h on 4×A100; single-GPU training works with fewer envs.
- Docker: `Dockerfile.isaacgym` (CUDA 12.1, Ubuntu 20.04, py3.8) and `Dockerfile.newton` (CUDA 12.4, Ubuntu 22.04) — used by `train_slurm.py` for cluster runs.

## Environment Setup

```bash
git lfs install && git lfs pull        # REQUIRED — checkpoints/meshes/USD are LFS

# Per-simulator env (example: IsaacLab, the recommended training backend)
uv venv --python 3.11 env_isaaclab && source env_isaaclab/bin/activate
uv pip install torch==2.7.0 torchvision==0.22.0
uv pip install "isaaclab[isaacsim,all]==2.3.0" --extra-index-url https://pypi.nvidia.com
uv pip install -e . && uv pip install -r requirements_isaaclab.txt

# Other backends: same pattern with requirements_<sim>.txt (see docs/source/getting_started/installation.rst)
```

- Local conda envs already exist: `proto`, `env_isaaclab`, `env_isaacsim6`, `env_kimodo` (base has no torch — activate one first).
- PyRoki (retargeting, `pyroki/`) requires its **own env**, separate from ProtoMotions.
- IsaacGym `libpython` error fix: `export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH`.

## Checkpoints & Data

- **Pretrained models ship in-repo** (via LFS): `data/pretrained_models/<family>/<name>/` with `last.ckpt`, `resolved_configs_inference.pt`, and a `MODEL_CARD.md` (read it — most are IsaacLab-only; only the G1 deployment tracker transfers across sims/hardware).
- Small motion subsets ship in `data/motion_for_trackers/` (g1/h1_2 tiny subsets) and `examples/data/` (SMPL subset).
- Full datasets are external and gated: **AMASS** + **SMPL/SMPL-X body models** need registration at their sites; **BONES-SEED** from HuggingFace (`bones-studio/seed`). Conversion pipelines in docs `getting_started/*_preparation.rst`: raw → `.motion` files → packaged `.pt` MotionLib.
- No env vars required; everything is path-based via CLI flags. Training outputs go to `results/<experiment_name>/`.

## Commands

```bash
# Train
python protomotions/train_agent.py \
    --robot-name g1 --simulator isaaclab \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name my_exp \
    --motion-file path/to/motions.pt \
    --num-envs 4096 --batch-size 16384 --ngpu 1

# Inference / visualize (works with shipped checkpoints out of the box)
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --simulator isaaclab            # add --headless on servers; keys: J=push, R=reset, L=record, Q=quit

# Config overrides (scalars only; format config_type.field=value)
--overrides "agent.num_mini_epochs=4" "env.max_episode_length=500"

# Tests (pure pytest, no simulator needed) / lint / smoke test
pytest protomotions/tests/                      # single: pytest path/to/test.py -k name
pre-commit run --all-files                      # Ruff + typos + license headers
./scripts/smoke_test.sh <python_bin> <experiment.py> <robot> <simulator>   # 60s training sanity check

# Retarget AMASS → robot (needs BOTH interpreters)
./scripts/retarget_amass_to_robot.sh <proto_python> <pyroki_python> <amass.pt> g1
```

## Training Modes

`train_agent.py` has two orthogonal notions of "mode": **how the run starts** (checkpoint mode, chosen automatically by `detect_checkpoint_mode()`) and **what is trained** (the algorithm, chosen by `--experiment-path`).

### Start modes (automatic, based on what exists on disk)

Resolved in `train_agent.py:detect_checkpoint_mode()` — you never pass a mode flag, you trigger one by what you provide:

| Mode | Triggered by | What happens |
|---|---|---|
| **fresh** | `results/<experiment_name>/last.ckpt` absent **and** no `--checkpoint` | Loads the experiment file, builds configs, applies `--overrides`, trains from random init, writes `resolved_configs*.pt`. |
| **warm_start** | `--checkpoint path/to/x.ckpt`, with no `last.ckpt` in the run dir | Same config build as fresh (experiment file *is* re-executed, `--overrides` apply), then loads **weights only** — no optimizer/epoch state. This is how you fine-tune a pretrained tracker or migrate configs onto old weights. |
| **resume** | `results/<experiment_name>/last.ckpt` exists (takes priority over `--checkpoint`) | Does **not** load the experiment file. Configs come verbatim from `resolved_configs.pt`, CLI args are overwritten by the saved `config.yaml`, `--overrides` are **ignored with a warning**, training state (optimizer/epoch/wandb id) is restored, and the first policy update is skipped. |
| **create_config_only** | `--create-config-only` | Forces the fresh config-build path, writes `resolved_configs.pt/.yaml` + `resolved_configs_inference.pt/.yaml` + `experiment_config.py`, then exits before any simulator/agent is created. Use to regenerate configs for stale checkpoints, then rerun with `--checkpoint <old_weights>` (warm start). |

Consequences worth remembering: to change any config on an existing run you must use a **new `--experiment-name`** (or delete/move `last.ckpt`) — resume can't be overridden; and `--use-slurm` adds the autoresume callback that relies on this resume path (`agents/callbacks/slurm_autoresume_srun.py`, 12600 s).

### Algorithm modes (`--experiment-path`)

Every file under `examples/experiments/` is a self-contained training mode. All take the common flags (`--robot-name --simulator --num-envs --batch-size --motion-file --experiment-name`); the table lists only what's *extra*. Files exposing `additional_experiment_arguments()` add their own CLI flags, which show up in `--help` only once `--experiment-path` is given.

| Experiment file | Agent class | Trains | Extra flags |
|---|---|---|---|
| `mimic/mlp.py` | `ppo.agent.PPO` | Baseline full-body motion tracker (pose+velocity reward, early termination). | — |
| `mimic/mlp_complex_terrain.py` | PPO | Same but on generated rough terrain with terrain obs. | — |
| `mimic/mlp_bm_l2c2.py` | PPO | BeyondMimic-style tracker + L2C2 smoothness penalty (clean/noisy obs pair); the recipe behind the shipped G1 deploy model. | — |
| `mimic/fsq.py` | PPO | Tracker with an **FSQ-quantized latent** — the prerequisite for the GPC prior. | — |
| `mimic/g1_pick_box.py` | `amp.agent.AMP` | Fork of the g1-bones-deploy tracker + box-tracking rewards/terminations (local WIP). | `--scenes-file <boxes.pt>` |
| `add/mlp.py` | `mimic.agent_add.MimicADD` | ADD — tracking with an adversarial differential discriminator instead of hand-tuned reward weights. | — |
| `amp/mlp.py` | AMP | Style-only motion prior (discriminator reward, no per-frame tracking). | — |
| `ase/mlp.py` | `ase.agent.ASE` | AMP + learned latent skill space (skill-conditioned policy). | — |
| `steering/mlp.py` | AMP | Heading/speed steering task on top of an AMP style reward. | — |
| `path_follower/mlp.py` | AMP | Path-following task on top of an AMP style reward. | — |
| `masked_mimic/transformer.py` | `supervised.agent.SupervisedAgent` (MaskedMimic preset) | MaskedMimic distillation — transformer inpainting controller supervised by a full-body expert. | `--expert-model-path <tracker.ckpt>` |
| `gpc/prior.py` | `supervised.agent.SupervisedAgent` (latent-prior preset) | Autoregressive discrete prior over the FSQ tracker's codes (supervised, not RL). | `--tracker-checkpoint <fsq_tracker.ckpt>` |
| `gpc/sft_target_prior_peft.py` | `peft.sft_agent.DiscretePriorPEFTSFTAgent` | SFT bootstrap of a PEFT adapter for target reaching (cross-entropy against tracker codes). | `--prior-checkpoint`, `--tracker-checkpoint` (defaults to the shipped soma tracker) |
| `gpc/task_target_prior_peft.py` | `peft.prior_agent.DiscretePriorPEFTRLFTAgent` | RLFT of the PEFT adapter for random target reaching over a frozen prior. | `--prior-checkpoint`; usually `--checkpoint <sft_run>/last.ckpt` |
| `gpc/task_target_prior_peft_amp.py` | `…PEFTRLFTAMPAgent` | Same, plus an AMP discriminator reward/critic. | same |
| `gpc/task_steering_headvel_prior_peft.py` | `…PEFTRLFTAgent` | RLFT adapter for heading+velocity steering. | `--prior-checkpoint` |
| `gpc/task_steering_headvel_prior_peft_amp.py` | `…PEFTRLFTAMPAgent` | Steering RLFT + AMP rewards. | `--prior-checkpoint` |

GPC is a **staged pipeline**, each stage warm-starting from the previous: FSQ tracker (`mimic/fsq.py`) → prior (`gpc/prior.py --tracker-checkpoint`) → SFT (`gpc/sft_target_prior_peft.py --prior-checkpoint`) → RLFT (`gpc/task_*_peft*.py --prior-checkpoint --checkpoint <sft>`). Full worked commands: `docs/source/user_guide/gpc.rst`. Prose overview of each algorithm: `docs/source/user_guide/experiments.rst`.

Sanity-check any mode in ~60 s before committing GPU hours:
`./scripts/smoke_test.sh <python_bin> examples/experiments/<file>.py <robot> <simulator>`.

## Architecture

- **Entry points**: `protomotions/train_agent.py`, `protomotions/inference_agent.py`, `protomotions/train_slurm.py`. They parse args, import the simulator, then hand off to the experiment file.
- **Config system = experiment files** (`examples/experiments/<task>/<variant>.py`) — plain Python, no Hydra/YAML. Each defines `configure_robot_and_simulator()`, `env_config()`, `agent_config()`, `motion_lib_config()`, `terrain_config()`, `scene_lib_config()`, `apply_inference_overrides()`. Configs are dataclasses living in a `config.py` next to each module.
- **Composable MDP** (`protomotions/envs/`): tasks are assembled from `MdpComponent`s, not monolithic env classes. A component = pure tensor `compute_func` (kernels in `envs/obs|rewards|terminations|control/`) + `dynamic_vars` (`FieldPath` objects from class-level `EnvContext` access, e.g. `EnvContext.current.dof_pos`) + `static_params` (constants/metadata like `weight`). `ComponentManager` runs them with torch.compile caching; prebuilt factories in `component_factories.py`. This design lets ONNX export bake obs computation into the deployed model.
- **Agents** (`protomotions/agents/`): one dir per algorithm (`ppo`, `mimic`, `amp`, `ase`, `masked_mimic`, `peft`, `supervised`, ...) with `agent.py`/`config.py`/`model.py`. New algos subclass existing ones — `agents/mimic/agent_add.py` is a ~50-line example.
- **Simulators** (`protomotions/simulator/`): `base_simulator/` is the abstract API; one dir per backend; `factory.py` maps `--simulator` string → config class.
- **Robots** (`protomotions/robot_configs/`): `g1.py`, `h1_2.py`, `smpl.py`, ...; register new ones in `factory.py`; MJCF assets in `protomotions/data/assets/mjcf/`.
- **Shared** (`protomotions/components/`): `motion_lib` (motion data), `pose_lib`, `scene_lib`, `terrains`. SceneLib API reference (object model, `.pt` format, replication/subsetting, runtime getters): `docs/scene_lib_api.md` — read it before touching scene code.
- **Reproducibility**: every run writes `results/<name>/resolved_configs.pt` (pickled configs — the authoritative source for resume/inference), `resolved_configs.yaml` (human-readable), `experiment_config.py` (copy), `last.ckpt`. Full run-output reference (every artifact, its writer, cadence, and what's safe to prune): `docs/training_outputs.md`. Key points: `score_based.ckpt` (best eval score) is what you evaluate/export, not `last.ckpt`; `env_<motion_file>.ckpt` is motion sampling weights, not a model; `inference_agent.py` requires `resolved_configs_inference.pt` next to the checkpoint; `failed_motions/*.txt` and `results/predicted_motion_lib_epoch_*.pt` are re-loadable inputs, not just logs.

## Conventions

- **Multi-GPU**: Lightning Fabric DDP via `--ngpu N --nodes M` — no torchrun/accelerate wrapper. Cluster launch via `train_slurm.py` + Dockerfiles.
- **Seeding**: `--seed` (default 0), applied per rank as `seed + rank`; `--torch-deterministic` for strict determinism.
- **Logging**: wandb opt-in via `--use-wandb` (`wandb login` first). Watch `Eval/gt_err`, `Eval/success_rate` (unbiased) over `Train/episode_reward` (biased by prioritized sampling); keep `Train/clip_frac` under ~0.3.
- Every file starts with the SPDX Apache-2.0 header (pre-commit auto-inserts); Ruff for lint+format.
- Keep public code Python 3.8-compatible (IsaacGym): no `dict[...]`-style builtin generics without `from __future__ import annotations`; guard torch.compile. Enforced by `test_gpc_python38_annotations.py`.
- Upstream PRs require DCO sign-off: `git commit -s`.

## Gotchas

- **License**: code is **Apache-2.0** (permissive — unusual for NVlabs). But bundled assets carry third-party terms (`legal/`): SMPL/SMPL-H, Unitree, IsaacLab, BeyondMimic. IsaacGym itself and AMASS/SMPL datasets have their own restrictive/registration-gated licenses.
- **Import order**: `isaacgym`/`isaaclab` must be imported *before* torch. Entry scripts parse args first, then call `protomotions.utils.simulator_imports.import_simulator_before_torch()` — preserve this pattern in new scripts.
- **`--overrides` are permanent**: saved into `resolved_configs.pt` and reused on every resume. For temporary changes, use a new `--experiment-name`. Inference does NOT re-run `configure_robot_and_simulator()` — it loads frozen configs, then `apply_inference_overrides()` + CLI on top.
- **LFS pointer files**: errors like `is not a valid usda layer` mean assets weren't pulled — run `git lfs pull`.
- **`dynamic_vars` vs `static_params`**: a concrete tensor in `dynamic_vars` raises `AttributeError` — it must be a `FieldPath`; tensors/scalars go in `static_params`.
- Checkpoints are generally **not portable across simulators** — check the MODEL_CARD.md before evaluating in a different backend.
- Newton on Python 3.10 compiles `imgui-bundle` from source (10–20 min); use 3.11+ for prebuilt wheels.
