# Training Run Outputs — `results/<experiment_name>/`

Every `train_agent.py` run writes its artifacts to `results/<experiment_name>/`
(gitignored). This document explains what each file is, which code writes it,
and which ones you actually need.

Nothing here is written by a generic Lightning callback — each artifact comes
from a specific place in the codebase, listed below so you can check the
cadence and payload yourself.

## Configs and provenance

Small (tens of KB), and the only record of *how* a run was launched. Keep all
of them.

| File | Written by | Contents |
| --- | --- | --- |
| `config.yaml` | `train_agent.py` | Verbatim dump of the CLI args: robot, simulator, `--num-envs`, `--batch-size`, `--motion-file`, `--scenes-file`, `--checkpoint` (i.e. whether this run was a fine-tune), `--overrides`, seed. |
| `experiment_config.py` | `save_configs()` | Frozen copy of the `examples/experiments/.../<name>.py` used at launch. Your working tree copy may have drifted since; this is the version that produced the weights. |
| `resolved_configs.pt` / `.yaml` | `save_configs()` | The fully resolved dataclass tree (robot / simulator / env / agent / terrain / motion_lib / scene_lib). |
| `resolved_configs_inference.pt` / `.yaml` | `save_configs(file_name="resolved_configs_inference")` | Same tree with `apply_inference_overrides()` already applied. |

`resolved_configs.pt` is **authoritative for resume** — `configure_robot_and_simulator()`
is never re-run, so a resumed run reads this file, not your experiment file.
This is also why `--overrides` are permanent: they are baked in here.

`resolved_configs_inference.pt` is **required by `inference_agent.py`**, which
looks for exactly that filename next to `--checkpoint` (see
`utils/config_utils.py`, which falls back to `resolved_configs.pt`). Typical
differences from the training config: `reset_noise` set to `None`, the whole
`domain_randomization` block (action noise, friction buckets, pushes) stripped,
and some termination components removed — i.e. deterministic playback.

### `.pt` vs `.yaml` — same configs, different fidelity

Both files are written in the same call (`train_agent.py::save_configs`) from
the same seven config objects, so they describe the same configuration. They are
**not** interchangeable:

- `.pt` is `torch.save` of the live dataclass instances — a round-trippable
  object graph. `save_epoch_checkpoint_every` is the `int` `1000`;
  `MdpComponent.compute_func` is the actual function object.
- `.yaml` is `clean_dict_for_storage(asdict(cfg))`, whose fallback branch is
  `str(value)`. Every leaf becomes a **string**: `1000` → `'1000'`, `False` →
  `'False'`, `None` → `'None'`, tuples → `'(-0.05, 0.05)'`. Callables collapse
  to `value.__name__` with no module path, so the function cannot be
  reconstructed from the YAML.

Two practical consequences:

1. Nothing in the codebase ever *reads* `resolved_configs.yaml` — resume and
   inference load the `.pt`. Hand-editing the YAML changes nothing.
2. The YAML write is wrapped in `try/except` with a warning (genuinely
   "best-effort"), so a run may legitimately have the `.pt` and no `.yaml`.
   That is not corruption.

The YAML is for reading and diffing: it is the right thing to grep for
hyperparameters, and diffing `resolved_configs.yaml` against
`resolved_configs_inference.yaml` is the quickest way to see what inference
changes.

## Checkpoints

All model checkpoints have the same structure (`BaseAgent.get_state_dict`):

```
model                 # actor + critic (+ discriminator for AMP/ASE),
                      # including the running observation-normalizer stats
actor_optimizer       # Adam state — first/second moments
critic_optimizer
discriminator_optimizer / disc_critic_optimizer   # AMP/ASE only
epoch, step_count, run_start_time
best_evaluated_score
evaluator             # {"eval_count": ...}
```

The optimizer moments dominate the file size, which is why a small MLP policy
still produces a few hundred MB per checkpoint. These are *training*
checkpoints, not deployment artifacts.

| File | Cadence | Notes |
| --- | --- | --- |
| `last.ckpt` | `agent.save_last_checkpoint_every` (default **10** epochs) | The resume point. Overwritten in place. |
| `score_based.ckpt` | On every new best eval score | Snapshot at the highest `best_evaluated_score` seen. **This is the one to evaluate or export from**, not `last.ckpt` — training continues past the peak. |
| `epoch_<N>.ckpt` | `agent.save_epoch_checkpoint_every` (default **1000** epochs, `None` disables) | Periodic snapshots. Useful for bisecting a regression or comparing behaviour mid-training; otherwise the main disk cost of a run. |
| `inference_<name>.ckpt` | Only if `agent.save_inference_checkpoint=True` (default **False**) | Slimmed state dict without optimizer/training-only state. Off by default, so most runs have none and inference must load a full checkpoint. |
| `env_<task_id>.ckpt` | Alongside every save | **Not a model.** `task_id` is the motion filename (`env/base_env/env.py::get_task_id`), so it reads e.g. `env_my_motions.pt.ckpt`. Payload is `{"motion_manager": {"motion_file_name", "motion_weights"}}` — the prioritized-sampling weights per motion. Needed for a faithful resume; `motion_manager.load_state_dict` refuses to apply it if the motion filename does not match. |

Multi-GPU: rank 0 writes the model checkpoints; one env checkpoint is written
per *unique* task id (relevant for co-training over heterogeneous motion
libraries), and `failed_motions` files are per-rank.

## Evaluation outputs

Written by `agents/evaluators/mimic_evaluator.py`, which runs every
`evaluator.eval_metrics_every` epochs (default **200**).

**`failed_motions/failed_motions_epoch_<N>_rank_<R>.txt`**
One motion ID per line: the motions that failed tracking at that eval. An empty
file means everything succeeded. These are not just a log — `motion_manager.py`
can read the directory back (it scans for `failed_motions_epoch_*_rank_0.txt`
and picks the highest epoch) to seed sampling weights or to build a filtered
"hard motions" subset for a follow-up run. Point the motion manager at the run
directory and it finds the subdirectory itself.

**`results/predicted_motion_lib_epoch_<N>.pt`**
Written every `evaluator.save_predicted_motion_lib_every` evals (default **3**,
so every 600 epochs at the default eval cadence). Each file is the trajectory
the policy **actually executed**, repacked in MotionLib format (`gts`, `grs`,
`gvs`, `gavs`, `dps`, `dvs`, `contacts`, with lengths/dt/weights copied from the
ground-truth MotionLib). Because it is MotionLib-compatible it can be loaded
straight back as a motion file — useful for visualising policy output against
the reference, or for feeding rollouts into a distillation/supervised stage.

## Logs

**`lightning_logs/version_<N>/events.out.tfevents.*`**
TensorBoard scalars, always written (wandb is opt-in via `--use-wandb`). For a
run without wandb this is the only record of the learning curves. Watch
`Eval/gt_err` and `Eval/success_rate` (unbiased) over `Train/episode_reward`
(biased by prioritized sampling), and keep `Train/clip_frac` under ~0.3.

```bash
tensorboard --logdir results/<experiment_name>/lightning_logs
```

Note this directory grows large over a long run — often the second-biggest item
after the checkpoints.

## What to keep

- **Deploy / evaluate**: `score_based.ckpt` (or `last.ckpt`) +
  `resolved_configs_inference.pt` + `experiment_config.py`.
- **Resume training**: add `resolved_configs.pt`, `last.ckpt` and
  `env_<task_id>.ckpt`.
- **Prune first**: the `epoch_*.ckpt` series — especially once the eval score
  has saturated — and `lightning_logs/` if the curves have been exported.

## Worked example: `results/g1_box`

Fine-tune of the shipped `g1-bones-deploy` tracker on a box-pickup motion with
a scene (`examples/experiments/mimic/g1_pick_box.py`, AMP agent + mimic
evaluator, IsaacLab, 4096 envs). Stopped at epoch 12270 / 1.6 B env steps,
~3.2 GB total.

| Item | Size | Comment |
| --- | --- | --- |
| 14 × `.ckpt` (`last`, `score_based`, `epoch_1000`…`epoch_12000`) | 228 MB each, ~3.1 GB | Effectively the entire footprint. `best_evaluated_score` = 1.0. |
| `env_kimodo_g1_motions.pt.ckpt` | 1.9 KB | Motion sampling weights for `kimodo_g1_motions.pt`. |
| `lightning_logs/` | 98 MB | TensorBoard; wandb was disabled. |
| `results/predicted_motion_lib_epoch_*.pt` | 21 files, 12 MB | Every 600 epochs. |
| `failed_motions/` | 62 files, **all 0 bytes** | No motion ever failed — consistent with the 1.0 score, and unsurprising for a single small motion set. Carries no information for this run. |
| configs (`config.yaml`, `experiment_config.py`, `resolved_configs*`) | ~160 KB | Records the fine-tune source checkpoint and scene file. |

`save_inference_checkpoint` was `False`, so there is no `inference_*.ckpt`;
inference and ONNX export have to go through a full checkpoint plus
`resolved_configs_inference.pt`.
