# ONNX Export and Deployment — `unified_pipeline.onnx` + `.yaml`

Deploying a trained tracker means turning `results/<exp>/last.ckpt` into two
files that a robot-side process can consume without PyTorch, without the
ProtoMotions package, and without a simulator:

```
unified_pipeline.onnx    the compiled policy — observation math, network, and
                         action processing fused into one static graph
unified_pipeline.yaml    the deployment contract — everything the graph cannot
                         express: names, orderings, gains, rates, frames
```

Neither file is produced during training. Export is a separate, offline,
CPU-only step. This document covers how each file is written, what is actually
inside them, and exactly which parts are read back at deployment time.

## 1. Export

### Entry point

```bash
python deployment/export_bm_tracker_onnx.py \
    --checkpoint results/my_exp/last.ckpt \
    [--output <dir>]        # default: <checkpoint_dir>/compiled_models/
    [--no-validate]         # skip the onnxruntime round-trip check
```

No GPU, no simulator import, no `--robot-name`/`--simulator` flags — everything
is recovered from the checkpoint directory. Worked example with the shipped G1
model: `docs/source/tutorials/workflows/g1_deployment.rst`.

### What `export_tracker()` does

All of the following lives in `deployment/export_bm_tracker_onnx.py:144`.

1. **Load frozen configs** (`:182`). Reads `resolved_configs_inference.pt` from
   the checkpoint's directory, falling back to `resolved_configs.pt` with a
   warning. The fallback matters: the training config still has domain
   randomization active, so the exported graph would carry observation noise.
2. **Auto-detect the observation set** (`:205`) from
   `agent_config.model.actor.in_keys`. Nothing is hardcoded per robot — the
   actor declares what it consumes.
3. **Extract dimensions** (`:212-238`): DOF/body counts and the anchor body from
   `robot_config.kinematic_info`, `future_steps` from the `mimic` control
   component, action-history depth from the obs component's `static_params`.
4. **Resolve MuJoCo timing** (`:240-268`) by re-running
   `update_simulator_config_for_test(..., new_simulator="mujoco")`. The exported
   contract therefore advertises *MuJoCo* rates (1 kHz physics, decimation 20),
   not the rates the policy was trained at under IsaacLab.
5. **Build a `MockContext`** (`:116`) — fake `current` / `mimic` / `historical`
   tensors of the right shapes. This is what lets the MDP components be traced
   with no environment in existence.
6. **Assemble three modules** from `protomotions/utils/export_utils.py`:
   `ObservationExportModule` (`:1065`) → the actor → `ActionExportModule`
   (`:321`), composed by `UnifiedPipelineModule` (`:379`). The actor is rebuilt
   from `get_class(agent_config.model.actor._target_)`, given one mock forward
   pass to materialise `nn.LazyLinear` shapes, then loaded with the
   `_actor.`-prefixed weights out of `ckpt["model"]`.
7. **Trace to ONNX** (`:403`) at opset 17, `dynamo=False`, with dim 0 marked
   dynamic so any batch size works.
8. **Validate** (`:447`, unless `--no-validate`): re-run through onnxruntime and
   compare against the PyTorch outputs at 1e-4.
9. **Write the YAML** (`:501-533`) — built field by field by `_build_yaml()`
   (`:543`), not dumped from a config object.

### Why the observation math ends up inside the graph

This is the payoff of the composable-MDP design (`protomotions/envs/`). An
`MdpComponent` is a pure tensor `compute_func` plus `dynamic_vars` declared as
`FieldPath` objects. Because the compute function is pure and its inputs are
named paths, `ObservationExportModule` can call it with mock tensors and let
`torch.onnx.export` trace straight through it. The deployment side therefore
never reimplements quaternion frames, normalisation, or history stacking — a
notorious source of sim-to-real drift.

## 2. What is inside the `.onnx`

A protobuf `ModelProto`. No Python, no pickle, no code execution. Measured on
the shipped `data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx`
(22.6 MB):

| Part | Value |
| --- | --- |
| `ir_version` / `producer` | 8 / `pytorch 2.10.0` |
| `opset_import` | `ai.onnx` v17 |
| `graph.node` | 630 operator nodes, a static DAG |
| `graph.initializer` | 21 constant tensors, 5.64 M float32 values (≈ the whole file) |
| `metadata_props` | empty — the sidecar YAML carries all metadata |

Inputs and outputs, with `batch_size` as a symbolic dimension:

```
current_anchor_rot            [batch, 4]      actions            [batch, 29]
current_dof_pos               [batch, 29]     joint_pos_targets  [batch, 29]
current_dof_vel               [batch, 29]     stiffness_targets  [batch, 29]
current_root_local_ang_vel    [batch, 3]      damping_targets    [batch, 29]
historical_processed_actions  [batch, 1, 29]
mimic_future_anchor_rot       [batch, 4, 4]
mimic_future_dof_pos          [batch, 4, 29]
mimic_future_dof_vel          [batch, 4, 29]
```

The node histogram shows the three fused stages:

- **Observation math**, inlined under the `/observation_module/` name prefix:
  259 `Constant`, 42 `Concat`, 42 `Mul`, 40 `Slice`, 38 `Shape`, plus
  `Gather`/`Reshape`/`Sub`. Frame transforms become elementwise ops.
- **Policy**: 7 `Gemm` + 6 `Relu` — a 349 → 1024×6 → 29 MLP holding all 5.6 M
  parameters (`policy_module.mu.mlp.*`).
- **Action processing**: `action_module.pd_action_offset` and
  `action_module.action_scale` as baked-in constants.

Two things are constant-folded in that are easy to overlook:

- **Observation normalisation** — initializers `onnx::Sub_787` and
  `onnx::Div_790`, both `[349]`, are the running mean/std of the obs
  normaliser. There is no separate normaliser file to ship, and no chance of
  forgetting to apply it.
- **PD gains** — shipped as `[1, 29]` constants that are merely `Expand`ed to
  the batch, which is why `stiffness_targets` / `damping_targets` are graph
  outputs despite never varying.

There are **no `Loop` / `If` / `Scan` nodes**: the graph is fully static. Every
call does identical work, which is what makes latency predictable enough for a
50 Hz control loop.

What the graph deliberately does *not* contain: the MJCF path, joint or body
names, control rates, or future-step indices. A `[batch, 29]` tensor does not
tell you which 29 joints, in which order. That gap is the YAML's entire reason
to exist.

## 3. What is inside the `.yaml`

Not a standard format. ONNX defines no sidecar convention (only the flat
`metadata_props` map inside the file), and nothing in IsaacLab produces or reads
this schema. It is ProtoMotions-specific: the values in
`_runtime.onnx_name_to_in_key` are literally `EnvContext` attribute paths from
this repo's MDP design.

Current schema, as emitted by `_build_yaml()`:

| Key | Contents |
| --- | --- |
| `type`, `dt` | Format tag; policy control period. |
| `joint_names`, `body_names` | Canonical orderings — the meaning of every per-DOF and per-body axis. |
| `default_joint_stiffness` / `_damping` | Per-joint PD gains. |
| `policy_inputs` | Per ONNX input: name, semantic key, shape, a `kind` tag, and `element_names` (joint names, quaternion `x,y,z,w` order). |
| `policy_outputs` | Same for the four outputs. |
| `_runtime` | `onnx_in_names`, `onnx_out_names`, `onnx_name_to_in_key`, `passthrough_keys`, `obs_context_keys`. |
| `metadata` | Source checkpoint, control type. |
| `robot` | `mjcf_path`, body/DOF counts, anchor and root body name **and index**. |
| `control` | `stiffness`, `damping`, `effort_limits`, `pd_target_max_accel`, `action_ema_alpha`. |
| `timing` | `control_dt`, `physics_dt`, `decimation`. |
| `motion` | `future_step_indices` and their dt in seconds. |

Two details worth knowing about how it is built:

- **ONNX input names are read back from a live session** (`:424-442`), not
  assumed. The exporter sanitises `current.dof_pos` → `current_dof_pos`, but
  ONNX may append `.1`/`_1` suffixes; `onnx_name_to_in_key` is the reconciled
  mapping and is the only reliable way back to semantic keys.
- **`passthrough_keys` explains an input-count mismatch.** `obs_context_keys`
  can list more entries than the graph has inputs — the difference is context
  values that were constant-folded away during tracing.

### Legacy schema

`data/pretrained_models/motion_tracker/soma-bones/compiled_models/unified_pipeline.yaml`
uses a **different, older schema**: `deploy_inputs` instead of `policy_inputs`,
an extra `conventions` block, `robot.usd_path`, `control.armature` /
`velocity_limits`, per-simulator `timing.{mujoco,isaaclab,isaacgym}` blocks, and
no `type` / `dt` / `policy_outputs`. No code in the repo emits it — grep for
`deploy_inputs` or `conventions` and you get nothing — so it cannot be
regenerated. Its own `MODEL_CARD.md` calls the export legacy and explicitly not
a support contract. Treat it as a historical artifact; `_build_yaml()` is the
live format.

## 4. Consumption at deployment

Reference consumer: `deployment/test_tracker_mujoco.py`. Its full dependency set
is `mujoco, onnxruntime, numpy, pyyaml, torch` — and torch only for
`torch.load` on the motion file. On real hardware you drop MuJoCo and read state
from the robot SDK instead.

The YAML path is derived from the ONNX path (`:480`, `.onnx` → `.yaml`), so the
two files must sit together with the same stem. One `yaml.safe_load` at `:486`,
then five top-level sections are indexed (`:488-492`).

### Keys that are read

| Key | Fields | Consumed by |
| --- | --- | --- |
| `robot` | `mjcf_path` | `load_mujoco_model` (`:572`) → `_resolve_mjcf_path` (`:158`) |
| | `anchor_body_index` | `read_robot_state` (`:668`), `compute_anchor_rot_np` (`:407`), initial motion alignment (`:672`), and the slice producing `mimic_future_anchor_rot` (`:414`) |
| | `root_body_index` | `read_robot_state`, where root rotation is taken from the free-joint qpos rather than `xquat` (`:347`) |
| | `num_dofs` | zeroing `prev_actions` on the first step (`:411`) |
| | `num_bodies`, `anchor_body_name`, `root_body_name` | log lines only (`:517-519`); the names use `.get()` and are optional |
| `control` | `stiffness`, `damping` | asserted against `model.nu` (`:296`), then written into `actuator_gainprm` / `actuator_biasprm` as MuJoCo implicit PD (`:301-310`) |
| | `pd_target_max_accel` | second-derivative clamp on PD targets (`:703-707`); `null` disables it |
| | `action_ema_alpha` | EMA filter on PD targets (`:718`); `1.0` disables it; overridable by `--action-ema-alpha` |
| `timing` | `physics_dt` | `model.opt.timestep` (`:278`), overriding the MJCF's own timestep |
| | `decimation` | inner `mj_step` loop (`:729`) |
| | `control_dt` | `MotionPlayer(...)` (`:549`) and wall-clock pacing (`:748`) |
| `motion` | `future_step_indices` | `player.get_future_references(...)` (`:677`) |
| `_runtime` | `onnx_name_to_in_key` | the input-assembly loop (`:431`) |

### The control loop

```python
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
# each control step (control_dt = 0.02 s):
#   read MuJoCo state           -> read_robot_state(data, anchor_idx, root_idx)
#   slice reference motion      -> player.get_future_references(frame, step_indices)
#   map semantic keys -> tensors, per onnx_name_to_in_key
#   outs = session.run(out_names, onnx_inputs)
#   apply pd_target_max_accel clamp and action_ema_alpha filter (external!)
#   write outs[1] (joint_pos_targets) to data.ctrl
#   step physics `decimation` times
#   feed outs[0] (actions) back in as historical_processed_actions
```

Two properties of that loop follow from the graph being stateless:

- **The action history is a feedback loop closed outside the model.** The YAML
  flags it via `policy_inputs[].output_key: robot_action`.
- **The reference motion is sliced by the caller.** You provide all
  `future_step_indices` frames; the traced obs function selects internally.

And two post-processing steps are deliberately *not* baked into the graph —
`pd_target_max_accel` and `action_ema_alpha` must be applied by the deploy
script, which is exactly what the `control.note` field in the legacy soma YAML
spells out.

### Keys that are never read

`type`, `dt`, `joint_names`, `body_names`, `default_joint_stiffness`,
`default_joint_damping`, `policy_inputs`, `policy_outputs`, `metadata` — and,
inside `_runtime`, everything except `onnx_name_to_in_key`.

Why each is inert, since "unused" does not mean "safe to delete":

- `dt` duplicates `timing.control_dt`; the script uses the latter, and nothing
  checks that they agree.
- `policy_outputs` is bypassed by **positional** indexing — `ort_out[1]` is
  assumed to be `joint_pos_targets` (`:700`). Output names are never matched.
- `policy_inputs` is superseded by `_runtime.onnx_name_to_in_key`. Its `kind`
  tags and `element_names` exist for other consumers (e.g. robojudo), which do
  read them.
- `_runtime.onnx_in_names` / `onnx_out_names` are re-read from the live session
  instead (`:528-529`).
- `joint_names` / `body_names` are **never validated against the MJCF** — see
  below.

## 5. Gotchas

- **`_resolve_mjcf_path` resolves relative to the ProtoMotions repo root**
  (`:158`), *not* to the YAML's own directory. It tries `<repo>/<mjcf_path>`
  then `<repo>/protomotions/data/assets/<mjcf_path>`. A YAML moved outside the
  repo still works, but only because of that second candidate.
- **Joint and body ordering is assumed, not checked.** Only the actuator *count*
  is asserted (`:296`). Body order is assumed to match `data.xquat[1:]` — MuJoCo
  body 0 is the world body, hence the `[1:]` at `:341`. So an
  `anchor_body_index: 16` meaning `torso_link` is an unverified claim about that
  MJCF's body ordering. Regenerate the MJCF with a different body order and
  nothing errors; you silently track the wrong body.
- **`effort_limits` is exported but not consumed** by the MuJoCo runner
  (explicit TODO at `:315`), so `actuator_forcelimited` stays at MuJoCo's
  default of disabled and joints run with unbounded torque in validation. Also
  note the field is `effort_limit` on `control_info`, not `effort` — reading the
  wrong one silently produced `effort_limits: null` in every export.
- **A missing semantic key is only a warning.** The input-assembly loop
  (`:431`) logs and skips unknown keys; the failure surfaces later inside
  `session.run` as a missing input. If you re-export with a different
  observation set, watch for that warning.
- **The runner is BM-specific.** `key_to_array` (`:415-424`) is a hardcoded dict
  of BeyondMimic keys. A max-coords model wanting `current.rigid_body_*` or
  `ground_heights` will not run with it, regardless of what its YAML says.
- **Exported timing is MuJoCo timing**, converted at export time — it is not the
  rate the policy trained at.
- **Checkpoints are not portable across simulators**; check `MODEL_CARD.md`
  before exporting a model for a backend it was not trained on.

## 6. Runtime notes

`onnxruntime` is not PyTorch — it is an independent C++ library (declared deps:
`flatbuffers, numpy, packaging, protobuf`; ≈50 MB installed against torch's
≈990 MB). Session creation parses the protobuf, validates the opset, applies
graph optimisations (constant folding, `Gemm`+`Relu` fusion, layout choice),
partitions nodes across execution providers, and builds a static memory plan;
`run()` then just fills pre-planned buffers and executes kernels.

Measured on the shipped G1 export, single-threaded CPU, batch 1: **0.505 ms per
step** against a 20 ms control budget, i.e. ~40× headroom for the entire
pipeline including observation math. Execution providers (CUDA, TensorRT,
CoreML, OpenVINO, …) swap the backend without touching the file; for on-robot
NVIDIA hardware the usual last step is `trtexec` to a hardware-specific
TensorRT `.plan`, though at 0.5 ms there is rarely a reason to bother.

The case for this over just running the `.ckpt` in PyTorch is mostly *not* raw
speed — eager PyTorch at batch 1 also makes a 20 ms deadline. It is that the
`.ckpt` needs the ProtoMotions package and config dataclasses just to rebuild
the module (~200 lines of the exporter before weights can even be loaded),
pickle is code execution, ~1 GB of dependencies plus a Python interpreter is a
lot to put on robot compute, and the GIL/allocator/GC give a long latency tail.
For a fixed-rate control loop the p99.9 is what you design against.

## See also

- `docs/source/tutorials/workflows/g1_deployment.rst` — end-to-end worked
  example, plus the angular-velocity frame convention.
- `docs/training_outputs.md` — where `resolved_configs_inference.pt` comes from
  and why the exporter needs it.
- `protomotions/envs/action/action_functions.py` — the contract an action
  function must satisfy to be exportable.
