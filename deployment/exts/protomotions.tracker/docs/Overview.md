# ProtoMotions Tracker Driver

Run a ProtoMotions ONNX tracker policy inside an already-running Isaac Sim 6
app. This is a **library-API-only** extension: it adds no menu and no window,
just two importable names.

## Enabling it

The extension lives in-tree, so point Kit at its folder:

```bash
~/isaacsim/isaac-sim.sh \
  --ext-folder ~/dev_nv/ProtoMotions/deployment/exts \
  --enable protomotions.tracker
```

Or copy/symlink `deployment/exts/protomotions.tracker` into
`~/isaacsim/extsUser/`, which is already on the search path, and enable it from
*Window -> Extensions*.

## Using it

From **Window -> Script Editor**:

```python
from protomotions_tracker import TrackerSession, TrackerRuntimeConfig

s = TrackerSession(TrackerRuntimeConfig(
    onnx="results/soma_sit_1/compiled_models/unified_pipeline.onnx",
    motion="configs/soma_sit_0/soma_sit_motion_seatcontacts.pt",
    scenes_file="configs/soma_sit_0/soma_sit_chair.pt",
    resolved_configs="results/soma_sit_1/resolved_configs_inference.pt",
    robot="soma23",
    device="cuda:0",
))
s.load()   # builds the stage, adopts World.instance(), starts the sim
s.play()   # hands physics to the timeline
```

Relative paths resolve against the app's working directory. `s.metrics` reports
live counters and the worst tracking error; `s.summary()` logs the end-of-run
stats; `s.close()` detaches everything so you can load another run in the same
session.

## Throwing things at the robot

`ProtoMotions`' simulator binds **J** to throwing a cube at the humanoid, for
eyeballing how well a policy takes a hit. The same thing works here on an
exported ONNX policy -- configure a pool, then call `throw()`:

```python
from protomotions_tracker import TrackerSession, TrackerRuntimeConfig
from deployment.tracker_runtime import ProjectileConfig

s = TrackerSession(TrackerRuntimeConfig(
    onnx=..., motion=..., robot="g1", device="cuda:0",
    projectiles=ProjectileConfig(num_projectiles=4),
))
s.load(); s.play()
s.throw()   # cube launches on the next app update
```

Cubes appear at a random bearing 4-5 m out, aimed at the robot's root with a
little noise and led by its horizontal velocity, and are parked below the floor
`hide_delay` seconds later. The pool is reused round-robin, so `num_projectiles`
is how many can be airborne at once; cube size grows with pool index, giving a
light-to-heavy set off one call. `ProjectileConfig` mirrors ProtoMotions'
`ProjectileConfig` field for field -- read its docstring for the knobs, and for
why `speed_range` is the one default that deliberately differs.

**No key is bound for you.** This extension installs no input handling, so bind
`session.throw()` to whatever your app already uses. The CLI, which owns its
app, does bind **J**:

```bash
~/isaacsim/python.sh deployment/test_tracker_isaacsim.py \
    --robot g1 --onnx <export>/unified_pipeline.onnx \
    --motion data/motion_for_trackers/g1_bones_seed_mini.pt \
    --projectiles 4
```

`TrackerRuntimeConfig` is a plain dataclass -- read its field comments in
`deployment/tracker_runtime.py` for what each knob does. Flags that are the
*host app's* decision (headless, renderer, resolution, anti-aliasing) have no
field here, and the diagnostic modes (`--action-tape`, `--drive-probe`,
`--dump-physx-properties`, tracing, profiling) stay CLI-only because each owns
its own stepping loop, which is incompatible with Kit's timeline owning it. For
those, use the CLI:

```bash
~/isaacsim/python.sh deployment/test_tracker_isaacsim.py --help
```

## Layout, and the naming trap

| Path | Role |
|---|---|
| `deployment/tracker_runtime.py` | The whole driver. No argparse, no `SimulationApp`, no `logging.basicConfig`, no stepping loop -- importable inside a live app. |
| `deployment/tracker_session.py` | `TrackerSession`: timeline + app-update subscription + teardown. Kit-only. |
| `deployment/test_tracker_isaacsim.py` | The standalone CLI. Owns the app and the `while` loop. |
| `deployment/exts/protomotions.tracker/` | This extension: a shim that puts the repo root on `sys.path`. |

The extension **id** is `protomotions.tracker` (dotted, as Kit expects for the
directory name) but its python module is the flat **`protomotions_tracker`**.
The mismatch is deliberate: Kit puts an extension's root directory on
`sys.path`, so a python package named `protomotions/` here would shadow the real
ProtoMotions package and break every `import protomotions.*` in the app.

## Requirements

- Isaac Sim 6.0 (tested against 6.0.1). Depends on `isaacsim.core.api`,
  `isaacsim.core.prims`, `isaacsim.core.utils` -- shipped but in
  `extsDeprecated/`, so expect migration warnings.
- torch comes from `omni.isaac.ml_archive`, pyyaml from `omni.pip.compute`,
  onnxruntime from Kit's own site-packages. All declared as dependencies; no
  pip install needed.
- A ProtoMotions checkout. Derived from this extension's own location in-tree;
  override with the carb setting `exts."protomotions.tracker".repoRoot` if you
  moved the extension out of the repo.

## Known limits

- **`close()` leaves the stage alone.** Removing prims is the host's call, so a
  second `load()` in the same session re-references the same prim paths
  (`/World/Robot`, the scene objects, the ground plane) rather than spawning
  fresh ones. It runs, but the second run starts from whatever the first left on
  the stage -- expect slightly different numbers. For a clean repeat, make a new
  stage, or pass a different `prim_path`.
- **The callback stepping path only.** The CLI defaults to
  `--control-in-loop` (action in the outer loop, then `decimation` substeps, as
  IsaacLab does) because it scores better: 0.0454 rad vs 0.0537 for the
  physics-callback path on the G1 reference clip. That path owns the stepping
  loop, and inside Kit the timeline owns it, so `TrackerSession` rejects
  `control_in_loop=True` rather than pretending. A session therefore tracks
  slightly worse than the equivalent CLI run; use the CLI when that matters.
- One physics rate. Tracking is only valid at the rate the policy trained at, so
  if the host app already has a `World` with a different `physics_dt`,
  `load()` raises `TrackerConfigError` rather than silently running at the
  host's rate.
- `build_robot_config()` briefly `os.chdir()`s to the repo root while importing a
  robot config. It is `try`/`finally`-guarded, but it is a process-global
  mutation inside a long-lived app.
