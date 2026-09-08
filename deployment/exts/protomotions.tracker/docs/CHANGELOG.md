# Changelog

## [0.2.0] - 2026-09-08

### Added

- Cube projectiles: `TrackerRuntimeConfig.projectiles` spawns a pool of rigid
  cubes, and `TrackerSession.throw()` launches one at the robot. The deployment
  counterpart of the simulator's J-key perturbation
  (`protomotions/simulator/base_simulator/simulator.py:_throw_projectile`), so an
  exported policy can be prodded the same way a training one is. Off unless a
  pool is configured. The CLI exposes it as `--projectiles N` and binds **J**;
  no key is bound here, because a library extension must not grab the host's
  input.

### Changed

- `TrackerRuntime.drain_deferred()` replaces `drain_pending_reset()` as the
  once-per-frame housekeeping call (it does the reset drain plus the projectile
  throw and expiry timer). `drain_pending_reset()` is unchanged and still
  public; `TrackerSession` now calls `drain_deferred()`.

## [0.1.0] - 2026-09-03

### Added

- Initial release. Library-API-only extension exposing `TrackerSession` and
  `TrackerRuntimeConfig`, so a ProtoMotions ONNX tracker policy can be driven
  from inside any running Isaac Sim 6 app (GUI launcher, Script Editor, another
  extension, a custom `.kit` app).
- The driver itself moved out of `deployment/test_tracker_isaacsim.py` into the
  importable `deployment/tracker_runtime.py`; the CLI is now a thin wrapper over
  it and is behaviourally unchanged.
