# Changelog

## [0.1.0] - 2026-09-03

### Added

- Initial release. Library-API-only extension exposing `TrackerSession` and
  `TrackerRuntimeConfig`, so a ProtoMotions ONNX tracker policy can be driven
  from inside any running Isaac Sim 6 app (GUI launcher, Script Editor, another
  extension, a custom `.kit` app).
- The driver itself moved out of `deployment/test_tracker_isaacsim.py` into the
  importable `deployment/tracker_runtime.py`; the CLI is now a thin wrapper over
  it and is behaviourally unchanged.
