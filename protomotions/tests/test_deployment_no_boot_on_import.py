# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``deployment/tracker_runtime.py`` must stay importable inside a live Kit app.

The whole point of splitting the driver out of
``deployment/test_tracker_isaacsim.py`` is that the core no longer boots or owns
an application. Four things would silently take that back:

* ``argparse`` at module scope -- ``--onnx``/``--motion`` are required, so a bare
  import would raise ``SystemExit``;
* ``SimulationApp(...)`` -- a second app inside a running Kit is invalid;
* ``logging.basicConfig(...)`` -- reconfigures the *root* logger process-wide,
  displacing Kit's own handler and redirecting every other extension's output;
* ``raise SystemExit`` -- swallowed at best inside a Kit callback, and it can
  take the host app down. ``TrackerConfigError`` exists for this.

The checks are static (AST over the source text), so this test needs no
simulator and runs in the ordinary ``pytest protomotions/tests/`` sweep --
``tracker_runtime`` itself cannot be imported here, since it does import
``isaacsim.*`` and ``pxr`` at module scope.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = _REPO_ROOT / "deployment" / "tracker_runtime.py"
SESSION = _REPO_ROOT / "deployment" / "tracker_session.py"
CLI = _REPO_ROOT / "deployment" / "test_tracker_isaacsim.py"


def _tree(path: Path) -> ast.Module:
    assert path.is_file(), f"missing {path}"
    return ast.parse(path.read_text(), filename=str(path))


def _called_names(node: ast.AST) -> set:
    """Every callee spelled out in ``node``, as source text (``a.b.c``)."""
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            names.add(ast.unparse(sub.func))
    return names


def _imported(tree: ast.Module) -> set:
    """Every module and imported name, as dotted source text."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.add(module)
            names |= {f"{module}.{alias.name}" for alias in node.names}
    return names


def _module_level_calls(tree: ast.Module) -> set:
    """Callees reached at import time, i.e. outside any def/class body."""
    names = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        names |= _called_names(stmt)
    return names


@pytest.mark.parametrize(
    "forbidden",
    [
        "SimulationApp",
        "isaacsim.SimulationApp",
        "logging.basicConfig",
        "basicConfig",
        "argparse.ArgumentParser",
        "p.parse_args",
        "parse_args",
        "_parse_args",
    ],
)
def test_runtime_does_not_boot_an_app(forbidden: str) -> None:
    """Nothing that owns the process may run when the core is imported."""
    assert forbidden not in _module_level_calls(_tree(RUNTIME))


def test_runtime_never_imports_argparse_or_simulationapp() -> None:
    """Not even inside a function: those belong to the CLI wrapper."""
    imported = _imported(_tree(RUNTIME))
    assert "argparse" not in imported
    assert "isaacsim.SimulationApp" not in imported


def test_runtime_calls_no_app_owning_function_anywhere() -> None:
    """Deferred inside a helper would be just as fatal as at module scope."""
    calls = _called_names(_tree(RUNTIME))
    for name in ("SimulationApp", "logging.basicConfig", "argparse.ArgumentParser"):
        assert name not in calls, f"{name}() leaked back into tracker_runtime.py"


def test_runtime_raises_tracker_config_error_not_system_exit() -> None:
    """``SystemExit`` from a Kit callback is swallowed at best."""
    offenders = [
        node.lineno
        for node in ast.walk(_tree(RUNTIME))
        if isinstance(node, ast.Raise)
        and node.exc is not None
        and isinstance(node.exc, ast.Call)
        and ast.unparse(node.exc.func) == "SystemExit"
    ]
    assert not offenders, f"raise SystemExit at tracker_runtime.py:{offenders}"


def test_session_does_not_import_the_cli() -> None:
    """The Kit layer must not pull in the module that creates an app."""
    imported = _imported(_tree(SESSION))
    assert not any("test_tracker_isaacsim" in name for name in imported)


def test_cli_keeps_owning_the_app_and_the_root_logger() -> None:
    """The inverse guard: the CLI is still the thing that boots and configures."""
    calls = _module_level_calls(_tree(CLI))
    assert "SimulationApp" in calls
    assert "logging.basicConfig" in calls
    assert "_parse_args" in calls
