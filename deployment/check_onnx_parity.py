# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check an exported ONNX policy against IsaacLab's own outputs, on real observations.

Answers one question: **is the deployed graph computing the action training
computes?**  If it is, a deployment failure is downstream physics; if it is not,
no amount of physics tuning will help.

Why this exists
---------------
The exporter's own self-check
(``deployment/export_bm_tracker_onnx_isaacsim.py``) compares onnxruntime against
the *same traced module* on one ``torch.randn`` batch, and never raises.  That
catches a broken ONNX conversion and nothing else -- not a wrong
context-to-input mapping, not stale baked normalizer constants, not a
mismatched ``ActionExportModule``.  All three survive it because both sides of
its comparison share the defect.

This script replaces the random batch with the **real observation trajectory**
recorded by ``deployment/trace_tracker_isaaclab.py``, and the traced module
with **IsaacLab's recorded outputs**, so the two sides are independent.

What it compares, per control step
-----------------------------------
=================================  ============================================
``e_a`` = |onnx.actions − mean_action|∞     the policy network
``e_p`` = |onnx.joint_pos_targets            the policy *and* the baked
          − processed_action|∞               ``ActionExportModule`` constants
normalizer                                   the baked mean/std against the
                                             recorded pre/post-norm vectors
=================================  ============================================

Interpreting the result
-----------------------
========================================  =================================
signature                                 implicates
========================================  =================================
all ≈ 0                                   export is correct; look downstream
normalizer mismatch                       baked constants ≠ this checkpoint
``e_a`` small but ``e_p`` large            ``ActionExportModule``'s baked
                                          ``default_dof_pos`` /
                                          ``effort_limit/stiffness``
both large, normalizer fine                the policy weights or the
                                          context→input mapping
========================================  =================================

``--diagnose`` (automatic on failure) re-runs with every intermediate tensor
promoted to a graph output and localizes a mismatch to a contiguous slice of
the observation vector, which maps back to one actor ``in_key``.

Usage
-----
::

    python deployment/check_onnx_parity.py \\
        --npz  /tmp/stage1/custom/context_isaaclab.npz \\
        --onnx results/g1_walk_box/compiled_models/unified_pipeline.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

#: Pass thresholds.  Float32 CUDA (IsaacLab) versus CPU onnxruntime through a
#: 6x1024 MLP accumulates a few 1e-4; anything above these is a real defect,
#: not arithmetic.
ACTION_TOL = 1e-3  # rad, on the raw policy output
PD_TARGET_TOL = 2e-3  # rad, on the commanded joint position
NORM_TOL = 1e-3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare an exported ONNX policy against IsaacLab's outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--npz",
        required=True,
        help="context_isaaclab.npz from trace_tracker_isaaclab.py",
    )
    p.add_argument("--onnx", required=True, help="Path to unified_pipeline.onnx")
    p.add_argument(
        "--report-out", default=None, help="Write the per-step errors as JSON here"
    )
    p.add_argument(
        "--diagnose",
        action="store_true",
        default=False,
        help="Always run the intermediate-tensor localization (automatic on failure)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------


def build_onnx_inputs(session, npz, onnx_name_to_key: dict) -> dict:
    """Assemble a full-trajectory ONNX input dict from the recorded context arrays.

    Every input carries a dynamic batch axis (the exporter sets
    ``dynamic_axes`` on dim 0 for all of them), so the entire trace runs as one
    batch instead of 251 single-step calls.

    The history input needs slicing: ``historical.processed_actions`` is
    recorded as the full ``[steps, 32, num_dofs]`` state-history buffer, while
    the graph declares ``[batch, history_steps, num_dofs]`` with
    ``history_steps == 1``.  ``compute_historical_actions_from_state`` selects
    with ``narrow(1, 0, N)`` -- the *first* N slots, most-recent-first -- so the
    slice has to be taken from the front, not the back.

    Args:
        session: An ``ort.InferenceSession`` for the graph.
        npz: The loaded ``context_isaaclab.npz``.
        onnx_name_to_key: ``_runtime.onnx_name_to_in_key`` from the YAML sidecar.

    Returns:
        Dict of ONNX input name -> float32 array with the trace's batch length.
    """
    declared = {i.name: i.shape for i in session.get_inputs()}
    inputs = {}
    for onnx_name, sem_key in onnx_name_to_key.items():
        arr = npz[f"ctx__{sem_key.replace('.', '_')}"]
        shape = declared[onnx_name]
        # Trailing (non-batch) dims are static in the export; trust them over
        # whatever the recorder happened to store.
        want = [d for d in shape[1:] if isinstance(d, int)]
        if len(want) == arr.ndim - 1 and list(arr.shape[1:]) != want:
            if len(want) == 2 and arr.shape[2] == want[1] and arr.shape[1] > want[0]:
                arr = arr[:, : want[0], :]
            else:
                raise ValueError(
                    f"Cannot reconcile recorded {sem_key} {arr.shape} with the "
                    f"graph's declared {shape} for input '{onnx_name}'"
                )
        inputs[onnx_name] = np.ascontiguousarray(arr, dtype=np.float32)
    missing = set(declared) - set(inputs)
    if missing:
        raise ValueError(f"No recorded context for ONNX inputs: {sorted(missing)}")
    return inputs


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


def check_normalizer(onnx_path: str, npz) -> dict:
    """Verify the baked observation normalizer against the recorded pre/post vectors.

    The exporter folds the running observation normalizer into the graph as a
    ``Sub`` (mean) → ``Div`` (std) → ``Clip`` chain, the last of which applies
    ``NormObsBaseConfig.norm_clamp_value`` (5.0 by default).  A
    checkpoint/export mismatch in those constants shows up here and nowhere
    else: the graph would still run and its outputs would still look plausible.

    The whole chain has to be reproduced, clamp included.  Skipping the clamp
    makes every dimension whose baked std is small report a spurious error --
    on a single-clip policy some std collapses to ~0.004, so the pre-clamp value
    routinely leaves [-5, 5] and the comparison is meaningless without it.

    Args:
        onnx_path: Path to the ONNX file.
        npz: The loaded recording; needs ``obs_concat`` and ``obs_norm``.

    Returns:
        Dict with the max error and the recovered constants, or a ``skipped``
        reason when the recording or the graph does not support the check.
    """
    if "obs_norm" not in npz.files or "obs_concat" not in npz.files:
        return {"skipped": "recording has no obs_concat/obs_norm"}

    obs_concat = npz["obs_concat"]
    obs_norm = npz["obs_norm"]
    width = obs_concat.shape[1]

    model = onnx.load(onnx_path)
    consts = {
        init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer
    }
    # Constant *nodes* carry the clamp bounds (they are not initializers).
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    consts[node.output[0]] = onnx.numpy_helper.to_array(attr.t)

    # Walk the chain by data dependency, not by name -- ONNX numbers these
    # ("onnx::Sub_787") and the numbering shifts between exports.
    sub_operand = div_operand = None
    sub_out = None
    for node in model.graph.node:
        if node.op_type != "Sub":
            continue
        operands = [consts.get(i) for i in node.input]
        match = [a for a in operands if a is not None and a.shape == (width,)]
        if match:
            sub_operand, sub_out = match[0], node.output[0]
            break
    if sub_operand is None:
        return {"skipped": "could not locate the baked normalizer mean"}

    div_out = None
    for node in model.graph.node:
        if node.op_type == "Div" and sub_out in node.input:
            match = [
                consts[i]
                for i in node.input
                if i in consts and consts[i].shape == (width,)
            ]
            if match:
                div_operand, div_out = match[0], node.output[0]
            break
    if div_operand is None:
        return {"skipped": "could not locate the baked normalizer std"}

    clamp = None
    for node in model.graph.node:
        if node.op_type == "Clip" and div_out in node.input:
            bounds = [consts[i] for i in node.input[1:] if i in consts]
            if len(bounds) == 2:
                clamp = (float(bounds[0]), float(bounds[1]))
            break

    expected = (obs_concat - sub_operand) / div_operand
    if clamp is not None:
        expected = np.clip(expected, clamp[0], clamp[1])
    else:
        log.warning(
            "No Clip node follows the normalizer Div; comparing unclamped. Any "
            "dimension with a small baked std will report a spurious error."
        )

    err = np.abs(expected - obs_norm)
    return {
        "max_err": float(err.max()),
        "mean_err": float(err.mean()),
        "worst_index": int(np.unravel_index(err.argmax(), err.shape)[1]),
        "min_std": float(div_operand.min()),
        "max_std": float(div_operand.max()),
        "clamp": clamp,
        "clamp_fraction": (
            float(np.mean(np.abs((obs_concat - sub_operand) / div_operand) > clamp[1]))
            if clamp is not None
            else None
        ),
        "passed": bool(err.max() < NORM_TOL),
    }


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def localize_obs_mismatch(onnx_path: str, inputs: dict, npz, actor_in_keys) -> None:
    """Promote every intermediate tensor to an output and localize an obs mismatch.

    Only worth the extra graph copy when something already failed: it re-runs
    the whole graph with several hundred extra outputs to find the internal
    tensor that corresponds to the assembled observation vector, then reports
    which contiguous slice -- i.e. which actor ``in_key`` -- differs.
    """
    if "obs_concat" not in npz.files:
        log.warning("Recording has no obs_concat; cannot localize.")
        return

    recorded = npz["obs_concat"]
    width = recorded.shape[1]

    model = onnx.load(onnx_path)
    existing = {o.name for o in model.graph.output}
    produced = [name for node in model.graph.node for name in node.output]
    for name in produced:
        if name not in existing:
            model.graph.output.extend([onnx.helper.ValueInfoProto(name=name)])

    try:
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        log.warning(f"Could not build the instrumented graph: {e}")
        return

    names = [o.name for o in session.get_outputs()]
    outs = session.run(names, inputs)

    best = None
    for name, value in zip(names, outs):
        value = np.asarray(value)
        if value.ndim != 2 or value.shape != recorded.shape:
            continue
        err = float(np.abs(value - recorded).max())
        if best is None or err < best[1]:
            best = (name, err, value)

    if best is None:
        log.warning(f"No intermediate tensor of shape {recorded.shape} found.")
        return

    name, err, value = best
    log.info(
        f"Closest {width}-wide intermediate is '{name}': "
        f"max|onnx - recorded obs_concat| = {err:.3e}"
    )

    per_dim = np.abs(value - recorded).max(axis=0)
    offset = 0
    log.info("  Per-observation-component error:")
    for key in actor_in_keys:
        arr = npz[f"obs__{key}"]
        n = arr.shape[1]
        sl = per_dim[offset : offset + n]
        flag = "  <-- MISMATCH" if sl.max() > 1e-3 else ""
        log.info(
            f"    {key:46s} dims [{offset:3d}:{offset + n:3d}] max={sl.max():.3e}{flag}"
        )
        offset += n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    yaml_path = args.onnx.replace(".onnx", ".yaml")
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    onnx_name_to_key = meta["_runtime"]["onnx_name_to_in_key"]

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]

    inputs = build_onnx_inputs(session, npz, onnx_name_to_key)
    num_steps = next(iter(inputs.values())).shape[0]
    log.info(f"Replaying {num_steps} recorded control steps through {args.onnx}")

    outs = dict(zip(out_names, session.run(out_names, inputs)))
    onnx_actions = outs["actions"]
    onnx_targets = outs["joint_pos_targets"]

    mean_action = npz["mean_action"]
    processed_action = npz["processed_action"]

    e_a = np.abs(onnx_actions - mean_action).max(axis=1)
    e_p = np.abs(onnx_targets - processed_action).max(axis=1)

    joint_names = [str(n) for n in npz["meta__joint_names"]]
    worst_a, worst_p = int(e_a.argmax()), int(e_p.argmax())
    worst_joint = joint_names[
        int(np.abs(onnx_targets - processed_action)[worst_p].argmax())
    ]

    log.info(
        f"\n=== ONNX vs IsaacLab, {num_steps} control steps ===\n"
        f"  e_a  |actions - mean_action|inf         "
        f"max={e_a.max():.3e}  mean={e_a.mean():.3e}  (worst step {worst_a})\n"
        f"  e_p  |joint_pos_targets - processed|inf "
        f"max={e_p.max():.3e}  mean={e_p.mean():.3e}  (worst step {worst_p}, "
        f"joint {worst_joint})"
    )

    norm = check_normalizer(args.onnx, npz)
    if "skipped" in norm:
        log.warning(f"  normalizer check skipped: {norm['skipped']}")
    else:
        log.info(
            f"  normalizer  max_err={norm['max_err']:.3e} "
            f"(worst dim {norm['worst_index']})  baked std range "
            f"[{norm['min_std']:.4f}, {norm['max_std']:.4f}]  "
            f"clamp={norm['clamp']}  "
            f"clamped {100.0 * (norm['clamp_fraction'] or 0.0):.2f}% of elements"
        )
        if norm["min_std"] < 0.01:
            log.warning(
                f"  Baked normalizer std reaches {norm['min_std']:.4f}: that "
                f"observation dimension is amplified {1.0 / norm['min_std']:.0f}x. "
                "Near-zero variance in training makes the policy hypersensitive "
                "to any deployment mismatch in that dimension."
            )

    action_ok = e_a.max() < ACTION_TOL
    target_ok = e_p.max() < PD_TARGET_TOL
    norm_ok = norm.get("passed", True)
    passed = action_ok and target_ok and norm_ok

    if passed:
        log.info("\nPASS -- the exported graph reproduces training's action.")
    else:
        log.error(
            f"\nFAIL -- actions_ok={action_ok} targets_ok={target_ok} "
            f"normalizer_ok={norm_ok}"
        )
        if action_ok and not target_ok:
            log.error(
                "  e_a small but e_p large: suspect ActionExportModule's baked "
                "constants (default_dof_pos, effort_limit/stiffness)."
            )

    if args.diagnose or not passed:
        localize_obs_mismatch(
            args.onnx, inputs, npz, [str(k) for k in npz["meta__actor_in_keys"]]
        )

    if args.report_out:
        with open(args.report_out, "w") as f:
            json.dump(
                {
                    "num_steps": int(num_steps),
                    "e_a": e_a.tolist(),
                    "e_p": e_p.tolist(),
                    "normalizer": norm,
                    "passed": bool(passed),
                },
                f,
            )
        log.info(f"Per-step report -> {args.report_out}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
