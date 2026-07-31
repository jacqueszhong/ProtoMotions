#!/usr/bin/env python3
"""
inspect_motionlib_scene.py

Inspect a .pt file that may contain either a MotionLib or SceneLib object
(from NVlabs ProtoMotions). Reports:
  - Detected type (MotionLib, SceneLib, or unknown)
  - All relevant metadata for whichever type is found
  - Tensor details (shapes, dtypes, stats) for each field

Usage:
    python inspect_motionlib_scene.py path/to/file.pt
    python inspect_motionlib_scene.py path/to/file.pt --allow-unsafe-load
"""

import argparse
import collections
import os
import sys

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def tensor_stats(t):
    """Return a dict of descriptive stats for a floating-point or bool tensor."""
    info = {}
    if t.dtype.is_floating_point:
        info["min"] = f"{t.min().item():.6f}"
        info["max"] = f"{t.max().item():.6f}"
        info["mean"] = f"{t.mean().item():.6f}"
    elif t.dtype == torch.bool:
        n_on = int(t.sum())
        n_total = t.numel()
        info["True"] = str(n_on)
        info["False"] = str(n_total - n_on)
        info["frac_true"] = f"{n_on / n_total:.2%}" if n_total else "N/A"
    return info


def describe_tensor(name, t, indent="  "):
    """Print a single tensor with shape, dtype, and statistics."""
    print(f"{indent}{name}:")
    print(f"{indent}  shape={tuple(t.shape)}  dtype={t.dtype}  device={t.device}")
    stats = tensor_stats(t)
    if stats:
        parts = [f"{k}={v}" for k, v in stats.items()]
        print(f"{indent}  stats: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# MotionLib inspector
# ---------------------------------------------------------------------------

def describe_motionlib(sd, indent="  "):
    """Report MotionLib-specific details."""
    num_motions = int(sd["motion_num_frames"].numel())
    total_frames = int(sd["gts"].shape[0]) if "gts" in sd else "?"
    num_bodies = int(sd["gts"].shape[1]) if "gts" in sd and sd["gts"].ndim >= 2 else "?"
    num_dofs = int(sd["dps"].shape[-1]) if "dps" in sd and sd["dps"].ndim >= 1 else "?"

    print(f"{indent}Number of motion clips: {num_motions}")
    print(f"{indent}Total frames (all clips): {total_frames:,}")
    dt_val = float(sd["motion_dt"]) if isinstance(sd["motion_dt"], torch.Tensor) else sd["motion_dt"]
    fps = int(1 / dt_val) if dt_val else 0
    print(f"{indent}Frame dt: {dt_val:.6f}s ({fps} FPS)")
    if "motion_lengths" in sd and isinstance(sd["motion_lengths"], torch.Tensor):
        total_dur = float(sd["motion_lengths"].sum())
    elif "motion_lengths" in sd:
        total_dur = sum(sd["motion_lengths"]) if isinstance(sd["motion_lengths"], (list, tuple)) else float(sd["motion_lengths"])
    else:
        total_dur = "?"
    print(f"{indent}Total duration: {total_dur:.3f}s")
    print(f"{indent}Inferred humanoid: {num_bodies} bodies, {num_dofs} dofs")

    if "motion_weights" in sd and isinstance(sd["motion_weights"], torch.Tensor):
        weights_list = sd["motion_weights"].cpu().tolist()
        preview = ", ".join(f"{w:.4f}" for w in weights_list[:5])
        tail = "..." if len(weights_list) > 5 else ""
        print(f"{indent}Motion weights present: Yes (sample=[{preview}]{tail})")

    if "motion_files" in sd:
        files = sd["motion_files"]
        if isinstance(files, (list, tuple)):
            print(f"{indent}Source file(s):")
            for fpath in files:
                size = os.path.getsize(fpath) if os.path.exists(fpath) else "?"
                size_str = human_size(size) if isinstance(size, int) else str(size)
                print(f"{indent}  - {fpath} ({size_str})")
        else:
            print(f"{indent}Source file(s): {files}")

    # Clip-level summary for multi-motion
    if num_motions > 1 and "length_starts" in sd:
        print()
        print(f"{indent}Clip breakdown:")
        print(f"{indent}  {'#':>3s}  {'frames':>7s}  {'duration(s)':>12s}  {'start_frame':>10s}")
        starts = sd["length_starts"].cpu().tolist() if isinstance(sd["length_starts"], torch.Tensor) else sd["length_starts"]
        lengths = sd["motion_lengths"].cpu().tolist() if isinstance(sd["motion_lengths"], torch.Tensor) else sd["motion_lengths"]
        nframes = sd["motion_num_frames"].cpu().tolist() if isinstance(sd["motion_num_frames"], torch.Tensor) else sd["motion_num_frames"]
        for i in range(num_motions):
            print(f"{indent}  {i:3d}  {nframes[i]:7d}  {float(lengths[i]):12.4f}  {starts[i]:10d}")

    # Per-body contact stats for single-clip files
    if "contacts" in sd and isinstance(sd["contacts"], torch.Tensor):
        contacts = sd["contacts"]
        print()
        n_bodies = contacts.shape[1] if contacts.ndim >= 2 else "?"
        total_true = int(contacts.sum())
        total_cells = int(contacts.numel())
        print(f"{indent}Contact statistics:")
        print(f"{indent}  Body count: {n_bodies}")
        print(f"{indent}  Overall: {total_true:,}/{total_cells:,} frame-body contacts ({total_true / total_cells:.2%})")

        if num_motions == 1 and "length_starts" in sd:
            start = int(sd["length_starts"][0].item())
            n_frames = int(sd["motion_num_frames"][0].item())
            clip_contacts = contacts[start : start + n_frames]
            per_body_true = clip_contacts.sum(dim=0)
            print(f"{indent}  Per-body contact ratio:")
            for i in range(int(n_bodies)):
                cnt = int(per_body_true[i].item())
                pct = f"{cnt / n_frames:.2%}" if n_frames else "N/A"
                bar = "#" * int(cnt / max(n_frames, 1) * 40)
                print(f"{indent}    body {i:2d}: {pct:>6s}  [{bar}]")


# ---------------------------------------------------------------------------
# SceneLib inspector
# ---------------------------------------------------------------------------

def _format_translation(val):
    """Format translation as single point or 'N frames x 3 tensor'."""
    if isinstance(val, (list, tuple)) and len(val) > 0:
        first = val[0]
        if isinstance(first, (list, tuple)) and len(first) == 3:
            return f"{len(val)} frames → {first} (first), shape=({len(val)}, 3)"
    # single point
    return str(val)


def _get_obj_field(obj, key):
    """Safely get a field from an object dict."""
    v = obj.get(key)
    if isinstance(v, (list, tuple)) and len(v) > 0:
        first = v[0]
        if isinstance(first, (list, tuple)) and len(first) == 3:
            return f"{len(v)} frames → {first} (first), shape=({len(v)}, 4)"
    if v is None:
        return "<None>"
    s = str(v)[:80]
    return s if len(s) < 80 else s + "..."


def describe_scene_object(obj, idx=0):
    """Describe a single SceneObject dict."""
    otype = obj.get("type", "?")
    print(f"  Object {idx} ({otype}):")

    # Geometry params (unique per primitive type)
    if otype in ("BoxSceneObject", "PrimitiveSceneObject"):
        w = obj.get("width")
        d = obj.get("depth")
        h = obj.get("height")
        if w is not None and d is not None and h is not None:
            print(f"    Size: {w:.4f} x {d:.4f} x {h:.4f}")

    elif otype in ("SphereSceneObject",):
        r = obj.get("radius")
        if r is not None:
            print(f"    Radius: {r:.4f}")

    elif otype in ("CylinderSceneObject",):
        r = obj.get("radius")
        h = obj.get("height")
        if r is not None and h is not None:
            print(f"    Radius: {r:.4f}, Height: {h:.4f}")

    # Mesh path
    mesh_path = obj.get("object_path")
    if mesh_path:
        size = os.path.getsize(mesh_path) if os.path.exists(mesh_path) else "?"
        size_str = human_size(size) if isinstance(size, int) else str(size)
        print(f"    Mesh: {mesh_path} ({size_str})")

    # Translation & rotation (motion-aware primitives carry multi-frame data)
    tr = obj.get("translation")
    rot = obj.get("rotation")
    print(f"    Translation: {_format_translation(tr)}")
    if rot is not None:
        print(f"    Rotation: {_get_obj_field(obj, 'rotation')}")

    # Other fields
    fps = obj.get("fps")
    if fps is not None:
        fps_val = float(fps) if isinstance(fps, (int, float)) else fps
        print(f"    FPS: {fps_val:.2f}")

    dims = obj.get("object_dims")
    if dims is not None:
        d_str = str(dims)[:80]
        label = "AABB min/max corners" if len(str(dims).replace(",", "").split()) == 6 else f"{len(str(dims).replace(',', ''))} values"
        print(f"    Object dims ({label}): {d_str}")

    opt = obj.get("options")
    if opt and isinstance(opt, dict):
        # Filter out None/empty vhacd_params
        clean_opt = {}
        for kk, vv in opt.items():
            if vv is None or (isinstance(vv, dict) and all(v == "" or v is None for v in vv.values())):
                continue
            if isinstance(vv, str):
                clean_opt[kk] = f'"{vv}"'
            elif kk == "vhacd_params" and isinstance(vv, dict):
                clean_opt[kk] = str(vv)
            else:
                s = str(vv)[:60]
                clean_opt[kk] = s if len(s) < 60 else s + "..."
        if clean_opt:
            print(f"    Options:")
            for kk, vv in clean_opt.items():
                print(f"      {kk}: {vv}")

    # Point clouds (if present / large files)
    pc = obj.get("object_pointcloud")
    if pc is not None and isinstance(pc, list):
        n_pts = len(pc)
        preview_pt = pc[0] if pc else "?"
        print(f"    Point cloud: {n_pts} points (first pt preview: {preview_pt})")


def describe_scelib(sd, indent="  "):
    """Report SceneLib-specific details."""
    num_scenes = sd.get("num_original_scenes", len(sd.get("original_scenes", [])))
    n_objects = sd.get("num_objects_per_scene", "?")

    print(f"{indent}Number of original scenes: {num_scenes}")
    print(f"{indent}Objects per scene: {n_objects}")
    print()

    scenes = sd.get("original_scenes", [])
    for i, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            print(f"{indent}Scene {i}: (not a dict)")
            continue

        offset = scene.get("offset", "(none)")
        motion_id = scene.get("humanoid_motion_id", -1)
        objs = scene.get("objects", [])

        print(f"{indent}Scene {i}:")
        if isinstance(offset, (list, tuple)):
            print(f"{indent}  Offset: ({offset[0]:.3f}, {offset[1]:.3f})" if len(offset) >= 2 else f"{indent}  Offset: {offset}")
        else:
            print(f"{indent}  Offset: {offset}")
        print(f"{indent}  Humanoid motion ID: {motion_id}")
        print(f"{indent}  Objects ({len(objs)}):")

        # Object type summary
        type_counts = collections.Counter()
        for obj in objs:
            if isinstance(obj, dict):
                type_counts[obj.get("type", "?")] += 1
        for tname, cnt in type_counts.items():
            print(f"{indent}    {tname}: {cnt}")

        # Detailed description (up to first 3 objects)
        for j, obj in enumerate(objs[:3]):
            if isinstance(obj, dict):
                describe_scene_object(obj, idx=j)

        if len(objs) > 3:
            print(f"{indent}    ... and {len(objs) - 3} more object(s)")


# ---------------------------------------------------------------------------
# Detection & dispatch
# ---------------------------------------------------------------------------

def is_motionlib_dict(obj):
    """Heuristic: does this dict look like a MotionLib save?"""
    core_fields = {"gts", "grs", "dvs", "dps", "contacts",
                   "length_starts", "motion_num_frames"}
    return isinstance(obj, (dict, collections.OrderedDict)) and core_fields.issubset(set(obj.keys()))


def is_scelib_dict(obj):
    """Heuristic: does this dict look like a SceneLib save?"""
    return isinstance(obj, (dict, collections.OrderedDict)) and all(
        k in obj for k in ("original_scenes", "num_original_scenes")
    )


def describe_motionlib_core(sd, indent="  "):
    """Check for missing core MotionLib fields and print summary."""
    core = {"gts", "grs", "gvs", "gavs", "dps", "dvs", "contacts",
            "length_starts", "motion_num_frames", "motion_lengths", "motion_dt"}
    present = core & set(sd.keys())
    missing = core - present

    print(f"{indent}Core fields:")
    for k in sorted(core):
        if k not in sd:
            print(f"{indent}  {k}: <missing>")
        else:
            v = sd[k]
            if isinstance(v, torch.Tensor):
                print(f"{indent}  {k}: ✓ shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"{indent}  {k}: present ({type(v).__name__})")

    # Tensor details for all fields
    print()
    print(f"{indent}Tensor details:")
    for k in sorted(sd.keys()):
        v = sd[k]
        if isinstance(v, torch.Tensor):
            describe_tensor(k, v, indent=indent)


def describe_scelib_core(sd, indent="  "):
    """Check SceneLib top-level keys."""
    core = {"original_scenes", "num_original_scenes", "num_objects_per_scene"}

    print(f"{indent}Top-level fields:")
    for k in sorted(core):
        v = sd.get(k)
        if v is None:
            print(f"{indent}  {k}: <missing>")
        elif isinstance(v, int):
            print(f"{indent}  {k}: ✓ {v}")
        elif isinstance(v, list):
            print(f"{indent}  {k}: ✓ list len={len(v)}")
        else:
            s = str(v)[:80]
            print(f"{indent}  {k}: {type(v).__name__} = {s}")


# ---------------------------------------------------------------------------
# Main inspect pipeline
# ---------------------------------------------------------------------------

def inspect(path, allow_unsafe_load):
    print(f"File: {path}")
    print(f"Size: {human_size(os.path.getsize(path))}")
    print("-" * 60)

    # --- Step 1: try TorchScript first (safe) ---
    try:
        scripted = torch.jit.load(path, map_location="cpu")
        print("Detected type: TorchScript model")
        print()
        print(scripted)
        return
    except Exception:
        pass

    # --- Step 2: load the dict / pickled object ---
    loaded_safely = False
    obj = None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        loaded_safely = True
    except Exception as e:
        if not allow_unsafe_load:
            print("Could not load with weights_only=True.")
            print(f"Reason: {e}")
            print()
            print("Re-run with --allow-unsafe-load ONLY if you trust this file's source.")
            return
        else:
            print("weights_only=True failed, falling back to weights_only=False")
            obj = torch.load(path, map_location="cpu", weights_only=False)

    loaded_mode = "True (safe)" if loaded_safely else "False (unsafe)"
    print(f"Loaded with weights_only={loaded_mode}")
    print()

    indent = "  "

    # --- Step 3: classify & report ---
    if is_motionlib_dict(obj):
        print("Detected type: MotionLib motion collection (.pt save)")
        print()
        describe_motionlib_core(obj)
        print()
        describe_motionlib(obj)
        # Optional fields
        _core_motionlib = {"gts", "grs", "gvs", "gavs", "dps", "dvs", "contacts",
                           "length_starts", "motion_num_frames", "motion_lengths",
                           "motion_dt", "motion_weights", "motion_files"}
        optional_keys = [k for k in obj.keys() if k not in _core_motionlib]
        if optional_keys:
            print()
            print(f"{indent}Optional fields:")
            for k in optional_keys:
                v = obj[k]
                if isinstance(v, torch.Tensor):
                    describe_tensor(k, v, indent="    ")
                elif isinstance(v, (list, tuple)):
                    print(f"    {k}: {type(v).__name__} len={len(v)}")
                else:
                    s = str(v)[:120]
                    print(f"    {k}: {type(v).__name__} = {s}")

    elif is_scelib_dict(obj):
        print("Detected type: SceneLib scene collection (.pt save)")
        print()
        describe_scelib_core(obj)
        print()
        describe_scelib(obj)

    elif isinstance(obj, dict):
        print("Detected type: generic dict (not a MotionLib or SceneLib save)")
        print()
        print(f"Top-level keys ({len(obj)}):")
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
            elif isinstance(v, (list, tuple)):
                s = str(v)[:100]
                label = f"len={len(v)}"
                print(f"  {k}: {type(v).__name__} {label}")
            else:
                s = str(v)[:120]
                print(f"  {k}: {type(v).__name__} = {s}")

    elif isinstance(obj, torch.Tensor):
        print("Detected type: single tensor")
        print(f"  shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}")
        stats = tensor_stats(obj)
        if stats:
            parts = [f"{k}={v}" for k, v in stats.items()]
            print(f"  stats: {', '.join(parts)}")

    else:
        print("Detected type: full pickled object")
        print(f"  Python type: {type(obj).__name__}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a .pt file (MotionLib or SceneLib).",
    )
    parser.add_argument("path", help="Path to the .pt file")
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        help="Allow weights_only=False fallback (executes arbitrary code -- only use for trusted files)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"Error: file not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    inspect(args.path, args.allow_unsafe_load)


if __name__ == "__main__":
    main()
