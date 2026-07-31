# custom_box.npz Analysis Log

Generated: 2026-07-30

## File Structure

| Key | Shape | Dtype | Size | Description |
|-----|-------|-------|------|-------------|
| `posed_joints` | (155, 77, 3) | float32 | ~139 KB | Per-frame joint pose vectors |
| `global_rot_mats` | (155, 77, 3, 3) | float32 | ~481 KB | Global-space rotation matrices |
| `local_rot_mats` | (155, 77, 3, 3) | float32 | ~481 KB | Local (parent-relative) rotation matrices |
| `root_positions` | (155, 3) | float32 | ~2 KB | Base origin (x, y, z) per frame |
| `foot_contacts` | (155, 6) | bool | <1 KB | Foot contact flags per frame |

**Total: ~982 KB, 155 frames at ~60fps → ~2.6 seconds of motion.**

---

## Key Data Formats

### `posed_joints` — Rotation Vectors (Axis-Angle)
Each of the **77 dimensions** stores a 3D rotation vector: magnitude = rotation angle (radians), direction = rotation axis. All 77 dims vary across frames. The per-dimension rotation magnitudes break into **5 groups**:

| Group | Dims | Mean Angle | Std | Interpretation |
|-------|------|-----------|-----|----------------|
| A | 0–15 | ~1.28 rad | 0.29 | Large-DoF joints (hips/torso) |
| B | 16–38 | ~0.73 rad | **0.046** | **Nearly constant** — likely gripper/fingers held in fixed pose |
| C | 39–66 | ~0.92 rad | 0.14 | Arm/leg swing joints |
| D | 67–71 | ~0.51 rad | 0.20 | Fine motor joints (wrists/fingers) |
| E | 72–76 | ~0.50 rad | 0.20 | Fine motor joints |

The **stable Group B** (std < 0.05 across all 155 frames) is the signature of a closed/holding gripper — consistent with the "box" in the filename (SOMA holding a box fixed).

### `global_rot_mats` / `local_rot_mats`
- Both are valid rotation matrices (orthogonality error ~5×10⁻⁶, det ≈ 1.0)
- Joint[0] has **identical** local and global rotations (root body joint at base of kinematic chain)
- Subsequent joints diverge as expected from parent-chain propagation

### `foot_contacts` — 6 contact sensors
All 6 are active (~60% on-contact each), suggesting a bipedal stance with heel-toe contacts on both feet. 6–9 state transitions per sensor across the clip, consistent with a walking/gait cycle.

---

## Motion Profile

| Property | Value |
|----------|-------|
| Net X (forward) displacement | **1.89 m** |
| Net Z (lateral) displacement | **2.40 m** |
| Root height range | **32 cm** (0.68 → 1.00 m) — active vertical motion |
| Total trajectory length | **3.70 m** |
| Avg speed | ~1.4 m/s |
| Max speed | ~2.7 m/s |

**Interpretation**: A lateral/circular walk with crouching/jumping motion (32 cm height change is large for normal walking). The combination of significant vertical oscillation + lateral travel suggests a dynamic maneuver — possibly a step-over, obstacle crossing, or jump while holding the box.

---

## Mapping to SOMA 23-DoF

The 77 dimensions map to the **full kinematic tree** (not just actuated DoFs), including:
- **Left leg** (~6 body joints) → Group A/C
- **Right leg** (~6 body joints) → Group A/C  
- **Torso/chest/neck/head** (~4–5 joints) → Group A
- **Left arm** (~4 joints) → Group C
- **Right arm** (~4 joints) → Group C
- **Gripper/fingers** (~10–15 joints) → Group B (fixed pose, holding box)
- **Extra wrist/foot articulation** (~5–8 joints) → Group D/E

Group B being ~23 dimensions with near-zero variance is the telltale sign: these are the gripper fingers held in a fixed grasp configuration throughout the entire motion clip.
