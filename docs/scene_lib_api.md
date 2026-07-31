# SceneLib API

Reference for `protomotions/components/scene_lib.py` — object spawning, placement, and
motion for scenes attached to environments. Read this before touching scene code; line
numbers drift, names don't.

## Object model

```
Scene(objects=[SceneObject, ...], offset=(x, y), humanoid_motion_id=-1)
```

`humanoid_motion_id = -1` means "no specific motion pairing". All scenes in a library
**must have the same number of objects** — this is validated on save and assumed
everywhere (object order within a scene is the index, so it matters).

`SceneObject` subclasses, and the fields each adds:

| Class | Extra fields |
|---|---|
| `BoxSceneObject` | `width`, `depth`, `height` |
| `SphereSceneObject` | `radius` |
| `CylinderSceneObject` | `radius`, `height` |
| `MeshSceneObject` | `object_path`, `scale` (3-tuple) |

Every object carries `translation`, `rotation` (quaternion, **xyzw**), `fps`,
`object_dims`, and an `ObjectOptions`. Translation/rotation accept tuple, list, ndarray,
or tensor and are converted to tensors in `__post_init__`. A **static** object has a
single frame (3-vector / 4-quaternion); a **moving** object has `(N,3)` / `(N,4)` with
matching frame counts, where frame 0 is the initial state. `fps` is required when the
object has motion and defaults to 1.0 otherwise.

Static vs dynamic is `options.fix_base_link`: True = static, single frame, no motion.

`ObjectOptions` fields: `fix_base_link`, `vhacd_enabled`, `vhacd_params`, `density`
(kg/m³) **or** `mass` (kg) but never both (`__post_init__` raises), `angular_damping`,
`linear_damping`, `max_angular_velocity`, `static_friction`, `dynamic_friction`,
`restitution`, `texture_path`, `color`. Unset defaults to density = `DEFAULT_OBJECT_DENSITY`.

## The `.pt` file format

`SceneLib.save_scenes_to_file(scenes, path, asset_root=None)` is a **static method** — no
instance or config needed. It `torch.save`s exactly:

```python
{
    "original_scenes": [...],     # list of serialized scene dicts
    "num_original_scenes": int,
    "num_objects_per_scene": int, # identical across all scenes (enforced)
}
```

Each scene dict is `{"offset", "humanoid_motion_id", "objects": [...]}`. Each object dict
is `{"type", "translation", "rotation", "fps", "object_dims", "options"}` plus the
type-specific fields from the table above. `type` is the class-name string used to
reconstruct the right class on load. `options` contains **only non-None fields**.

Things that bite:

- **Despite the `.pt` extension, nothing is a tensor.** Everything goes through
  `.cpu().numpy().tolist()` — the file is pure Python primitives.
- **Mesh paths are stored relative to `asset_root`**, defaulting to the scene file's
  parent directory. This is what makes the file portable. Absolute paths on a different
  Windows drive silently stay absolute.
- **Config is NOT saved.** `replicate_method`, `subset_method`,
  `pointcloud_samples_per_object` etc. live only in `SceneLibConfig` — supply a fresh one
  when loading.
- `object_dims` is precomputed and stored, so meshes don't re-read the `.obj` on load.
  Mesh dimension calc swaps `.urdf`/`.usda`/`.usd` → `.obj`, so the `.obj` must sit
  alongside.
- Load re-runs each object's `__post_init__` and re-resolves mesh paths against
  `asset_root`.

Creation scripts: `data/scripts/create_box_scene.py`, `data/scripts/create_mesh_scene.py`
(the latter passes an explicit `asset_root` because the config default is the
*grandparent* of the scenes file while `save_scenes_to_file` defaults to its *parent* —
they must be made to agree).

## Construction

```python
SceneLib(config: SceneLibConfig, num_envs=0, scenes=None, device="cpu",
         terrain=None, scene_weights=None)
```

Three sources of scenes, mutually exclusive — passing two raises `ValueError`:
`config.scene_file`, `config.inline_scenes`, or the `scenes` argument.

With no scene source at all you get the **Null Object** empty library
(`num_objects_per_scene=0`); `SceneLib.empty(num_envs, device, terrain)` is the explicit
form. Otherwise `num_envs > 0` is required. `scene_weights` must match the original scene
count and is only used by `weighted` replication.

`SceneLibConfig` fields worth knowing: `scene_file`, `inline_scenes`, `asset_root`,
`scene_indices` (pre-filter applied at load, *before* replication/subsetting — handy for
inspecting specific scenes), `subset_method`, `replicate_method`,
`pointcloud_samples_per_object` (None disables pointclouds *and* bbox extents),
`num_objects_per_env`, and the IsaacLab-only mesh collision overrides
(`mesh_collision_approximation` — `convexDecomposition`/`convexHull`/`boundingCube`/
`boundingSphere` — plus `max_convex_hulls`, `hull_vertex_limit`, `voxel_resolution`).

Experiment files expose this through `scene_lib_config(args)`, reading `args.scenes_file`
(CLI `--scenes-file`). See `examples/experiments/format.py` and
`examples/tutorial/3_scene_creation.py`.

## Original vs replicated scenes

The core invariant: **scenes are fitted to `num_envs`, one scene per env**, so
`num_scenes() == len(self.scenes) == num_envs`. But heavy data (motion, pointclouds) is
combined from **original scenes only**, exactly like MotionLib, and shared via
`_scene_to_original_scene_id: (num_envs,)`. Five unique scenes replicated to 4096 envs
store motion five times, not 4096.

`_create_scenes` order: subset if `len(scenes) > num_envs` → replicate if `<` → assign
offsets → build static mask → combine pointclouds/motions → bbox extents → valid mask →
class IDs.

- `replicate_method`: `first` (clone scene 0), `sequential` (round-robin), `random`
  (uniform), `weighted` (uses `scene_weights`; default).
- `subset_method`: `first` (default), `last`, `random`, `sequential` (same as `first`), or
  a plain **list of indices**.

Note `random` replication is just `weighted` with weights dropped, and subsetting slices
`scene_weights` alongside the scenes.

## Runtime queries

Poses (all return `ObjectState`, quaternions xyzw, interpolated with lerp + slerp):

- `get_object_pose(object_indices, time)` → batched `(num_envs, 3)` / `(num_envs, 4)`.
  Requires `combine_object_motions()` to have run (it does, in `_create_scenes`).
- `get_scene_pose(scene_indices, time, respawn_offset=0.0)` — all objects in the given
  scenes; accepts replicated indices.
- `get_default_object_state(device)` — start poses with zero velocities.

Geometry — these all take optional `scene_indices` (replicated; mapped to originals
internally) and return everything when it's None:

- `get_scene_neutral_pointcloud()` / `get_scene_neutral_pointcloud_normals()` — neutral
  (unposed) local-coordinate points, `(num_orig_scenes, objs_per_scene, num_points, 3)`.
- `get_object_scales(device)` → `[..., objs_per_scene, 1, 3]`. Meshes carry a real
  `scale`; primitives bake dims into the pointcloud so theirs is `(1,1,1)`.
- `get_object_bbox_extents()` → `(..., objs_per_scene, 3)` as (width, depth, height).
- `get_per_object_valid_mask()` — False marks padding. A `BoxSceneObject` with **all dims
  < 0.01** is treated as a padding dummy; all other primitives and meshes are always valid.
- `get_object_class_ids()` — for voxel observations.

Pointclouds, normals, and bbox extents **only exist if `pointcloud_samples_per_object` is
set**; otherwise the getters raise.

Placement: `scene_offsets` property, `get_scene_positions(terrain, device)` → `[x, y, 0]`.
Scenes always land in the terrain's flat object-playground region (z=0), see `terrains/`.

Motion bookkeeping (`_object_translations`/`_object_rotations` concatenated over all
original objects, indexed by `_motion_starts`, `_motion_lengths`, `_motion_dts`,
`_motion_num_frames`, all sized `num_original_scenes * num_objects_per_scene`) mirrors
MotionLib — read the `SceneLib` class docstring for the full tensor inventory.

## Tests

`protomotions/tests/test_scene_lib_{empty,objects,scenes,loading_and_assets,inline_scenes,runtime}.py`
— pure pytest, no simulator required.
