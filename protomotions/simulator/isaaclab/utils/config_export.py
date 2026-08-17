# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the IsaacLab configs ProtoMotions builds as an IsaacLab-style ``env.yaml``.

ProtoMotions drives IsaacLab through its scene layer only -- ``SimulationCfg`` ->
``SimulationContext`` and ``SceneCfg(InteractiveSceneCfg)`` -> ``InteractiveScene`` --
so there is no ``ManagerBasedEnvCfg``/``DirectRLEnvCfg`` to hand to
:func:`isaaclab.utils.io.dump_yaml`. This module assembles the equivalent dict from
the two config objects that do exist, in the layout emitted by IsaacLab's own workflow
scripts (``dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)``).

The output is consumable by Isaac Sim's ``PolicyController``
(``isaacsim.robot.policy.examples``), which reads only four groups of keys:

- ``decimation``, ``sim.dt``, ``sim.render_interval``
- ``scene.robot.actuators.*.{joint_names_expr,stiffness,damping,effort_limit,velocity_limit}``
- ``scene.robot.init_state.{joint_pos,joint_vel}``
- ``scene.robot.spawn.articulation_props.*``

The manager-layer sections of a real ``ManagerBasedRLEnvCfg`` dump (``observations``,
``actions``, ``rewards``, ``terminations``, ...) have no ProtoMotions equivalent and are
never read by ``PolicyController``, so they are absent here.
"""

import logging
from dataclasses import MISSING
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from protomotions.robot_configs.base import ControlType

# isaaclab is imported lazily inside the functions that need it: importing it at module
# scope requires a running SimulationApp, which would make the pure-python helpers below
# untestable and would pull a heavy import into any module that touches this one.

log = logging.getLogger(__name__)


def _to_plain(value: Any) -> Any:
    """Recursively coerce a ``class_to_dict`` result into YAML-safe primitives.

    ``class_to_dict`` returns tensors untouched and leaves ``MISSING`` sentinels in
    place, both of which ``yaml.dump`` would emit as opaque ``!!python/object`` tags.
    Tuples are preserved as tuples so they round-trip as ``!!python/tuple``, matching
    the workflow-script output that Isaac Sim's loader has a constructor for.
    """
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_plain(item) for item in value)
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if value is MISSING:
        return None
    return value


def _elide_terrain_mesh(scene_dict: Dict[str, Any]) -> None:
    """Replace the inlined trimesh with a size summary.

    ``TrimeshTerrainImporterCfg`` carries the full vertex/face arrays (see
    ``utils/scene.py``), which on generated terrain is megabytes of YAML and is of no
    use to any consumer of this file.
    """
    terrain = scene_dict.get("terrain")
    if not isinstance(terrain, dict):
        return
    for key, count_key in (
        ("terrain_vertices", "num_vertices"),
        ("terrain_faces", "num_faces"),
    ):
        mesh_data = terrain.get(key)
        if mesh_data is None:
            continue
        try:
            count = len(mesh_data)
        except TypeError:
            count = None
        terrain[key] = {count_key: count, "elided": True}


def _patch_actuator_limits(robot_dict: Dict[str, Any]) -> None:
    """Mirror ``*_limit_sim`` onto the bare ``*_limit`` keys.

    ``effort_limit``/``effort_limit_sim`` are distinct fields on ``ActuatorBaseCfg``,
    both defaulting to ``None``. ProtoMotions sets only the ``_sim`` variants, but
    Isaac Sim's ``get_robot_joint_properties`` reads the bare keys and silently falls
    back to ``sys.maxsize`` when they are ``None``.
    """
    actuators = robot_dict.get("actuators")
    if not isinstance(actuators, dict):
        return
    for actuator in actuators.values():
        if not isinstance(actuator, dict):
            continue
        for key in ("effort_limit", "velocity_limit"):
            if actuator.get(key) is None and actuator.get(key + "_sim") is not None:
                actuator[key] = actuator[key + "_sim"]


def _patch_actuator_gains(robot_dict: Dict[str, Any], robot_config) -> None:
    """Restore the real PD gains when ProtoMotions applies control itself.

    ``utils/scene.py`` deliberately zeroes stiffness and damping for every control type
    other than ``BUILT_IN_PD``, because the torque is computed in ProtoMotions rather
    than by PhysX. That is correct for simulation but leaves a consumer of this file
    with zero gains, so the true per-DOF values are read back from ``control_info``.
    """
    if robot_config.control.control_type == ControlType.BUILT_IN_PD:
        return
    actuators = robot_dict.get("actuators")
    if not isinstance(actuators, dict):
        return
    control_info = robot_config.control.control_info
    for dof_name, actuator in actuators.items():
        info = control_info.get(dof_name)
        if not isinstance(actuator, dict) or info is None:
            continue
        if info.stiffness is not None:
            actuator["stiffness"] = float(info.stiffness)
        if info.damping is not None:
            actuator["damping"] = float(info.damping)


def _patch_init_joint_pos(robot_dict: Dict[str, Any], robot_config) -> None:
    """Fill in the default pose, which the scene cfg hardcodes to zeros.

    ProtoMotions resets from motion data, so ``ArticulationCfg.InitialStateCfg`` is
    built with ``{".*": 0.0}``. The real default pose lives on the robot config as a
    tensor ordered by ``kinematic_info.dof_names``.
    """
    init_state = robot_dict.get("init_state")
    if not isinstance(init_state, dict):
        return
    default_dof_pos = getattr(robot_config, "default_dof_pos", None)
    if default_dof_pos is None:
        return
    dof_names = robot_config.kinematic_info.dof_names
    if isinstance(default_dof_pos, torch.Tensor):
        values = default_dof_pos.detach().cpu().tolist()
    else:
        values = list(default_dof_pos)
    if len(values) != len(dof_names):
        log.warning(
            "Skipping init_state.joint_pos export: default_dof_pos has %d entries but "
            "the robot has %d DOFs",
            len(values),
            len(dof_names),
        )
        return
    init_state["joint_pos"] = {
        name: float(value) for name, value in zip(dof_names, values)
    }


def build_env_cfg_dict(
    sim_cfg, scene_cfg, decimation: int, robot_config
) -> Dict[str, Any]:
    """Assemble the IsaacLab-style env config dict from ProtoMotions' IsaacLab objects.

    Args:
        sim_cfg: The ``isaaclab.sim.SimulationCfg`` handed to ``SimulationContext``.
        scene_cfg: The ``SceneCfg`` handed to ``InteractiveScene``.
        decimation: Physics steps per control step (``config.sim.decimation``).
        robot_config: The ProtoMotions ``RobotConfig``, used to recover the values the
            scene cfg intentionally leaves blank or zeroed.

    Returns:
        A dict with ``decimation``, ``sim`` and ``scene`` keys, ready for ``dump_yaml``.
    """
    from isaaclab.utils.dict import class_to_dict

    sim_dict = _to_plain(class_to_dict(sim_cfg))
    scene_dict = _to_plain(class_to_dict(scene_cfg))

    _elide_terrain_mesh(scene_dict)

    robot_dict = scene_dict.get("robot")
    if isinstance(robot_dict, dict):
        _patch_actuator_limits(robot_dict)
        _patch_actuator_gains(robot_dict, robot_config)
        _patch_init_joint_pos(robot_dict, robot_config)
    else:
        log.warning("No 'robot' entry in the scene config; export will be incomplete")

    return {"decimation": decimation, "sim": sim_dict, "scene": scene_dict}


def export_env_cfg_yaml(
    log_dir: str,
    sim_cfg,
    scene_cfg,
    decimation: int,
    robot_config,
    file_name: str = "env.yaml",
) -> Optional[Path]:
    """Write ``<log_dir>/params/<file_name>``, mirroring IsaacLab's workflow scripts.

    Returns the written path, or ``None`` if the export failed. Export problems are
    logged and swallowed: a missing config dump must never take down a run.
    """
    from isaaclab.utils.io import dump_yaml

    output_path = Path(log_dir) / "params" / file_name
    try:
        env_cfg_dict = build_env_cfg_dict(
            sim_cfg=sim_cfg,
            scene_cfg=scene_cfg,
            decimation=decimation,
            robot_config=robot_config,
        )
        # dump_yaml creates missing directories and uses the non-safe dumper, so tuples
        # emit as !!python/tuple exactly as in IsaacLab's own params/env.yaml.
        dump_yaml(str(output_path), env_cfg_dict)
    except Exception:
        log.exception("Failed to export IsaacLab env config to %s", output_path)
        return None

    # print rather than log.info: SimulationContext has already run IsaacLab's
    # configure_logging() at WARNING by this point, which would swallow an info record.
    print(f"[INFO]: Exported IsaacLab env config to {output_path}")
    return output_path
