# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA seated balance on a chair — full fine-tune of the soma-bones tracker.

Sibling of examples/experiments/mimic/soma_pick_box.py, and like it a copy of
data/pretrained_models/motion_tracker/soma-bones/experiment_config.py (plain PPO
max-coords tracker) so that `--checkpoint .../soma-bones/last.ckpt` warm-starts.
The task changes are:

1. scene_lib_config takes --scenes-file and enables pointcloud sampling.
2. All object-tracking terms are REMOVED (see below).
3. Two anti-squat terms: a torque-squared power penalty and thigh contact bodies.

THE CHAIR IS STATIC, WHICH INVERTS THE BOX FILE'S ADVICE. soma_pick_box.py warns
that a single-frame object makes ``object_pos_rew`` pay the box for not moving.
A chair is *supposed* to be single-frame and ``fix_base_link=True``, so that
reward is not merely useless here, it is a free 100% that dilutes the tracking
signal — and ``object_pos_error_term`` can never fire on a pinned object. So
this file carries no object reward, no object termination and no object metric,
exactly like the vault obstacle in examples/data/rigv1-vaulting/. The sitting
signal comes entirely from body tracking plus the terms below.

WHY THE POLICY WOULD OTHERWISE LEARN TO SQUAT
---------------------------------------------
The tracking reward scores body positions only. A policy can hold the seated
pose as an isometric wall-sit — thighs hovering a centimetre above the seat,
legs carrying everything — and collect full reward without ever loading the
chair. Three things stop that, in the order they cost anything:

1. Seat fit. data/scripts/create_chair_scene.py puts the seat top at the lowest
   thigh/pelvis *collision* surface over the clip, so the reference pose is
   already resting on it and there is no gap to hover in. Verify with
   ``--inspect``: "clearance (thigh - seat)" must be ~0.

2. pow_rew with use_torque_squared=True. This is the load-bearing change.
   ``power_consumption_sum`` (protomotions/envs/rewards/base.py) computes
   sum|tau * qdot| by default, and the soma-bones tracker uses that default —
   but a static wall-sit has qdot ~ 0, so the shipped penalty is blind to
   precisely this failure. sum(tau^2) is large when the knee and hip extensors
   fight gravity and near-zero once the seat carries the weight, which makes
   sitting strictly cheaper than squatting. The weight below is the one knob to
   sweep; too high and it eats into tracking.

3. Thigh contact bodies + contact_match_rew. Contact sensors only exist for
   bodies listed in robot_cfg.contact_bodies (the IsaacLab simulator zero-fills
   rigid_body_contact_forces for everything else), so thigh contact is invisible
   until LeftLeg/RightLeg are added. Safe for the warm start: the observations
   below use observe_contacts=False, so contact_bodies does not enter any
   observation width.

   THE MOTION FILE MUST CARRY RELABELLED THIGH CONTACTS. The packaged labels
   come from ground-plane contact detection, so a seated clip marks only the
   feet and leaves the thighs free — and then contact_match_rew *penalises*
   sitting. Generate the relabelled copy alongside the chair:

       python data/scripts/create_chair_scene.py --output <chair.pt> \
           --motion-file <sit_motion.pt> --motion-id 0 \
           --relabel-contacts <sit_motion_seatcontacts.pt>

   and train against the relabelled file. Without it, drop LeftLeg/RightLeg from
   contact_bodies below.

The decisive check is an ablation, not a curve: evaluate the trained policy
twice, once with --scenes-file and once with `--scenes-file none`. A policy that
really sits falls over without the chair. A squatter scores the same both ways.

On the reference data: the SOMA sit clips are seated *throughout* — the pelvis
height is constant for the whole 5s — so this trains seated balance, not a
standing-to-seated transition. Every frame is therefore a valid RSI point, which
is why init_start_prob stays at the tracker's 0.2 rather than being raised the
way soma_pick_box.py raises it (there, mid-clip RSI spawns the character
mid-grasp with its hands off the box; here it just spawns it seated).

The observation components and both network configs below must stay identical to
the pretrained config. The networks use LazyLinear and lazy observation
normalizers that materialize from the checkpoint's tensor shapes before a strict
load_state_dict, so any change to an observation width makes the checkpoint
unloadable rather than merely degraded. Rewards, terminations, scenes, contact
bodies and the motion manager are not part of that contract.

The policy does not observe the chair: its pose is fixed per motion through
Scene.humanoid_motion_id, and sitting comes from tracking the reference motion.
Adding a chair observation would change obs widths and break the warm start.
SOMA's mimic_target_poses is built with with_relative=True, so the actor already
sees each reference body's offset from its own current body.

Run it as a warm start from the tracker checkpoint:

    python protomotions/train_agent.py --robot-name soma23 --simulator isaaclab \
        --experiment-path examples/experiments/mimic/soma_sit_chair.py \
        --checkpoint data/pretrained_models/motion_tracker/soma-bones/last.ckpt \
        --motion-file <sit_motion_seatcontacts.pt> --scenes-file <chair.pt> \
        --num-envs 4096 --batch-size 16384 --experiment-name soma_sit_chair_v1
"""

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration.

    pointcloud_samples_per_object is required, not an optimization: the env only
    populates object state in the MDP context when pointclouds exist. Nothing in
    this file reads that state today, but leaving it unset silently empties
    EnvContext.scene, so any diagnostic or reward added later would read zeros
    with no error. 8 samples on a box returns its exact corners.
    """
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file, pointcloud_samples_per_object=8)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults).

    Observations, control and action config are unchanged from the soma-bones
    tracker — they are the checkpoint's contract. The rewards carry the sitting
    task.
    """
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        action_smoothness_factory,
        mimic_tracking_rewards_factory,
        pow_rew_factory,
        contact_match_rew_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.action import make_pd_action_config

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True
        ),
    }

    termination_components = {
        # Max per-body world-frame error. Covers falls and world drift, which is
        # what the G1 files need three separate anchor-relative terms for. It is
        # also what makes a badly fitted chair visible: a seat 1cm too high
        # pushes the character off its reference until this fires.
        "tracking_error": tracking_error_term_factory(threshold=0.5),
    }

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gv_weight=0.1,
            gav_weight=0.2,
            rh_weight=0.2,
            gt_coef=-25.0,
            gr_coef=-5.0,
            gv_coef=-0.5,
            gav_coef=-0.1,
            rh_coef=-100.0,
        ),
        # Anti-squat, term 2 of the docstring. Both changes from the tracker's
        # (-1e-5, use_torque_squared=False) matter: the flag is what makes the
        # penalty see a static hold at all, and the weight is what makes it
        # outrank the effort of hovering. Sweep with
        #   --overrides "env.reward_components.pow_rew.static_params.weight=-3e-5"
        "pow_rew": pow_rew_factory(
            weight=-1e-4, min_value=-0.5, use_torque_squared=True
        ),
        # With LeftLeg/RightLeg in contact_bodies and relabelled reference
        # contacts, this is the direct "thighs must be on the seat" term. Read
        # its logged value to tell sitting from squatting: ~0 means the thighs
        # are down, ~-0.1 means they are not.
        "contact_match_rew": contact_match_rew_factory(
            weight=-0.1, zero_during_grace_period=True
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            # The tracker's default, deliberately not soma_pick_box.py's 0.5.
            # The clip is seated end to end, so every RSI point is a valid
            # seated pose and mid-clip starts are free state coverage rather
            # than the broken-grasp starts the box task had to avoid.
            init_start_prob=0.2,
            resample_on_reset=True,
            # Must stay False: realigning the reference to the humanoid each
            # step would slide the motion off the world-fixed chair.
            realign_motion_with_humanoid_on_each_step=False,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Build agent configuration.

    Identical to the soma-bones tracker — the networks and optimizers are the
    checkpoint's contract, and a static chair admits no object eval metric.
    """
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "previous_actions",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "previous_actions",
            ],
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot contact sensors for foot and thigh contact tracking.

    The thighs are the anti-squat term: on SOMA they, not the buttocks, carry the
    weight when seated (the thigh capsule undersides sit 4-6cm below the pelvis
    sphere's). Contact sensors are only created for bodies named here, so without
    them contact_match_rew cannot see the seat at all.

    Raw body names are fine alongside the common-naming keys — see
    abstract_names_to_body_names in protomotions/robot_configs/base.py.
    """
    robot_cfg.update_fields(
        contact_bodies=[
            "all_left_foot_bodies",
            "all_right_foot_bodies",
            "LeftLeg",
            "RightLeg",
        ]
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
