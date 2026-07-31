# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 box pickup — full fine-tune of the g1-bones-deploy tracker.

Copy of data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py
(BeyondMimic + L2C2 + AMP discriminator) with the task changes needed to make the
box actually track its reference trajectory:

1. scene_lib_config takes --scenes-file and enables pointcloud sampling.
2. Push randomization is off; a shove mid-grasp is unrecoverable noise.
3. Box position tracking at two scales, plus a termination and an eval metric on
   box error, so a dropped box ends the rollout and shows up in eval.
4. Global anchor position reward + drift termination: the box lives in world
   coordinates, and every inherited tracking term is anchor-relative.
5. Softer reset noise and more start-of-clip resets, so RSI does not spawn the
   robot mid-grasp with its hands off the box.

THE SCENES FILE MUST CARRY A BOX TRAJECTORY. The reference the box is scored
against is ``scene_lib.get_scene_pose(env_ids, motion_times)``, which comes from
the object's own motion frames. A single-frame object (what
``data/scripts/create_box_scene.py`` produces without ``--motion-file``) yields a
constant reference, and then this reward pays the box for *not moving* — a
successful pickup scores zero. Author the box with per-frame translation/rotation
at the paired motion's fps and set ``humanoid_motion_id``; verify with
``python data/scripts/create_box_scene.py --inspect <scenes.pt>``.

Run it as a warm start from the deploy checkpoint:

    python protomotions/train_agent.py --robot-name g1 --simulator isaaclab \
        --experiment-path examples/experiments/mimic/g1_pick_box.py \
        --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
        --motion-file <pickup_motion.pt> --scenes-file <box_scene.pt> \
        --num-envs 4096 --batch-size 16384 --experiment-name g1_pick_box_v1

The observation components and all four network configs below must stay
identical to the pretrained config. The networks use LazyLinear and lazy
observation normalizers that materialize from the checkpoint's tensor shapes
before a strict load_state_dict, so any change to an observation width makes the
checkpoint unloadable rather than merely degraded. Rewards, terminations, scenes
and domain randomization are not part of that contract.

The policy does not observe the box: its pose is fixed per motion through
Scene.humanoid_motion_id, and the pickup comes from tracking the reference
motion. Same approach as the vault obstacle in examples/data/rigv1-vaulting/.

Known limitation of that approach here: the actor obs are drift-blind
(include_xy_offset=False, no object channel), so the policy cannot see where the
box is relative to itself. The rewards and terminations below make drift costly,
but they cannot close the loop on it. If eval/box_pos_error stays high while the
body-tracking metrics look healthy, drift is the binding constraint and the fix
is an obs channel (corrupted_xy_offset_factory, include_xy_offset=True, or a
box-relative observation) — which changes obs widths and breaks the warm start
above, so it needs a from-scratch run or zero-padded first-layer weights.

Reward weighting (unnormalized):
- task_reward_w = 0.5 (BM tracking rewards, ~3-4 magnitude)
- discriminator_reward_w = 2.0 (AMP disc reward, 0-1 magnitude)
  -> roughly equalizes their contribution to advantages
- box_pos_coarse + box_pos_fine = 2.0 of the ~7.5 task-reward budget, comparable
  to one body-tracking term pair

Discriminator sees clean (noise-free) historical observations, same as critic.
"""

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import (
    SimulatorConfig,
    ActionNoiseDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    RobotNoiseConfig,
    DomainRandomizationConfig,
)
from protomotions.components.terrains.config import (
    TerrainConfig,
    TerrainSimConfig,
    CombineMode,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse


# History steps for discriminator temporal context
DISC_HISTORY_STEPS = [1, 2, 3, 4, 8, 16, 32]


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    terrain_cfg = TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=1,
            dynamic_friction=1,
            restitution=0.0,
            combine_mode=CombineMode.MULTIPLY,
        )
    )
    return terrain_cfg


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration.

    pointcloud_samples_per_object is required, not an optimization: the env only
    populates object state in the MDP context when pointclouds exist, so the box
    reward would silently see no objects without it. 8 samples on a box returns
    its exact corners.
    """
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file, pointcloud_samples_per_object=8)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration.

    Actor sees noisy reduced coords + multi-horizon target poses.
    Critic sees clean max coords + target poses.
    Discriminator sees clean historical max coords.
    L2C2 clean counterparts provided for actor regularization.
    Rewards use BM-style tracking + AMP discriminator.
    """
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.component_factories import (
        reduced_coords_obs_factory,
        mimic_target_poses_reduced_coords_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        historical_max_coords_obs_factory,
        previous_actions_factory,
        action_smoothness_factory,
        global_anchor_ori_rew_factory,
        global_anchor_pos_rew_factory,
        relative_body_pos_rew_factory,
        relative_body_ori_rew_factory,
        global_body_lin_vel_rew_factory,
        global_body_ang_vel_rew_factory,
        object_pos_rew_factory,
        anchor_height_error_term_factory,
        anchor_pos_error_term_factory,
        object_pos_error_term_factory,
        relative_body_pos_error_term_factory,
    )
    from protomotions.envs.rewards import compute_soft_pos_limit_rew
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.action import make_bm_pd_action_config

    # Control components configuration
    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=[1, 2, 4, 8],
        )
    }

    # Observation components
    observation_components = {
        # Actor observations (noisy) — reduced coords proprioception
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=True,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        # Actor target poses — reduced coords, no XY offset, multi-horizon
        "noisy_mimic_reduced_coords_target_poses": mimic_target_poses_reduced_coords_factory(
            use_noisy=True,
            include_dof_vel=True,
            include_xy_offset=False,
        ),
        # Clean counterparts for L2C2
        "clean_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "clean_mimic_reduced_coords_target_poses": mimic_target_poses_reduced_coords_factory(
            use_noisy=False,
            include_dof_vel=True,
            include_xy_offset=False,
        ),
        # Critic observations (clean) — full max coords
        "max_coords_obs": max_coords_obs_factory(
            use_noisy=False,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "mimic_max_coords_target_poses": mimic_target_poses_max_coords_factory(
            use_noisy=False,
            with_velocities=True,
            with_relative=True,
        ),
        # Historical observations for AMP discriminator (clean)
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            use_noisy=False,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=DISC_HISTORY_STEPS,
        ),
        # Common observations (processed actions after tanh/clamp)
        "historical_previous_processed_actions": previous_actions_factory(
            history_steps=1, processed=True
        ),
    }

    # Termination components
    termination_components = {
        "fall": anchor_height_error_term_factory(threshold=0.25),
        "bad_motion_body_pos": relative_body_pos_error_term_factory(threshold=0.25),
        # The box sits at a fixed world position, so a drifted robot reaches for
        # empty air. The deploy tracker tolerates drift because nothing in its
        # world depends on absolute position; here it decides the grasp.
        "drifted": anchor_pos_error_term_factory(threshold=0.35),
        # A dropped box ends the rollout. Without this the episode keeps earning
        # the discriminator reward, and the evaluator's failure discount never
        # sees the pickup fail.
        "box_lost": object_pos_error_term_factory(threshold=0.3),
    }

    # Reward components (BM tracking — task rewards)
    reward_components = {
        # Global anchor (root) orientation
        "global_anchor_ori": global_anchor_ori_rew_factory(weight=0.5, sigma=0.4),
        # Global anchor position. Every other tracking term below is
        # anchor-relative, so without this nothing penalizes world-frame drift —
        # and the box only exists in the world frame.
        "global_anchor_pos": global_anchor_pos_rew_factory(weight=1.0, sigma=0.3),
        # Relative body position and orientation (region-weighted)
        "relative_body_pos": relative_body_pos_rew_factory(
            weight=1.0,
            sigma=0.3,
        ),
        "relative_body_ori": relative_body_ori_rew_factory(
            weight=1.0,
            sigma=0.4,
        ),
        # Global body velocities (region-weighted)
        "body_lin_vel": global_body_lin_vel_rew_factory(
            weight=1.0,
            sigma=1.0,
        ),
        "body_ang_vel": global_body_ang_vel_rew_factory(
            weight=1.0,
            sigma=3.14,
        ),
        # Box tracking, two scales. exp(-e^2/sigma^2) at sigma=0.1 is already
        # 0.018 at 20cm and 1e-4 at 30cm, so a fine term alone is effectively
        # binary: it cannot tell "drifting off" from "box on the floor" and
        # offers no gradient back. The coarse term carries the approach and any
        # recovery, the fine term buys the last few centimetres where 10cm of
        # error is a failed pickup rather than a wobble.
        "box_pos_coarse": object_pos_rew_factory(
            weight=1.0, sigma=0.4, zero_during_grace_period=True
        ),
        "box_pos_fine": object_pos_rew_factory(
            weight=1.0, sigma=0.1, zero_during_grace_period=True
        ),
        "action_rate": action_smoothness_factory(weight=-0.1),
        "limits_dof_pos": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,
            },
            static_params={
                "weight": -10.0,
                "dof_limits_lower": robot_cfg.kinematic_info.dof_limits_lower,
                "dof_limits_upper": robot_cfg.kinematic_info.dof_limits_upper,
            },
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=max(DISC_HISTORY_STEPS),
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_bm_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            # Higher than the deploy tracker's 0.2. RSI drops the robot into the
            # middle of the clip with reset noise applied, while the box is
            # placed exactly on its reference with zero velocity — mid-grasp that
            # means hands that are not around the box. Half the resets start from
            # the beginning so the grasp is actually practised from the approach.
            init_start_prob=0.5,
            resample_on_reset=True,
            # Must stay False: realigning the reference to the humanoid each step
            # would slide the motion away from the world-fixed box.
            realign_motion_with_humanoid_on_each_step=False,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
):
    """Build AMP agent configuration with L2C2 regularization."""
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
    )
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        AdaptiveLRConfig,
        AdvantageNormalizationConfig,
        L2C2Config,
    )
    from protomotions.agents.amp.config import (
        AMPAgentConfig,
        AMPModelConfig,
        DiscriminatorConfig,
        AMPParametersConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        anchor_ori_metric_factory,
        relative_body_pos_metric_factory,
        anchor_height_error_metric_factory,
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
        object_pos_error_metric_factory,
    )
    from protomotions.envs.obs import compute_historical_max_coords_from_motion_lib
    from protomotions.envs.mdp_component import MdpComponent

    # Actor configuration — obs normalization ON, learnable std from -2.9
    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        learnable_std=True,
        in_keys=[
            "noisy_reduced_coords_obs",
            "noisy_mimic_reduced_coords_target_poses",
            "historical_previous_processed_actions",
        ],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "noisy_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                "historical_previous_processed_actions",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
        ),
    )

    # Critic configuration — obs normalization ON
    critic_config = MLPWithConcatConfig(
        in_keys=[
            "max_coords_obs",
            "mimic_max_coords_target_poses",
            "historical_previous_processed_actions",
        ],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5.0,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    # Discriminator — sees clean historical max coords
    discriminator_config = DiscriminatorConfig(
        in_keys=["historical_max_coords_obs"],
        out_keys=["disc_logits"],
        models=[
            MLPWithConcatConfig(
                in_keys=["historical_max_coords_obs"],
                out_keys=["disc_logits"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            )
        ],
    )

    # Discriminator critic — same inputs as discriminator
    disc_critic_config = ModuleContainerConfig(
        in_keys=["max_coords_obs", "historical_max_coords_obs"],
        out_keys=["disc_value"],
        models=[
            MLPWithConcatConfig(
                in_keys=["max_coords_obs", "historical_max_coords_obs"],
                out_keys=["disc_value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=512, activation="relu"),
                    MLPLayerConfig(units=256, activation="relu"),
                ],
            )
        ],
    )

    # Reference observation components for discriminator expert data
    reference_obs_components = {
        "historical_max_coords_obs": MdpComponent(
            compute_func=compute_historical_max_coords_from_motion_lib,
            dynamic_vars={},  # motion_lib, motion_ids, motion_times, dt injected by agent
            static_params={
                "num_state_history_steps": max(DISC_HISTORY_STEPS),
                "history_steps": DISC_HISTORY_STEPS,
            },
        ),
    }

    agent_cfg = AMPAgentConfig(
        model=AMPModelConfig(
            in_keys=[
                # Noisy observations for actor
                "noisy_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses",
                # Clean observations for L2C2
                "clean_reduced_coords_obs",
                "clean_mimic_reduced_coords_target_poses",
                # Clean observations for critic
                "max_coords_obs",
                "mimic_max_coords_target_poses",
                # Historical observations for discriminator (clean)
                "historical_max_coords_obs",
                # Shared observations
                "historical_previous_processed_actions",
            ],
            out_keys=[
                "action",
                "mean_action",
                "neglogp",
                "value",
                "disc_logits",
                "disc_value",
            ],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_config,
            disc_critic=disc_critic_config,
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=2e-5, betas=(0.95, 0.99)
            ),
            critic_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4, betas=(0.95, 0.99)
            ),
            discriminator_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
        ),
        reference_obs_components=reference_obs_components,
        normalize_rewards=False,
        task_reward_w=0.5,
        amp_parameters=AMPParametersConfig(
            discriminator_reward_w=2.0,
            discriminator_reward_threshold=0.02,
        ),
        adaptive_lr=AdaptiveLRConfig(enabled=False),
        batch_size=args.batch_size,
        num_mini_epochs=2,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        l2c2=L2C2Config(
            enabled=True,
            lambda_l2c2=1.0,
            obs_pairs={
                "noisy_reduced_coords_obs": "clean_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses": "clean_mimic_reduced_coords_target_poses",
            },
        ),
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "anchor_ori": anchor_ori_metric_factory(),
                "relative_body_pos": relative_body_pos_metric_factory(threshold=0.5),
                "anchor_height_error": anchor_height_error_metric_factory(
                    threshold=0.25
                ),
                "gt_error": gt_error_factory(),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
                # Makes box tracking part of eval/success_rate, which is what
                # drives the motion weights below.
                "box_pos_error": object_pos_error_metric_factory(threshold=0.25),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True
        ),
    )
    return agent_cfg


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot and simulator — same as the base L2C2 experiment."""
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )

    # Trimmed relative to the deploy tracker (dof 0.1, root_pos 5cm). The box is
    # respawned on its reference pose regardless of how the robot is perturbed,
    # so reset noise translates directly into hands that miss the box on an RSI
    # reset. Costs some robustness margin; that is a task-specific trade, and
    # domain randomization is not part of the checkpoint's obs contract.
    robot_cfg.reset_noise = RobotNoiseConfig(
        dof_pos_noise=0.03,
        root_pos_noise=[0.02, 0.02, 0.01],
        root_rot_noise=[0.05, 0.05, 0.1],
        root_vel_noise=[0.1, 0.1, 0.05],
        root_ang_vel_noise=[0.1, 0.1, 0.1],
    )

    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.025, 0.025), dof_names=[".*"], dof_indices=None
        ),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.3, 1.6),
            dynamic_friction_range=(0.3, 1.2),
            restitution_range=(0.0, 0.5),
            body_names=[".*"],
            body_indices=None,
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],
            body_indices=None,
        ),
        observation_noise=RobotNoiseConfig(
            dof_pos_noise=0.01,
            dof_vel_noise=0.5,
            anchor_ang_vel_noise=0.2,
            anchor_rot_noise=0.05,
        ),
        # No push randomization: the deploy tracker is shoved every 1-3s, which
        # rips the box out of the hands mid-grasp. The rest of the deploy
        # randomization stays so the sim2real contract survives.
        push=None,
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
    """Apply inference overrides.

    Removes clean L2C2 obs, disables noise, disables termination,
    and disables discriminator reward threshold.
    """
    from protomotions.envs.component_factories import (
        reduced_coords_obs_factory,
        mimic_target_poses_reduced_coords_factory,
    )

    # Disable all termination components for inference
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
    robot_cfg.reset_noise = None

    terrain_cfg.sim_config = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        combine_mode=CombineMode.AVERAGE,
    )
    simulator_cfg.domain_randomization = None

    # Swap noisy observations for clean
    env_cfg.observation_components["noisy_reduced_coords_obs"] = (
        reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=False,
            root_vel_obs=False,
        )
    )
    env_cfg.observation_components["noisy_mimic_reduced_coords_target_poses"] = (
        mimic_target_poses_reduced_coords_factory(
            use_noisy=False,
            include_dof_vel=True,
            include_xy_offset=False,
        )
    )

    # Remove clean obs (not needed at inference — noise is disabled)
    for key in [
        "clean_reduced_coords_obs",
        "clean_mimic_reduced_coords_target_poses",
    ]:
        env_cfg.observation_components.pop(key, None)

    # Disable discriminator reward threshold at inference
    if hasattr(agent_cfg, "amp_parameters"):
        agent_cfg.amp_parameters.discriminator_reward_threshold = 0.0
