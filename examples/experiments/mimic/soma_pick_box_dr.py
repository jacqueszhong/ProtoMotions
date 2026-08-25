# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA box pickup — full fine-tune of the soma-bones tracker.

The SOMA counterpart of examples/experiments/mimic/g1_pick_box.py. It is a copy
of data/pretrained_models/motion_tracker/soma-bones/experiment_config.py (plain
PPO max-coords tracker) with the task changes needed to make the box actually
track its reference trajectory:

1. scene_lib_config takes --scenes-file and enables pointcloud sampling.
2. Box position tracking at two scales, plus a termination and an eval metric on
   box error, so a dropped box ends the rollout and shows up in eval.
3. More start-of-clip resets, so RSI does not spawn the character mid-grasp with
   its hands off the box.

Note this is NOT a port of the G1 file's config: the two pretrained trackers are
different experiments. g1-bones-deploy is BeyondMimic + L2C2 + AMP on noisy
reduced-coordinate observations with a full sim2real randomization recipe;
soma-bones is bare PPO on clean max-coordinate observations with no domain
randomization and no reset noise at all. The base here is soma-bones, so that
`--checkpoint .../soma-bones/last.ckpt` warm-starts.

Several of the G1 file's task components are deliberately absent, because SOMA's
tracking terms are already world-frame where G1's BeyondMimic terms were
anchor-relative:

- No global_anchor_pos reward. gt_rew already compares current against reference
  rigid_body_pos in world coordinates, so drift is priced in.
- No "drifted"/"fall"/"bad_motion_body_pos" terminations. tracking_error is max
  per-body world-frame error over 0.5m, which fires on a fall and on a drift.
- No push randomization, for the same reason the G1 file drops it: a shove every
  1-3s rips the box out of the hands mid-grasp.
- No reset noise. Unlike the G1 file, which inherited it and trimmed it, there is
  nothing here to trim, and adding it would be a net negative: ``apply_reset_noise``
  (envs/obs/observation_noise.py) perturbs only the robot's ``ResetState``, never
  ``new_object_states``, while the box is respawned exactly on its reference. Every
  radian of it is pure robot/box desync on an RSI reset. init_start_prob=0.5 below
  already covers the "practise the approach" case that reset noise would serve.

DOMAIN RANDOMIZATION (configure_robot_and_simulator, below). soma-bones was
trained with none — its MODEL_CARD says so outright, and warns that cross-sim
transfer should not be assumed. Two things are being bought here:

- Box generalization: ``object_assets`` randomizes the box's mass, friction,
  restitution and centre of mass. Without it the policy can memorise one exact
  3 kg box. Note this is the ONLY DR that reaches the box; robot-body friction
  does not touch scene objects.
- Cross-sim transfer: friction, centre-of-mass and action noise, the portable
  part of the G1 recipe. ``convert_friction_for_simulator`` rewrites both the
  terrain config and these friction ranges per backend (IsaacGym forces AVERAGE,
  Newton forces MAX, IsaacLab is configurable), so the effective distribution is
  preserved when the same config is run elsewhere.

All of it is width-neutral, so the warm start still loads strictly. Three caveats
worth knowing before reading the eval curves:

1. Only IsaacLab and IsaacGym can run this experiment at all. MuJoCo asserts
   ``scene_lib.num_scenes() == 0``, Genesis stubs out object spawning, and Newton
   does not spawn scene-lib objects — and Newton additionally ignores
   ``object_assets`` outright. So "cross-sim" here realistically means
   IsaacLab -> IsaacGym, and IsaacGym applies friction to all shapes at once and
   ignores ``dynamic_friction_range`` (it has a single friction property).
2. Object and robot properties are sampled into buckets ONCE at setup and
   assigned round-robin (``arange(num_envs) % num_buckets``), not resampled per
   episode. Each env keeps its box for the whole run; variety is across envs.
   With 4096 envs and 64 buckets that is 64 distinct boxes, ~64 envs each.
3. The evaluator's ``_disable_perturbations`` nulls only ``reset_noise`` and the
   push flag — friction, CoM, action noise and object DR all stay live during
   eval. So eval/box_pos_error and eval/success_rate are measured on randomized
   boxes and will read worse than a no-DR baseline. That is honest rather than
   broken, but it feeds motion_weights_update_failure_discount=0 below, so a
   motion that only fails on the extreme buckets still gets its weight zeroed.

Observation noise is off. Copying the G1 ``observation_noise`` block verbatim
would be a silent no-op: its ``dof_pos``/``dof_vel``/``anchor_*`` fields never
enter a max-coordinate observation. The fields that would actually bite here are
``body_pos_noise``/``body_rot_noise``/``body_vel_noise``/``body_ang_vel_noise``,
and using them also means flipping ``use_noisy=True`` on the two obs factories
below. That is width-neutral too, but it feeds noise to the critic as well as the
actor, since both read the same keys — an asymmetric split would need separate
clean keys for the critic, which is safe (the normalizer is one RunningMeanStd
over the concatenated in_keys, so names do not matter, only widths and order).

THE SCENES FILE MUST CARRY A BOX TRAJECTORY. The reference the box is scored
against is ``scene_lib.get_scene_pose(env_ids, motion_times)``, which comes from
the object's own motion frames. A single-frame object (what
``data/scripts/create_box_scene.py`` produces without ``--motion-file``) yields a
constant reference, and then this reward pays the box for *not moving* — a
successful pickup scores zero. Author the box with per-frame translation/rotation
at the paired motion's fps and set ``humanoid_motion_id``; verify with
``python data/scripts/create_box_scene.py --inspect <scenes.pt>``, which warns on
exactly that case, on ``fix_base_link=True``, and on an unpaired scene.

create_box_scene.py defaults are G1-shaped — ``--robot-name`` defaults to ``g1``
and the default carry bodies are ``left_rubber_hand``/``right_rubber_hand``,
neither of which exists on SOMA. The SOMA invocation is:

    python data/scripts/create_box_scene.py --output <box_scene.pt> \
        --robot-name soma23 --carry-body LeftHand --carry-body RightHand \
        --motion-file <soma_pickup_motion.pt> --motion-id 0 \
        --grasp-start <t0_seconds> --grasp-end <t1_seconds>

Omit --fixed: a pinned box cannot be picked up.

Existing SOMA box-pickup material in this repo, all of it currently unfetched
git-LFS pointer stubs — run ``git lfs pull`` before touching any of it:
``data/soma-kimodo-generated/soma_walk_box.pt``,
``data/soma-kimodo-generated/proto/custom_box.motion``,
``data/out-kimodo-soma/proto/custom_box.motion``, and
``data/pick-box-0/{output_soma.bvh, save_soma/}``. The ``.pt`` names are
ambiguous — some are scenes files, some are authored trajectories for
``--trajectory-file``. ``--inspect`` tells you which.

Run it as a warm start from the tracker checkpoint:

    python protomotions/train_agent.py --robot-name soma23 --simulator isaaclab \
        --experiment-path examples/experiments/mimic/soma_pick_box.py \
        --checkpoint data/pretrained_models/motion_tracker/soma-bones/last.ckpt \
        --motion-file <soma_pickup_motion.pt> --scenes-file <box_scene.pt> \
        --num-envs 4096 --batch-size 16384 --experiment-name soma_pick_box_v1

The observation components and both network configs below must stay identical to
the pretrained config. The networks use LazyLinear and lazy observation
normalizers that materialize from the checkpoint's tensor shapes before a strict
load_state_dict, so any change to an observation width makes the checkpoint
unloadable rather than merely degraded. Rewards, terminations, scenes and the
motion manager are not part of that contract, and neither is domain
randomization: it changes physics and action values, never a tensor width.

The policy does not observe the box: its pose is fixed per motion through
Scene.humanoid_motion_id, and the pickup comes from tracking the reference
motion. Same approach as the vault obstacle in examples/data/rigv1-vaulting/,
though that obstacle is kinematic and single-frame while this box must be
dynamic and per-frame. Scene i pins env i to that scene's motion, so a
single-scene file authored with --motion-id 0 runs motion 0 in every env.

Unlike the G1 file, the actor here is not drift-blind: mimic_target_poses is
built with with_relative=True, so the policy observes each reference body's
offset from its own current body and can close its own tracking error. The box
is rigidly bound to that reference, so tracking well is carrying the box.

Reward weighting (weights are normalized by the agent, so only ratios matter):
- SOMA's positive tracking budget is gt 0.5 + gr 0.3 + gv 0.1 + gav 0.2 + rh 0.2
  = 1.3, against G1's BeyondMimic budget of 5.5.
- box_pos_coarse + box_pos_fine = 0.5 gives the box ~28% of the task reward, the
  same share the G1 file's 2.0-of-7.5 gave it. Copying the G1 weights verbatim
  would hand the box 61% and wreck the warm start.
"""

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import (
    SimulatorConfig,
    ActionNoiseDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    DomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    ObjectAssetDomainRandomizationConfig,
)
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
    """Build environment configuration (training defaults).

    Observations, control and action config are unchanged from the soma-bones
    tracker — they are the checkpoint's contract. Rewards, terminations and the
    motion manager carry the box task.
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
        object_pos_rew_factory,
        tracking_error_term_factory,
        object_pos_error_term_factory,
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
        # what the G1 file needed three separate anchor-relative terms for.
        "tracking_error": tracking_error_term_factory(threshold=0.5),
        # A dropped box ends the rollout. Without this the episode keeps earning
        # the tracking reward, and the evaluator's failure discount never sees
        # the pickup fail.
        "box_lost": object_pos_error_term_factory(threshold=0.3),
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
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        "contact_match_rew": contact_match_rew_factory(
            weight=-0.1, zero_during_grace_period=True
        ),
        # Box tracking, two scales. exp(-e^2/sigma^2) at sigma=0.1 is already
        # 0.018 at 20cm and 1e-4 at 30cm, so a fine term alone is effectively
        # binary: it cannot tell "drifting off" from "box on the floor" and
        # offers no gradient back. The coarse term carries the approach and any
        # recovery, the fine term buys the last few centimetres where 10cm of
        # error is a failed pickup rather than a wobble.
        "box_pos_coarse": object_pos_rew_factory(
            weight=0.25, sigma=0.4, zero_during_grace_period=True
        ),
        "box_pos_fine": object_pos_rew_factory(
            weight=0.25, sigma=0.1, zero_during_grace_period=True
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
            # Higher than the tracker's 0.2. RSI drops the character into the
            # middle of the clip, while the box is placed exactly on its
            # reference with zero velocity — mid-grasp that means hands that are
            # not around the box. Half the resets start from the beginning so
            # the grasp is actually practised from the approach.
            init_start_prob=0.5,
            resample_on_reset=True,
            # Must stay False: realigning the reference to the humanoid each step
            # would slide the motion away from the world-fixed box.
            realign_motion_with_humanoid_on_each_step=False,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Build agent configuration.

    Identical to the soma-bones tracker apart from the box eval metric — the
    networks and optimizers are the checkpoint's contract.
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
        object_pos_error_metric_factory,
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
                # Makes box tracking part of eval/success_rate, which is what
                # drives the motion weights below. The threshold is not
                # optional: without it the metric is reported but carries no
                # fail criterion.
                "box_pos_error": object_pos_error_metric_factory(threshold=0.25),
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
    """Configure contact sensors and domain randomization.

    soma-bones ships with none of this — see the module docstring for why the
    recipe below is not a copy of the G1 one, and what deliberately stays off.
    All of it is width-neutral, so `--checkpoint .../soma-bones/last.ckpt` still
    loads strictly: DR touches physics and action values, never an observation
    tensor's shape.
    """
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )

    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        # Same magnitude as the G1 deploy recipe. Applied in the PD action space
        # by the shared base-simulator path, so it is backend-independent.
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.025, 0.025), dof_names=[".*"], dof_indices=None
        ),
        # Robot-body friction. terrain_config() above is a bare TerrainConfig(),
        # i.e. combine_mode=AVERAGE against ground friction 1.0, so the effective
        # range is (robot + 1.0) / 2 = (0.65, 1.3) rather than the (0.3, 1.6)
        # written here. That compression is wanted: soma-bones has never seen a
        # friction distribution at all, and this is a fine-tune, not a fresh run.
        # To get the full G1 spread instead, give terrain_config() a
        # TerrainSimConfig(..., combine_mode=CombineMode.MULTIPLY) — at the
        # nominal 1.0 ground both modes agree, so that swap changes only the
        # width of the distribution, not its centre.
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
        # The box itself. This is the half of the recipe that serves the task
        # rather than the transfer: without it the policy is free to memorise one
        # 3 kg, friction-1.0 box, which is exactly the failure mode
        # `create_box_scene.py --friction` warns about ("grasp reliability rides
        # on this value"). Ranges are ABSOLUTE and override the scene's
        # ObjectOptions, so they are written around that script's defaults
        # (mass 3.0, friction 1.0) rather than as multipliers.
        #
        # Deliberately narrower than the robot-side ranges: mass and grip
        # friction sit directly in the reward path, and box_pos_fine at
        # sigma=0.1 has no gradient left past ~20cm of error, so a box that
        # cannot be held at all just produces a dead reward signal.
        object_assets=ObjectAssetDomainRandomizationConfig(
            num_buckets=64,
            mass_range=(2.0, 4.5),
            static_friction_range=(0.7, 1.3),
            dynamic_friction_range=(0.7, 1.3),
            restitution_range=(0.0, 0.15),
            center_of_mass_range={
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
            },
        ),
        # Off on purpose — see the module docstring. Copying the G1 block here
        # would be a silent no-op: its dof_*/anchor_* fields never reach a
        # max-coords observation.
        observation_noise=None,
        # Off for the same reason as the G1 file: a shove every 1-3s rips the box
        # out of the hands mid-grasp.
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
    """Apply evaluation-specific overrides."""
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0

    # Watch the nominal 3 kg / friction-1.0 box on nominal physics, not a
    # randomized draw. Also what deployment/trace_tracker_isaaclab.py requires
    # before it will trace a checkpoint: it refuses while domain_randomization
    # is not None.
    simulator_cfg.domain_randomization = None
