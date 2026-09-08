# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Kit-lifecycle wrapper around :mod:`deployment.tracker_runtime`.

This module imports ``omni.timeline`` / ``omni.kit.app``, so it is **only valid
inside a running Kit app**. The standalone CLI
(``deployment/test_tracker_isaacsim.py``) never imports it.

The key inversion versus the CLI: **Kit's timeline drives physics**, so there is
no ``while`` loop, no ``world.step()`` and no ``time.sleep()`` pacing --
real-time comes from ``World(physics_dt=..., rendering_dt=control_dt)``, which
:func:`~deployment.tracker_runtime.build_runtime` already sets up. What is left
for this layer is:

- a subscription to the app's update stream, which does what the CLI loop body
  did *outside* the physics callback: drain deferred clip restarts, and stop the
  timeline when the clip run is finished;
- teardown, so a session can be closed and a new one opened in the same
  long-lived process.

Diagnostic modes are deliberately out of scope here -- ``--control-in-loop``,
``--action-tape``, ``--drive-probe``, ``--dump-physx-properties`` and friends own
their own stepping, which is incompatible with the timeline owning it. They stay
CLI-only; their implementations still live in ``tracker_runtime``.

Usage, from the Isaac Sim Script Editor::

    from protomotions_tracker import TrackerSession, TrackerRuntimeConfig

    s = TrackerSession(TrackerRuntimeConfig(
        onnx="results/soma_sit_1/compiled_models/unified_pipeline.onnx",
        motion="configs/soma_sit_0/soma_sit_motion_seatcontacts.pt",
        scenes_file="configs/soma_sit_0/soma_sit_chair.pt",
        resolved_configs="results/soma_sit_1/resolved_configs_inference.pt",
        robot="soma23", device="cuda:0"))
    s.load()
    s.play()
    ...
    print(s.metrics)
    s.close()
"""

from __future__ import annotations

import logging

import omni.kit.app
import omni.timeline
from isaacsim.core.api import World

from deployment.tracker_runtime import TrackerRuntimeConfig, build_runtime

log = logging.getLogger(__name__)


class TrackerSession:
    """One tracker run, driven by Kit's timeline.

    Lifecycle: :meth:`load` -> :meth:`play` / :meth:`pause` / :meth:`reset` ->
    :meth:`close`. Nothing touches the stage until ``load()``.

    Args:
        cfg: The run description.
        on_done: Called once, with this session, when the configured number of
            clip loops has played out. The timeline is already stopped by then.
    """

    def __init__(self, cfg: TrackerRuntimeConfig, *, on_done=None):
        if cfg.control_in_loop:
            raise ValueError(
                "TrackerSession cannot run control_in_loop=True: that path owns "
                "the stepping loop, and inside Kit the timeline owns it. Note "
                "this is the *CLI's* default and the better-scoring path "
                "(0.0454 rad vs 0.0537), so a session tracks slightly worse "
                "than the equivalent test_tracker_isaacsim.py run. Use the CLI "
                "when that matters."
            )
        self.cfg = cfg
        self.on_done = on_done
        self.runtime = None
        self._timeline = omni.timeline.get_timeline_interface()
        self._update_sub = None
        self._done_fired = False

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> "TrackerSession":
        """Build the run on the current stage and start the sim.

        Reuses the host app's ``World`` when there is one, so this composes with
        whatever else the app has already set up.
        """
        if self.runtime is not None:
            raise RuntimeError("Session already loaded; call close() first.")
        self.runtime = build_runtime(self.cfg, world=World.instance())
        self.runtime.initialize()
        log.info("TrackerSession loaded; call play() to run it.")
        return self

    def play(self) -> None:
        """Hand physics to the timeline and let the policy run."""
        self._require_loaded()
        self.runtime.add_physics_callback()
        if self._update_sub is None:
            # Runs once per rendered frame, i.e. *between* physics steps -- the
            # only place a clip restart is safe (see TrackerPolicy.reset_episode).
            self._update_sub = (
                omni.kit.app.get_app()
                .get_update_event_stream()
                .create_subscription_to_pop(
                    self._on_update, name="protomotions.tracker.session"
                )
            )
        self._timeline.play()

    def pause(self) -> None:
        """Freeze physics. The policy keeps its state; :meth:`play` resumes."""
        self._timeline.pause()

    def stop(self) -> None:
        """Stop the timeline. Physics views are invalidated by Kit on stop."""
        self._timeline.stop()

    def reset(self) -> None:
        """Restart the clip from frame 0 without rebuilding anything."""
        self._require_loaded()
        self.runtime.policy.reset_episode()
        self._done_fired = False

    def close(self) -> None:
        """Stop, unsubscribe and detach the runtime. Safe to call twice."""
        if self._update_sub is not None:
            self._update_sub.unsubscribe()
            self._update_sub = None
        self._timeline.stop()
        if self.runtime is not None:
            self.runtime.teardown()
            self.runtime = None

    # -- reporting -----------------------------------------------------------

    @property
    def metrics(self) -> dict:
        """Live counters and the worst tracking error so far.

        Returns an empty dict before :meth:`load`.
        """
        if self.runtime is None or self.runtime.policy is None:
            return {}
        policy = self.runtime.policy
        return {
            "steps": int(policy._total_steps),
            "loop": int(policy._loop_idx),
            "frame": int(policy._frame_idx),
            "max_ref_err_rad": float(policy.max_ref_err_run),
            "onnx_ms_per_step": policy.total_ort_ms / max(policy._total_steps, 1),
            "done": bool(policy.done),
        }

    def summary(self) -> None:
        """Log the end-of-run stats, as the CLI prints on exit."""
        self._require_loaded()
        self.runtime.policy.log_summary(self.runtime.total_sim_ms)

    # -- internals -----------------------------------------------------------

    def _require_loaded(self) -> None:
        if self.runtime is None:
            raise RuntimeError("Session not loaded; call load() first.")

    def _on_update(self, event) -> None:
        """Per-frame housekeeping: the CLI loop body minus the stepping."""
        if self.runtime is None:
            return
        self.runtime.drain_pending_reset()
        if self.runtime.done and not self._done_fired:
            self._done_fired = True
            self._timeline.stop()
            log.info(f"TrackerSession finished: {self.metrics}")
            if self.on_done is not None:
                self.on_done(self)
