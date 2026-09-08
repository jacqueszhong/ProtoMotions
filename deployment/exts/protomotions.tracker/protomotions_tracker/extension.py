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
"""Kit entry point for ``protomotions.tracker``.

The extension does exactly one thing: make ``import deployment.tracker_*`` work
inside the host app by putting the ProtoMotions repo root on ``sys.path``. It
imports nothing heavy, so enabling it costs nothing -- the driver, torch and
onnxruntime only load when you actually build a session.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import carb.settings
import omni.ext

EXT_ID = "protomotions.tracker"
_REPO_ROOT_SETTING = f"/exts/{EXT_ID}/repoRoot"

log = logging.getLogger(__name__)

#: Repo root this extension added to ``sys.path``, or None. Read by
#: ``protomotions_tracker.__init__`` so the API also works if someone imports it
#: before/without Kit having started the extension.
repo_root: Path | None = None


def resolve_repo_root() -> Path:
    """Locate the ProtoMotions repo root.

    Precedence: the ``exts."protomotions.tracker".repoRoot`` carb setting, then
    this file's location. In-tree the extension lives at
    ``<repo>/deployment/exts/protomotions.tracker/protomotions_tracker/``, hence
    ``parents[4]``.
    """
    override = carb.settings.get_settings().get(_REPO_ROOT_SETTING)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def ensure_on_sys_path() -> Path:
    """Prepend the repo root to ``sys.path`` if absent. Returns the root."""
    global repo_root
    root = resolve_repo_root()
    if not (root / "deployment" / "tracker_runtime.py").is_file():
        raise RuntimeError(
            f"{root} does not look like a ProtoMotions checkout "
            f"(no deployment/tracker_runtime.py). Set the carb setting "
            f"'{_REPO_ROOT_SETTING}' to the repo root."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        log.info(f"protomotions.tracker: added {root} to sys.path")
    repo_root = root
    return root


class ProtoMotionsTrackerExtension(omni.ext.IExt):
    """Library-API-only extension: no menu, no window, no timeline hooks."""

    def on_startup(self, ext_id: str) -> None:
        self._sessions = []
        root = ensure_on_sys_path()
        log.info(
            f"protomotions.tracker started (repo root {root}). "
            "API: from protomotions_tracker import TrackerSession, "
            "TrackerRuntimeConfig"
        )

    def on_shutdown(self) -> None:
        # Close anything the user registered with us so a reload does not leave
        # a physics callback pointing at a dead policy.
        for session in self._sessions:
            try:
                session.close()
            except Exception as e:
                log.warning(f"protomotions.tracker: session close failed: {e}")
        self._sessions.clear()
        global repo_root
        if repo_root is not None and str(repo_root) in sys.path:
            sys.path.remove(str(repo_root))
        repo_root = None

    # -- optional convenience ------------------------------------------------

    def track_session(self, session) -> None:
        """Ask the extension to close ``session`` on shutdown."""
        self._sessions.append(session)
