# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate keyboard capture for the MuJoCo passive viewer.

MuJoCo owns keyboard focus for its viewer and reserves many keys for camera
and UI commands. This module keeps application controls in a small Tk window,
so the simulator can forward raw transitions without competing with viewer
callbacks.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import string
from typing import List, Tuple


_CLOSE_EVENT = "__window_closed__"
_ERROR_EVENT = "__window_error__"
_KEYS = frozenset(string.ascii_letters + string.digits + string.punctuation + " ")


def _printable_key(char: str, keysym: str) -> str:
    """Return the one-character key represented by a Tk event.

    Tk normally fills ``event.char`` for key presses, but key-release events
    often expose only ``event.keysym``. Falling back to ``keysym`` prevents a
    dropped release from latching a key and suppressing later clicks.
    """
    key = (char or keysym or "").lower()
    if len(key) != 1 or key not in _KEYS:
        return ""
    return key


def _run_keyboard_window(events, ready) -> None:
    """Run the Tk event loop in a process that owns its GUI resources."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.title("ProtoMotions MuJoCo controls")
        root.geometry("420x100")
        root.resizable(False, False)
        label = tk.Label(
            root,
            text=(
                "Click here, then press your registered ProtoMotions keys.\n"
                "This window owns task keys; MuJoCo remains visible for viewing."
            ),
            padx=12,
            pady=12,
        )
        label.pack(fill=tk.BOTH, expand=True)

        def emit(event, pressed: bool) -> None:
            key = _printable_key(
                getattr(event, "char", ""), getattr(event, "keysym", "")
            )
            if not key:
                return
            try:
                events.put_nowait((key, pressed))
            except queue.Full:
                # A held key can generate OS repeat events. Dropping a
                # repeated transition is preferable to blocking the GUI.
                pass

        root.bind_all("<KeyPress>", lambda event: emit(event, True))
        root.bind_all("<KeyRelease>", lambda event: emit(event, False))

        def close() -> None:
            try:
                events.put_nowait((_CLOSE_EVENT, False))
            except queue.Full:
                pass
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", close)
        # Keep task-key focus within this Tk application without grabbing the
        # entire desktop. A global grab would also intercept window-manager
        # shortcuts and make the user's session appear frozen.
        root.deiconify()
        root.update_idletasks()
        root.update()
        root.grab_set()
        root.focus_force()
        ready.set()
        root.mainloop()
    except BaseException as exc:
        try:
            events.put_nowait((_ERROR_EVENT, repr(exc)))
        except queue.Full:
            pass
        ready.set()


class MujocoKeyboardWindow:
    """Own a separate process that captures printable key transitions."""

    def __init__(self) -> None:
        context = mp.get_context("spawn")
        self._events = context.Queue(maxsize=256)
        self._ready = context.Event()
        self._process = context.Process(
            target=_run_keyboard_window,
            args=(self._events, self._ready),
            daemon=True,
        )
        self._process.start()
        if not self._ready.wait(timeout=5.0):
            self.close()
            raise RuntimeError("MuJoCo keyboard window did not start")
        if not self._process.is_alive():
            error = self._read_startup_error()
            self.close()
            raise RuntimeError(f"MuJoCo keyboard window failed to start: {error}")

    def _read_startup_error(self) -> str:
        try:
            event = self._events.get_nowait()
        except queue.Empty:
            return "keyboard window process exited"
        if event[0] == _ERROR_EVENT:
            return event[1]
        return repr(event)

    def drain_events(self) -> List[Tuple[str, bool]]:
        """Return all queued ``(key, pressed)`` transitions."""
        events: List[Tuple[str, bool]] = []
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break

            key, value = event
            if key == _ERROR_EVENT:
                raise RuntimeError(f"MuJoCo keyboard window failed: {value}")
            if key == _CLOSE_EVENT:
                raise RuntimeError("MuJoCo keyboard window was closed")
            events.append((key, bool(value)))

        if not self._process.is_alive():
            raise RuntimeError("MuJoCo keyboard window exited unexpectedly")
        return events

    def close(self) -> None:
        """Stop the child process and release its IPC resources."""
        process = getattr(self, "_process", None)
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
        events = getattr(self, "_events", None)
        if events is not None:
            events.close()
            events.join_thread()
