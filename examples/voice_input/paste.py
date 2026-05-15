"""Inject text into the focused application via clipboard + Cmd+V."""
from __future__ import annotations

import subprocess
import time

from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
)

# Virtual key code for "v" on US layout.
KEY_V = 9


def set_clipboard(text: str) -> None:
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(text.encode("utf-8"))


def press_cmd_v() -> None:
    down = CGEventCreateKeyboardEvent(None, KEY_V, True)
    up = CGEventCreateKeyboardEvent(None, KEY_V, False)
    CGEventSetFlags(down, kCGEventFlagMaskCommand)
    CGEventSetFlags(up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, down)
    CGEventPost(kCGHIDEventTap, up)


def paste_text(text: str, settle_ms: int = 50) -> None:
    """Copy text to pasteboard and synthesize Cmd+V."""
    if not text:
        return
    set_clipboard(text)
    # Small delay so the receiving app sees the new pasteboard contents.
    time.sleep(settle_ms / 1000.0)
    press_cmd_v()
