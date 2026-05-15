"""Global Fn hold/release detection via Quartz CGEventTap.

Requires the running process to have "Input Monitoring" permission
(System Settings → Privacy & Security → Input Monitoring).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from Quartz import (
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetCurrent,
    CFRunLoopRun,
    CFRunLoopStop,
    CGEventGetFlags,
    CGEventTapCreate,
    CGEventTapEnable,
    kCFRunLoopCommonModes,
    kCGEventFlagsChanged,
    kCGEventTapOptionListenOnly,
    kCGHeadInsertEventTap,
    kCGSessionEventTap,
)

# Fn modifier flag (NSEventModifierFlagFunction).
NS_EVENT_MOD_FN = 0x800000

log = logging.getLogger(__name__)


class FnHotkey:
    """Listen for Fn-down / Fn-up via flagsChanged events."""

    def __init__(self, on_press: Callable[[], None], on_release: Callable[[], None]):
        self.on_press = on_press
        self.on_release = on_release
        self._fn_down = False
        self._tap = None
        self._runloop = None
        self._thread: Optional[threading.Thread] = None

    def _callback(self, proxy, event_type, event, refcon):  # noqa: D401
        if event_type != kCGEventFlagsChanged:
            return event
        flags = CGEventGetFlags(event)
        is_fn = bool(flags & NS_EVENT_MOD_FN)
        if is_fn and not self._fn_down:
            self._fn_down = True
            try:
                self.on_press()
            except Exception:
                log.exception("on_press handler raised")
        elif not is_fn and self._fn_down:
            self._fn_down = False
            try:
                self.on_release()
            except Exception:
                log.exception("on_release handler raised")
        return event

    def _run(self) -> None:
        mask = 1 << kCGEventFlagsChanged
        self._tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionListenOnly,
            mask,
            self._callback,
            None,
        )
        if self._tap is None:
            log.error(
                "CGEventTapCreate returned None — likely missing 'Input Monitoring' permission. "
                "Grant access in System Settings → Privacy & Security → Input Monitoring."
            )
            return
        source = CFMachPortCreateRunLoopSource(None, self._tap, 0)
        self._runloop = CFRunLoopGetCurrent()
        CFRunLoopAddSource(self._runloop, source, kCFRunLoopCommonModes)
        CGEventTapEnable(self._tap, True)
        CFRunLoopRun()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="FnHotkey", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._runloop is not None:
            CFRunLoopStop(self._runloop)
        if self._tap is not None:
            CGEventTapEnable(self._tap, False)
        self._thread = None
