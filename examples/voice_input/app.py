"""Menu-bar push-to-talk voice input app for macOS.

Press and hold Fn to record; release to transcribe and paste at the cursor.
"""
from __future__ import annotations

import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np
import rumps

from .audio_capture import AudioCapture, SAMPLE_RATE
from .engine import Engine
from .hotkey import FnHotkey
from .paste import paste_text

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

ICON_IDLE = "🎤"
ICON_RECORDING = "🔴"
ICON_TRANSCRIBING = "✨"

MIN_RECORD_SEC = 0.3  # discard accidental taps shorter than this

# Persist the most recent Fn recording for debugging/observability. Overwritten
# on every Fn release; only the latest is kept.
LAST_RECORDING_PATH = Path(__file__).resolve().parent / "last_recording.wav"


def _save_last_recording(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    if audio.size == 0:
        return
    LAST_RECORDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    tmp = LAST_RECORDING_PATH.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    tmp.replace(LAST_RECORDING_PATH)


class VoiceInputApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Voice", title=ICON_IDLE, quit_button=None)
        self.engine = Engine()
        self.capture = AudioCapture()
        self.hotkey = FnHotkey(on_press=self._on_fn_down, on_release=self._on_fn_up)

        self._record_started: float | None = None
        self._lock = threading.Lock()
        self._last_text = ""

        self.menu = [
            rumps.MenuItem("Status: loading…", callback=None),
            rumps.MenuItem("Last: (none)", callback=None),
            None,
            rumps.MenuItem("Test microphone", callback=self._test_mic),
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]
        self._set_status("loading")
        threading.Thread(target=self._boot, name="boot", daemon=True).start()

    def _boot(self) -> None:
        log.info("loading model…")
        load_t = self.engine.load()
        log.info("model loaded in %.2fs; warming up…", load_t)
        warm_t = self.engine.warmup()
        log.info("warmup done in %.2fs; opening mic…", warm_t)
        self.capture.open()
        log.info("mic open; arming Fn hotkey")
        self.hotkey.start()
        self._set_status("idle")

    # ---- state helpers ----
    def _set_status(self, state: str) -> None:
        text = {
            "loading": "Status: loading…",
            "idle": "Status: idle (hold Fn to record)",
            "recording": "Status: recording…",
            "transcribing": "Status: transcribing…",
        }.get(state, f"Status: {state}")
        for item in self.menu.values():
            if isinstance(item, rumps.MenuItem) and item.title.startswith("Status:"):
                item.title = text
                break
        self.title = {
            "loading": ICON_IDLE,
            "idle": ICON_IDLE,
            "recording": ICON_RECORDING,
            "transcribing": ICON_TRANSCRIBING,
        }.get(state, ICON_IDLE)

    def _set_last_text(self, text: str) -> None:
        self._last_text = text
        snippet = text if len(text) <= 40 else text[:37] + "…"
        for item in self.menu.values():
            if isinstance(item, rumps.MenuItem) and item.title.startswith("Last:"):
                item.title = f"Last: {snippet}" if snippet else "Last: (none)"
                break

    # ---- hotkey handlers (called from Fn watcher thread) ----
    def _on_fn_down(self) -> None:
        with self._lock:
            if self._record_started is not None:
                return
            self._record_started = time.perf_counter()
        self.capture.start()
        self._set_status("recording")
        log.info("Fn down → recording")

    def _on_fn_up(self) -> None:
        with self._lock:
            if self._record_started is None:
                return
            held = time.perf_counter() - self._record_started
            self._record_started = None
        audio = self.capture.stop()
        try:
            _save_last_recording(audio)
        except Exception:
            log.exception("failed to save last recording")
        if held < MIN_RECORD_SEC or len(audio) < int(MIN_RECORD_SEC * SAMPLE_RATE):
            log.info("Fn up after %.2fs (too short); discarding", held)
            self._set_status("idle")
            return
        self._set_status("transcribing")
        threading.Thread(
            target=self._transcribe_and_paste, args=(audio, held), daemon=True
        ).start()

    def _transcribe_and_paste(self, audio, held_sec: float) -> None:
        try:
            text, t_asr = self.engine.transcribe(audio)
        except Exception:
            log.exception("transcribe failed")
            self._set_status("idle")
            return
        log.info(
            "held %.2fs → asr %.2fs (RTF %.2f) → '%s'",
            held_sec,
            t_asr,
            t_asr / max(held_sec, 1e-6),
            text,
        )
        if text.strip():
            paste_text(text)
            self._set_last_text(text)
        self._set_status("idle")

    # ---- menu callbacks ----
    def _test_mic(self, _) -> None:
        rumps.notification(
            "Voice", "Microphone test", "Hold Fn to start recording.", sound=False
        )


def main() -> None:
    VoiceInputApp().run()


if __name__ == "__main__":
    main()
