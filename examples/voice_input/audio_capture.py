"""Microphone capture via sounddevice; produces float32 mono 16 kHz numpy array."""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
CHANNELS = 1
BLOCKSIZE = 1024  # ~64 ms per callback at 16 kHz

log = logging.getLogger(__name__)


class AudioCapture:
    """Open a single InputStream, append blocks to a list while recording."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._stream: Optional[sd.InputStream] = None
        self._chunks: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: D401
        if status:
            log.debug("sounddevice status: %s", status)
        with self._lock:
            if self._recording:
                # indata is (frames, channels) float32; squeeze to mono.
                self._chunks.append(indata[:, 0].copy())

    def open(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def start(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._recording = True

    def stop(self) -> np.ndarray:
        with self._lock:
            self._recording = False
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._chunks)
            self._chunks.clear()
        return audio.astype(np.float32, copy=False)
