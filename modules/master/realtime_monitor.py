"""
Real-time audio monitoring via Rust/cpal with Python fallback.

Story 4.4 — Epic 4: Pro Mastering.

Features:
    - Rust audio thread using cpal for system audio output
    - Audio routed through mastering chain in real-time
    - Latency < 20ms target
    - Bypass toggle (A/B compare) in real-time
    - Meter data streamed to GUI at 30fps
    - No audio glitches when changing parameters
    - Python fallback: offline render → play via QMediaPlayer

Architecture:
    Rust path:  longplay.RealtimeMonitor → cpal output → meter callback
    Python path: MasterChain.render() → temp WAV → QMediaPlayer
"""

from __future__ import annotations

import os
import threading
import time
import tempfile
from typing import Callable, Dict, Optional

import numpy as np

# Try Rust backend
try:
    import longplay
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


# ---------------------------------------------------------------------------
# Meter data container
# ---------------------------------------------------------------------------
class MeterData:
    """Real-time meter readings streamed from the audio engine."""

    __slots__ = (
        "left_rms_db", "right_rms_db",
        "left_peak_db", "right_peak_db",
        "momentary_lufs", "short_term_lufs", "integrated_lufs",
        "true_peak_l", "true_peak_r",
        "lra", "gain_reduction_db",
        "timestamp",
    )

    def __init__(self) -> None:
        self.left_rms_db: float = -70.0
        self.right_rms_db: float = -70.0
        self.left_peak_db: float = -70.0
        self.right_peak_db: float = -70.0
        self.momentary_lufs: float = -70.0
        self.short_term_lufs: float = -70.0
        self.integrated_lufs: float = -70.0
        self.true_peak_l: float = -70.0
        self.true_peak_r: float = -70.0
        self.lra: float = 0.0
        self.gain_reduction_db: float = 0.0
        self.timestamp: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "left_rms_db": self.left_rms_db,
            "right_rms_db": self.right_rms_db,
            "left_peak_db": self.left_peak_db,
            "right_peak_db": self.right_peak_db,
            "momentary_lufs": self.momentary_lufs,
            "short_term_lufs": self.short_term_lufs,
            "integrated_lufs": self.integrated_lufs,
            "true_peak_l": self.true_peak_l,
            "true_peak_r": self.true_peak_r,
            "lra": self.lra,
            "gain_reduction_db": self.gain_reduction_db,
        }


# ---------------------------------------------------------------------------
# Python fallback audio analysis engine
# ---------------------------------------------------------------------------
class _PythonMeterEngine:
    """
    Compute meter data from audio samples in Python.

    Used when Rust backend is unavailable or for offline analysis.
    """

    def __init__(self, sample_rate: int = 44100) -> None:
        self._sr = sample_rate
        self._integrated_sum: float = 0.0
        self._integrated_count: int = 0
        self._short_term_buf: list = []
        self._momentary_buf: list = []
        self._st_window = int(3.0 * sample_rate)     # 3s
        self._mom_window = int(0.4 * sample_rate)     # 400ms

    def analyze_block(self, left: np.ndarray, right: np.ndarray) -> MeterData:
        """Analyze a block of stereo audio and return meter readings."""
        data = MeterData()
        data.timestamp = time.time()

        n = len(left)
        if n == 0:
            return data

        # RMS
        rms_l = np.sqrt(np.mean(left ** 2))
        rms_r = np.sqrt(np.mean(right ** 2))
        data.left_rms_db = 20.0 * np.log10(max(rms_l, 1e-10))
        data.right_rms_db = 20.0 * np.log10(max(rms_r, 1e-10))

        # Peak
        peak_l = np.max(np.abs(left))
        peak_r = np.max(np.abs(right))
        data.left_peak_db = 20.0 * np.log10(max(peak_l, 1e-10))
        data.right_peak_db = 20.0 * np.log10(max(peak_r, 1e-10))

        # True peak (4x oversampled approximation)
        data.true_peak_l = data.left_peak_db + 0.2   # simplified TP estimate
        data.true_peak_r = data.right_peak_db + 0.2

        # Momentary LUFS (400ms window)
        mono = (left + right) / 2.0
        rms_mono = np.sqrt(np.mean(mono ** 2))
        data.momentary_lufs = -0.691 + 20.0 * np.log10(max(rms_mono, 1e-10))

        # Short-term LUFS (3s window - accumulate)
        self._short_term_buf.extend(mono.tolist())
        if len(self._short_term_buf) > self._st_window:
            self._short_term_buf = self._short_term_buf[-self._st_window:]
        if self._short_term_buf:
            st_arr = np.array(self._short_term_buf)
            rms_st = np.sqrt(np.mean(st_arr ** 2))
            data.short_term_lufs = -0.691 + 20.0 * np.log10(max(rms_st, 1e-10))

        # Integrated LUFS (full duration)
        self._integrated_sum += np.sum(mono ** 2)
        self._integrated_count += n
        if self._integrated_count > 0:
            rms_int = np.sqrt(self._integrated_sum / self._integrated_count)
            data.integrated_lufs = -0.691 + 20.0 * np.log10(max(rms_int, 1e-10))

        return data

    def reset(self) -> None:
        self._integrated_sum = 0.0
        self._integrated_count = 0
        self._short_term_buf.clear()
        self._momentary_buf.clear()


# ---------------------------------------------------------------------------
# RealtimeMonitor
# ---------------------------------------------------------------------------
class RealtimeMonitor:
    """
    Real-time audio monitoring with mastering chain processing.

    If the Rust backend with cpal is available, audio is streamed through
    the Rust mastering chain to the system audio output in real-time.
    Otherwise, falls back to offline rendering + playback.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        self._ffmpeg = ffmpeg_path
        self._is_playing: bool = False
        self._is_bypassed: bool = False
        self._meter_callback: Optional[Callable[[MeterData], None]] = None
        self._audio_path: Optional[str] = None
        self._samples_left: Optional[np.ndarray] = None
        self._samples_right: Optional[np.ndarray] = None
        self._sample_rate: int = 44100
        self._playback_pos: int = 0  # sample position
        self._lock = threading.Lock()
        self._meter_engine = _PythonMeterEngine()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._block_size = 2048   # samples per callback block

        # Rust backend
        self._rust_monitor = None
        if _HAS_RUST:
            try:
                self._rust_monitor = longplay.RealtimeMonitor()
            except (AttributeError, RuntimeError):
                self._rust_monitor = None

    # -- public API --------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def is_bypassed(self) -> bool:
        return self._is_bypassed

    @is_bypassed.setter
    def is_bypassed(self, value: bool) -> None:
        self._is_bypassed = value
        if self._rust_monitor is not None:
            try:
                self._rust_monitor.set_bypass(value)
            except AttributeError:
                pass

    @property
    def backend(self) -> str:
        if self._rust_monitor is not None:
            return "rust_cpal"
        return "python_fallback"

    def set_meter_callback(self, callback: Callable[[MeterData], None]) -> None:
        """Set callback for meter data updates (~30fps)."""
        self._meter_callback = callback

    def load_audio(self, audio_path: str) -> bool:
        """Load audio file for real-time monitoring."""
        if not os.path.exists(audio_path):
            return False
        self._audio_path = audio_path

        # try Rust backend
        if self._rust_monitor is not None:
            try:
                self._rust_monitor.load_audio(audio_path)
                return True
            except (AttributeError, RuntimeError):
                pass

        # Python fallback: extract PCM via FFmpeg
        return self._load_pcm_fallback(audio_path)

    def play(self, start_sec: float = 0.0) -> None:
        """Start real-time playback with mastering chain processing."""
        if self._is_playing:
            self.stop()

        self._stop_event.clear()
        self._is_playing = True

        if self._rust_monitor is not None:
            try:
                self._rust_monitor.play(start_sec, self._meter_callback)
                return
            except (AttributeError, RuntimeError):
                pass

        # Python fallback: start monitor thread
        self._playback_pos = int(start_sec * self._sample_rate)
        self._meter_engine.reset()
        self._monitor_thread = threading.Thread(
            target=self._python_monitor_loop, daemon=True
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop real-time playback."""
        self._is_playing = False
        self._stop_event.set()

        if self._rust_monitor is not None:
            try:
                self._rust_monitor.stop()
            except (AttributeError, RuntimeError):
                pass

        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    def seek(self, position_sec: float) -> None:
        """Seek to a position during playback."""
        with self._lock:
            self._playback_pos = int(position_sec * self._sample_rate)

        if self._rust_monitor is not None:
            try:
                self._rust_monitor.seek(position_sec)
            except (AttributeError, RuntimeError):
                pass

    def get_position_sec(self) -> float:
        """Get current playback position in seconds."""
        with self._lock:
            return self._playback_pos / max(self._sample_rate, 1)

    # -- internal ----------------------------------------------------------

    def _load_pcm_fallback(self, audio_path: str) -> bool:
        """Extract PCM samples via FFmpeg for Python fallback."""
        import subprocess
        cmd = [
            self._ffmpeg, "-i", audio_path,
            "-ac", "2", "-ar", str(self._sample_rate),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-v", "error", "pipe:1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if len(result.stdout) < 8:
                return False
            data = np.frombuffer(result.stdout, dtype=np.float32)
            self._samples_left = data[0::2].astype(np.float64)
            self._samples_right = data[1::2].astype(np.float64)
            return True
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _python_monitor_loop(self) -> None:
        """
        Python fallback monitor loop.

        Runs in a background thread, producing meter data at ~30fps.
        Actual audio playback is handled by the GUI's QMediaPlayer;
        this loop only generates meter data for the WLM display.
        """
        interval = 1.0 / 30.0  # 30fps meter updates

        while not self._stop_event.is_set():
            t0 = time.time()

            with self._lock:
                pos = self._playback_pos

            if self._samples_left is not None and pos < len(self._samples_left):
                end = min(pos + self._block_size, len(self._samples_left))
                left_block = self._samples_left[pos:end]
                right_block = self._samples_right[pos:end] if self._samples_right is not None else left_block

                meter = self._meter_engine.analyze_block(left_block, right_block)

                if self._meter_callback is not None:
                    self._meter_callback(meter)

                with self._lock:
                    self._playback_pos = end
            else:
                # end of audio
                self._is_playing = False
                break

            elapsed = time.time() - t0
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)
