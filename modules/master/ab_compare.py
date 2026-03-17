"""
A/B comparison with instant toggle for mastering.

Story 4.5 — Epic 4: Pro Mastering.

Features:
    - Instant A/B toggle between processed and unprocessed audio
    - Real-time switching during playback (no gap/glitch)
    - Loudness-matched comparison to avoid volume bias
    - State management for the comparison toggle
    - Meter snapshots for side-by-side display
    - Integration with RealtimeMonitor bypass toggle
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

from .realtime_monitor import MeterData, RealtimeMonitor


# ---------------------------------------------------------------------------
# AB state
# ---------------------------------------------------------------------------
class ABState:
    """Holds captured meter snapshots for A and B states."""

    def __init__(self) -> None:
        self.lufs: float = -70.0
        self.true_peak: float = -70.0
        self.lra: float = 0.0
        self.rms_l: float = -70.0
        self.rms_r: float = -70.0
        self.timestamp: float = 0.0

    def capture(self, meter: MeterData) -> None:
        self.lufs = meter.integrated_lufs
        self.true_peak = max(meter.true_peak_l, meter.true_peak_r)
        self.lra = meter.lra
        self.rms_l = meter.left_rms_db
        self.rms_r = meter.right_rms_db
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, float]:
        return {
            "lufs": self.lufs,
            "true_peak": self.true_peak,
            "lra": self.lra,
            "rms_l": self.rms_l,
            "rms_r": self.rms_r,
        }


# ---------------------------------------------------------------------------
# ABComparison
# ---------------------------------------------------------------------------
class ABComparison:
    """
    A/B comparison controller for mastering.

    State A = processed (mastering chain active)
    State B = bypassed (dry / original audio)

    Supports loudness-matched comparison: when enabled, the processed
    signal is gain-adjusted so that A and B have the same perceived
    loudness, removing volume bias from the comparison.
    """

    def __init__(self, monitor: Optional[RealtimeMonitor] = None) -> None:
        self._monitor = monitor
        self._is_state_a: bool = True       # True = processed, False = bypassed
        self._loudness_match: bool = False
        self._match_gain_db: float = 0.0
        self._state_a = ABState()
        self._state_b = ABState()
        self._on_toggle: Optional[Callable[[bool], None]] = None

    # -- public API --------------------------------------------------------

    @property
    def is_state_a(self) -> bool:
        """True when listening to the processed (A) state."""
        return self._is_state_a

    @property
    def current_label(self) -> str:
        return "A" if self._is_state_a else "B"

    @property
    def loudness_matched(self) -> bool:
        return self._loudness_match

    @loudness_matched.setter
    def loudness_matched(self, value: bool) -> None:
        self._loudness_match = value
        self._recalculate_match_gain()

    @property
    def match_gain_db(self) -> float:
        """Gain applied to the processed signal to match loudness of bypass."""
        return self._match_gain_db

    @property
    def state_a(self) -> ABState:
        return self._state_a

    @property
    def state_b(self) -> ABState:
        return self._state_b

    def set_toggle_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback invoked on toggle. Receives True if now in state A."""
        self._on_toggle = callback

    def toggle(self) -> bool:
        """
        Toggle between A and B states.

        Returns True if now in state A (processed).
        """
        self._is_state_a = not self._is_state_a

        if self._monitor is not None:
            self._monitor.is_bypassed = not self._is_state_a

        if self._on_toggle is not None:
            self._on_toggle(self._is_state_a)

        return self._is_state_a

    def set_state(self, is_a: bool) -> None:
        """Explicitly set the state."""
        self._is_state_a = is_a
        if self._monitor is not None:
            self._monitor.is_bypassed = not is_a
        if self._on_toggle is not None:
            self._on_toggle(is_a)

    def capture_a(self, meter: MeterData) -> None:
        """Capture current meter data as state A snapshot."""
        self._state_a.capture(meter)
        self._recalculate_match_gain()

    def capture_b(self, meter: MeterData) -> None:
        """Capture current meter data as state B snapshot."""
        self._state_b.capture(meter)
        self._recalculate_match_gain()

    def get_comparison(self) -> Dict:
        """Return side-by-side comparison of A and B states."""
        delta_lufs = self._state_a.lufs - self._state_b.lufs
        delta_tp = self._state_a.true_peak - self._state_b.true_peak

        return {
            "current": "A" if self._is_state_a else "B",
            "state_a": self._state_a.to_dict(),
            "state_b": self._state_b.to_dict(),
            "delta_lufs": round(delta_lufs, 1),
            "delta_true_peak": round(delta_tp, 1),
            "loudness_matched": self._loudness_match,
            "match_gain_db": round(self._match_gain_db, 2),
        }

    def reset(self) -> None:
        """Reset comparison state."""
        self._state_a = ABState()
        self._state_b = ABState()
        self._match_gain_db = 0.0
        self._is_state_a = True
        if self._monitor is not None:
            self._monitor.is_bypassed = False

    # -- internal ----------------------------------------------------------

    def _recalculate_match_gain(self) -> None:
        """
        Recalculate the gain offset for loudness-matched comparison.

        The processed signal (A) is typically louder due to limiting/maximizing.
        We reduce A by the LUFS difference so the comparison is fair.
        """
        if not self._loudness_match:
            self._match_gain_db = 0.0
            return

        if self._state_a.lufs > -60 and self._state_b.lufs > -60:
            self._match_gain_db = self._state_b.lufs - self._state_a.lufs
        else:
            self._match_gain_db = 0.0
