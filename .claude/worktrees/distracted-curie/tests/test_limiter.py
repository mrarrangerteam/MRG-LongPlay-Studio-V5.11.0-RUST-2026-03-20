"""Tests for Look-ahead True Peak Limiter."""

import pytest
import numpy as np
from modules.master.limiter import LookAheadLimiter


class TestLimiterInit:
    def test_init(self):
        lim = LookAheadLimiter()
        assert lim is not None

    def test_init_with_ceiling(self):
        lim = LookAheadLimiter(ceiling_db=-1.0)
        assert lim.ceiling_db == -1.0


class TestLimiterBehavior:
    def test_ceiling_enforcement(self):
        lim = LookAheadLimiter(ceiling_db=-1.0)
        # Create a hot signal
        t = np.linspace(0, 0.1, 4410)
        hot = np.column_stack([np.sin(2 * np.pi * 1000 * t) * 2.0] * 2)
        limited = lim.process(hot, 44100)
        # Peak should be at or below ceiling
        peak = np.max(np.abs(limited))
        peak_db = 20 * np.log10(max(peak, 1e-10))
        assert peak_db <= 0.0  # Should not exceed 0 dBFS

    def test_silence_passthrough(self):
        lim = LookAheadLimiter(ceiling_db=-1.0)
        silence = np.zeros((4410, 2))
        result = lim.process(silence, 44100)
        assert np.allclose(result, 0.0, atol=1e-10)

    def test_quiet_signal_unchanged(self):
        lim = LookAheadLimiter(ceiling_db=-1.0)
        t = np.linspace(0, 0.1, 4410)
        quiet = np.column_stack([np.sin(2 * np.pi * 1000 * t) * 0.1] * 2)
        result = lim.process(quiet, 44100)
        # Quiet signal should pass through nearly unchanged
        diff = np.max(np.abs(result - quiet))
        assert diff < 0.01
