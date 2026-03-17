"""Tests for LUFS measurement and True Peak detection."""

import pytest
import numpy as np
import tempfile
import os
from modules.master.loudness import LoudnessMeter, LoudnessAnalysis


class TestLoudnessAnalysis:
    def test_init(self):
        a = LoudnessAnalysis()
        assert a is not None

    def test_has_fields(self):
        a = LoudnessAnalysis()
        assert hasattr(a, 'integrated_lufs')
        assert hasattr(a, 'true_peak_dbtp')


class TestLoudnessMeter:
    def test_init(self):
        meter = LoudnessMeter()
        assert meter is not None

    def test_analyze_with_file(self):
        """Test analysis with a real WAV file."""
        import shutil
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not available")

        meter = LoudnessMeter()
        # Create a temporary WAV file with sine wave
        t = np.linspace(0, 1, 44100)
        sine = np.column_stack([np.sin(2 * np.pi * 1000 * t)] * 2) * 0.5

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            sf.write(path, sine, 44100)
            result = meter.analyze(path)
            assert result is not None
            assert hasattr(result, 'integrated_lufs')
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_analyze_silence(self):
        """Test analysis of silent audio."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        meter = LoudnessMeter()
        silence = np.zeros((44100, 2))

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            sf.write(path, silence, 44100)
            result = meter.analyze(path)
            if result:
                assert result.integrated_lufs <= -60.0
        finally:
            if os.path.exists(path):
                os.unlink(path)
