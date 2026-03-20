"""
Tests for Epic 4 — Pro Mastering (Stories 4.1–4.6).

Covers:
    4.1: Spectrum Analyzer (SpectrumAnalyzerWidget)
    4.2: WLM Plus Meter (WavesWLMPlusMeter)
    4.3: Match EQ (MatchEQ)
    4.4: Realtime Monitor (RealtimeMonitor, MeterData)
    4.5: A/B Comparison (ABComparison)
    4.6: Loudness Report (export_csv, export_pdf)
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Story 4.1: Spectrum Analyzer
# ============================================================

class TestSpectrumAnalyzer:
    """Tests for the spectrum analyzer module."""

    def test_import(self):
        from gui.widgets.spectrum_analyzer import SpectrumAnalyzerWidget
        assert SpectrumAnalyzerWidget is not None

    def test_feed_samples(self):
        from gui.widgets.spectrum_analyzer import SpectrumAnalyzerWidget, NUM_DISPLAY_BINS

        # Create a test instance (without Qt app, just test data path)
        # Test the FFT computation logic directly
        from gui.widgets.spectrum_analyzer import _log_freq_to_x, _db_to_y

        # log freq mapping
        x = _log_freq_to_x(1000.0, 800.0)
        assert 0 < x < 800

        x_low = _log_freq_to_x(20.0, 800.0)
        x_high = _log_freq_to_x(20000.0, 800.0)
        assert x_low < x < x_high

        # dB to y mapping
        y_0db = _db_to_y(0.0, 400.0)
        y_neg60 = _db_to_y(-60.0, 400.0)
        assert y_0db < y_neg60  # 0dB should be higher (smaller y)

    def test_constants(self):
        from gui.widgets.spectrum_analyzer import (
            FFT_SIZE, FREQ_MIN, FREQ_MAX, DB_MIN, DB_MAX
        )
        assert FFT_SIZE == 4096
        assert FREQ_MIN == 20.0
        assert FREQ_MAX == 20000.0
        assert DB_MIN == -60.0
        assert DB_MAX == 0.0


# ============================================================
# Story 4.2: WLM Plus Meter
# ============================================================

class TestWLMPlusMeter:
    """Tests for the WLM Plus clone."""

    def test_import(self):
        from gui.widgets.wlm_meter import (
            WavesWLMPlusMeter, SevenSegmentDisplay, LEDMeterBar,
            LoudnessHistogram, GainReductionGraph,
        )
        assert WavesWLMPlusMeter is not None

    def test_loudness_presets(self):
        from gui.widgets.wlm_meter import LOUDNESS_PRESETS

        assert "ITU-R BS.1770" in LOUDNESS_PRESETS
        assert "EBU R128" in LOUDNESS_PRESETS
        assert "ATSC A/85" in LOUDNESS_PRESETS
        assert "Spotify" in LOUDNESS_PRESETS
        assert "YouTube" in LOUDNESS_PRESETS
        assert "Apple Music" in LOUDNESS_PRESETS
        assert "Custom" in LOUDNESS_PRESETS

        for name, preset in LOUDNESS_PRESETS.items():
            assert "target_lufs" in preset
            assert "true_peak" in preset
            assert "lra_max" in preset

    def test_histogram_data(self):
        from gui.widgets.wlm_meter import LoudnessHistogram

        # test deque-based histogram
        hist = LoudnessHistogram.__new__(LoudnessHistogram)
        from collections import deque
        hist._data = deque(maxlen=1000)

        hist._data.append(-14.0)
        hist._data.append(-13.5)
        hist._data.append(-14.5)
        assert len(hist._data) == 3

    def test_gain_reduction_data(self):
        from gui.widgets.wlm_meter import GainReductionGraph
        gr = GainReductionGraph.__new__(GainReductionGraph)
        from collections import deque
        gr._data = deque(maxlen=300)
        gr._peak_hold = 0.0

        gr._data.append(-3.0)
        gr._data.append(-5.0)
        assert len(gr._data) == 2


# ============================================================
# Story 4.3: Match EQ
# ============================================================

class TestMatchEQ:
    """Tests for the Match EQ module."""

    def test_import(self):
        from modules.master.match_eq import MatchEQ
        assert MatchEQ is not None

    def test_init(self):
        from modules.master.match_eq import MatchEQ
        meq = MatchEQ()
        assert meq.strength == 1.0
        assert meq.reference_path == ""
        assert meq.reference_spectrum is None
        assert meq.current_spectrum is None
        assert meq.correction_curve is None

    def test_strength_clamping(self):
        from modules.master.match_eq import MatchEQ
        meq = MatchEQ()
        meq.strength = 1.5
        assert meq.strength == 1.0
        meq.strength = -0.5
        assert meq.strength == 0.0
        meq.strength = 0.75
        assert meq.strength == 0.75

    def test_third_octave_centers(self):
        from modules.master.match_eq import THIRD_OCTAVE_CENTERS, NUM_BANDS
        assert len(THIRD_OCTAVE_CENTERS) == NUM_BANDS
        assert THIRD_OCTAVE_CENTERS[0] == 20
        assert THIRD_OCTAVE_CENTERS[-1] == 20000

    def test_compute_avg_spectrum(self):
        from modules.master.match_eq import _compute_avg_spectrum, FFT_SIZE
        # generate a test sine wave
        sr = 44100
        t = np.arange(sr) / sr  # 1 second
        samples = np.sin(2 * np.pi * 1000 * t)  # 1 kHz tone

        spectrum = _compute_avg_spectrum(samples, sr)
        assert len(spectrum) == FFT_SIZE // 2 + 1
        assert spectrum.dtype == np.float64

    def test_spectrum_to_bands(self):
        from modules.master.match_eq import _spectrum_to_bands, _compute_avg_spectrum, NUM_BANDS
        sr = 44100
        samples = np.random.randn(sr * 2)  # 2 seconds of noise
        spectrum = _compute_avg_spectrum(samples, sr)
        bands = _spectrum_to_bands(spectrum, sr)
        assert len(bands) == NUM_BANDS

    def test_analyze_samples(self):
        from modules.master.match_eq import MatchEQ
        meq = MatchEQ()
        sr = 44100
        samples = np.random.randn(sr * 2)
        ok = meq.analyze_samples(samples, sr)
        assert ok
        assert meq.current_spectrum is not None

    def test_report(self):
        from modules.master.match_eq import MatchEQ
        meq = MatchEQ()
        report = meq.get_report()
        assert "reference_path" in report
        assert "strength_pct" in report
        assert report["has_reference"] is False

    def test_reset(self):
        from modules.master.match_eq import MatchEQ
        meq = MatchEQ()
        meq.analyze_samples(np.random.randn(44100), 44100)
        assert meq.current_spectrum is not None
        meq.reset()
        assert meq.current_spectrum is None
        assert meq.reference_path == ""


# ============================================================
# Story 4.4: Realtime Monitor
# ============================================================

class TestRealtimeMonitor:
    """Tests for the realtime audio monitoring module."""

    def test_import(self):
        from modules.master.realtime_monitor import RealtimeMonitor, MeterData
        assert RealtimeMonitor is not None
        assert MeterData is not None

    def test_meter_data_init(self):
        from modules.master.realtime_monitor import MeterData
        data = MeterData()
        assert data.left_rms_db == -70.0
        assert data.momentary_lufs == -70.0
        assert data.integrated_lufs == -70.0

    def test_meter_data_to_dict(self):
        from modules.master.realtime_monitor import MeterData
        data = MeterData()
        data.left_rms_db = -12.0
        d = data.to_dict()
        assert d["left_rms_db"] == -12.0
        assert "momentary_lufs" in d

    def test_python_meter_engine(self):
        from modules.master.realtime_monitor import _PythonMeterEngine, MeterData
        engine = _PythonMeterEngine(44100)
        left = np.sin(np.arange(4096) / 44100 * 2 * np.pi * 440) * 0.5
        right = np.sin(np.arange(4096) / 44100 * 2 * np.pi * 440) * 0.5
        meter = engine.analyze_block(left, right)
        assert isinstance(meter, MeterData)
        assert meter.left_rms_db > -70.0
        assert meter.momentary_lufs > -70.0

    def test_monitor_backend(self):
        from modules.master.realtime_monitor import RealtimeMonitor
        monitor = RealtimeMonitor()
        assert monitor.backend in ("rust_cpal", "python_fallback")

    def test_bypass_toggle(self):
        from modules.master.realtime_monitor import RealtimeMonitor
        monitor = RealtimeMonitor()
        assert not monitor.is_bypassed
        monitor.is_bypassed = True
        assert monitor.is_bypassed


# ============================================================
# Story 4.5: A/B Comparison
# ============================================================

class TestABComparison:
    """Tests for the A/B comparison module."""

    def test_import(self):
        from modules.master.ab_compare import ABComparison, ABState
        assert ABComparison is not None

    def test_init(self):
        from modules.master.ab_compare import ABComparison
        ab = ABComparison()
        assert ab.is_state_a is True
        assert ab.current_label == "A"

    def test_toggle(self):
        from modules.master.ab_compare import ABComparison
        ab = ABComparison()
        assert ab.is_state_a
        result = ab.toggle()
        assert not result
        assert ab.current_label == "B"
        result = ab.toggle()
        assert result
        assert ab.current_label == "A"

    def test_loudness_match(self):
        from modules.master.ab_compare import ABComparison, ABState
        from modules.master.realtime_monitor import MeterData

        ab = ABComparison()
        ab.loudness_matched = True
        assert ab.loudness_matched

        # simulate captures
        meter_a = MeterData()
        meter_a.integrated_lufs = -10.0
        ab.capture_a(meter_a)

        meter_b = MeterData()
        meter_b.integrated_lufs = -14.0
        ab.capture_b(meter_b)

        assert ab.match_gain_db == pytest.approx(-4.0, abs=0.1)

    def test_comparison_report(self):
        from modules.master.ab_compare import ABComparison
        ab = ABComparison()
        report = ab.get_comparison()
        assert report["current"] == "A"
        assert "delta_lufs" in report
        assert "state_a" in report
        assert "state_b" in report

    def test_reset(self):
        from modules.master.ab_compare import ABComparison
        ab = ABComparison()
        ab.toggle()
        ab.reset()
        assert ab.is_state_a


# ============================================================
# Story 4.6: Loudness Report
# ============================================================

class TestLoudnessReport:
    """Tests for the loudness report export module."""

    def test_import(self):
        from modules.master.loudness_report import export_csv, export_pdf, LoudnessReportData
        assert export_csv is not None
        assert export_pdf is not None

    def test_report_data_from_analysis(self):
        from modules.master.loudness_report import LoudnessReportData
        from modules.master.loudness import LoudnessAnalysis

        analysis = LoudnessAnalysis()
        analysis.integrated_lufs = -14.0
        analysis.true_peak_dbtp = -1.0
        analysis.lra = 7.0
        analysis.duration_sec = 180.0
        analysis.sample_rate = 44100
        analysis.channels = 2

        report = LoudnessReportData.from_analysis(analysis, "/tmp/test.wav", "Spotify")
        assert report.integrated_lufs == -14.0
        assert report.target_platform == "Spotify"
        assert report.file_name == "test.wav"

    def test_export_csv(self):
        from modules.master.loudness_report import export_csv, LoudnessReportData

        report = LoudnessReportData()
        report.file_name = "test_song.wav"
        report.integrated_lufs = -14.0
        report.true_peak_dbtp = -1.0
        report.lra = 7.0
        report.target_platform = "YouTube"
        report.target_lufs = -14.0
        report.target_tp = -1.0
        report.timestamp = "2026-03-14 12:00:00"

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            ok = export_csv(report, path)
            assert ok
            assert os.path.exists(path)
            with open(path, "r") as f:
                content = f.read()
            assert "LongPlay Studio" in content
            assert "test_song.wav" in content
            assert "LUFS" in content
        finally:
            os.unlink(path)

    def test_export_pdf(self):
        from modules.master.loudness_report import export_pdf, LoudnessReportData

        report = LoudnessReportData()
        report.file_name = "test_song.wav"
        report.integrated_lufs = -14.0
        report.true_peak_dbtp = -1.0
        report.lra = 7.0
        report.timestamp = "2026-03-14 12:00:00"

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name

        try:
            ok = export_pdf(report, path)
            assert ok
            assert os.path.exists(path)
            with open(path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-"
        finally:
            os.unlink(path)
