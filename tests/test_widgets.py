"""Tests for QPainter widget instantiation (no display needed)."""

import pytest
import os
import sys

# Skip if no display available (CI environments)
pytestmark = pytest.mark.skipif(
    os.environ.get("DISPLAY") is None and sys.platform != "darwin",
    reason="No display available"
)


class TestOzoneRotaryKnob:
    def test_import(self):
        from modules.widgets.rotary_knob import OzoneRotaryKnob
        assert OzoneRotaryKnob is not None

    def test_value_range(self):
        from modules.widgets.rotary_knob import OzoneRotaryKnob
        knob = OzoneRotaryKnob(name="TEST", min_val=0, max_val=100, default=50)
        assert knob.value() == 50.0
        knob.setValue(75)
        assert knob.value() == 75.0
        knob.setValue(200)  # Over max — should clamp
        assert knob.value() == 100.0

    def test_default_reset(self):
        from modules.widgets.rotary_knob import OzoneRotaryKnob
        knob = OzoneRotaryKnob(name="TEST", min_val=0, max_val=100, default=50)
        knob.setValue(90)
        knob.mouseDoubleClickEvent(None)  # Reset to default


class TestVectorscopeWidget:
    def test_import(self):
        from modules.widgets.vectorscope import VectorscopeWidget
        assert VectorscopeWidget is not None

    def test_set_audio_data(self):
        import numpy as np
        from modules.widgets.vectorscope import VectorscopeWidget
        vs = VectorscopeWidget()
        left = np.sin(np.linspace(0, 1, 1000))
        right = np.cos(np.linspace(0, 1, 1000))
        vs.set_audio_data(left, right)

    def test_reset(self):
        from modules.widgets.vectorscope import VectorscopeWidget
        vs = VectorscopeWidget()
        vs.reset()


class TestTransferCurveWidget:
    def test_import(self):
        from modules.widgets.transfer_curve import TransferCurveWidget
        assert TransferCurveWidget is not None

    def test_set_params(self):
        from modules.widgets.transfer_curve import TransferCurveWidget
        tc = TransferCurveWidget()
        tc.set_params(threshold=-20, ratio=4.0, knee=6.0, makeup=2.0)


class TestSpectrumAnalyzerWidget:
    def test_import(self):
        from modules.widgets.spectrum_analyzer import SpectrumAnalyzerWidget
        assert SpectrumAnalyzerWidget is not None

    def test_set_audio_data(self):
        import numpy as np
        from modules.widgets.spectrum_analyzer import SpectrumAnalyzerWidget
        sa = SpectrumAnalyzerWidget()
        data = np.sin(np.linspace(0, 10, 8192))
        sa.set_audio_data(data, 44100)

    def test_reset(self):
        from modules.widgets.spectrum_analyzer import SpectrumAnalyzerWidget
        sa = SpectrumAnalyzerWidget()
        sa.reset()


class TestLoudnessHistoryWidget:
    def test_import(self):
        from modules.widgets.loudness_history import LoudnessHistoryWidget
        assert LoudnessHistoryWidget is not None

    def test_append_levels(self):
        from modules.widgets.loudness_history import LoudnessHistoryWidget
        lh = LoudnessHistoryWidget(target_lufs=-14.0)
        lh.append_levels(-14.0, -13.5, -14.2)
        lh.append_levels(-13.0, -12.5, -14.1)
        assert len(lh._momentary) == 2

    def test_reset(self):
        from modules.widgets.loudness_history import LoudnessHistoryWidget
        lh = LoudnessHistoryWidget()
        lh.append_levels(-14.0, -13.5, -14.2)
        lh.reset()
        assert len(lh._momentary) == 0


class TestLogicChannelMeter:
    def test_import(self):
        from modules.master.ui_panel import LogicChannelMeter
        assert LogicChannelMeter is not None

    def test_set_before_after(self):
        from modules.master.ui_panel import LogicChannelMeter
        m = LogicChannelMeter(ceiling_db=-1.0)
        m.set_before(l_peak=-6.0, r_peak=-5.0, l_rms=-12.0, r_rms=-11.0)
        m.set_after(l_peak=-1.5, r_peak=-1.2, l_rms=-8.0, r_rms=-7.5)

    def test_set_ceiling(self):
        from modules.master.ui_panel import LogicChannelMeter
        m = LogicChannelMeter()
        m.set_ceiling(-0.5)
        assert m.ceiling_db == -0.5

    def test_reset(self):
        from modules.master.ui_panel import LogicChannelMeter
        m = LogicChannelMeter()
        m.set_before(l_peak=-3.0, r_peak=-2.0)
        m.reset()


class TestUndoSystem:
    def test_import(self):
        from modules.master.undo import CommandHistory, Command
        assert CommandHistory is not None

    def test_push_undo(self):
        from modules.master.undo import CommandHistory, Command
        h = CommandHistory()
        h.push(Command("maximizer", "gain_db", 0.0, 5.0, "Gain"))
        assert h.can_undo()

    def test_undo_redo(self):
        from modules.master.undo import CommandHistory, Command
        h = CommandHistory()
        h.push(Command("maximizer", "gain_db", 0.0, 5.0, "Gain"))
        cmd = h.undo()
        assert cmd.old_val == 0.0
        assert cmd.new_val == 5.0
        assert h.can_redo()
        cmd2 = h.redo()
        assert cmd2.new_val == 5.0

    def test_max_size(self):
        from modules.master.undo import CommandHistory, Command
        h = CommandHistory()
        for i in range(60):
            h.push(Command("m", "p", i, i + 1))
        assert len(h._undo_stack) == 50


class TestPipelineDialog:
    def test_import(self):
        from modules.master.pipeline import ProductionPipelineDialog
        assert ProductionPipelineDialog is not None
