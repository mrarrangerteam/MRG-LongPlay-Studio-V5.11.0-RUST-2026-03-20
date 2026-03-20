"""Tests for Equalizer module."""

import pytest
from modules.master.equalizer import Equalizer, EQ_TONE_PRESETS


class TestEqualizerInit:
    def test_init(self):
        eq = Equalizer()
        assert eq is not None

    def test_default_bands(self):
        eq = Equalizer()
        assert len(eq.bands) == 8

    def test_band_default_gain(self):
        eq = Equalizer()
        for band in eq.bands:
            assert band.gain == 0.0


class TestEqualizerPresets:
    def test_tone_presets_exist(self):
        assert len(EQ_TONE_PRESETS) >= 10

    def test_load_preset_flat(self):
        eq = Equalizer()
        eq.load_tone_preset("Flat")
        for band in eq.bands:
            assert band.gain == 0.0

    def test_load_preset_warm(self):
        eq = Equalizer()
        eq.load_tone_preset("Warm")
        # Warm should boost lows
        assert eq.bands[0].gain > 0 or eq.bands[1].gain > 0

    def test_load_invalid_preset(self):
        eq = Equalizer()
        # Should not crash
        eq.load_tone_preset("NonexistentPreset")


class TestEqualizerBands:
    def test_set_band_gain(self):
        eq = Equalizer()
        eq.bands[0].gain = 6.0
        assert eq.bands[0].gain == 6.0

    def test_band_frequency(self):
        eq = Equalizer()
        assert eq.bands[0].freq > 0

    def test_enabled(self):
        eq = Equalizer()
        eq.enabled = False
        assert eq.enabled is False
