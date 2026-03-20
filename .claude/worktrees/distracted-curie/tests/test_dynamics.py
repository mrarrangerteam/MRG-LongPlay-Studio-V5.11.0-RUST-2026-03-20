"""Tests for Dynamics (compressor) module."""

import pytest
from modules.master.dynamics import Dynamics, DYNAMICS_PRESETS


class TestDynamicsInit:
    def test_init(self):
        d = Dynamics()
        assert d is not None

    def test_default_threshold(self):
        d = Dynamics()
        assert d.single_band.threshold < 0

    def test_default_ratio(self):
        d = Dynamics()
        assert d.single_band.ratio >= 1.0


class TestDynamicsPresets:
    def test_presets_exist(self):
        assert len(DYNAMICS_PRESETS) >= 5

    def test_preset_keys(self):
        for name, preset in DYNAMICS_PRESETS.items():
            assert "threshold" in preset
            assert "ratio" in preset

    def test_load_preset(self):
        d = Dynamics()
        d.load_preset("Standard Master")


class TestDynamicsParams:
    def test_set_threshold(self):
        d = Dynamics()
        d.single_band.threshold = -20.0
        assert d.single_band.threshold == -20.0

    def test_set_ratio(self):
        d = Dynamics()
        d.single_band.ratio = 4.0
        assert d.single_band.ratio == 4.0

    def test_multiband(self):
        d = Dynamics()
        d.multiband = True
        assert d.multiband is True

    def test_enabled(self):
        d = Dynamics()
        d.enabled = False
        assert d.enabled is False
