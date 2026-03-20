"""Tests for Imager (stereo width) module."""

import pytest
from modules.master.imager import Imager, IMAGER_PRESETS


class TestImagerInit:
    def test_init(self):
        img = Imager()
        assert img is not None

    def test_default_width(self):
        img = Imager()
        assert img.width == 100


class TestImagerParams:
    def test_set_width(self):
        img = Imager()
        img.width = 150
        assert img.width == 150

    def test_set_mono_bass(self):
        img = Imager()
        img.mono_bass_freq = 200
        assert img.mono_bass_freq == 200

    def test_set_balance(self):
        img = Imager()
        img.balance = 0.5
        assert img.balance == 0.5

    def test_enabled(self):
        img = Imager()
        img.enabled = False
        assert img.enabled is False


class TestImagerPresets:
    def test_presets_exist(self):
        assert len(IMAGER_PRESETS) >= 3

    def test_load_preset(self):
        img = Imager()
        img.load_preset("Wide Master")
