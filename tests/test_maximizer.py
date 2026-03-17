"""Tests for Maximizer module."""

import pytest
from modules.master.maximizer import Maximizer


class TestMaximizerInit:
    def test_init(self):
        m = Maximizer()
        assert m is not None

    def test_default_gain(self):
        m = Maximizer()
        assert m.gain_db == 0.0

    def test_default_ceiling(self):
        m = Maximizer()
        assert m.ceiling == -1.0

    def test_default_irc_mode(self):
        m = Maximizer()
        assert "IRC" in m.irc_mode


class TestMaximizerGain:
    def test_set_gain(self):
        m = Maximizer()
        m.set_gain(5.0)
        assert m.gain_db == 5.0

    def test_set_gain_zero(self):
        m = Maximizer()
        m.set_gain(0.0)
        assert m.gain_db == 0.0

    def test_set_gain_max(self):
        m = Maximizer()
        m.set_gain(20.0)
        assert m.gain_db == 20.0


class TestMaximizerCeiling:
    def test_set_ceiling(self):
        m = Maximizer()
        m.set_ceiling(-0.5)
        assert m.ceiling == -0.5

    def test_set_ceiling_default(self):
        m = Maximizer()
        m.set_ceiling(-1.0)
        assert m.ceiling == -1.0


class TestMaximizerIRC:
    def test_set_irc_mode(self):
        m = Maximizer()
        m.set_irc_mode("IRC 3")
        assert m.irc_mode == "IRC 3"

    def test_set_irc_with_sub_mode(self):
        m = Maximizer()
        m.set_irc_mode("IRC 3", "Balanced")
        assert m.irc_mode == "IRC 3"
        assert m.irc_sub_mode == "Balanced"

    def test_set_irc_sub_mode_direct(self):
        m = Maximizer()
        m.set_irc_sub_mode("Crisp")
        assert m.irc_sub_mode == "Crisp"

    def test_effective_irc_key(self):
        m = Maximizer()
        m.set_irc_mode("IRC 3", "Pumping")
        key = m.get_effective_irc_key()
        assert "IRC 3" in key


class TestMaximizerCharacter:
    def test_set_character(self):
        m = Maximizer()
        m.set_character(5.0)
        assert m.character == 5.0

    def test_set_character_zero(self):
        m = Maximizer()
        m.set_character(0.0)
        assert m.character == 0.0


class TestMaximizerAdvanced:
    def test_upward_compress(self):
        m = Maximizer()
        m.set_upward_compress(6.0)
        assert m.upward_compress_db == 6.0

    def test_soft_clip(self):
        m = Maximizer()
        m.set_soft_clip(True, 50)
        assert m.soft_clip_enabled is True
        assert m.soft_clip_pct == 50

    def test_true_peak(self):
        m = Maximizer()
        m.true_peak = True
        assert m.true_peak is True

    def test_getstate(self):
        m = Maximizer()
        m.set_gain(8.0)
        m.set_ceiling(-0.5)
        assert m.gain_db == 8.0
        assert m.ceiling == -0.5
