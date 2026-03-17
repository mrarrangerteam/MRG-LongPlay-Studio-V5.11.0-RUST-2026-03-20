"""Tests for genre profiles and platform targets."""

import pytest
from modules.master.genre_profiles import (
    GENRE_PROFILES, PLATFORM_TARGETS, IRC_MODES, IRC_TOP_MODES,
    TONE_PRESETS, MASTERING_PRESETS,
    get_genre_list, get_genre_profile, get_irc_mode, get_irc_sub_modes,
    get_tone_preset,
)


class TestGenreProfiles:
    def test_minimum_50_genres(self):
        assert len(GENRE_PROFILES) >= 50

    def test_all_genres_have_required_keys(self):
        required = {"category", "target_lufs", "true_peak_ceiling", "irc_mode"}
        for name, profile in GENRE_PROFILES.items():
            for key in required:
                assert key in profile, f"{name} missing {key}"

    def test_genre_lufs_range(self):
        for name, profile in GENRE_PROFILES.items():
            lufs = profile["target_lufs"]
            assert -30.0 <= lufs <= 0.0, f"{name}: LUFS {lufs} out of range"

    def test_genre_irc_mode_valid(self):
        valid_modes = set(IRC_TOP_MODES)
        for name, profile in GENRE_PROFILES.items():
            mode = profile["irc_mode"]
            assert mode in valid_modes, f"{name}: invalid IRC mode {mode}"

    def test_get_genre_list(self):
        categories = get_genre_list()
        assert isinstance(categories, dict)
        assert len(categories) > 0

    def test_get_genre_profile(self):
        profile = get_genre_profile("Pop")
        assert profile is not None
        assert "target_lufs" in profile

    def test_get_genre_profile_fallback(self):
        profile = get_genre_profile("Nonexistent Genre 12345")
        assert profile is not None  # Should return All-Purpose


class TestPlatformTargets:
    def test_minimum_11_platforms(self):
        assert len(PLATFORM_TARGETS) >= 11

    def test_all_platforms_have_required_keys(self):
        for name, target in PLATFORM_TARGETS.items():
            assert "target_lufs" in target, f"{name} missing target_lufs"
            assert "true_peak" in target, f"{name} missing true_peak"

    def test_youtube(self):
        assert PLATFORM_TARGETS["YouTube"]["target_lufs"] == -14.0

    def test_spotify(self):
        assert PLATFORM_TARGETS["Spotify"]["target_lufs"] == -14.0

    def test_apple_music(self):
        assert PLATFORM_TARGETS["Apple Music"]["target_lufs"] == -16.0

    def test_deezer(self):
        assert "Deezer" in PLATFORM_TARGETS

    def test_vinyl(self):
        assert "Vinyl" in PLATFORM_TARGETS


class TestIRCModes:
    def test_irc_modes_exist(self):
        assert len(IRC_MODES) > 0

    def test_irc_top_modes(self):
        assert len(IRC_TOP_MODES) >= 5

    def test_get_irc_mode(self):
        mode = get_irc_mode("IRC 3")
        assert mode is not None

    def test_get_irc_sub_modes(self):
        subs = get_irc_sub_modes("IRC 3")
        assert isinstance(subs, list)
        assert len(subs) >= 3

    def test_irc4_sub_modes(self):
        subs = get_irc_sub_modes("IRC 4")
        assert isinstance(subs, list)
        assert len(subs) >= 2


class TestTonePresets:
    def test_tone_presets_exist(self):
        assert len(TONE_PRESETS) > 0

    def test_get_tone_preset(self):
        preset = get_tone_preset("Transparent")
        assert preset is not None

    def test_get_tone_preset_fallback(self):
        preset = get_tone_preset("Nonexistent")
        assert preset is not None


class TestMasteringPresets:
    def test_mastering_presets_exist(self):
        assert len(MASTERING_PRESETS) >= 10
