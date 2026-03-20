"""
Tests for Epic 3: CapCut Features — keyframes, text, transitions, effects,
speed ramp, and export presets.

Covers Stories 3.1–3.6.
"""

import sys
import os
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===== Story 3.1: Keyframe animation system =====

from gui.models.keyframes import (
    KeyframeType, Keyframe, KeyframeTrack, bezier_interpolate, interpolate_value,
)


class TestKeyframeType:
    def test_enum_members(self):
        assert KeyframeType.LINEAR is not None
        assert KeyframeType.EASE_IN is not None
        assert KeyframeType.EASE_OUT is not None
        assert KeyframeType.EASE_IN_OUT is not None
        assert KeyframeType.BEZIER is not None


class TestKeyframe:
    def test_defaults(self):
        kf = Keyframe()
        assert kf.time == 0.0
        assert kf.value == 0.0
        assert kf.interpolation == KeyframeType.LINEAR
        assert len(kf.id) == 12

    def test_custom_values(self):
        kf = Keyframe(time=2.5, value=100.0, interpolation=KeyframeType.EASE_IN)
        assert kf.time == 2.5
        assert kf.value == 100.0


class TestBezierInterpolate:
    def test_start_and_end(self):
        assert abs(bezier_interpolate(0.0, (0.42, 0.0), (0.58, 1.0))) < 0.01
        assert abs(bezier_interpolate(1.0, (0.42, 0.0), (0.58, 1.0)) - 1.0) < 0.01

    def test_midpoint_linear(self):
        # Linear bezier: cp1=(0.33, 0.33), cp2=(0.67, 0.67)
        val = bezier_interpolate(0.5, (0.33, 0.33), (0.67, 0.67))
        assert abs(val - 0.5) < 0.05


class TestKeyframeTrack:
    def test_empty_track(self):
        track = KeyframeTrack("opacity")
        assert track.get_value_at(0.0) is None

    def test_single_keyframe(self):
        track = KeyframeTrack("opacity")
        track.add_keyframe(Keyframe(time=0.0, value=50.0))
        assert track.get_value_at(0.0) == 50.0
        assert track.get_value_at(5.0) == 50.0

    def test_linear_interpolation(self):
        track = KeyframeTrack("opacity")
        track.add_keyframe(Keyframe(time=0.0, value=0.0, interpolation=KeyframeType.LINEAR))
        track.add_keyframe(Keyframe(time=10.0, value=100.0))
        assert abs(track.get_value_at(5.0) - 50.0) < 0.1
        assert abs(track.get_value_at(0.0) - 0.0) < 0.01
        assert abs(track.get_value_at(10.0) - 100.0) < 0.01

    def test_hold_before_and_after(self):
        track = KeyframeTrack("scale")
        track.add_keyframe(Keyframe(time=2.0, value=1.0))
        track.add_keyframe(Keyframe(time=8.0, value=2.0))
        assert track.get_value_at(0.0) == 1.0   # before
        assert track.get_value_at(20.0) == 2.0  # after

    def test_remove_keyframe(self):
        track = KeyframeTrack("x")
        kf = Keyframe(time=1.0, value=10.0)
        track.add_keyframe(kf)
        removed = track.remove_keyframe(kf.id)
        assert removed is kf
        assert track.get_value_at(1.0) is None

    def test_sort(self):
        track = KeyframeTrack("y")
        track.add_keyframe(Keyframe(time=5.0, value=50.0))
        track.add_keyframe(Keyframe(time=1.0, value=10.0))
        track.add_keyframe(Keyframe(time=3.0, value=30.0))
        assert track.keyframes[0].time == 1.0
        assert track.keyframes[-1].time == 5.0

    def test_ease_in_interpolation(self):
        track = KeyframeTrack("alpha")
        track.add_keyframe(Keyframe(time=0.0, value=0.0, interpolation=KeyframeType.EASE_IN))
        track.add_keyframe(Keyframe(time=1.0, value=1.0))
        # Ease-in should be slower at start
        val_early = track.get_value_at(0.2)
        val_linear = 0.2  # what linear would give
        assert val_early < val_linear + 0.05  # ease-in lags behind linear


# ===== Story 3.2: Text layer (model only — no Qt) =====

from gui.timeline.text_layer import TextClip, TextAnimation


class TestTextClip:
    def test_defaults(self):
        tc = TextClip()
        assert tc.text_content == "Your Text Here"
        assert tc.font_size == 48
        assert tc.duration == 5.0

    def test_end_time(self):
        tc = TextClip(start_time=2.0, duration=3.0)
        assert tc.end_time == 5.0

    def test_clone(self):
        tc = TextClip(text_content="Hello", font_size=72)
        tc2 = tc.clone()
        assert tc2.text_content == "Hello"
        assert tc2.id != tc.id

    def test_ffmpeg_filter(self):
        tc = TextClip(text_content="Test", start_time=1.0, duration=3.0)
        filt = tc.to_ffmpeg_filter()
        assert "drawtext=" in filt
        assert "Test" in filt
        assert "enable=" in filt


class TestTextAnimation:
    def test_all_presets_exist(self):
        expected = {
            "NONE", "FADE_IN", "FADE_OUT", "TYPEWRITER",
            "SLIDE_LEFT", "SLIDE_RIGHT", "SLIDE_UP", "SLIDE_DOWN",
            "BOUNCE", "ZOOM_IN", "SCALE_PULSE",
        }
        actual = {a.name for a in TextAnimation}
        assert expected.issubset(actual)


# ===== Story 3.3: Transitions =====

from gui.models.transitions import TransitionType, Transition, EasingType


class TestTransitionType:
    def test_12_types(self):
        assert len(TransitionType) >= 12

    def test_ffmpeg_values(self):
        assert TransitionType.CROSSFADE.value == "fade"
        assert TransitionType.DISSOLVE.value == "dissolve"
        assert TransitionType.WIPE_LEFT.value == "wipeleft"


class TestTransition:
    def test_defaults(self):
        t = Transition()
        assert t.type == TransitionType.CROSSFADE
        assert t.duration == 1.0

    def test_duration_clamping(self):
        t = Transition(duration=0.01)
        assert t.duration == 0.1
        t2 = Transition(duration=99.0)
        assert t2.duration == 5.0

    def test_to_ffmpeg_filter(self):
        t = Transition(type=TransitionType.DISSOLVE, duration=1.5)
        filt = t.to_ffmpeg_filter(offset=5.0)
        assert "xfade=transition=dissolve" in filt
        assert "duration=1.500" in filt
        assert "offset=5.000" in filt

    def test_label(self):
        t = Transition(type=TransitionType.IRIS)
        assert t.label == "Iris (Circle)"


# ===== Story 3.4: Effects =====

from gui.models.effects import EffectType, Effect


class TestEffectType:
    def test_10_types(self):
        assert len(EffectType) >= 10


class TestEffect:
    def test_defaults_populated(self):
        e = Effect(type=EffectType.BRIGHTNESS)
        assert "brightness" in e.parameters
        assert e.parameters["brightness"] == 0.0

    def test_ffmpeg_brightness(self):
        e = Effect(type=EffectType.BRIGHTNESS, parameters={"brightness": 0.5})
        filt = e.to_ffmpeg_filter()
        assert "eq=brightness=0.500" in filt

    def test_ffmpeg_blur(self):
        e = Effect(type=EffectType.BLUR, parameters={"radius": 10.0})
        filt = e.to_ffmpeg_filter()
        assert "boxblur=" in filt

    def test_disabled_effect(self):
        e = Effect(type=EffectType.HUE, enabled=False)
        assert e.to_ffmpeg_filter() == ""

    def test_all_effects_have_filters(self):
        for et in EffectType:
            e = Effect(type=et)
            filt = e.to_ffmpeg_filter()
            # Color temperature at 6500K returns empty (neutral)
            if et != EffectType.COLOR_TEMPERATURE:
                assert filt != "", f"{et.name} produced empty filter"

    def test_keyframe_override(self):
        e = Effect(type=EffectType.BRIGHTNESS)
        from gui.models.keyframes import KeyframeTrack, Keyframe
        track = KeyframeTrack("brightness")
        track.add_keyframe(Keyframe(time=0.0, value=-0.5))
        track.add_keyframe(Keyframe(time=10.0, value=0.5))
        e.keyframe_tracks["brightness"] = track
        val = e.get_param_at("brightness", 5.0)
        assert abs(val - 0.0) < 0.1  # should be ~0.0 midpoint


# ===== Story 3.5: Speed ramp =====

from gui.timeline.speed_ramp import SpeedRamp, SpeedPreset


class TestSpeedRamp:
    def test_normal_preset(self):
        ramp = SpeedRamp.from_preset(SpeedPreset.NORMAL)
        assert abs(ramp.get_speed_at(0.0) - 1.0) < 0.01
        assert abs(ramp.get_speed_at(0.5) - 1.0) < 0.01
        assert abs(ramp.get_speed_at(1.0) - 1.0) < 0.01

    def test_bullet_time_slow(self):
        ramp = SpeedRamp.from_preset(SpeedPreset.BULLET_TIME)
        mid_speed = ramp.get_speed_at(0.5)
        assert mid_speed < 0.5  # should be very slow in middle

    def test_flash_fast(self):
        ramp = SpeedRamp.from_preset(SpeedPreset.FLASH)
        mid_speed = ramp.get_speed_at(0.5)
        assert mid_speed > 2.0  # should be fast in middle

    def test_speed_clamped(self):
        ramp = SpeedRamp(control_points=[(0.0, 0.001), (1.0, 999.0)])
        assert ramp.get_speed_at(0.0) >= 0.1
        assert ramp.get_speed_at(1.0) <= 10.0

    def test_ffmpeg_filter_normal(self):
        ramp = SpeedRamp.from_preset(SpeedPreset.NORMAL)
        filt = ramp.to_ffmpeg_filter()
        # Normal speed = 1.0x, no filter needed
        assert filt == ""

    def test_ffmpeg_filter_nondefault(self):
        ramp = SpeedRamp.from_preset(SpeedPreset.FLASH)
        filt = ramp.to_ffmpeg_filter()
        assert "setpts=" in filt

    def test_all_presets(self):
        for sp in SpeedPreset:
            ramp = SpeedRamp.from_preset(sp)
            for p in (0.0, 0.25, 0.5, 0.75, 1.0):
                speed = ramp.get_speed_at(p)
                assert 0.1 <= speed <= 10.0, f"{sp.name} at {p} = {speed}"


# ===== Story 3.6: Export presets =====

from gui.models.export_presets import ExportPreset, BUILTIN_PRESETS


class TestExportPreset:
    def test_builtin_count(self):
        assert len(BUILTIN_PRESETS) >= 10

    def test_youtube_1080p(self):
        yt = [p for p in BUILTIN_PRESETS if p.name == "YouTube 1080p"][0]
        assert yt.resolution == (1920, 1080)
        assert yt.video_codec == "libx264"
        assert yt.container == "mp4"

    def test_audio_only_presets(self):
        audio_presets = [p for p in BUILTIN_PRESETS if p.is_audio_only]
        assert len(audio_presets) >= 3  # Spotify, Apple Music, WAV

    def test_prores_master(self):
        pr = [p for p in BUILTIN_PRESETS if "ProRes" in p.name][0]
        assert pr.video_codec == "prores_ks"
        assert pr.container == "mov"

    def test_to_ffmpeg_args(self):
        preset = ExportPreset(
            name="Test",
            resolution=(1280, 720),
            fps=24.0,
            video_codec="libx264",
            audio_codec="aac",
            video_bitrate="5M",
            audio_bitrate="128k",
            container="mp4",
        )
        args = preset.to_ffmpeg_args("in.mp4", "out.mp4")
        assert args[0] == "ffmpeg"
        assert "-c:v" in args
        assert "libx264" in args
        assert "out.mp4" in args

    def test_audio_only_ffmpeg_args(self):
        preset = ExportPreset(
            name="Audio",
            resolution=(0, 0),
            fps=0,
            video_codec="",
            audio_codec="pcm_s24le",
            video_bitrate="0",
            audio_bitrate="0",
            container="wav",
        )
        args = preset.to_ffmpeg_args("in.wav", "out.wav")
        assert "-c:v" not in args
        assert "-c:a" in args

    def test_platform_names(self):
        platforms = {p.platform_name for p in BUILTIN_PRESETS if p.platform_name}
        assert "YouTube" in platforms
        assert "Instagram" in platforms
        assert "TikTok" in platforms
        assert "Spotify" in platforms
        assert "Master" in platforms
