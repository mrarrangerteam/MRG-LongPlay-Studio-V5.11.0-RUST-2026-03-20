"""
Tests for Epic 5 — Polish & Production (Stories 5.1–5.6).

Covers:
    5.1: GPU Preview (FrameCache, FrameDecoder, ProxyManager, GPUPreviewCompositor)
    5.2: Audio I/O (audio_io module with symphonia integration)
    5.3: Auto-save (AutoSaveManager, project serialization)
    5.4: Vintage Theme (VintageTheme, theme registry)
    5.5: Profiler (Profiler, FPSMonitor, MemoryTracker, RenderCache)
    5.6: Build script (check_environment)
"""

import json
import os
import sys
import tempfile
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Story 5.1: GPU-Accelerated Video Preview
# ============================================================

class TestGPUPreview:
    """Tests for the GPU preview module."""

    def test_import(self):
        from gui.video.gpu_preview import (
            FrameCache, FrameDecoder, ProxyManager, GPUPreviewCompositor,
        )
        assert FrameCache is not None

    def test_frame_cache_lru(self):
        from gui.video.gpu_preview import FrameCache

        cache = FrameCache(max_size=3)
        frame = np.zeros((100, 100, 4), dtype=np.uint8)

        cache.put("a", frame)
        cache.put("b", frame)
        cache.put("c", frame)
        assert cache.size == 3

        cache.put("d", frame)  # should evict "a"
        assert cache.size == 3
        assert cache.get("a") is None
        assert cache.get("b") is not None

    def test_frame_cache_clear(self):
        from gui.video.gpu_preview import FrameCache
        cache = FrameCache()
        cache.put("test", np.zeros((10, 10, 4), dtype=np.uint8))
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0

    def test_compositor_blank(self):
        from gui.video.gpu_preview import GPUPreviewCompositor

        comp = GPUPreviewCompositor(320, 180)
        result = comp.composite([])  # no layers
        assert result.shape == (180, 320, 4)
        assert result.dtype == np.uint8

    def test_compositor_single_layer(self):
        from gui.video.gpu_preview import GPUPreviewCompositor

        comp = GPUPreviewCompositor(320, 180)
        red_frame = np.zeros((180, 320, 4), dtype=np.uint8)
        red_frame[:, :, 0] = 255  # red
        red_frame[:, :, 3] = 255  # opaque

        result = comp.composite([(red_frame, 1.0)])
        assert result.shape == (180, 320, 4)
        assert result[90, 160, 0] == 255  # red channel preserved

    def test_compositor_opacity(self):
        from gui.video.gpu_preview import GPUPreviewCompositor

        comp = GPUPreviewCompositor(100, 100)
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        frame[:, :] = [200, 100, 50, 255]

        result = comp.composite([(frame, 0.5)])
        # at 50% opacity, values should be roughly half
        assert 80 < result[50, 50, 0] < 120

    def test_proxy_manager_init(self):
        from gui.video.gpu_preview import ProxyManager
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProxyManager(proxy_dir=tmpdir)
            # non-existent source should return source path
            assert pm.get_proxy("/nonexistent/file.mp4") == "/nonexistent/file.mp4"

    def test_decoder_hw_accel_detection(self):
        from gui.video.gpu_preview import FrameDecoder
        decoder = FrameDecoder()
        assert decoder.hw_accel_name in ("videotoolbox", "vaapi", "dxva2", "software")


# ============================================================
# Story 5.2: Audio I/O with symphonia
# ============================================================

class TestAudioIO:
    """Tests for the unified audio I/O module."""

    def test_import(self):
        from modules.master.audio_io import read_audio, get_info, get_backend_name
        assert read_audio is not None

    def test_backend_name(self):
        from modules.master.audio_io import get_backend_name
        name = get_backend_name()
        assert name in ("Rust (symphonia)", "Rust (longplay)",
                        "soundfile (libsndfile)", "FFmpeg (subprocess)")

    def test_supported_formats(self):
        from modules.master.audio_io import SUPPORTED_FORMATS
        assert "wav" in SUPPORTED_FORMATS
        assert "mp3" in SUPPORTED_FORMATS
        assert "flac" in SUPPORTED_FORMATS
        assert "ogg" in SUPPORTED_FORMATS

    def test_read_nonexistent(self):
        from modules.master.audio_io import read_audio
        result = read_audio("/nonexistent/file.wav")
        assert result is None

    def test_get_info_nonexistent(self):
        from modules.master.audio_io import get_info
        result = get_info("/nonexistent/file.wav")
        assert result is None

    def test_audio_file_info(self):
        from modules.master.audio_io import AudioFileInfo
        info = AudioFileInfo()
        assert info.sample_rate == 44100
        assert info.channels == 2
        d = info.to_dict()
        assert "sample_rate" in d
        assert "duration_sec" in d

    def test_resample(self):
        from modules.master.audio_io import _resample
        samples = np.sin(np.arange(44100) / 44100 * 2 * np.pi * 440)
        resampled = _resample(samples, 44100, 22050)
        assert len(resampled) == 22050  # half the length


# ============================================================
# Story 5.3: Auto-save and Crash Recovery
# ============================================================

class TestAutoSave:
    """Tests for the auto-save system."""

    def test_import(self):
        from gui.models.autosave import AutoSaveManager, project_to_dict, dict_to_project
        assert AutoSaveManager is not None

    def test_project_serialization(self):
        from gui.models.autosave import project_to_dict, dict_to_project
        from gui.models.track import Project, Track, Clip, TrackType

        project = Project(fps=30.0, resolution=(1920, 1080))
        track = Track(name="Video 1", type=TrackType.VIDEO)
        clip = Clip(start_time=1.0, duration=5.0, source_path="/tmp/test.mp4", name="Test Clip")
        track.add_clip(clip)
        project.add_track(track)

        data = project_to_dict(project)
        assert data["version"] == "5.10"
        assert len(data["tracks"]) == 1
        assert data["tracks"][0]["name"] == "Video 1"
        assert len(data["tracks"][0]["clips"]) == 1

        restored = dict_to_project(data)
        assert len(restored.tracks) == 1
        assert restored.tracks[0].name == "Video 1"
        assert len(restored.tracks[0].clips) == 1
        assert restored.tracks[0].clips[0].name == "Test Clip"
        assert restored.tracks[0].clips[0].start_time == 1.0

    def test_roundtrip_json(self):
        from gui.models.autosave import project_to_dict, dict_to_project
        from gui.models.track import Project, Track, Clip, TrackType

        project = Project()
        for i in range(3):
            t = Track(name=f"Track {i}", type=TrackType.AUDIO)
            for j in range(2):
                c = Clip(start_time=j * 5.0, duration=4.0, name=f"Clip {i}-{j}")
                t.add_clip(c)
            project.add_track(t)

        data = project_to_dict(project)
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = dict_to_project(restored_data)

        assert len(restored.tracks) == 3
        assert len(restored.tracks[0].clips) == 2

    def test_autosave_manager(self):
        from gui.models.autosave import AutoSaveManager
        from gui.models.track import Project

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = AutoSaveManager(autosave_dir=tmpdir, interval=600)
            project = Project()
            mgr.set_project(project)

            path = mgr.save_now()
            assert path is not None
            assert os.path.exists(path)
            assert mgr.save_count == 1

    def test_recovery(self):
        from gui.models.autosave import AutoSaveManager
        from gui.models.track import Project, Track, TrackType

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = AutoSaveManager(autosave_dir=tmpdir)
            project = Project()
            track = Track(name="Recovery Test", type=TrackType.VIDEO)
            project.add_track(track)
            mgr.set_project(project)
            mgr.save_now()

            # simulate crash (create lock file)
            lock_path = os.path.join(tmpdir, "session.lock")
            with open(lock_path, "w") as f:
                f.write("12345\n")

            # new manager should find recovery
            mgr2 = AutoSaveManager(autosave_dir=tmpdir)
            assert mgr2.has_recovery()

            recovered = mgr2.recover()
            assert recovered is not None
            assert len(recovered.tracks) == 1
            assert recovered.tracks[0].name == "Recovery Test"

    def test_cleanup(self):
        from gui.models.autosave import AutoSaveManager
        from gui.models.track import Project

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = AutoSaveManager(autosave_dir=tmpdir)
            mgr.set_project(Project())
            mgr.save_now()
            mgr.save_now()

            count = mgr.cleanup()
            assert count >= 1


# ============================================================
# Story 5.4: Vintage Hardware UI Polish
# ============================================================

class TestVintageTheme:
    """Tests for the vintage theme system."""

    def test_import(self):
        from gui.styles_vintage import VintageTheme, MidnightTheme, WarmConsoleTheme
        assert VintageTheme is not None

    def test_theme_names(self):
        from gui.styles_vintage import get_theme_names, THEMES
        names = get_theme_names()
        assert len(names) == 3
        assert "Classic Dark" in names
        assert "Midnight" in names
        assert "Warm Console" in names

    def test_get_theme(self):
        from gui.styles_vintage import get_theme
        theme = get_theme("classic_dark")
        assert theme.name == "Classic Dark"
        theme = get_theme("midnight")
        assert theme.name == "Midnight"

    def test_stylesheet_generation(self):
        from gui.styles_vintage import VintageTheme
        css = VintageTheme.get_global_stylesheet()
        assert "QMainWindow" in css
        assert "QPushButton" in css
        assert "QSlider" in css
        assert "QComboBox" in css
        assert len(css) > 500

    def test_theme_colors_exist(self):
        from gui.styles_vintage import VintageTheme
        assert VintageTheme.CHASSIS
        assert VintageTheme.AMBER
        assert VintageTheme.TEAL
        assert VintageTheme.METER_GREEN
        assert VintageTheme.METER_RED

    def test_theme_variants_differ(self):
        from gui.styles_vintage import VintageTheme, MidnightTheme, WarmConsoleTheme
        assert VintageTheme.CHASSIS != MidnightTheme.CHASSIS
        assert VintageTheme.TEAL != WarmConsoleTheme.TEAL


# ============================================================
# Story 5.5: Performance Optimization
# ============================================================

class TestProfiler:
    """Tests for the profiler and optimization utilities."""

    def test_import(self):
        from gui.utils.profiler import Profiler, FPSMonitor, MemoryTracker, RenderCache
        assert Profiler is not None

    def test_profiler_decorator(self):
        from gui.utils.profiler import Profiler

        p = Profiler()

        @p.profile
        def slow_func():
            total = 0
            for i in range(1000):
                total += i
            return total

        result = slow_func()
        assert result == 499500

        report = p.report_dict()
        assert len(report) == 1
        assert report[0]["calls"] == 1
        assert report[0]["total_ms"] >= 0

    def test_profiler_section(self):
        from gui.utils.profiler import Profiler

        p = Profiler()
        with p.section("test_section"):
            x = sum(range(100))

        report = p.report_dict()
        assert any(r["name"] == "test_section" for r in report)

    def test_profiler_report_text(self):
        from gui.utils.profiler import Profiler

        p = Profiler()

        @p.profile
        def func():
            pass

        func()
        func()

        text = p.report()
        assert "Performance Report" in text
        report = p.report_dict()
        assert len(report) == 1
        assert report[0]["calls"] == 2

    def test_fps_monitor(self):
        from gui.utils.profiler import FPSMonitor

        fps = FPSMonitor(window_size=10)
        # simulate 60fps
        for i in range(10):
            fps.tick()
            time.sleep(0.001)  # minimal sleep

        assert fps.fps > 0

    def test_memory_tracker(self):
        from gui.utils.profiler import MemoryTracker
        rss = MemoryTracker.get_rss_mb()
        assert rss >= 0  # may be 0 if resource module unavailable

    def test_render_cache(self):
        from gui.utils.profiler import RenderCache

        cache = RenderCache(max_size_mb=1.0)
        data = b"x" * 1024
        cache.put("key1", data)
        assert cache.entry_count == 1
        assert cache.size_mb > 0

        retrieved = cache.get("key1")
        assert retrieved == data

        cache.clear()
        assert cache.entry_count == 0

    def test_render_cache_eviction(self):
        from gui.utils.profiler import RenderCache

        cache = RenderCache(max_size_mb=0.001)  # ~1KB max
        cache.put("a", b"x" * 512)
        cache.put("b", b"x" * 512)
        cache.put("c", b"x" * 512)  # should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") is not None or cache.get("c") is not None

    def test_profiler_disabled(self):
        from gui.utils.profiler import Profiler

        p = Profiler()
        p.enabled = False

        @p.profile
        def func():
            return 42

        result = func()
        assert result == 42
        assert len(p.report_dict()) == 0


# ============================================================
# Story 5.6: Build Script
# ============================================================

class TestBuildScript:
    """Tests for the build script."""

    def test_import(self):
        import build_app
        assert hasattr(build_app, "check_environment")
        assert hasattr(build_app, "APP_VERSION")
        assert build_app.APP_VERSION == "5.10.0"

    def test_constants(self):
        import build_app
        assert build_app.APP_NAME == "LongPlay Studio"
        assert build_app.BUNDLE_ID == "com.mrg.longplay-studio"
        assert build_app.MIN_MACOS == "13.0"
