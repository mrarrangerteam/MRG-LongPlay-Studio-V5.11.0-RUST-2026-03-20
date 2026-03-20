"""
Tests for Content Factory pipeline.

Tests cover: models, planner, batch_master, metadata, and workers.
Video builder tests are skipped if ffmpeg is not available.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path

from gui.content_factory.models import (
    SongEntry, BackgroundVideo, LongVideoPlan, ShortVideoPlan,
    ProductionPlan, BatchMasterConfig, ContentFactoryConfig, ContentJob,
    VideoFormat, VerticalStrategy, CrossfadeCurve, JobStatus,
)
from gui.content_factory.planner import ContentPlanner
from gui.content_factory.metadata import MetadataGenerator


# ─── Model Tests ─────────────────────────────────────────────────────────

class TestSongEntry:
    def test_default_title_from_path(self):
        s = SongEntry(file_path="/music/My Song.wav")
        assert s.title == "My Song"

    def test_explicit_title(self):
        s = SongEntry(file_path="/music/track.wav", title="Custom Title")
        assert s.title == "Custom Title"

    def test_effective_path_original(self):
        s = SongEntry(file_path="/music/track.wav")
        assert s.effective_path == "/music/track.wav"

    def test_effective_path_mastered(self):
        s = SongEntry(file_path="/music/track.wav", mastered_path="/out/track_mastered.wav")
        assert s.effective_path == "/out/track_mastered.wav"


class TestBackgroundVideo:
    def test_is_landscape(self):
        bg = BackgroundVideo(file_path="bg.mp4", width=1920, height=1080)
        assert bg.is_landscape is True

    def test_is_portrait(self):
        bg = BackgroundVideo(file_path="bg.mp4", width=1080, height=1920)
        assert bg.is_landscape is False

    def test_aspect_ratio(self):
        bg = BackgroundVideo(file_path="bg.mp4", width=1920, height=1080)
        assert abs(bg.aspect_ratio - 16 / 9) < 0.01


class TestLongVideoPlan:
    def test_auto_video_id(self):
        lv = LongVideoPlan()
        assert lv.video_id.startswith("long_")

    def test_estimated_duration_empty(self):
        lv = LongVideoPlan()
        assert lv.estimated_duration == 0.0

    def test_estimated_duration_with_songs(self):
        songs = [
            SongEntry(file_path=f"/s{i}.wav", duration_sec=180.0)
            for i in range(5)
        ]
        lv = LongVideoPlan(songs=songs, crossfade_sec=3.0)
        # 5 songs × 180s = 900s, minus 4 crossfades × 3s = 12s = 888s
        assert lv.estimated_duration == 888.0

    def test_build_chapters(self):
        songs = [
            SongEntry(file_path="/a.wav", title="Song A", artist="Artist X", duration_sec=180),
            SongEntry(file_path="/b.wav", title="Song B", artist="Artist Y", duration_sec=200),
            SongEntry(file_path="/c.wav", title="Song C", duration_sec=150),
        ]
        lv = LongVideoPlan(songs=songs, crossfade_sec=3.0)
        lv.build_chapters()

        assert len(lv.chapters) == 3
        assert lv.chapters[0] == (0.0, "Artist X — Song A")
        assert lv.chapters[1] == (177.0, "Artist Y — Song B")  # 180 - 3
        assert lv.chapters[2] == (374.0, "Song C")  # 177 + 200 - 3


class TestShortVideoPlan:
    def test_auto_video_id(self):
        sv = ShortVideoPlan()
        assert sv.video_id.startswith("short_")

    def test_max_duration_clamped(self):
        sv = ShortVideoPlan(max_duration_sec=120.0)
        assert sv.max_duration_sec == 60.0

    def test_default_strategy(self):
        sv = ShortVideoPlan()
        assert sv.vertical_strategy == VerticalStrategy.BLUR_FILL


class TestBatchMasterConfig:
    def test_defaults(self):
        c = BatchMasterConfig()
        assert c.enabled is True
        assert c.target_lufs == -14.0
        assert c.imager_width == 100

    def test_lufs_clamped(self):
        c = BatchMasterConfig(target_lufs=-30.0)
        assert c.target_lufs == -24.0

    def test_imager_width_clamped(self):
        c = BatchMasterConfig(imager_width=300)
        assert c.imager_width == 200


class TestContentFactoryConfig:
    def test_defaults(self):
        c = ContentFactoryConfig()
        assert c.long_count == 1
        assert c.short_count == 0
        assert c.output_dir != ""

    def test_short_max_clamped(self):
        c = ContentFactoryConfig(short_max_sec=90.0)
        assert c.short_max_sec == 60.0


class TestContentJob:
    def test_auto_job_id(self):
        job = ContentJob()
        assert job.job_id.startswith("job_")

    def test_add_error(self):
        job = ContentJob()
        job.add_error("Error 1")
        job.add_error("Error 2")
        assert len(job.errors) == 2
        assert "Error 1" in job.errors

    def test_status_flow(self):
        job = ContentJob()
        assert job.status == JobStatus.PENDING
        job.status = JobStatus.MASTERING
        assert job.status == JobStatus.MASTERING
        job.status = JobStatus.COMPLETED
        assert job.status == JobStatus.COMPLETED


class TestProductionPlan:
    def test_total_videos(self):
        plan = ProductionPlan()
        plan.long_videos = [LongVideoPlan() for _ in range(2)]
        plan.shorts = [ShortVideoPlan() for _ in range(5)]
        assert plan.total_videos == 7

    def test_total_songs_used(self):
        s1 = SongEntry(file_path="/a.wav")
        s2 = SongEntry(file_path="/b.wav")
        lv = LongVideoPlan(songs=[s1, s2])
        sv = ShortVideoPlan(song=s1)  # Reuse s1
        plan = ProductionPlan(long_videos=[lv], shorts=[sv])
        assert plan.total_songs_used == 2  # Unique files


# ─── Planner Tests ───────────────────────────────────────────────────────

class TestContentPlanner:
    def _make_songs(self, n, duration=180.0):
        return [
            SongEntry(
                file_path=f"/music/song_{i}.wav",
                title=f"Song {i}",
                artist=f"Artist {i}",
                duration_sec=duration,
            )
            for i in range(n)
        ]

    def _make_config(self, n_songs=10, n_bg=1, long_count=1, short_count=0):
        songs = self._make_songs(n_songs)
        bg_videos = [
            BackgroundVideo(file_path=f"/bg/bg_{i}.mp4", duration_sec=600.0)
            for i in range(n_bg)
        ]
        return ContentFactoryConfig(
            songs=songs,
            bg_videos=bg_videos,
            long_count=long_count,
            short_count=short_count,
            channel_name="Test Channel",
            channel_genre="Chill",
        )

    def test_basic_plan(self):
        config = self._make_config(n_songs=20, long_count=1, short_count=0)
        planner = ContentPlanner()
        plan = planner.create_plan(config)

        assert len(plan.long_videos) == 1
        assert len(plan.long_videos[0].songs) == 20
        assert plan.shorts  # One per song by default (short_count=0)

    def test_multiple_long_videos(self):
        config = self._make_config(n_songs=40, long_count=2)
        planner = ContentPlanner()
        plan = planner.create_plan(config)

        assert len(plan.long_videos) == 2
        assert len(plan.long_videos[0].songs) == 20
        assert len(plan.long_videos[1].songs) == 20

    def test_shorts_one_per_song(self):
        config = self._make_config(n_songs=5, short_count=0)
        planner = ContentPlanner()
        plan = planner.create_plan(config)

        assert len(plan.shorts) == 5

    def test_explicit_short_count(self):
        config = self._make_config(n_songs=10, short_count=3)
        planner = ContentPlanner()
        plan = planner.create_plan(config)

        assert len(plan.shorts) == 3

    def test_empty_songs(self):
        config = ContentFactoryConfig(songs=[], long_count=1)
        planner = ContentPlanner()
        plan = planner.create_plan(config)

        assert len(plan.long_videos) == 0
        assert len(plan.shorts) == 0

    def test_no_bg_videos(self):
        config = self._make_config(n_songs=5, n_bg=0)
        planner = ContentPlanner()
        plan = planner.create_plan(config)

        assert len(plan.long_videos) == 1
        assert len(plan.long_videos[0].bg_videos) == 0

    def test_chapters_generated(self):
        # Use songs_per_long=5 to avoid song reuse
        config = self._make_config(n_songs=5, long_count=1)
        config.songs_per_long = 5
        planner = ContentPlanner()
        plan = planner.create_plan(config)

        lv = plan.long_videos[0]
        assert len(lv.chapters) == 5
        assert lv.chapters[0][0] == 0.0

    def test_100_songs(self):
        """Verify planner handles 100+ songs without error."""
        config = self._make_config(n_songs=100, long_count=5)
        planner = ContentPlanner()
        plan = planner.create_plan(config)

        assert len(plan.long_videos) == 5
        total_assigned = sum(len(lv.songs) for lv in plan.long_videos)
        assert total_assigned == 100


# ─── Metadata Tests ──────────────────────────────────────────────────────

class TestMetadataGenerator:
    def test_long_metadata(self):
        songs = [
            SongEntry(file_path="/a.wav", title="Song A", artist="Artist 1", duration_sec=180),
            SongEntry(file_path="/b.wav", title="Song B", artist="Artist 2", duration_sec=200),
        ]
        lv = LongVideoPlan(songs=songs, crossfade_sec=3.0)
        lv.build_chapters()

        config = ContentFactoryConfig(
            songs=songs,
            channel_name="Chill Vibes",
            channel_genre="Chill",
        )
        meta = MetadataGenerator(config)
        meta.generate_long_metadata(lv)

        assert lv.title != ""
        assert lv.description != ""
        assert len(lv.tags) > 0

    def test_short_metadata(self):
        song = SongEntry(file_path="/a.wav", title="Hook Song", artist="DJ Cool")
        sv = ShortVideoPlan(song=song)

        config = ContentFactoryConfig(
            songs=[song],
            channel_name="Music Shorts",
            channel_genre="Electronic",
        )
        meta = MetadataGenerator(config)
        meta.generate_short_metadata(sv)

        assert sv.title != ""
        assert "#Shorts" in sv.description or "#shorts" in sv.description.lower()
        assert len(sv.tags) > 0


# ─── BatchMasterLite Tests ───────────────────────────────────────────────

class TestBatchMasterLite:
    def test_init_default(self):
        from gui.content_factory.batch_master import BatchMasterLite
        bm = BatchMasterLite()
        assert bm._config.enabled is True

    def test_init_disabled(self):
        from gui.content_factory.batch_master import BatchMasterLite
        config = BatchMasterConfig(enabled=False)
        bm = BatchMasterLite(config)
        assert bm._config.enabled is False

    def test_has_rust_backend_returns_bool(self):
        from gui.content_factory.batch_master import BatchMasterLite
        result = BatchMasterLite.has_rust_backend()
        assert isinstance(result, bool)

    def test_cancel(self):
        from gui.content_factory.batch_master import BatchMasterLite
        bm = BatchMasterLite()
        bm.cancel()
        assert bm._cancelled.is_set()

    def test_master_batch_disabled(self):
        from gui.content_factory.batch_master import BatchMasterLite
        config = BatchMasterConfig(enabled=False)
        bm = BatchMasterLite(config)
        songs = [SongEntry(file_path="/fake.wav")]
        result = bm.master_batch(songs, "/tmp/out")
        assert result == 1  # Returns len(songs) when disabled


# ─── Integration: Planner → Metadata ──────────────────────────────────

class TestPlannerMetadataIntegration:
    def test_full_pipeline_plan_to_metadata(self):
        """Test complete flow: Config → Plan → Metadata for all videos."""
        songs = [
            SongEntry(
                file_path=f"/music/track_{i}.wav",
                title=f"Track {i}",
                artist="Various",
                duration_sec=180.0,
            )
            for i in range(10)
        ]
        bg_videos = [BackgroundVideo(file_path="/bg/bg.mp4", duration_sec=600)]

        config = ContentFactoryConfig(
            songs=songs,
            bg_videos=bg_videos,
            long_count=1,
            short_count=3,
            channel_name="LongPlay Music",
            channel_genre="Chill",
        )

        # Plan
        planner = ContentPlanner()
        plan = planner.create_plan(config)

        assert len(plan.long_videos) == 1
        assert len(plan.shorts) == 3

        # Metadata
        meta = MetadataGenerator(config)
        for lv in plan.long_videos:
            meta.generate_long_metadata(lv)
            assert lv.title
            assert lv.description
            assert len(lv.tags) >= 3

        for sv in plan.shorts:
            meta.generate_short_metadata(sv)
            assert sv.title
            assert sv.description

    def test_large_batch_500_songs(self):
        """Verify planner handles 500 songs without performance issues."""
        songs = [
            SongEntry(
                file_path=f"/music/song_{i}.wav",
                title=f"Song {i}",
                duration_sec=180.0,
            )
            for i in range(500)
        ]

        config = ContentFactoryConfig(
            songs=songs,
            long_count=25,  # 25 videos × 20 songs each
            short_count=100,
        )

        planner = ContentPlanner()
        plan = planner.create_plan(config)

        assert len(plan.long_videos) == 25
        assert len(plan.shorts) == 100

        total_long_songs = sum(len(lv.songs) for lv in plan.long_videos)
        assert total_long_songs == 500
