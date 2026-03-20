"""
Content Factory — Content Planner

Allocates songs and background videos into Long Video and Short plans.
Handles automatic song distribution, duration calculation, and hook extraction scheduling.
"""

from __future__ import annotations

import itertools
import logging
import subprocess
import json
import os
from typing import List, Optional, Tuple

from .models import (
    ContentFactoryConfig, ProductionPlan,
    LongVideoPlan, ShortVideoPlan,
    SongEntry, BackgroundVideo,
    VerticalStrategy,
)

logger = logging.getLogger(__name__)


class ContentPlanner:
    """
    Plans content production by distributing songs across Long Videos
    and assigning Shorts (one per song or a fixed count).
    """

    def __init__(self):
        self._errors: List[str] = []

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    # ─── Main Entry ──────────────────────────────────────────────

    def create_plan(self, config: ContentFactoryConfig) -> ProductionPlan:
        """Create a complete production plan from config."""
        self._errors.clear()
        plan = ProductionPlan(
            channel_name=config.channel_name,
            channel_genre=config.channel_genre,
        )

        # Step 1: Probe all songs for duration
        self._probe_songs(config.songs)

        # Step 2: Probe background videos
        self._probe_videos(config.bg_videos)

        # Step 3: Plan long videos
        if config.long_count > 0:
            plan.long_videos = self._plan_long_videos(config)

        # Step 4: Plan shorts
        plan.shorts = self._plan_shorts(config)

        return plan

    # ─── Long Video Planning ─────────────────────────────────────

    def _plan_long_videos(self, config: ContentFactoryConfig) -> List[LongVideoPlan]:
        """Distribute songs across N long videos using round-robin."""
        songs = config.songs
        if not songs:
            self._errors.append("No songs provided for long videos")
            return []

        n_longs = config.long_count
        songs_per = config.songs_per_long

        # If songs_per_long is 0 (auto), distribute evenly
        if songs_per <= 0:
            songs_per = max(1, len(songs) // max(n_longs, 1))

        # Warn if songs will be reused
        total_needed = songs_per * n_longs
        if total_needed > len(songs):
            logger.warning(
                f"[Planner] Need {total_needed} song slots but only have "
                f"{len(songs)} songs — songs will be reused across videos"
            )

        # Round-robin song distribution using itertools.cycle
        song_cycle = itertools.cycle(songs)
        plans = []

        for i in range(n_longs):
            batch = [next(song_cycle) for _ in range(songs_per)]

            # Each long video gets all bg videos (they'll be looped/cycled)
            bg_for_video = list(config.bg_videos) if config.bg_videos else []

            lv = LongVideoPlan(
                songs=batch,
                bg_videos=bg_for_video,
                crossfade_sec=config.crossfade_sec,
                crossfade_curve=config.crossfade_curve,
                title=f"Compilation #{i + 1}",
            )
            # Auto-generate chapter timestamps
            lv.build_chapters()
            plans.append(lv)

        return plans

    # ─── Short Video Planning ────────────────────────────────────

    def _plan_shorts(self, config: ContentFactoryConfig) -> List[ShortVideoPlan]:
        """Create Short plans — one per song or N total."""
        songs = config.songs
        if not songs:
            return []

        if config.short_count <= 0:
            # Default: one short per song
            target_count = len(songs)
        else:
            target_count = config.short_count

        plans = []
        bg_count = len(config.bg_videos)

        for i in range(target_count):
            song = songs[i % len(songs)]

            # Round-robin background video
            bg = config.bg_videos[i % bg_count] if bg_count > 0 else None

            sp = ShortVideoPlan(
                song=song,
                bg_video=bg,
                max_duration_sec=config.short_max_sec,
                vertical_strategy=config.vertical_strategy,
                text_preset=config.short_text_preset,
            )
            plans.append(sp)

        return plans

    # ─── Media Probing ───────────────────────────────────────────

    def _probe_songs(self, songs: List[SongEntry]) -> None:
        """Probe each song for duration using ffprobe."""
        for song in songs:
            if song.duration_sec > 0:
                continue
            dur = self._ffprobe_duration(song.file_path)
            if dur > 0:
                song.duration_sec = dur
            else:
                song.duration_sec = 180.0  # fallback 3 min
                self._errors.append(f"Could not probe duration: {song.title}")

    def _probe_videos(self, videos: List[BackgroundVideo]) -> None:
        """Probe each video for duration and resolution."""
        for vid in videos:
            if vid.duration_sec > 0:
                continue
            info = self._ffprobe_video_info(vid.file_path)
            if info:
                vid.duration_sec = info.get("duration", 0.0)
                vid.width = info.get("width", 1920)
                vid.height = info.get("height", 1080)

    @staticmethod
    def _ffprobe_duration(file_path: str) -> float:
        """Get media duration via ffprobe."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", file_path],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, FileNotFoundError):
            return 0.0

    @staticmethod
    def _ffprobe_video_info(file_path: str) -> Optional[dict]:
        """Get video duration + resolution via ffprobe."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", file_path],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(result.stdout)
            info = {"duration": float(data.get("format", {}).get("duration", 0))}

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["width"] = int(stream.get("width", 1920))
                    info["height"] = int(stream.get("height", 1080))
                    break

            return info
        except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, FileNotFoundError):
            return None

    # ─── Utility ─────────────────────────────────────────────────

    @staticmethod
    def estimate_long_duration(songs: List[SongEntry], crossfade_sec: float) -> float:
        """Estimate total long video duration."""
        if not songs:
            return 0.0
        total = sum(s.duration_sec for s in songs)
        total -= crossfade_sec * max(0, len(songs) - 1)
        return max(total, 0.0)
