"""
Content Factory — Data Models

Defines all data structures used throughout the Content Factory pipeline.
Uses Python dataclasses for lightweight, type-safe models.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
from pathlib import Path


# ─── Enums ───────────────────────────────────────────────────────────

class VideoFormat(Enum):
    """Output video format."""
    LONG = "long"        # 16:9, ~1 hour compilation
    SHORT = "short"      # 9:16, ≤29 seconds hook clip


class VerticalStrategy(Enum):
    """How to convert 16:9 → 9:16 for Shorts."""
    BLUR_FILL = "blur_fill"      # Blurred scaled bg + sharp center crop
    SCALE_CROP = "scale_crop"    # Center crop to fill 9:16
    SCALE_FIT = "scale_fit"      # Letterbox with black bars (not recommended)


class CrossfadeCurve(Enum):
    """Audio crossfade curve type for FFmpeg acrossfade."""
    TRI = "tri"              # Linear
    EXP = "exp"              # Exponential
    ESIN = "esin"            # Equal power (sine)
    HSIN = "hsin"            # Half-sine
    LOG = "log"              # Logarithmic
    SQRT = "sqrt"            # Square root (equal power)


class JobStatus(Enum):
    """Status of a content production job."""
    PENDING = "pending"
    MASTERING = "mastering"
    RENDERING = "rendering"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ─── Song Entry ──────────────────────────────────────────────────────

@dataclass
class SongEntry:
    """A single song in the import list."""
    file_path: str
    title: str = ""
    artist: str = ""
    duration_sec: float = 0.0
    genre: str = ""

    # Set after mastering
    mastered_path: str = ""

    # Set after hook extraction (for Shorts)
    hook_start_sec: float = 0.0
    hook_end_sec: float = 0.0
    hook_duration_sec: float = 0.0
    hook_confidence: float = 0.0
    hook_file_path: str = ""

    def __post_init__(self):
        if not self.title:
            self.title = Path(self.file_path).stem

    @property
    def effective_path(self) -> str:
        """Return mastered path if available, else original."""
        return self.mastered_path if self.mastered_path else self.file_path


# ─── Background Video ────────────────────────────────────────────────

@dataclass
class BackgroundVideo:
    """A background video file for visual content."""
    file_path: str
    duration_sec: float = 0.0
    width: int = 1920
    height: int = 1080

    @property
    def is_landscape(self) -> bool:
        return self.width >= self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)


# ─── Long Video Plan ─────────────────────────────────────────────────

@dataclass
class LongVideoPlan:
    """Plan for a single long video compilation."""
    video_id: str = ""
    songs: List[SongEntry] = field(default_factory=list)
    bg_videos: List[BackgroundVideo] = field(default_factory=list)
    crossfade_sec: float = 3.0
    crossfade_curve: CrossfadeCurve = CrossfadeCurve.EXP
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # YouTube chapters: list of (timestamp_sec, song_title)
    chapters: List[Tuple[float, str]] = field(default_factory=list)

    # Output
    output_path: str = ""
    total_duration_sec: float = 0.0
    status: JobStatus = JobStatus.PENDING

    def __post_init__(self):
        if not self.video_id:
            self.video_id = f"long_{uuid.uuid4().hex[:8]}"

    @property
    def estimated_duration(self) -> float:
        """Estimate total duration from songs + crossfades."""
        if not self.songs:
            return 0.0
        total = sum(s.duration_sec for s in self.songs)
        total -= self.crossfade_sec * max(0, len(self.songs) - 1)
        return max(total, 0.0)

    def build_chapters(self) -> None:
        """Generate chapter timestamps from song list."""
        self.chapters.clear()
        pos = 0.0
        for i, song in enumerate(self.songs):
            label = f"{song.artist} — {song.title}" if song.artist else song.title
            self.chapters.append((pos, label))
            pos += song.duration_sec
            if i < len(self.songs) - 1:
                pos -= self.crossfade_sec


# ─── Short Video Plan ────────────────────────────────────────────────

@dataclass
class ShortVideoPlan:
    """Plan for a single YouTube Short."""
    video_id: str = ""
    song: Optional[SongEntry] = None
    bg_video: Optional[BackgroundVideo] = None
    max_duration_sec: float = 29.0
    vertical_strategy: VerticalStrategy = VerticalStrategy.BLUR_FILL
    text_preset: str = "Song Title"     # Must match TEXT_STYLE_PRESETS key
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Output
    output_path: str = ""
    status: JobStatus = JobStatus.PENDING

    def __post_init__(self):
        if not self.video_id:
            self.video_id = f"short_{uuid.uuid4().hex[:8]}"
        # YouTube Shorts hard limit is 60 seconds
        self.max_duration_sec = min(self.max_duration_sec, 60.0)


# ─── Production Plan ─────────────────────────────────────────────────

@dataclass
class ProductionPlan:
    """Complete production plan for a batch of content."""
    long_videos: List[LongVideoPlan] = field(default_factory=list)
    shorts: List[ShortVideoPlan] = field(default_factory=list)
    channel_name: str = ""
    channel_genre: str = "Chill"

    @property
    def total_videos(self) -> int:
        return len(self.long_videos) + len(self.shorts)

    @property
    def total_songs_used(self) -> int:
        songs = set()
        for lv in self.long_videos:
            for s in lv.songs:
                songs.add(s.file_path)
        for sv in self.shorts:
            if sv.song:
                songs.add(sv.song.file_path)
        return len(songs)


# ─── Master Config (Dynamics + Loudness + Imager only) ───────────────

@dataclass
class BatchMasterConfig:
    """Mastering config for Content Factory — NO EQ (different tones)."""
    enabled: bool = True
    target_lufs: float = -14.0          # YouTube standard
    true_peak_limit: float = -1.0       # dBTP
    dynamics_enabled: bool = True
    dynamics_ratio: float = 2.5         # Gentle compression
    dynamics_threshold: float = -18.0   # dB
    imager_enabled: bool = True
    imager_width: int = 100             # 0=mono, 100=original, 200=super-wide
    maximizer_enabled: bool = True
    maximizer_ceiling: float = -0.3     # dB

    def __post_init__(self):
        # Validate LUFS range (-24 to 0)
        self.target_lufs = max(-24.0, min(0.0, self.target_lufs))
        # Validate imager width (0-200)
        self.imager_width = max(0, min(200, self.imager_width))


# ─── Content Factory Config ──────────────────────────────────────────

@dataclass
class ContentFactoryConfig:
    """Top-level configuration for Content Factory."""
    # Input
    songs: List[SongEntry] = field(default_factory=list)
    bg_videos: List[BackgroundVideo] = field(default_factory=list)

    # Long video settings
    long_count: int = 1                 # How many long videos to produce
    songs_per_long: int = 20            # Songs per long video (0 = auto)
    crossfade_sec: float = 3.0
    crossfade_curve: CrossfadeCurve = CrossfadeCurve.EXP
    target_duration_min: float = 60.0   # Target ~1 hour

    # Short video settings
    short_count: int = 0                # 0 = one per song, N = exactly N shorts
    short_max_sec: float = 29.0
    vertical_strategy: VerticalStrategy = VerticalStrategy.BLUR_FILL
    short_text_preset: str = "Song Title"  # Must match TEXT_STYLE_PRESETS key

    # Mastering
    master_config: BatchMasterConfig = field(default_factory=BatchMasterConfig)

    # Channel info (for metadata)
    channel_name: str = ""
    channel_genre: str = "Chill"

    # Output
    output_dir: str = ""

    # Upload
    auto_upload: bool = False
    upload_privacy: str = "private"     # private / unlisted / public

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = str(Path.home() / "LongPlay_Output")
        # Validate counts
        self.long_count = max(0, self.long_count)
        self.short_count = max(0, self.short_count)
        self.short_max_sec = min(self.short_max_sec, 60.0)


# ─── Content Job (tracks overall progress) ───────────────────────────

@dataclass
class ContentJob:
    """A single Content Factory job tracking overall progress."""
    job_id: str = ""
    config: ContentFactoryConfig = field(default_factory=ContentFactoryConfig)
    plan: ProductionPlan = field(default_factory=ProductionPlan)
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0               # 0-100
    current_step: str = ""
    errors: List[str] = field(default_factory=list)   # Multiple error tracking
    output_files: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.job_id:
            self.job_id = f"job_{uuid.uuid4().hex[:8]}"

    def add_error(self, msg: str) -> None:
        """Log an error without stopping the job."""
        self.errors.append(msg)
