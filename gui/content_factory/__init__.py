"""
Content Factory — Automated batch content production pipeline.

Produces Long Videos (16:9, ~1hr compilations) and Shorts (9:16, ≤29s hooks)
from imported songs + background videos, with auto-mastering and YouTube upload.
"""

from .models import (
    ContentJob, ProductionPlan, ContentFactoryConfig,
    SongEntry, BackgroundVideo, LongVideoPlan, ShortVideoPlan,
    BatchMasterConfig, VideoFormat, VerticalStrategy, CrossfadeCurve,
    JobStatus,
)
from .planner import ContentPlanner
from .batch_master import BatchMasterLite
from .long_builder import LongVideoBuilder
from .short_builder import ShortVideoBuilder
from .metadata import MetadataGenerator
from .workers import check_ffmpeg, get_ffmpeg_version

__all__ = [
    "ContentJob",
    "ProductionPlan",
    "ContentFactoryConfig",
    "SongEntry",
    "BackgroundVideo",
    "LongVideoPlan",
    "ShortVideoPlan",
    "BatchMasterConfig",
    "VideoFormat",
    "VerticalStrategy",
    "CrossfadeCurve",
    "JobStatus",
    "ContentPlanner",
    "BatchMasterLite",
    "LongVideoBuilder",
    "ShortVideoBuilder",
    "MetadataGenerator",
    "check_ffmpeg",
    "get_ffmpeg_version",
]
