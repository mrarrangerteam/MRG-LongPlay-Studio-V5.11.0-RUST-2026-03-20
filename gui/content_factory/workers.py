"""
Content Factory — QThread Workers

Background workers for the Content Factory pipeline.
Handles mastering, rendering, and uploading in separate threads
to keep the UI responsive.

Progress aggregation:
  Mastering:  0-25%
  Long video: 25-55%
  Shorts:     55-80%
  Upload:     80-100%
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from typing import Optional, List

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    from PySide6.QtCore import QThread, Signal as pyqtSignal

from .models import (
    ContentJob, ContentFactoryConfig, ProductionPlan,
    JobStatus, LongVideoPlan, ShortVideoPlan,
)
from .planner import ContentPlanner
from .batch_master import BatchMasterLite
from .long_builder import LongVideoBuilder
from .short_builder import ShortVideoBuilder
from .metadata import MetadataGenerator

logger = logging.getLogger(__name__)

# YouTube upload (optional)
try:
    from modules.upload.youtube_upload import YouTubeUploader
    HAS_YOUTUBE = True
except ImportError:
    HAS_YOUTUBE = False


def check_ffmpeg() -> bool:
    """Check if ffmpeg and ffprobe are available on the system PATH."""
    for cmd in ("ffmpeg", "ffprobe"):
        if shutil.which(cmd) is None:
            return False
    return True


def get_ffmpeg_version() -> Optional[str]:
    """Get ffmpeg version string, or None if unavailable."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10,
        )
        first_line = result.stdout.split("\n")[0]
        return first_line
    except Exception:
        return None


class ContentFactoryWorker(QThread):
    """
    Main Content Factory worker — runs the full pipeline in a background thread.

    Signals:
        progress_changed(float, str) — (0-100, status message)
        phase_changed(str) — current phase name
        job_completed(ContentJob) — emitted when done (success or partial)
        job_failed(str) — emitted on fatal error
    """

    progress_changed = pyqtSignal(float, str)
    phase_changed = pyqtSignal(str)
    job_completed = pyqtSignal(object)    # ContentJob
    job_failed = pyqtSignal(str)

    def __init__(self, config: ContentFactoryConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._job = ContentJob(config=config)
        self._cancelled = threading.Event()

    @property
    def job(self) -> ContentJob:
        return self._job

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled.set()
        self._job.status = JobStatus.CANCELLED

    def run(self) -> None:
        """Execute the full Content Factory pipeline."""
        job = self._job
        job.status = JobStatus.PENDING

        try:
            # ── Pre-flight: Check ffmpeg ──
            if not check_ffmpeg():
                job.status = JobStatus.FAILED
                job.add_error(
                    "ffmpeg/ffprobe not found. Install ffmpeg to use Content Factory. "
                    "https://ffmpeg.org/download.html"
                )
                self.job_failed.emit("ffmpeg not found")
                return

            ffmpeg_ver = get_ffmpeg_version()
            if ffmpeg_ver:
                logger.info(f"[ContentFactory] {ffmpeg_ver}")

            # ── Phase 1: Plan (0-5%) ──
            self._emit_progress(0.0, "Planning content...")
            self.phase_changed.emit("Planning")

            planner = ContentPlanner()
            plan = planner.create_plan(self._config)
            job.plan = plan

            if planner.errors:
                for err in planner.errors:
                    job.add_error(f"[Planner] {err}")

            if self._cancelled.is_set():
                return

            # ── Phase 2: Master audio (5-25%) ──
            if self._config.master_config.enabled:
                self._emit_progress(5.0, "Mastering audio...")
                self.phase_changed.emit("Mastering")
                job.status = JobStatus.MASTERING

                master = BatchMasterLite(self._config.master_config)
                all_songs = self._config.songs
                total_songs = len(all_songs)

                def master_progress(idx, total, msg):
                    pct = 5.0 + (idx / max(total, 1)) * 20.0
                    self._emit_progress(pct, msg)

                success = master.master_batch(
                    all_songs, self._config.output_dir + "/mastered",
                    master_progress,
                )
                if success < total_songs:
                    job.add_error(f"Mastering: {total_songs - success} songs failed")
                logger.info(f"[BatchMaster] Mastered {success}/{total_songs} songs")

            if self._cancelled.is_set():
                return

            # ── Phase 3: Generate metadata ──
            self._emit_progress(25.0, "Generating metadata...")
            meta_gen = MetadataGenerator(self._config)

            for lv in plan.long_videos:
                meta_gen.generate_long_metadata(lv)
            for sv in plan.shorts:
                meta_gen.generate_short_metadata(sv)

            if self._cancelled.is_set():
                return

            # ── Phase 4: Build long videos (25-55%) ──
            self._emit_progress(25.0, "Building long videos...")
            self.phase_changed.emit("Long Videos")
            job.status = JobStatus.RENDERING

            long_builder = LongVideoBuilder()
            n_longs = len(plan.long_videos)

            for i, lv_plan in enumerate(plan.long_videos):
                if self._cancelled.is_set():
                    return

                # Use default arg to capture loop variable correctly
                def long_progress(pct, msg, _i=i, _n=n_longs):
                    base = 25.0 + (_i / max(_n, 1)) * 30.0
                    scaled = base + (pct / 100.0) * (30.0 / max(_n, 1))
                    self._emit_progress(scaled, msg)

                try:
                    output = long_builder.build(
                        lv_plan, self._config.output_dir + "/long",
                        long_progress,
                    )
                    if output:
                        job.output_files.append(output)
                    else:
                        job.add_error(f"Long video #{i + 1} failed to build")
                except Exception as e:
                    logger.error(f"[ContentFactory] Long video #{i + 1} error: {e}")
                    job.add_error(f"Long video #{i + 1} error: {e}")
                    # Continue to next video — don't abort the whole pipeline

            if self._cancelled.is_set():
                return

            # ── Phase 5: Build shorts (55-80%) ──
            self._emit_progress(55.0, "Building shorts...")
            self.phase_changed.emit("Shorts")

            short_builder = ShortVideoBuilder()
            n_shorts = len(plan.shorts)

            for i, sv_plan in enumerate(plan.shorts):
                if self._cancelled.is_set():
                    return

                # Use default arg to capture loop variable correctly
                def short_progress(pct, msg, _i=i, _n=n_shorts):
                    base = 55.0 + (_i / max(_n, 1)) * 25.0
                    scaled = base + (pct / 100.0) * (25.0 / max(_n, 1))
                    self._emit_progress(scaled, msg)

                try:
                    output = short_builder.build(
                        sv_plan, self._config.output_dir + "/shorts",
                        short_progress,
                    )
                    if output:
                        job.output_files.append(output)
                    else:
                        job.add_error(f"Short #{i + 1} failed to build")
                except Exception as e:
                    logger.error(f"[ContentFactory] Short #{i + 1} error: {e}")
                    job.add_error(f"Short #{i + 1} error: {e}")
                    # Continue — don't abort pipeline for one failed short

            if self._cancelled.is_set():
                return

            # ── Phase 6: Upload (80-100%) ──
            if self._config.auto_upload and HAS_YOUTUBE:
                self._emit_progress(80.0, "Uploading to YouTube...")
                self.phase_changed.emit("Uploading")
                job.status = JobStatus.UPLOADING

                self._upload_videos(job, plan)
            else:
                self._emit_progress(80.0, "Skipping upload (disabled)")

            # ── Done ──
            if job.errors:
                job.status = JobStatus.COMPLETED
                self._emit_progress(
                    100.0,
                    f"Done with {len(job.errors)} warning(s)"
                )
            else:
                job.status = JobStatus.COMPLETED
                self._emit_progress(100.0, "Content Factory complete!")

            job.progress = 100.0
            self.job_completed.emit(job)

        except Exception as e:
            logger.error(f"[ContentFactory] Fatal error: {e}")
            job.status = JobStatus.FAILED
            job.add_error(f"Fatal: {e}")
            self.job_failed.emit(str(e))

    # ─── Upload ──────────────────────────────────────────────────

    def _upload_videos(self, job: ContentJob, plan: ProductionPlan) -> None:
        """Upload completed videos to YouTube."""
        uploader = YouTubeUploader()

        if not uploader.is_authenticated():
            if not uploader.authenticate():
                job.add_error("YouTube authentication failed")
                return

        privacy = self._config.upload_privacy
        all_videos: List = []

        # Long videos first
        for lv in plan.long_videos:
            if lv.output_path and lv.status == JobStatus.COMPLETED:
                all_videos.append(("long", lv))

        # Then shorts
        for sv in plan.shorts:
            if sv.output_path and sv.status == JobStatus.COMPLETED:
                all_videos.append(("short", sv))

        total = len(all_videos)
        for i, (vtype, vplan) in enumerate(all_videos):
            if self._cancelled.is_set():
                return

            pct = 80.0 + (i / max(total, 1)) * 20.0
            self._emit_progress(pct, f"Uploading {i + 1}/{total}: {vplan.title}")

            def upload_progress(upload_pct):
                inner_pct = pct + (upload_pct / 100.0) * (20.0 / max(total, 1))
                self._emit_progress(inner_pct, f"Uploading: {upload_pct:.0f}%")

            try:
                url = uploader.upload_video(
                    video_path=vplan.output_path,
                    title=vplan.title,
                    description=vplan.description,
                    tags=vplan.tags,
                    category_id="10",   # Music
                    privacy=privacy,
                    progress_callback=upload_progress,
                )
                if url:
                    logger.info(f"[Upload] Success: {url}")
                else:
                    job.add_error(f"Upload failed: {vplan.title}")
            except Exception as e:
                job.add_error(f"Upload error ({vplan.title}): {e}")

    # ─── Helpers ─────────────────────────────────────────────────

    def _emit_progress(self, pct: float, msg: str) -> None:
        """Emit progress signal and update job."""
        self._job.progress = pct
        self._job.current_step = msg
        self.progress_changed.emit(pct, msg)
