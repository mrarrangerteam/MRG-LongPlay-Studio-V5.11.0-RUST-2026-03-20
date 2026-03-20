"""
Content Factory — Long Video Builder

Builds long compilation videos (16:9, ~1 hour) from mastered songs + background videos.

Pipeline (2-pass approach):
  Pass 1: Concat mastered audio with crossfades → single WAV
  Pass 2: Loop background video(s) to match audio duration → mux audio + video → MP4

Uses FFmpeg subprocess for all video operations.
Audio crossfades use FFmpeg filter_complex with acrossfade filter.
Background videos are looped using -stream_loop for efficiency.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Callable

from .models import (
    LongVideoPlan, SongEntry, BackgroundVideo,
    CrossfadeCurve, JobStatus,
)

logger = logging.getLogger(__name__)


class LongVideoBuilder:
    """
    Builds a single long compilation video.

    Usage:
        builder = LongVideoBuilder()
        output = builder.build(plan, output_dir, progress_callback)
    """

    # Audio normalization: all inputs → 48kHz stereo before crossfade
    SAMPLE_RATE = 48000
    CHANNELS = 2

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self._ffmpeg = ffmpeg_path
        self._ffprobe = ffmpeg_path.replace("ffmpeg", "ffprobe")

    # ─── Main Build ──────────────────────────────────────────────

    def build(
        self,
        plan: LongVideoPlan,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Optional[str]:
        """
        Build a long compilation video.

        Returns output MP4 path, or None on failure.
        """
        if not plan.songs:
            logger.error("[LongBuilder] No songs in plan")
            return None

        os.makedirs(output_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="longplay_long_")

        try:
            # Step 1: Normalize all audio to uniform format (10%)
            if progress_callback:
                progress_callback(5.0, "Normalizing audio formats...")

            norm_paths = self._normalize_audio(plan.songs, temp_dir)
            if not norm_paths:
                logger.error("[LongBuilder] Audio normalization failed")
                return None

            # Step 2: Concat audio with crossfades (30%)
            if progress_callback:
                progress_callback(15.0, "Concatenating audio with crossfades...")

            audio_path = os.path.join(temp_dir, "concat_audio.wav")
            if not self._concat_audio_crossfade(
                norm_paths, audio_path, plan.crossfade_sec, plan.crossfade_curve
            ):
                # Fallback: simple concat without crossfade
                logger.warning("[LongBuilder] Crossfade failed, using simple concat")
                if not self._concat_audio_simple(norm_paths, audio_path):
                    return None

            # Step 3: Get audio duration
            audio_duration = self._get_duration(audio_path)
            plan.total_duration_sec = audio_duration

            # Step 4: Prepare looped background video (50%)
            if progress_callback:
                progress_callback(35.0, "Looping background video...")

            video_path = None
            if plan.bg_videos:
                video_path = os.path.join(temp_dir, "looped_bg.mp4")
                if not self._loop_background(
                    plan.bg_videos, video_path, audio_duration
                ):
                    video_path = None

            # Step 5: Mux audio + video → final MP4 (80%)
            if progress_callback:
                progress_callback(55.0, "Encoding final video...")

            output_name = f"{plan.video_id}.mp4"
            output_path = os.path.join(output_dir, output_name)

            if video_path:
                success = self._mux_audio_video(audio_path, video_path, output_path)
            else:
                # Audio-only: generate black background
                success = self._audio_to_video_black(audio_path, output_path)

            if success:
                plan.output_path = output_path
                plan.status = JobStatus.COMPLETED
                if progress_callback:
                    progress_callback(100.0, "Long video complete")
                return output_path
            else:
                plan.status = JobStatus.FAILED
                return None

        except Exception as e:
            logger.error(f"[LongBuilder] Build failed: {e}")
            plan.status = JobStatus.FAILED
            return None
        finally:
            # Clean up temp files
            self._cleanup_temp(temp_dir)

    # ─── Audio Normalization ─────────────────────────────────────

    def _normalize_audio(
        self, songs: List[SongEntry], temp_dir: str
    ) -> List[str]:
        """Normalize all songs to 48kHz stereo WAV (required for acrossfade)."""
        paths = []
        for i, song in enumerate(songs):
            src = song.effective_path
            dst = os.path.join(temp_dir, f"norm_{i:04d}.wav")

            cmd = [
                self._ffmpeg, "-y", "-i", src,
                "-ar", str(self.SAMPLE_RATE),
                "-ac", str(self.CHANNELS),
                "-c:a", "pcm_s24le",
                "-v", "warning",
                dst,
            ]

            try:
                subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                if os.path.exists(dst):
                    paths.append(dst)
                else:
                    logger.error(f"[LongBuilder] Normalize failed: {song.title}")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                logger.error(f"[LongBuilder] Normalize error for {song.title}: {e}")

        return paths

    # ─── Audio Crossfade ─────────────────────────────────────────

    def _concat_audio_crossfade(
        self,
        audio_paths: List[str],
        output_path: str,
        crossfade_sec: float,
        curve: CrossfadeCurve,
    ) -> bool:
        """Concat audio files with acrossfade filter_complex."""
        if len(audio_paths) < 2:
            # Single file: just copy
            if audio_paths:
                return self._copy_file(audio_paths[0], output_path)
            return False

        # Build filter_complex for chained acrossfade
        # [0:a][1:a]acrossfade=d=3:c1=exp:c2=exp[a01];
        # [a01][2:a]acrossfade=d=3:c1=exp:c2=exp[a02]; ...
        inputs = []
        for p in audio_paths:
            inputs.extend(["-i", p])

        curve_name = curve.value
        filters = []
        n = len(audio_paths)

        for i in range(n - 1):
            if i == 0:
                src_left = "[0:a]"
            else:
                src_left = f"[a{i - 1:02d}]"

            src_right = f"[{i + 1}:a]"

            if i == n - 2:
                out_label = "[outa]"
            else:
                out_label = f"[a{i:02d}]"

            filters.append(
                f"{src_left}{src_right}acrossfade=d={crossfade_sec}"
                f":c1={curve_name}:c2={curve_name}{out_label}"
            )

        filter_str = ";".join(filters)

        cmd = [
            self._ffmpeg, "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[outa]",
            "-c:a", "pcm_s24le",
            "-ar", str(self.SAMPLE_RATE),
            "-v", "warning",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600, text=True)
            if result.returncode == 0 and os.path.exists(output_path):
                return True
            logger.error(f"[LongBuilder] Crossfade failed: {result.stderr[-500:]}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("[LongBuilder] Crossfade timed out")
            return False

    def _concat_audio_simple(
        self, audio_paths: List[str], output_path: str
    ) -> bool:
        """Fallback: simple concat using concat demuxer (no crossfade)."""
        # Create concat list in temp dir (not output dir) to avoid pollution
        concat_list = os.path.join(
            tempfile.gettempdir(), f"longplay_concat_{os.getpid()}.txt"
        )
        try:
            with open(concat_list, "w") as f:
                for p in audio_paths:
                    # Escape single quotes in paths for FFmpeg concat format
                    safe_p = p.replace("'", "'\\''")
                    f.write(f"file '{safe_p}'\n")

            cmd = [
                self._ffmpeg, "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list,
                "-c:a", "pcm_s24le",
                "-v", "warning",
                output_path,
            ]

            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            return os.path.exists(output_path)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"[LongBuilder] Simple concat failed: {e}")
            return False
        finally:
            # Always clean up temp concat file
            try:
                os.remove(concat_list)
            except OSError:
                pass

    # ─── Background Video Looping ────────────────────────────────

    def _loop_background(
        self,
        bg_videos: List[BackgroundVideo],
        output_path: str,
        target_duration: float,
    ) -> bool:
        """Loop background video(s) to match audio duration using -stream_loop."""
        if not bg_videos:
            return False

        # Use first background video and loop it
        bg = bg_videos[0]
        bg_duration = bg.duration_sec if bg.duration_sec > 0 else 30.0

        # Calculate loop count needed
        loop_count = max(1, int(target_duration / bg_duration) + 1)

        cmd = [
            self._ffmpeg, "-y",
            "-stream_loop", str(loop_count),
            "-i", bg.file_path,
            "-t", str(target_duration),
            "-c", "copy",         # Stream copy = instant
            "-an",                # No audio from video
            "-v", "warning",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
            if result.returncode == 0 and os.path.exists(output_path):
                return True

            # Fallback: re-encode if stream copy fails
            cmd_reencode = [
                self._ffmpeg, "-y",
                "-stream_loop", str(loop_count),
                "-i", bg.file_path,
                "-t", str(target_duration),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-an", "-v", "warning",
                output_path,
            ]
            subprocess.run(cmd_reencode, capture_output=True, timeout=600, check=True)
            return os.path.exists(output_path)

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"[LongBuilder] Video loop failed: {e}")
            return False

    # ─── Mux Audio + Video ───────────────────────────────────────

    def _mux_audio_video(
        self, audio_path: str, video_path: str, output_path: str
    ) -> bool:
        """Mux final audio with looped background video → MP4."""
        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            "-v", "warning",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=1800, check=True)
            return os.path.exists(output_path)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"[LongBuilder] Mux failed: {e}")
            return False

    def _audio_to_video_black(self, audio_path: str, output_path: str) -> bool:
        """Generate black 1920x1080 video from audio (no background video)."""
        duration = self._get_duration(audio_path)

        cmd = [
            self._ffmpeg, "-y",
            "-f", "lavfi", "-i", f"color=c=black:s=1920x1080:d={duration}:r=24",
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            "-v", "warning",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=1800, check=True)
            return os.path.exists(output_path)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"[LongBuilder] Black video generation failed: {e}")
            return False

    # ─── Utilities ───────────────────────────────────────────────

    def _get_duration(self, file_path: str) -> float:
        """Get media duration via ffprobe."""
        try:
            result = subprocess.run(
                [self._ffprobe, "-v", "quiet", "-print_format", "json",
                 "-show_format", file_path],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except Exception:
            return 0.0

    @staticmethod
    def _copy_file(src: str, dst: str) -> bool:
        """Copy a file."""
        try:
            shutil.copy2(src, dst)
            return True
        except OSError:
            return False

    @staticmethod
    def _cleanup_temp(temp_dir: str) -> None:
        """Remove temporary directory."""
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError:
            pass
