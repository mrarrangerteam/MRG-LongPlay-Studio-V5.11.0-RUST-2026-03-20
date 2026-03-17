"""
Content Factory — Short Video Builder

Builds YouTube Shorts (9:16, ≤29s) from song hooks + background videos.

Pipeline:
  1. Extract hook from song (using HookExtractor, default ≤29s)
  2. Convert background video 16:9 → 9:16 (blur_fill / scale_crop / scale_fit)
  3. Add text overlay (song title + artist) via FFmpeg drawtext
  4. Mux audio + video → MP4

Resolution: 1080x1920 (YouTube recommended for Shorts)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable

from .models import (
    ShortVideoPlan, SongEntry, BackgroundVideo,
    VerticalStrategy, JobStatus,
)

logger = logging.getLogger(__name__)

# HookExtractor for extracting song hooks
try:
    from hook_extractor import HookExtractor
    HAS_HOOK_EXTRACTOR = True
except ImportError:
    HAS_HOOK_EXTRACTOR = False


class ShortVideoBuilder:
    """
    Builds a single YouTube Short from a song hook + background video.

    Usage:
        builder = ShortVideoBuilder()
        output = builder.build(plan, output_dir, progress_callback)
    """

    WIDTH = 1080
    HEIGHT = 1920

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self._ffmpeg = ffmpeg_path
        self._hook_extractor: Optional[HookExtractor] = None
        if HAS_HOOK_EXTRACTOR:
            self._hook_extractor = HookExtractor(
                hook_duration=29.0,     # Match Shorts max
                min_hook_duration=10.0,
                max_hook_duration=29.0,
            )

    # ─── Main Build ──────────────────────────────────────────────

    def build(
        self,
        plan: ShortVideoPlan,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Optional[str]:
        """
        Build a single YouTube Short.

        Returns output MP4 path, or None on failure.
        """
        if not plan.song:
            logger.error("[ShortBuilder] No song in plan")
            return None

        os.makedirs(output_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="longplay_short_")

        try:
            # Step 1: Extract hook from song (20%)
            if progress_callback:
                progress_callback(10.0, f"Extracting hook: {plan.song.title}")

            hook_path = self._extract_hook(plan.song, temp_dir, plan.max_duration_sec)
            if not hook_path:
                logger.error(f"[ShortBuilder] Hook extraction failed: {plan.song.title}")
                plan.status = JobStatus.FAILED
                return None

            hook_duration = self._get_duration(hook_path)

            # Step 2: Process background video to vertical (50%)
            if progress_callback:
                progress_callback(30.0, "Creating vertical video...")

            vertical_video = None
            if plan.bg_video:
                vertical_video = os.path.join(temp_dir, "vertical_bg.mp4")
                if not self._make_vertical(
                    plan.bg_video, vertical_video,
                    hook_duration, plan.vertical_strategy
                ):
                    vertical_video = None

            # Step 3: Add text overlay + mux (80%)
            if progress_callback:
                progress_callback(60.0, "Adding text and encoding...")

            output_name = f"{plan.video_id}.mp4"
            output_path = os.path.join(output_dir, output_name)

            if vertical_video:
                success = self._compose_short(
                    hook_path, vertical_video, output_path,
                    plan.song.title, plan.song.artist, plan.text_preset
                )
            else:
                # No bg video: black background with text
                success = self._audio_to_vertical_black(
                    hook_path, output_path,
                    plan.song.title, plan.song.artist
                )

            if success:
                plan.output_path = output_path
                plan.status = JobStatus.COMPLETED
                if progress_callback:
                    progress_callback(100.0, "Short complete")
                return output_path
            else:
                plan.status = JobStatus.FAILED
                return None

        except Exception as e:
            logger.error(f"[ShortBuilder] Build failed: {e}")
            plan.status = JobStatus.FAILED
            return None
        finally:
            self._cleanup_temp(temp_dir)

    # ─── Hook Extraction ─────────────────────────────────────────

    def _extract_hook(
        self, song: SongEntry, temp_dir: str, max_duration: float
    ) -> Optional[str]:
        """Extract the hook section from a song."""
        # If hook already extracted, use it
        if song.hook_file_path and os.path.exists(song.hook_file_path):
            return song.hook_file_path

        src = song.effective_path

        if self._hook_extractor:
            # Use HookExtractor for intelligent hook detection
            result = self._hook_extractor.analyze_audio(src)
            song.hook_start_sec = result.hook_start_sec
            song.hook_end_sec = result.hook_end_sec
            song.hook_duration_sec = result.hook_duration_sec
            song.hook_confidence = result.hook_confidence

            hook_path = self._hook_extractor.extract_hook(src, temp_dir)
            if hook_path and os.path.exists(hook_path):
                song.hook_file_path = hook_path
                return hook_path

        # Fallback: extract middle section with FFmpeg
        duration = self._get_duration(src)
        if duration <= 0:
            duration = 180.0

        hook_dur = min(max_duration, duration)
        start = max(0, (duration - hook_dur) / 2)

        hook_path = os.path.join(temp_dir, f"{Path(src).stem}_hook.wav")
        cmd = [
            self._ffmpeg, "-y",
            "-i", src,
            "-ss", str(start),
            "-t", str(hook_dur),
            "-ar", "48000", "-ac", "2",
            "-c:a", "pcm_s24le",
            "-v", "warning",
            hook_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
            if os.path.exists(hook_path):
                song.hook_start_sec = start
                song.hook_end_sec = start + hook_dur
                song.hook_duration_sec = hook_dur
                song.hook_confidence = 0.3
                song.hook_file_path = hook_path
                return hook_path
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"[ShortBuilder] Hook extract fallback failed: {e}")

        return None

    # ─── Vertical Video Conversion ───────────────────────────────

    def _make_vertical(
        self,
        bg: BackgroundVideo,
        output_path: str,
        duration: float,
        strategy: VerticalStrategy,
    ) -> bool:
        """Convert 16:9 background video to 9:16 vertical."""
        w, h = self.WIDTH, self.HEIGHT

        if strategy == VerticalStrategy.BLUR_FILL:
            # Blurred background + sharp center crop overlay
            filter_str = (
                f"split[original][forblur];"
                f"[forblur]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h}:(iw-ow)/2:(ih-oh)/2,gblur=sigma=20[bg];"
                f"[original]scale={w}:-2:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[out]"
            )
        elif strategy == VerticalStrategy.SCALE_CROP:
            # Simple center crop to fill 9:16
            filter_str = (
                f"scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h}:(iw-ow)/2:(ih-oh)/2[out]"
            )
        else:  # SCALE_FIT
            # Scale to fit with black bars
            filter_str = (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black[out]"
            )

        # Loop if needed
        loop_count = max(0, int(duration / max(bg.duration_sec, 1)) + 1)

        cmd = [
            self._ffmpeg, "-y",
            "-stream_loop", str(loop_count),
            "-i", bg.file_path,
            "-t", str(duration),
            "-filter_complex", filter_str,
            "-map", "[out]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an",
            "-v", "warning",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            return os.path.exists(output_path)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"[ShortBuilder] Vertical conversion failed: {e}")
            return False

    # ─── Compose Short (video + audio + text) ────────────────────

    def _compose_short(
        self,
        audio_path: str,
        video_path: str,
        output_path: str,
        song_title: str,
        artist: str,
        text_preset: str,
    ) -> bool:
        """Mux vertical video + hook audio + text overlay → final Short."""
        # Build drawtext filter for song info
        title_text = self._escape_ffmpeg(song_title)
        artist_text = self._escape_ffmpeg(artist) if artist else ""

        # Title: large, centered near bottom
        drawtext_title = (
            f"drawtext=text='{title_text}'"
            f":fontsize=48:fontcolor=white"
            f":borderw=3:bordercolor=black"
            f":x=(w-text_w)/2:y=h-h/6"
        )

        # Artist: smaller, below title
        drawtext_artist = ""
        if artist_text:
            drawtext_artist = (
                f",drawtext=text='{artist_text}'"
                f":fontsize=32:fontcolor=0xCCCCCC"
                f":borderw=2:bordercolor=black"
                f":x=(w-text_w)/2:y=h-h/6+60"
            )

        filter_str = f"{drawtext_title}{drawtext_artist}"

        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", f"[0:v]{filter_str}[vout]",
            "-map", "[vout]",
            "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            "-v", "warning",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            return os.path.exists(output_path)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"[ShortBuilder] Compose failed: {e}")
            return False

    def _audio_to_vertical_black(
        self,
        audio_path: str,
        output_path: str,
        song_title: str,
        artist: str,
    ) -> bool:
        """Generate black 1080x1920 video with text from audio."""
        duration = self._get_duration(audio_path)
        title_text = self._escape_ffmpeg(song_title)
        artist_text = self._escape_ffmpeg(artist) if artist else ""

        drawtext = (
            f"drawtext=text='{title_text}'"
            f":fontsize=48:fontcolor=white"
            f":borderw=3:bordercolor=black"
            f":x=(w-text_w)/2:y=h/2-30"
        )
        if artist_text:
            drawtext += (
                f",drawtext=text='{artist_text}'"
                f":fontsize=32:fontcolor=0xCCCCCC"
                f":borderw=2:bordercolor=black"
                f":x=(w-text_w)/2:y=h/2+30"
            )

        cmd = [
            self._ffmpeg, "-y",
            "-f", "lavfi", "-i",
            f"color=c=0x111111:s={self.WIDTH}x{self.HEIGHT}:d={duration}:r=24",
            "-i", audio_path,
            "-filter_complex", f"[0:v]{drawtext}[vout]",
            "-map", "[vout]",
            "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            "-v", "warning",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            return os.path.exists(output_path)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.error(f"[ShortBuilder] Black vertical failed: {e}")
            return False

    # ─── Utilities ───────────────────────────────────────────────

    @staticmethod
    def _escape_ffmpeg(text: str) -> str:
        """Escape text for FFmpeg drawtext filter (textfile-compatible).

        FFmpeg drawtext uses : as delimiter and ' for quoting.
        We escape all special chars for the text= parameter.
        """
        # FFmpeg drawtext escaping order matters
        text = text.replace("\\", "\\\\")   # backslash first
        text = text.replace("'", "'\\\\\\''")  # single quote
        text = text.replace(":", "\\:")      # colon (delimiter)
        text = text.replace(";", "\\;")      # semicolon
        text = text.replace("%", "%%")       # percent (time variable)
        text = text.replace("[", "\\[")      # bracket
        text = text.replace("]", "\\]")
        return text

    def _get_duration(self, file_path: str) -> float:
        """Get media duration via ffprobe."""
        try:
            result = subprocess.run(
                [self._ffmpeg.replace("ffmpeg", "ffprobe"),
                 "-v", "quiet", "-print_format", "json",
                 "-show_format", file_path],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except Exception:
            return 0.0

    @staticmethod
    def _cleanup_temp(temp_dir: str) -> None:
        """Remove temporary directory."""
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError:
            pass
