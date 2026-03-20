"""
Multi-track video exporter — builds FFmpeg commands from a Project model.

Provides:
    FormatPreset         — Dataclass with codec / bitrate / container settings
    MultiTrackExporter   — Builds and runs FFmpeg for multi-track export
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from gui.models.track import Clip, Track, TrackType, Project


# ---------------------------------------------------------------------------
# Format presets
# ---------------------------------------------------------------------------
@dataclass
class FormatPreset:
    """Container for export format settings."""
    name: str = "MP4 1080p"
    container: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    video_bitrate: str = "8M"
    audio_bitrate: str = "192k"
    resolution: Tuple[int, int] = (1920, 1080)
    fps: float = 30.0
    extra_flags: List[str] = field(default_factory=list)


# Built-in presets
PRESETS: Dict[str, FormatPreset] = {
    "mp4_1080p": FormatPreset(
        name="MP4 1080p", container="mp4",
        video_codec="libx264", audio_codec="aac",
        video_bitrate="8M", audio_bitrate="192k",
        resolution=(1920, 1080), fps=30.0,
    ),
    "mp4_4k": FormatPreset(
        name="MP4 4K", container="mp4",
        video_codec="libx264", audio_codec="aac",
        video_bitrate="20M", audio_bitrate="256k",
        resolution=(3840, 2160), fps=30.0,
    ),
    "webm_1080p": FormatPreset(
        name="WebM 1080p", container="webm",
        video_codec="libvpx-vp9", audio_codec="libopus",
        video_bitrate="5M", audio_bitrate="128k",
        resolution=(1920, 1080), fps=30.0,
    ),
    "prores_1080p": FormatPreset(
        name="ProRes 1080p", container="mov",
        video_codec="prores_ks", audio_codec="pcm_s16le",
        video_bitrate="",  # prores doesn't use bitrate
        audio_bitrate="",
        resolution=(1920, 1080), fps=30.0,
        extra_flags=["-profile:v", "3"],  # HQ profile
    ),
}


# ---------------------------------------------------------------------------
# MultiTrackExporter
# ---------------------------------------------------------------------------
class MultiTrackExporter:
    """
    Builds an FFmpeg filter graph from a Project model and exports video.

    Supports:
        - Video track compositing via overlay filter
        - Audio track mixing via amix
        - Correct timing from clip start_time and in/out points
        - Progress callback (0.0 — 1.0)
    """

    def __init__(self) -> None:
        self._ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        self._process: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._cancelled = False

    # -- public API --------------------------------------------------------
    def export(
        self,
        project: Project,
        output_path: str,
        format_preset: Optional[FormatPreset] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """
        Export *project* to *output_path*.

        Returns True on success, False on failure/cancel.
        """
        preset = format_preset or PRESETS["mp4_1080p"]
        self._cancelled = False

        cmd = self._build_command(project, output_path, preset)
        if cmd is None:
            return False

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            duration = max(project.duration, 0.1)
            stderr_lines: List[str] = []

            # Read stderr for progress
            if self._process.stderr is not None:
                for line in self._process.stderr:
                    if self._cancelled:
                        self._process.terminate()
                        return False

                    stderr_lines.append(line)
                    # Parse FFmpeg time= output for progress
                    if "time=" in line:
                        time_str = self._parse_time(line)
                        if time_str is not None and progress_callback is not None:
                            frac = min(1.0, time_str / duration)
                            progress_callback(frac)

            self._process.wait()
            if progress_callback is not None:
                progress_callback(1.0)

            return self._process.returncode == 0

        except FileNotFoundError:
            # FFmpeg not found
            return False
        except OSError as exc:
            return False

    def export_async(
        self,
        project: Project,
        output_path: str,
        format_preset: Optional[FormatPreset] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        done_callback: Optional[Callable[[bool], None]] = None,
    ) -> threading.Thread:
        """Run export in a background thread."""
        def _run() -> None:
            result = self.export(project, output_path, format_preset, progress_callback)
            if done_callback is not None:
                done_callback(result)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def cancel(self) -> None:
        """Request cancellation of the current export."""
        self._cancelled = True
        if self._process is not None:
            try:
                self._process.terminate()
            except OSError:
                pass

    # -- command builder ---------------------------------------------------
    def _build_command(
        self,
        project: Project,
        output_path: str,
        preset: FormatPreset,
    ) -> Optional[List[str]]:
        """Build the full FFmpeg command list."""

        video_tracks = [t for t in project.tracks if t.type == TrackType.VIDEO]
        audio_tracks = [t for t in project.tracks if t.type == TrackType.AUDIO]

        video_clips: List[Tuple[Clip, Track]] = []
        audio_clips: List[Tuple[Clip, Track]] = []

        for t in video_tracks:
            for c in t.clips:
                if c.source_path and os.path.isfile(c.source_path):
                    video_clips.append((c, t))

        for t in audio_tracks:
            for c in t.clips:
                if c.source_path and os.path.isfile(c.source_path):
                    audio_clips.append((c, t))

        if not video_clips and not audio_clips:
            return None

        cmd: List[str] = [self._ffmpeg, "-y"]

        # Inputs
        input_idx = 0
        clip_input_map: Dict[str, int] = {}

        for clip, _track in video_clips + audio_clips:
            cmd.extend([
                "-ss", f"{clip.in_point:.3f}",
                "-t", f"{clip.duration:.3f}",
                "-i", clip.source_path,
            ])
            clip_input_map[clip.id] = input_idx
            input_idx += 1

        # Build filter graph
        filter_parts: List[str] = []
        res_w, res_h = preset.resolution

        # --- Video compositing ---
        if video_clips:
            # Create a base black canvas
            filter_parts.append(
                f"color=c=black:s={res_w}x{res_h}:d={project.duration:.3f}:r={preset.fps}[base]"
            )

            current_label = "base"
            for i, (clip, _track) in enumerate(video_clips):
                idx = clip_input_map[clip.id]
                scaled_label = f"v{i}scaled"
                overlay_label = f"v{i}out"

                # Scale input to resolution
                filter_parts.append(
                    f"[{idx}:v]scale={res_w}:{res_h}:force_original_aspect_ratio=decrease,"
                    f"pad={res_w}:{res_h}:(ow-iw)/2:(oh-ih)/2,"
                    f"setpts=PTS+{clip.start_time:.3f}/TB[{scaled_label}]"
                )

                # Overlay onto current composite
                enable_expr = (
                    f"between(t,{clip.start_time:.3f},{clip.end_time:.3f})"
                )
                filter_parts.append(
                    f"[{current_label}][{scaled_label}]overlay=0:0:"
                    f"enable='{enable_expr}'[{overlay_label}]"
                )
                current_label = overlay_label

            final_video = current_label
        else:
            # No video — generate black
            filter_parts.append(
                f"color=c=black:s={res_w}x{res_h}:d={project.duration:.3f}:r={preset.fps}[vout]"
            )
            final_video = "vout"

        # --- Audio mixing ---
        if audio_clips:
            delayed_labels: List[str] = []
            for i, (clip, _track) in enumerate(audio_clips):
                idx = clip_input_map[clip.id]
                lbl = f"a{i}del"
                delay_ms = int(clip.start_time * 1000)
                filter_parts.append(
                    f"[{idx}:a]adelay={delay_ms}|{delay_ms}[{lbl}]"
                )
                delayed_labels.append(f"[{lbl}]")

            if len(delayed_labels) > 1:
                inputs_str = "".join(delayed_labels)
                filter_parts.append(
                    f"{inputs_str}amix=inputs={len(delayed_labels)}:"
                    f"duration=longest:normalize=0[aout]"
                )
            else:
                # Single audio — rename
                filter_parts.append(
                    f"{delayed_labels[0]}anull[aout]"
                )
            final_audio = "aout"
        else:
            final_audio = None

        # Assemble filter_complex
        filter_graph = ";\n".join(filter_parts)
        cmd.extend(["-filter_complex", filter_graph])

        # Map outputs
        cmd.extend(["-map", f"[{final_video}]"])
        if final_audio:
            cmd.extend(["-map", f"[{final_audio}]"])

        # Codec / format
        cmd.extend(["-c:v", preset.video_codec])
        if preset.video_bitrate:
            cmd.extend(["-b:v", preset.video_bitrate])
        if final_audio:
            cmd.extend(["-c:a", preset.audio_codec])
            if preset.audio_bitrate:
                cmd.extend(["-b:a", preset.audio_bitrate])

        cmd.extend(preset.extra_flags)

        # Output
        cmd.append(output_path)

        return cmd

    # -- progress parsing --------------------------------------------------
    @staticmethod
    def _parse_time(line: str) -> Optional[float]:
        """Extract seconds from an FFmpeg stderr line containing 'time=HH:MM:SS.ss'."""
        try:
            idx = line.index("time=")
            time_part = line[idx + 5:].split()[0]
            if time_part == "N/A":
                return None
            parts = time_part.split(":")
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            return float(time_part)
        except (ValueError, IndexError):
            return None
