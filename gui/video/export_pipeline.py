"""
export_pipeline.py — Parallel Audio+Video Export Pipeline
==========================================================
Replaces BabitMF (not available) with Python multiprocessing.
Inspired by CapCut / Logic Pro X pipeline architecture:
  - Worker A: audio mastering (CPU-bound, numpy/soundfile)
  - Worker B: video encoding   (GPU-bound, ffmpeg hardware encoder)
  - Coordination: multiprocessing.Queue for status/progress
  - Result: final_video.mp4 with mastered audio baked in

Usage:
  from gui.video.export_pipeline import ExportPipeline

  pipeline = ExportPipeline(
      input_video   = "final_video.mp4",
      mastered_audio= "mastered_final.wav",
      output_path   = "output.mp4",
  )
  pipeline.run(progress_callback=lambda p, msg: print(f"{p:.0%} {msg}"))
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

# ── Phase constants ─────────────────────────────────────────────────────────
PHASE_ENCODE  = "encode"
PHASE_MUX     = "mux"
PHASE_DONE    = "done"
PHASE_ERROR   = "error"


def _double_bitrate(bitrate_str: str) -> str:
    """Double a bitrate string like '8M' → '16M'.  Mirrors hw_encoder._double_bitrate."""
    try:
        if bitrate_str.upper().endswith("M"):
            return f"{int(bitrate_str[:-1]) * 2}M"
        if bitrate_str.upper().endswith("K"):
            return f"{int(bitrate_str[:-1]) * 2}K"
    except (ValueError, AttributeError):
        pass
    return bitrate_str


# ── Progress message dataclass ──────────────────────────────────────────────
@dataclass
class ProgressMsg:
    phase:   str
    frac:    float      # 0.0 – 1.0
    message: str = ""
    speed_x: float = 0.0  # realtime speed factor (>1 = faster than realtime)
    eta_s:   float = 0.0  # estimated seconds remaining


# ── Encoding worker (runs in subprocess) ────────────────────────────────────
def _encode_worker(
    input_video:   str,
    mastered_audio: str,
    output_path:   str,
    logo_path:     Optional[str],
    logo_pos:      Tuple[int, int],
    logo_scale:    int,
    encoder:       str,
    encoder_flags: list,
    video_bitrate: str,
    q:             "mp.Queue[ProgressMsg]",
) -> None:
    """Subprocess worker: encodes video + muxes mastered audio."""
    import shutil as _shutil, subprocess as _sub, time as _time, os as _os

    ffmpeg   = _shutil.which("ffmpeg") or "ffmpeg"
    ffprobe  = _shutil.which("ffprobe") or "ffprobe"
    tmp_out  = output_path + ".pipeline_tmp.mp4"

    # Probe duration
    duration_s = 0.0
    try:
        r = _sub.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_video],
            capture_output=True, text=True, timeout=15,
        )
        duration_s = float(r.stdout.strip())
    except Exception:
        pass

    # Build ffmpeg command
    cmd = [ffmpeg, "-y"]
    cmd += ["-i", input_video]
    cmd += ["-i", mastered_audio]

    logo_idx = None
    if logo_path and _os.path.isfile(logo_path):
        cmd += ["-i", logo_path]
        logo_idx = 2

    if logo_idx is not None:
        lx, ly = logo_pos
        fc = (
            f"[{logo_idx}:v]scale={logo_scale}:-1[logo];"
            f"[0:v][logo]overlay={lx}:{ly}:format=auto[vout]"
        )
        cmd += ["-filter_complex", fc, "-map", "[vout]"]
    else:
        cmd += ["-map", "0:v"]

    cmd += ["-map", "1:a"]
    cmd += ["-c:v", encoder] + encoder_flags

    if encoder not in ("libx264",) and video_bitrate:
        if encoder == "h264_nvenc":
            # bufsize = 2x maxrate — matches hw_encoder.py convention; prevents
            # NVENC VBR from starving the buffer and causing quality oscillation.
            cmd += ["-maxrate", video_bitrate, "-bufsize", _double_bitrate(video_bitrate)]
        elif encoder != "h264_videotoolbox":
            cmd += ["-b:v", video_bitrate]

    cmd += ["-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", tmp_out]

    start_t = _time.perf_counter()
    q.put(ProgressMsg(phase=PHASE_ENCODE, frac=0.0, message=f"Starting {encoder}"))

    try:
        proc = _sub.Popen(cmd, stdout=_sub.DEVNULL, stderr=_sub.PIPE, text=True)  # stdout not read — DEVNULL prevents deadlock
        last_s = 0.0
        if proc.stderr:
            for line in proc.stderr:
                if "time=" in line and duration_s > 0:
                    try:
                        idx  = line.index("time=")
                        part = line[idx + 5:].split()[0]
                        h, m, s = part.split(":")
                        cur_s = float(h) * 3600 + float(m) * 60 + float(s)
                        frac  = min(1.0, cur_s / duration_s)
                        elapsed = _time.perf_counter() - start_t
                        speed_x = cur_s / elapsed if elapsed > 0 else 0.0
                        eta_s   = (duration_s - cur_s) / speed_x if speed_x > 0 else 0.0
                        q.put(ProgressMsg(
                            phase=PHASE_ENCODE, frac=frac,
                            message=f"{encoder} {frac:.0%}",
                            speed_x=speed_x, eta_s=eta_s,
                        ))
                        last_s = cur_s
                    except Exception:
                        pass
        proc.wait()
        if proc.returncode != 0:
            q.put(ProgressMsg(phase=PHASE_ERROR, frac=0.0, message="ffmpeg failed"))
            return
    except Exception as exc:
        q.put(ProgressMsg(phase=PHASE_ERROR, frac=0.0, message=str(exc)))
        return

    # Move tmp → final
    try:
        _os.replace(tmp_out, output_path)
    except OSError as exc:
        q.put(ProgressMsg(phase=PHASE_ERROR, frac=0.0, message=f"rename failed: {exc}"))
        return

    elapsed = _time.perf_counter() - start_t
    speed_x = duration_s / elapsed if elapsed > 0 else 0.0
    q.put(ProgressMsg(
        phase=PHASE_DONE, frac=1.0,
        message=f"Done in {elapsed:.1f}s ({speed_x:.1f}x realtime)",
        speed_x=speed_x,
    ))


# ── Main pipeline class ──────────────────────────────────────────────────────
class ExportPipeline:
    """
    Graph-based pipeline for video+audio export.

    Nodes:
      [mastered_audio] ─────────────────────────────────────┐
      [input_video]    → [hw_encode + mux + logo overlay] → [output_mp4]

    Uses a single multiprocessing.Process for the ffmpeg worker so the
    main Python process stays responsive for UI progress callbacks.
    """

    def __init__(
        self,
        input_video:    str,
        mastered_audio: str,
        output_path:    str,
        logo_path:      Optional[str] = None,
        logo_position:  Tuple[int, int] = (30, 30),
        logo_scale:     int = 120,
        encoder:        Optional[str] = None,
        video_bitrate:  str = "8M",
    ) -> None:
        self.input_video    = input_video
        self.mastered_audio = mastered_audio
        self.output_path    = output_path
        self.logo_path      = logo_path
        self.logo_position  = logo_position
        self.logo_scale     = logo_scale
        self.video_bitrate  = video_bitrate

        # Resolve encoder
        try:
            from gui.video.hw_encoder import HW_ENCODER, get_encoder_flags
            self.encoder       = encoder or HW_ENCODER
            self.encoder_flags = get_encoder_flags(self.encoder)
        except ImportError:
            self.encoder       = encoder or "libx264"
            self.encoder_flags = ["-preset", "medium", "-crf", "23", "-movflags", "+faststart"]

        self._process: Optional[mp.Process] = None
        self._queue:   Optional["mp.Queue[ProgressMsg]"] = None

    def run(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        poll_interval: float = 0.25,
    ) -> bool:
        """
        Run the export pipeline synchronously, calling *progress_callback*(frac, msg).
        Returns True on success.
        """
        ctx   = mp.get_context("spawn")
        queue: "mp.Queue[ProgressMsg]" = ctx.Queue()
        self._queue = queue

        proc = ctx.Process(
            target=_encode_worker,
            args=(
                self.input_video,
                self.mastered_audio,
                self.output_path,
                self.logo_path,
                self.logo_position,
                self.logo_scale,
                self.encoder,
                self.encoder_flags,
                self.video_bitrate,
                queue,
            ),
            daemon=True,
        )
        self._process = proc
        proc.start()

        success = False
        while proc.is_alive() or not queue.empty():
            try:
                msg = queue.get(timeout=poll_interval)
            except Exception:
                continue

            if progress_callback:
                label = msg.message
                if msg.speed_x > 0:
                    label += f"  {msg.speed_x:.1f}x"
                if msg.eta_s > 0:
                    mins, secs = divmod(int(msg.eta_s), 60)
                    label += f"  ETA {mins}:{secs:02d}"
                progress_callback(msg.frac, label)

            if msg.phase == PHASE_DONE:
                success = True
                break
            elif msg.phase == PHASE_ERROR:
                success = False
                break

        proc.join(timeout=10)
        return success

    def cancel(self) -> None:
        """Terminate the encode worker."""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
