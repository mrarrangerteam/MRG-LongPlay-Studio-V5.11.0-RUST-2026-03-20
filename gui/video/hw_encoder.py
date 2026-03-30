"""
hw_encoder.py — Hardware Encoder Auto-Detection & Fallback Chain
================================================================
Detects the best available video encoder at import time and exposes:
  - detect_hw_encoder()   → str  (encoder name for ffmpeg -c:v)
  - get_encoder_flags()   → List[str]  (extra ffmpeg flags for that encoder)
  - HW_ENCODER            → str  (cached result, set at import)

Fallback chain (CapCut / Logic Pro X / iZotope-style):
  1. h264_nvenc       — NVIDIA GPU (Windows/Linux)
  2. h264_videotoolbox — Apple VideoToolbox (macOS)
  3. h264_qsv         — Intel QuickSync
  4. libx264          — software fallback (always available)

Usage:
  from gui.video.hw_encoder import HW_ENCODER, get_encoder_flags, export_video
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


# ── Encoder-specific flags ─────────────────────────────────────────────────
ENCODER_FLAGS: Dict[str, List[str]] = {
    "h264_nvenc": [
        "-preset", "p4",          # balanced speed/quality (NVENC SDK preset)
        "-tune", "hq",            # high quality mode
        "-rc", "vbr",             # variable bitrate
        "-cq", "23",              # quality level (lower = better, like CRF)
        "-b:v", "0",              # let VBR control bitrate
        "-profile:v", "high",
        "-level", "4.1",
    ],
    "h264_videotoolbox": [
        "-allow_sw", "1",         # allow software fallback within VideoToolbox
        "-realtime", "0",         # non-realtime = better quality
        "-profile:v", "high",
        "-level", "4.1",
        "-q:v", "65",             # 0-100 quality scale (VideoToolbox)
    ],
    "h264_qsv": [
        "-preset", "medium",
        "-global_quality", "23",  # QSV quality (lower = better)
        "-profile:v", "high",
        "-level", "4.1",
    ],
    "libx264": [
        "-preset", "medium",
        "-crf", "23",
        "-profile:v", "high",
        "-level", "4.1",
        "-movflags", "+faststart",
    ],
}

# ── Detection ──────────────────────────────────────────────────────────────
_CANDIDATE_ORDER = [
    "h264_nvenc",
    "h264_videotoolbox",
    "h264_qsv",
    "libx264",
]


def _probe_ffmpeg_encoders() -> set:
    """Return set of encoder names available in this ffmpeg build."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    try:
        result = subprocess.run(
            [ffmpeg, "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.splitlines()
        encoders = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2 and len(parts[0]) >= 6:
                # FFmpeg encoder list format: " V..... encoder_name ..."
                # First char is media type (V=video, A=audio, S=subtitle)
                encoders.add(parts[1])
        return encoders
    except Exception:
        return {"libx264"}  # safe fallback


def _test_encoder_functional(encoder: str) -> bool:
    """Quick encode test: 1 black frame. Returns True if encoder works."""
    if encoder == "libx264":
        return True  # always available if ffmpeg has it

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    flags  = ENCODER_FLAGS.get(encoder, [])
    cmd    = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1:r=1",
        "-c:v", encoder,
    ] + flags + [
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def detect_hw_encoder() -> str:
    """
    Detect best available hardware encoder.

    Tests candidates in order: NVENC → VideoToolbox → QuickSync → libx264.
    Each candidate is first checked in -encoders output, then functionally tested.
    Returns the name of the first working encoder.
    """
    available = _probe_ffmpeg_encoders()

    for encoder in _CANDIDATE_ORDER:
        if encoder not in available:
            continue
        if _test_encoder_functional(encoder):
            return encoder

    return "libx264"  # absolute fallback


def get_encoder_flags(encoder: Optional[str] = None) -> List[str]:
    """Return the ffmpeg flags for *encoder* (defaults to HW_ENCODER)."""
    enc = encoder or _get_or_detect()
    return ENCODER_FLAGS.get(enc, ENCODER_FLAGS["libx264"])[:]


# ── Lazy module-level cache — detected on first access, NOT at import ──────
# Detection runs subprocess calls (ffmpeg -encoders + test encode) which is
# slow (~0.5–2 s). Deferring to first use avoids blocking import in unit tests
# and in modules that import hw_encoder but don't immediately need the encoder.

_HW_ENCODER_CACHE: Optional[str] = None
_HW_DETECTED_AT: float = 0.0


def _get_or_detect() -> str:
    """Return cached encoder name, running detection on first call."""
    global _HW_ENCODER_CACHE, _HW_DETECTED_AT
    if _HW_ENCODER_CACHE is None:
        _HW_ENCODER_CACHE = detect_hw_encoder()
        _HW_DETECTED_AT = time.time()
    return _HW_ENCODER_CACHE


def __getattr__(name: str) -> str:
    """Lazy HW_ENCODER — subprocess detection deferred to first attribute access."""
    if name == "HW_ENCODER":
        return _get_or_detect()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def encoder_info() -> str:
    """Human-readable info about the selected encoder."""
    labels = {
        "h264_nvenc":        "NVIDIA NVENC (GPU)",
        "h264_videotoolbox": "Apple VideoToolbox (GPU)",
        "h264_qsv":          "Intel QuickSync (GPU)",
        "libx264":           "libx264 (CPU software)",
    }
    enc = _get_or_detect()
    return f"{enc} — {labels.get(enc, 'unknown')}"


# ── High-level export function ─────────────────────────────────────────────
def export_video(
    input_video: str,
    input_audio: str,
    output_path: str,
    encoder: Optional[str] = None,
    video_bitrate: str = "8M",
    audio_codec: str = "aac",
    audio_bitrate: str = "320k",
    logo_path: Optional[str] = None,
    logo_position: Tuple[int, int] = (30, 30),
    logo_scale: int = 120,
    progress_callback: Optional[Callable[[float], None]] = None,
    timeout: int = 7200,
) -> bool:
    """
    Export video with hardware-accelerated encoding.

    Replaces video audio stream with *input_audio*.
    Optionally overlays *logo_path* PNG (with alpha) at *logo_position*.
    Calls *progress_callback*(0.0–1.0) as encoding progresses.

    Returns True on success, False on failure.
    """
    enc    = encoder or HW_ENCODER
    flags  = get_encoder_flags(enc)
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

    # Get video duration for progress
    duration_s = _probe_duration(input_video)

    tmp_out = output_path + ".tmp.mp4"

    # Build command
    cmd: List[str] = [ffmpeg, "-y"]

    # Inputs
    cmd += ["-i", input_video]   # 0: source video
    cmd += ["-i", input_audio]   # 1: mastered audio

    if logo_path and os.path.isfile(logo_path):
        cmd += ["-i", logo_path]  # 2: logo PNG (RGBA)
        logo_input_idx = 2
    else:
        logo_input_idx = None

    # Filter complex (logo overlay if available)
    if logo_input_idx is not None:
        lx, ly = logo_position
        filter_complex = (
            f"[{logo_input_idx}:v]scale={logo_scale}:-1[logo];"
            f"[0:v][logo]overlay={lx}:{ly}:format=auto[vout]"
        )
        cmd += ["-filter_complex", filter_complex, "-map", "[vout]"]
    else:
        cmd += ["-map", "0:v"]

    # Audio from mastered WAV
    cmd += ["-map", "1:a"]

    # Video encoder
    cmd += ["-c:v", enc] + flags

    # Bitrate override (for VBR encoders that support it)
    if enc in ("h264_nvenc", "libx264", "h264_qsv") and video_bitrate:
        # For NVENC VBR mode, maxrate/bufsize provide ceiling
        if enc == "h264_nvenc":
            cmd += ["-maxrate", video_bitrate, "-bufsize", _double_bitrate(video_bitrate)]
        elif enc == "libx264":
            pass  # CRF mode, bitrate override skipped
        else:
            cmd += ["-b:v", video_bitrate]

    # Audio
    cmd += ["-c:a", audio_codec]
    if audio_bitrate:
        cmd += ["-b:a", audio_bitrate]

    # Container optimizations
    cmd += ["-movflags", "+faststart"]
    cmd += [tmp_out]

    # Execute
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,  # not read — DEVNULL prevents deadlock on Windows
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        stderr_buf: List[str] = []
        if proc.stderr:
            for line in proc.stderr:
                stderr_buf.append(line)
                if "time=" in line and progress_callback and duration_s > 0:
                    secs = _parse_ffmpeg_time(line)
                    if secs is not None:
                        progress_callback(min(1.0, secs / duration_s))

        proc.wait()
        if progress_callback:
            progress_callback(1.0)

        if proc.returncode != 0:
            print(f"[hw_encoder] ffmpeg failed (rc={proc.returncode})")
            if stderr_buf:
                print("".join(stderr_buf[-20:]))
            return False

        os.replace(tmp_out, output_path)
        return True

    except FileNotFoundError:
        print("[hw_encoder] ffmpeg not found")
        return False
    except Exception as exc:
        print(f"[hw_encoder] export error: {exc}")
        return False
    finally:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass


# ── Helpers ────────────────────────────────────────────────────────────────
def _probe_duration(path: str) -> float:
    """Return duration in seconds using ffprobe."""
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _parse_ffmpeg_time(line: str) -> Optional[float]:
    """Extract seconds from ffmpeg stderr 'time=HH:MM:SS.ss' line."""
    try:
        idx  = line.index("time=")
        part = line[idx + 5:].split()[0]
        if part == "N/A":
            return None
        h, m, s = part.split(":")
        return float(h) * 3600 + float(m) * 60 + float(s)
    except (ValueError, IndexError):
        return None


def _double_bitrate(bitrate_str: str) -> str:
    """Double a bitrate string like '8M' → '16M'.

    Returns the original string unchanged if the format is unrecognised,
    and logs a warning so callers can detect the fallback.
    """
    try:
        if bitrate_str.upper().endswith("M"):
            return f"{int(bitrate_str[:-1]) * 2}M"
        if bitrate_str.upper().endswith("K"):
            return f"{int(bitrate_str[:-1]) * 2}K"
    except ValueError:
        pass
    _log.warning(
        "_double_bitrate: unrecognised bitrate format %r — "
        "NVENC bufsize will equal maxrate instead of 2x maxrate",
        bitrate_str,
    )
    return bitrate_str
