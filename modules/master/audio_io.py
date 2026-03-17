"""
Native audio I/O module with symphonia integration.

Story 5.2 — Epic 5: Polish & Production.

Provides a unified interface for reading audio files using:
    1. Rust/symphonia (fastest, native MP3/FLAC/OGG/AAC/WAV)
    2. soundfile/libsndfile (fast, WAV/FLAC/OGG)
    3. FFmpeg subprocess (fallback, any format)

The Rust backend uses the symphonia crate exposed via PyO3 in the
longplay module for native decoding without FFmpeg subprocess overhead.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional, Tuple

import numpy as np


# --- Backend detection ---

_BACKEND = "ffmpeg"  # default fallback

try:
    import longplay
    if hasattr(longplay, "PyAudioIO"):
        _BACKEND = "rust_symphonia"
    elif hasattr(longplay, "PyAudioAnalyzer"):
        _BACKEND = "rust"
except ImportError:
    pass

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False


# ---------------------------------------------------------------------------
# AudioInfo
# ---------------------------------------------------------------------------
class AudioFileInfo:
    """Audio file metadata."""

    __slots__ = ("sample_rate", "channels", "frames", "bit_depth",
                 "format", "duration_sec", "path")

    def __init__(self) -> None:
        self.sample_rate: int = 44100
        self.channels: int = 2
        self.frames: int = 0
        self.bit_depth: int = 16
        self.format: str = "unknown"
        self.duration_sec: float = 0.0
        self.path: str = ""

    def to_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "bit_depth": self.bit_depth,
            "format": self.format,
            "duration_sec": round(self.duration_sec, 3),
            "path": self.path,
        }


# ---------------------------------------------------------------------------
# Unified read function
# ---------------------------------------------------------------------------
def read_audio(
    path: str,
    target_sr: Optional[int] = None,
    mono: bool = False,
    ffmpeg_path: str = "ffmpeg",
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Read an audio file and return (samples, sample_rate).

    Samples are returned as a numpy float64 array:
        - mono: shape (N,)
        - stereo: shape (N, 2)

    Tries backends in order: Rust/symphonia → soundfile → FFmpeg.

    Args:
        path: Path to audio file.
        target_sr: Optional target sample rate for resampling.
        mono: If True, downmix to mono.
        ffmpeg_path: Path to ffmpeg binary.

    Returns:
        (samples, sample_rate) or None on failure.
    """
    if not os.path.exists(path):
        return None

    result = None

    # 1. Try Rust/symphonia
    if _BACKEND in ("rust_symphonia", "rust"):
        result = _read_rust(path)

    # 2. Try soundfile
    if result is None and _HAS_SOUNDFILE:
        result = _read_soundfile(path)

    # 3. FFmpeg fallback
    if result is None:
        result = _read_ffmpeg(path, ffmpeg_path, target_sr, mono)
        if result is not None:
            return result  # FFmpeg already handles sr/mono

    if result is None:
        return None

    samples, sr = result

    # mono downmix if requested
    if mono and samples.ndim > 1 and samples.shape[1] > 1:
        samples = np.mean(samples, axis=1)

    # resample if needed
    if target_sr and target_sr != sr:
        samples = _resample(samples, sr, target_sr)
        sr = target_sr

    return samples, sr


def get_info(path: str, ffmpeg_path: str = "ffmpeg") -> Optional[AudioFileInfo]:
    """
    Get audio file metadata without reading the full file.

    Args:
        path: Path to audio file.
        ffmpeg_path: Path to ffmpeg binary.

    Returns:
        AudioFileInfo or None on failure.
    """
    if not os.path.exists(path):
        return None

    info = AudioFileInfo()
    info.path = path
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    info.format = ext

    # Try soundfile
    if _HAS_SOUNDFILE:
        try:
            sf_info = sf.info(path)
            info.sample_rate = sf_info.samplerate
            info.channels = sf_info.channels
            info.frames = sf_info.frames
            info.duration_sec = sf_info.duration
            info.format = sf_info.format
            return info
        except RuntimeError:
            pass

    # FFmpeg probe
    try:
        cmd = [
            ffmpeg_path.replace("ffmpeg", "ffprobe"),
            "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        import json
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                info.sample_rate = int(stream.get("sample_rate", 44100))
                info.channels = int(stream.get("channels", 2))
                info.bit_depth = int(stream.get("bits_per_sample", 16))
                break
        fmt = data.get("format", {})
        info.duration_sec = float(fmt.get("duration", 0))
        info.frames = int(info.duration_sec * info.sample_rate)
        return info
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Format support query
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = {
    "wav": "WAV (PCM)",
    "mp3": "MP3",
    "flac": "FLAC",
    "ogg": "OGG Vorbis",
    "aac": "AAC",
    "m4a": "MPEG-4 Audio",
    "wma": "Windows Media Audio",
    "aiff": "AIFF",
    "opus": "Opus",
}


def get_backend_name() -> str:
    """Return the name of the active audio I/O backend."""
    if _BACKEND == "rust_symphonia":
        return "Rust (symphonia)"
    elif _BACKEND == "rust":
        return "Rust (longplay)"
    elif _HAS_SOUNDFILE:
        return "soundfile (libsndfile)"
    return "FFmpeg (subprocess)"


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------
def _read_rust(path: str) -> Optional[Tuple[np.ndarray, int]]:
    """Read via Rust/symphonia backend."""
    try:
        if hasattr(longplay, "PyAudioIO"):
            io = longplay.PyAudioIO()
            result = io.read(path)  # returns dict with 'samples', 'sample_rate', 'channels'
            samples = np.array(result["samples"], dtype=np.float64)
            sr = result["sample_rate"]
            channels = result.get("channels", 1)
            if channels > 1:
                frames = len(samples) // channels
                samples = samples[:frames * channels].reshape(frames, channels)
            return samples, sr
    except (AttributeError, RuntimeError, KeyError):
        pass
    return None


def _read_soundfile(path: str) -> Optional[Tuple[np.ndarray, int]]:
    """Read via soundfile/libsndfile."""
    try:
        data, sr = sf.read(path, dtype="float64")
        return data, sr
    except (RuntimeError, OSError):
        return None


def _read_ffmpeg(
    path: str,
    ffmpeg_path: str,
    target_sr: Optional[int],
    mono: bool,
) -> Optional[Tuple[np.ndarray, int]]:
    """Read via FFmpeg subprocess."""
    sr = target_sr or 44100
    channels = "1" if mono else "2"

    cmd = [
        ffmpeg_path,
        "-i", path,
        "-ac", channels,
        "-ar", str(sr),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-v", "error",
        "pipe:1",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if len(result.stdout) < 4:
            return None
        samples = np.frombuffer(result.stdout, dtype=np.float32).astype(np.float64)
        ch = int(channels)
        if ch > 1:
            frames = len(samples) // ch
            samples = samples[:frames * ch].reshape(frames, ch)
        return samples, sr
    except (subprocess.TimeoutExpired, OSError):
        return None


def _resample(samples: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Simple linear interpolation resampling."""
    if from_sr == to_sr:
        return samples

    ratio = to_sr / from_sr
    if samples.ndim == 1:
        new_len = int(len(samples) * ratio)
        indices = np.arange(new_len) / ratio
        idx0 = indices.astype(int)
        idx1 = np.minimum(idx0 + 1, len(samples) - 1)
        frac = indices - idx0
        return samples[idx0] * (1.0 - frac) + samples[idx1] * frac
    else:
        # multichannel
        new_len = int(samples.shape[0] * ratio)
        result = np.zeros((new_len, samples.shape[1]), dtype=samples.dtype)
        for ch in range(samples.shape[1]):
            result[:, ch] = _resample(samples[:, ch], from_sr, to_sr)
        return result
