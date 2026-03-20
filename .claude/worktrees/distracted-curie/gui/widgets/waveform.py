"""
Waveform and thumbnail caching for timeline display.

Classes:
    WaveformCache  — Generate and cache real audio peak data from ffmpeg
    ThumbnailCache — Extract and cache video frame thumbnails for filmstrip display
"""

import os
import subprocess
import tempfile
import random
from typing import Dict

from gui.utils.compat import QPixmap


class ThumbnailCache:
    """Extract and cache video frame thumbnails for filmstrip display on Timeline"""

    _cache: Dict[str, list] = {}  # path -> list of QPixmap

    @classmethod
    def get_thumbnails(cls, file_path: str, num_frames: int = 20, thumb_height: int = 35) -> list:
        """Get cached thumbnails. Returns list of QPixmap objects."""
        cache_key = f"{file_path}:{num_frames}:{thumb_height}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        thumbnails = cls._extract_thumbnails(file_path, num_frames, thumb_height)
        cls._cache[cache_key] = thumbnails
        return thumbnails

    @classmethod
    def _extract_thumbnails(cls, file_path: str, num_frames: int, thumb_height: int) -> list:
        """Extract frames from video/gif using ffmpeg, return as QPixmap list"""
        try:
            import json as _json

            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", file_path
            ]
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            duration = 5.0
            if probe.returncode == 0:
                info = _json.loads(probe.stdout)
                duration = float(info.get("format", {}).get("duration", 5.0))

            if duration <= 0:
                duration = 5.0

            thumb_width = int(thumb_height * 16 / 9)

            interval = max(0.1, duration / max(1, num_frames))
            fps_filter = f"fps=1/{interval}"

            with tempfile.TemporaryDirectory() as tmpdir:
                out_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
                cmd = [
                    "ffmpeg", "-i", file_path,
                    "-vf", f"{fps_filter},scale={thumb_width}:{thumb_height}:force_original_aspect_ratio=decrease,pad={thumb_width}:{thumb_height}:(ow-iw)/2:(oh-ih)/2:color=black",
                    "-q:v", "8",
                    "-v", "quiet",
                    out_pattern
                ]
                subprocess.run(cmd, timeout=60)

                thumbnails = []
                for i in range(1, num_frames + 50):
                    fpath = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
                    if os.path.exists(fpath):
                        pix = QPixmap(fpath)
                        if not pix.isNull():
                            thumbnails.append(pix)
                    else:
                        break

                if thumbnails:
                    print(f"[THUMBNAIL] Extracted {len(thumbnails)} frames from {os.path.basename(file_path)}")
                    return thumbnails

        except Exception as e:
            print(f"[THUMBNAIL] ffmpeg failed for {os.path.basename(file_path)}: {e}")

        return []

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()


class WaveformCache:
    """Generate and cache REAL waveform data from audio files using ffmpeg"""

    _cache: Dict[str, list] = {}

    @classmethod
    def get_waveform(cls, file_path: str, num_samples: int = 500) -> list:
        """Get cached waveform peaks. Returns list of floats 0.0-1.0"""
        cache_key = f"{file_path}:{num_samples}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        waveform = cls._generate_real_waveform(file_path, num_samples)
        cls._cache[cache_key] = waveform
        return waveform

    @classmethod
    def _generate_real_waveform(cls, file_path: str, num_samples: int) -> list:
        """Extract real audio peak data using ffmpeg raw PCM"""
        try:
            cmd = [
                "ffmpeg", "-i", file_path,
                "-ac", "1",
                "-ar", str(max(1000, num_samples * 4)),
                "-f", "u8",
                "-acodec", "pcm_u8",
                "-v", "quiet",
                "pipe:1"
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode != 0 or len(result.stdout) < 100:
                return cls._generate_deterministic_waveform(file_path, num_samples)

            raw = result.stdout
            total_samples = len(raw)
            chunk_size = max(1, total_samples // num_samples)

            peaks = []
            for i in range(num_samples):
                start = i * chunk_size
                end = min(start + chunk_size, total_samples)
                if start >= total_samples:
                    peaks.append(0.05)
                    continue
                chunk = raw[start:end]
                peak = max(abs(b - 128) for b in chunk) if chunk else 0
                peaks.append(peak / 128.0)

            max_peak = max(peaks) if peaks and max(peaks) > 0 else 1.0
            normalized = [max(0.05, min(1.0, p / max_peak)) for p in peaks]

            print(f"[WAVEFORM] Real waveform: {os.path.basename(file_path)} ({num_samples} peaks)")
            return normalized

        except Exception as e:
            print(f"[WAVEFORM] ffmpeg failed for {os.path.basename(file_path)}: {e}")
            return cls._generate_deterministic_waveform(file_path, num_samples)

    @classmethod
    def _generate_deterministic_waveform(cls, file_path: str, num_samples: int) -> list:
        """Fallback: deterministic fake waveform based on filename hash"""
        import hashlib
        seed = int(hashlib.md5(file_path.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        wave = []
        for i in range(num_samples):
            progress = i / num_samples
            if progress < 0.05:
                envelope = 0.3 + progress * 4
            elif progress > 0.95:
                envelope = 0.3 + (1.0 - progress) * 4
            elif 0.3 < progress < 0.5 or 0.6 < progress < 0.8:
                envelope = 0.85
            else:
                envelope = 0.6
            val = 0.4 + rng.uniform(0.0, 0.6) * envelope
            wave.append(max(0.05, min(1.0, val)))
        return wave

    @classmethod
    def _generate_fake_waveform(cls, num_samples: int) -> list:
        """Simple fallback waveform"""
        return [0.3 + random.Random(42).uniform(0, 0.5) for _ in range(num_samples)]

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
