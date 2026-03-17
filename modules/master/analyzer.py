"""
LongPlay Studio V5.0 — Audio Analyzer Module
Analyzes audio characteristics for AI Assist recommendations.

Features:
- Spectral balance analysis (Low/Mid/High energy ratios)
- Dynamic range analysis (Crest factor, peak-to-RMS)
- Stereo width analysis (Correlation coefficient)
- Genre detection hints
- Spectrum data for visualization

Uses: numpy (fast FFT), FFmpeg for PCM extraction
V5.3 FIX: Replaced pure-Python O(n²) DFT with numpy FFT
           Analysis time: 26.5s → <0.5s
"""

import subprocess
import os
import math
import time
from typing import Optional, Dict, List, Tuple

import numpy as np


class SpectralAnalysis:
    """Spectral balance analysis results."""

    def __init__(self):
        self.low_energy = 0.0     # 20-250 Hz energy ratio (0-1)
        self.mid_energy = 0.0     # 250-4000 Hz energy ratio (0-1)
        self.high_energy = 0.0    # 4000-20000 Hz energy ratio (0-1)
        self.sub_energy = 0.0     # 20-60 Hz energy (for bass-heavy detection)
        self.brightness = 0.0     # High/Low ratio (>1 = bright, <1 = dark)
        self.spectral_centroid = 0.0  # Hz (average frequency)

    def get_balance_description(self) -> str:
        """Describe the spectral balance in human terms."""
        if self.brightness > 1.5:
            return "Very bright — may need high-frequency reduction"
        elif self.brightness > 1.1:
            return "Bright — good for pop/electronic"
        elif self.brightness > 0.9:
            return "Balanced — good starting point"
        elif self.brightness > 0.6:
            return "Warm — good for acoustic/jazz"
        else:
            return "Dark — may need high-frequency boost"

    def to_dict(self) -> dict:
        return {
            "low_energy": round(self.low_energy, 3),
            "mid_energy": round(self.mid_energy, 3),
            "high_energy": round(self.high_energy, 3),
            "sub_energy": round(self.sub_energy, 3),
            "brightness": round(self.brightness, 3),
            "spectral_centroid": round(self.spectral_centroid, 1),
            "description": self.get_balance_description(),
        }


class DynamicAnalysis:
    """Dynamic range analysis results."""

    def __init__(self):
        self.peak_db = -100.0          # Peak level in dB
        self.rms_db = -100.0           # RMS level in dB
        self.crest_factor_db = 0.0     # Peak - RMS (higher = more dynamic)
        self.dynamic_range_db = 0.0    # Based on loudness range

    def get_dynamics_description(self) -> str:
        """Describe the dynamic range in human terms."""
        cf = self.crest_factor_db
        if cf > 20:
            return "Very dynamic — minimal compression needed"
        elif cf > 14:
            return "Dynamic — light compression recommended"
        elif cf > 10:
            return "Moderate dynamics — standard compression"
        elif cf > 6:
            return "Compressed — may already be mastered"
        else:
            return "Very compressed — may be over-limited"

    def to_dict(self) -> dict:
        return {
            "peak_db": round(self.peak_db, 1),
            "rms_db": round(self.rms_db, 1),
            "crest_factor_db": round(self.crest_factor_db, 1),
            "dynamic_range_db": round(self.dynamic_range_db, 1),
            "description": self.get_dynamics_description(),
        }


class StereoAnalysis:
    """Stereo field analysis results."""

    def __init__(self):
        self.correlation = 1.0       # -1 to +1 (1 = mono, 0 = decorrelated)
        self.width_pct = 0.0         # Estimated width percentage
        self.balance_lr = 0.0        # L-R balance (-1=left, 0=center, +1=right)
        self.is_mono = False

    def get_stereo_description(self) -> str:
        if self.is_mono:
            return "Mono — stereo widening recommended"
        elif self.correlation > 0.9:
            return "Very narrow — could benefit from widening"
        elif self.correlation > 0.7:
            return "Moderately narrow — slight widening possible"
        elif self.correlation > 0.3:
            return "Good stereo spread — balanced image"
        elif self.correlation > 0:
            return "Wide stereo — be careful with more widening"
        else:
            return "Very wide / out of phase — may need narrowing"

    def to_dict(self) -> dict:
        return {
            "correlation": round(self.correlation, 3),
            "width_pct": round(self.width_pct, 1),
            "balance_lr": round(self.balance_lr, 3),
            "is_mono": self.is_mono,
            "description": self.get_stereo_description(),
        }


class AudioAnalysis:
    """Complete audio analysis results."""

    def __init__(self):
        self.spectral = SpectralAnalysis()
        self.dynamic = DynamicAnalysis()
        self.stereo = StereoAnalysis()
        self.duration_sec = 0.0
        self.sample_rate = 48000
        self.channels = 2
        self.bit_depth = 16

    def to_dict(self) -> dict:
        return {
            "spectral": self.spectral.to_dict(),
            "dynamic": self.dynamic.to_dict(),
            "stereo": self.stereo.to_dict(),
            "duration_sec": round(self.duration_sec, 2),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
        }


class AudioAnalyzer:
    """
    Audio analyzer using FFmpeg for PCM extraction and numpy for
    fast spectral analysis via FFT.

    V5.3 FIX: Replaced pure-Python DFT (26.5s) with numpy FFT (<0.5s).
    For production use with large files, analysis is done on a
    representative sample (first 30 seconds).
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self.sample_duration = 30  # seconds per sample section

    def analyze(self, audio_path: str, full_analysis: bool = False) -> Optional[AudioAnalysis]:
        """
        Analyze audio file characteristics.

        Args:
            audio_path: Path to audio file
            full_analysis: If True, analyze entire file. If False, sample sections.

        Returns:
            AudioAnalysis object or None on error
        """
        if not os.path.exists(audio_path):
            print(f"[ANALYZER] File not found: {audio_path}")
            return None

        try:
            t0 = time.time()

            # Get file info first
            duration = self._get_duration(audio_path)
            if duration is None or duration <= 0:
                print(f"[ANALYZER] Could not get duration for: {audio_path}")
                return None

            # Extract PCM samples for analysis (as numpy arrays)
            samples_l, samples_r, sr = self._extract_pcm_samples(
                audio_path, duration, full_analysis
            )

            if samples_l is None:
                print(f"[ANALYZER] Could not extract PCM samples from: {audio_path}")
                return None

            analysis = AudioAnalysis()
            analysis.duration_sec = duration
            analysis.sample_rate = sr
            analysis.channels = 2 if samples_r is not None else 1

            # Spectral analysis (using numpy FFT — FAST)
            analysis.spectral = self._analyze_spectrum(samples_l, sr)

            # Dynamic analysis
            analysis.dynamic = self._analyze_dynamics(samples_l)

            # Stereo analysis
            if samples_r is not None:
                analysis.stereo = self._analyze_stereo(samples_l, samples_r)
            else:
                analysis.stereo.is_mono = True

            elapsed = time.time() - t0
            print(f"[ANALYZER] ✅ Analysis complete in {elapsed:.2f}s "
                  f"({len(samples_l)/sr:.1f}s audio, {sr}Hz)")

            return analysis

        except Exception as e:
            print(f"[ANALYZER] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_duration(self, audio_path: str) -> Optional[float]:
        """Get audio duration using ffprobe."""
        try:
            cmd = [
                self.ffmpeg_path.replace("ffmpeg", "ffprobe"),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except subprocess.TimeoutExpired:
            print(f"[ANALYZER] ffprobe duration query timed out for {audio_path}")
            return None
        except Exception:
            return None

    def _extract_pcm_samples(
        self, audio_path: str, duration: float, full: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Extract PCM samples from audio file using FFmpeg.
        Returns (left_samples, right_samples, sample_rate) as numpy arrays.
        Samples are normalized to [-1.0, 1.0].

        V5.3 FIX: Uses numpy for fast array conversion instead of struct.unpack loop.
        """
        sr = 22050  # Downsample for analysis efficiency

        if full or duration <= self.sample_duration * 3:
            # Analyze entire file
            ss_args = []
            dur_args = []
        else:
            # Analyze first 30s only (for speed)
            ss_args = []
            dur_args = ["-t", str(self.sample_duration)]

        cmd = [
            self.ffmpeg_path,
            "-i", audio_path,
            *ss_args,
            *dur_args,
            "-ac", "2",           # Force stereo
            "-ar", str(sr),       # Downsample
            "-f", "s16le",        # 16-bit PCM
            "-acodec", "pcm_s16le",
            "-v", "error",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=60,
            )

            raw = result.stdout
            if len(raw) < 4:
                return None, None, sr

            # V5.3 FIX: Use numpy for fast PCM parsing (was slow struct.unpack loop)
            # Parse interleaved stereo 16-bit PCM as numpy array
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

            # Limit to sample_duration worth of samples
            max_samples = sr * self.sample_duration * 2  # *2 for stereo interleave
            if len(samples) > max_samples:
                samples = samples[:max_samples]

            # De-interleave stereo
            if len(samples) >= 2:
                left = samples[0::2]
                right = samples[1::2]
            else:
                return None, None, sr

            return left, right, sr

        except subprocess.TimeoutExpired:
            print(f"[ANALYZER] PCM extraction timed out for {audio_path} (60s timeout)")
            return None, None, sr
        except Exception as e:
            print(f"[ANALYZER] PCM extraction error: {e}")
            return None, None, sr

    def _analyze_spectrum(self, samples: np.ndarray, sr: int) -> SpectralAnalysis:
        """
        Analyze spectral balance using numpy FFT on chunks.

        V5.3 FIX: Replaced pure-Python DFT (O(n²), 26.5 seconds)
        with numpy.fft.rfft (O(n log n), <0.1 seconds).
        """
        result = SpectralAnalysis()

        if len(samples) < 1024:
            return result

        chunk_size = 4096
        num_chunks = min(len(samples) // chunk_size, 20)  # Max 20 chunks

        if num_chunks == 0:
            return result

        low_total = 0.0     # 20-250 Hz
        mid_total = 0.0     # 250-4000 Hz
        high_total = 0.0    # 4000-sr/2 Hz
        sub_total = 0.0     # 20-60 Hz

        # Pre-compute frequency bins (same for all chunks)
        freqs = np.fft.rfftfreq(chunk_size, d=1.0 / sr)

        # Create frequency band masks
        mask_sub = (freqs >= 20) & (freqs <= 60)
        mask_low = (freqs >= 20) & (freqs <= 250)
        mask_mid = (freqs > 250) & (freqs <= 4000)
        mask_high = (freqs > 4000) & (freqs <= 20000)

        # Hann window for better spectral analysis
        window = np.hanning(chunk_size)

        # Weighted centroid accumulator
        centroid_num = 0.0
        centroid_den = 0.0

        for chunk_idx in range(num_chunks):
            start = chunk_idx * (len(samples) // max(num_chunks, 1))
            chunk = samples[start:start + chunk_size]
            if len(chunk) < chunk_size:
                break

            # Apply window and compute FFT
            windowed = chunk * window
            spectrum = np.abs(np.fft.rfft(windowed)) / chunk_size

            # Accumulate energy per band
            sub_total += np.sum(spectrum[mask_sub])
            low_total += np.sum(spectrum[mask_low])
            mid_total += np.sum(spectrum[mask_mid])
            high_total += np.sum(spectrum[mask_high])

            # Spectral centroid
            total_mag = np.sum(spectrum[1:])
            if total_mag > 0:
                centroid_num += np.sum(freqs[1:] * spectrum[1:])
                centroid_den += total_mag

        total = low_total + mid_total + high_total
        if total > 0:
            result.low_energy = low_total / total
            result.mid_energy = mid_total / total
            result.high_energy = high_total / total
            result.sub_energy = sub_total / total
            result.brightness = high_total / max(low_total, 0.001)

        if centroid_den > 0:
            result.spectral_centroid = centroid_num / centroid_den

        return result

    def _analyze_dynamics(self, samples: np.ndarray) -> DynamicAnalysis:
        """Analyze dynamic range using numpy vectorized operations."""
        result = DynamicAnalysis()

        if len(samples) == 0:
            return result

        # Peak
        peak = np.max(np.abs(samples))
        result.peak_db = 20 * np.log10(max(peak, 1e-10))

        # RMS
        rms = np.sqrt(np.mean(samples ** 2))
        result.rms_db = 20 * np.log10(max(rms, 1e-10))

        # Crest factor
        result.crest_factor_db = result.peak_db - result.rms_db

        # Approximate dynamic range (simplified)
        # Compute RMS in 1-second windows and find range
        window_size = 22050  # ~1 second at 22050 Hz
        num_windows = len(samples) // window_size
        if num_windows > 0:
            # Reshape into windows and compute RMS per window
            trimmed = samples[:num_windows * window_size]
            windows = trimmed.reshape(num_windows, window_size)
            window_rms = np.sqrt(np.mean(windows ** 2, axis=1))

            # Filter out silence
            valid = window_rms > 1e-8
            if np.sum(valid) > 2:
                window_rms_db = 20 * np.log10(window_rms[valid])
                window_rms_db.sort()
                p95_idx = int(len(window_rms_db) * 0.95)
                p10_idx = int(len(window_rms_db) * 0.10)
                result.dynamic_range_db = float(window_rms_db[p95_idx] - window_rms_db[p10_idx])

        return result

    def _analyze_stereo(self, left: np.ndarray, right: np.ndarray) -> StereoAnalysis:
        """Analyze stereo field using numpy vectorized operations."""
        result = StereoAnalysis()

        n = min(len(left), len(right))
        if n < 100:
            return result

        left = left[:n]
        right = right[:n]

        # Check if effectively mono
        diff_energy = np.sum((left - right) ** 2)
        sum_energy = np.sum((left + right) ** 2)

        if diff_energy < sum_energy * 0.001:
            result.is_mono = True
            result.correlation = 1.0
            result.width_pct = 0.0
            return result

        # Correlation coefficient
        sum_lr = np.sum(left * right)
        sum_ll = np.sum(left * left)
        sum_rr = np.sum(right * right)

        denom = math.sqrt(sum_ll * sum_rr)
        if denom > 0:
            result.correlation = float(sum_lr / denom)
        else:
            result.correlation = 1.0

        # Width estimation (based on correlation)
        result.width_pct = (1.0 - result.correlation) * 100.0

        # L-R balance
        left_energy = float(np.sum(left ** 2))
        right_energy = float(np.sum(right ** 2))
        total_energy = left_energy + right_energy
        if total_energy > 0:
            result.balance_lr = (right_energy - left_energy) / total_energy

        return result

    def quick_analyze(self, audio_path: str) -> Optional[Dict]:
        """Quick analysis returning summary dict."""
        analysis = self.analyze(audio_path, full_analysis=False)
        if analysis:
            return analysis.to_dict()
        return None
