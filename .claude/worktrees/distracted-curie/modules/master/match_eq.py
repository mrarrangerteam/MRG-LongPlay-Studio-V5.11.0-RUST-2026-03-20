"""
Match EQ — reference track frequency matching.

Story 4.3 — Epic 4: Pro Mastering.

Features:
    - Load reference audio file
    - Analyze reference spectrum (FFT average)
    - Analyze current audio spectrum
    - Calculate difference curve
    - Apply correction curve to EQ
    - Adjustable match strength (0-100%)
    - Visual overlay: reference spectrum vs current
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

from .equalizer import Equalizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FFT_SIZE = 8192
ANALYSIS_SR = 44100
NUM_BANDS = 31          # 1/3-octave bands for the correction curve
FREQ_MIN = 20.0
FREQ_MAX = 20000.0


def _iso_third_octave_centers() -> np.ndarray:
    """Return ISO 1/3-octave center frequencies from 20 Hz to 20 kHz."""
    base_freqs = [
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
        2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
        20000,
    ]
    return np.array(base_freqs[:NUM_BANDS], dtype=np.float64)


THIRD_OCTAVE_CENTERS = _iso_third_octave_centers()


# ---------------------------------------------------------------------------
# Spectrum analysis helpers
# ---------------------------------------------------------------------------
def _extract_mono_pcm(audio_path: str, ffmpeg_path: str = "ffmpeg",
                      max_seconds: float = 60.0) -> Optional[np.ndarray]:
    """Extract mono PCM samples from an audio file via FFmpeg."""
    if not os.path.exists(audio_path):
        return None

    cmd = [
        ffmpeg_path,
        "-i", audio_path,
        "-t", str(max_seconds),
        "-ac", "1",
        "-ar", str(ANALYSIS_SR),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-v", "error",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if len(result.stdout) < 4:
            return None
        samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float64) / 32768.0
        return samples
    except (subprocess.TimeoutExpired, OSError):
        return None


def _compute_avg_spectrum(samples: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """
    Compute average magnitude spectrum across overlapping chunks.

    Returns dB magnitudes at rfft frequency bins.
    """
    hop = FFT_SIZE // 2
    window = np.hanning(FFT_SIZE)
    n_chunks = max(1, (len(samples) - FFT_SIZE) // hop)
    acc = np.zeros(FFT_SIZE // 2 + 1, dtype=np.float64)

    for i in range(n_chunks):
        start = i * hop
        chunk = samples[start:start + FFT_SIZE]
        if len(chunk) < FFT_SIZE:
            break
        windowed = chunk * window
        mag = np.abs(np.fft.rfft(windowed)) / FFT_SIZE
        acc += mag

    acc /= max(n_chunks, 1)
    acc = np.maximum(acc, 1e-12)
    return 20.0 * np.log10(acc)


def _spectrum_to_bands(spectrum_db: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """
    Downsample a full-resolution spectrum to 1/3-octave bands.

    Returns array of length NUM_BANDS with average dB per band.
    """
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sr)
    band_values = np.zeros(NUM_BANDS, dtype=np.float64)

    for i, center in enumerate(THIRD_OCTAVE_CENTERS):
        # 1/3-octave bandwidth
        lo = center / (2 ** (1.0 / 6.0))
        hi = center * (2 ** (1.0 / 6.0))
        mask = (freqs >= lo) & (freqs <= hi)
        if np.any(mask):
            band_values[i] = np.mean(spectrum_db[mask])
        else:
            band_values[i] = -60.0

    return band_values


# ---------------------------------------------------------------------------
# MatchEQ
# ---------------------------------------------------------------------------
class MatchEQ:
    """
    Reference-track frequency matching module.

    Computes a correction EQ curve that, when applied, makes the current
    audio's spectral profile approximate the reference.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        self._ffmpeg = ffmpeg_path
        self._ref_spectrum: Optional[np.ndarray] = None    # dB per band
        self._cur_spectrum: Optional[np.ndarray] = None
        self._diff_curve: Optional[np.ndarray] = None
        self._strength: float = 1.0    # 0.0 – 1.0
        self._ref_path: str = ""

    # -- public API --------------------------------------------------------

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        self._strength = max(0.0, min(1.0, value))
        if self._ref_spectrum is not None and self._cur_spectrum is not None:
            self._compute_diff()

    @property
    def reference_path(self) -> str:
        return self._ref_path

    @property
    def band_centers(self) -> np.ndarray:
        return THIRD_OCTAVE_CENTERS.copy()

    @property
    def reference_spectrum(self) -> Optional[np.ndarray]:
        return self._ref_spectrum.copy() if self._ref_spectrum is not None else None

    @property
    def current_spectrum(self) -> Optional[np.ndarray]:
        return self._cur_spectrum.copy() if self._cur_spectrum is not None else None

    @property
    def correction_curve(self) -> Optional[np.ndarray]:
        """Return the correction curve (dB per 1/3-octave band), scaled by strength."""
        return self._diff_curve.copy() if self._diff_curve is not None else None

    def load_reference(self, audio_path: str) -> bool:
        """Analyze a reference audio file and store its spectral profile."""
        samples = _extract_mono_pcm(audio_path, self._ffmpeg)
        if samples is None or len(samples) < FFT_SIZE:
            return False
        self._ref_path = audio_path
        full_spec = _compute_avg_spectrum(samples)
        self._ref_spectrum = _spectrum_to_bands(full_spec)
        if self._cur_spectrum is not None:
            self._compute_diff()
        return True

    def analyze_current(self, audio_path: str) -> bool:
        """Analyze the current (source) audio file."""
        samples = _extract_mono_pcm(audio_path, self._ffmpeg)
        if samples is None or len(samples) < FFT_SIZE:
            return False
        full_spec = _compute_avg_spectrum(samples)
        self._cur_spectrum = _spectrum_to_bands(full_spec)
        if self._ref_spectrum is not None:
            self._compute_diff()
        return True

    def analyze_samples(self, samples: np.ndarray, sr: int = ANALYSIS_SR) -> bool:
        """Analyze current audio from raw samples (mono float64)."""
        if len(samples) < FFT_SIZE:
            return False
        full_spec = _compute_avg_spectrum(samples, sr)
        self._cur_spectrum = _spectrum_to_bands(full_spec, sr)
        if self._ref_spectrum is not None:
            self._compute_diff()
        return True

    def apply_to_equalizer(self, equalizer: Equalizer) -> bool:
        """
        Apply the correction curve to an Equalizer instance.

        Picks the 8 most impactful bands and maps them to the 8-band EQ.
        """
        if self._diff_curve is None:
            return False

        curve = self._diff_curve.copy()

        # pick top 8 by absolute magnitude
        indices = np.argsort(np.abs(curve))[::-1][:8]
        indices = np.sort(indices)  # put in frequency order

        for band_idx, eq_band_idx in enumerate(range(min(8, len(indices)))):
            src_idx = indices[eq_band_idx]
            freq = float(THIRD_OCTAVE_CENTERS[src_idx])
            gain = float(curve[src_idx])
            # clamp gain
            gain = max(-12.0, min(12.0, gain))

            if band_idx < len(equalizer.bands):
                band = equalizer.bands[band_idx]
                band.freq = freq
                band.gain = gain
                band.enabled = abs(gain) > 0.2
                band.band_type = "equalizer"
                band.width = 1.4  # moderate Q for gentle matching

        return True

    def get_report(self) -> Dict:
        """Return a summary of the match EQ analysis."""
        return {
            "reference_path": self._ref_path,
            "strength_pct": round(self._strength * 100, 1),
            "has_reference": self._ref_spectrum is not None,
            "has_current": self._cur_spectrum is not None,
            "has_correction": self._diff_curve is not None,
            "band_centers_hz": THIRD_OCTAVE_CENTERS.tolist() if self._diff_curve is not None else [],
            "correction_db": self._diff_curve.tolist() if self._diff_curve is not None else [],
        }

    def reset(self) -> None:
        """Clear all analysis data."""
        self._ref_spectrum = None
        self._cur_spectrum = None
        self._diff_curve = None
        self._ref_path = ""

    # -- internal ----------------------------------------------------------

    def _compute_diff(self) -> None:
        """Compute the difference curve (reference – current) * strength."""
        if self._ref_spectrum is None or self._cur_spectrum is None:
            return
        raw_diff = self._ref_spectrum - self._cur_spectrum
        # smooth the curve slightly
        kernel = np.array([0.15, 0.7, 0.15])
        smoothed = np.convolve(raw_diff, kernel, mode="same")
        self._diff_curve = smoothed * self._strength
