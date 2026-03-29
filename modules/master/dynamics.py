"""
LongPlay Studio V5.0 — Dynamics Module
Inspired by iZotope Ozone 12 Dynamics

Features:
- Single-band Compressor with full parameter control
- Multiband mode (Low/Mid/High with adjustable crossover)
- Preset modes: Gentle, Standard, Aggressive, Parallel
- Manual mode: All parameters adjustable
- Sidechain HPF option

Uses FFmpeg: acompressor, highpass, lowpass (for band splitting)
"""

import math
from typing import List, Dict, Optional

import numpy as np

from .genre_profiles import get_genre_profile


# Dynamics Presets
DYNAMICS_PRESETS = {
    "Bypass": {
        "description": "No compression",
        "threshold": 0, "ratio": 1.0, "attack": 20, "release": 200,
        "makeup": 0, "knee": 0,
    },
    "Gentle Glue": {
        "description": "Light compression for mix cohesion",
        "threshold": -20, "ratio": 1.5, "attack": 20, "release": 200,
        "makeup": 1.0, "knee": 6,
    },
    "Standard Master": {
        "description": "Moderate mastering compression",
        "threshold": -16, "ratio": 2.5, "attack": 10, "release": 100,
        "makeup": 2.0, "knee": 4,
    },
    "Punchy": {
        "description": "Emphasized transients, punchy sound",
        "threshold": -14, "ratio": 3.0, "attack": 15, "release": 60,
        "makeup": 3.0, "knee": 2,
    },
    "Aggressive": {
        "description": "Heavy compression for loudness",
        "threshold": -10, "ratio": 5.0, "attack": 3, "release": 40,
        "makeup": 5.0, "knee": 1,
    },
    "Parallel Crush": {
        "description": "Heavy parallel compression (blend with dry)",
        "threshold": -8, "ratio": 8.0, "attack": 1, "release": 30,
        "makeup": 0, "knee": 0,
        "parallel_mix": 0.3,  # 30% wet
    },
    "Vocal Control": {
        "description": "Smooth vocal dynamics control",
        "threshold": -18, "ratio": 2.0, "attack": 10, "release": 80,
        "makeup": 1.5, "knee": 6,
    },
    "Bass Tightener": {
        "description": "Tighten and control low-end",
        "threshold": -16, "ratio": 3.0, "attack": 5, "release": 50,
        "makeup": 2.0, "knee": 3,
        "sidechain_hpf": 80,
    },
}


class CompressorBand:
    """Single compressor band with full parameter control."""

    def __init__(self, name="Full", low_freq=20, high_freq=20000):
        self.name = name
        self.enabled = True
        self.low_freq = low_freq    # Hz (for multiband crossover)
        self.high_freq = high_freq  # Hz

        # Compressor parameters
        self.threshold = -16.0   # dB
        self.ratio = 2.5         # :1
        self.attack = 10.0       # ms
        self.release = 100.0     # ms
        self.makeup = 2.0        # dB
        self.knee = 4.0          # dB
        self.sidechain_hpf = 0   # Hz (0 = disabled)

        # Parallel compression
        self.parallel_mix = 1.0  # 0.0 = all dry, 1.0 = all wet

        # V5.11: Detection mode — how the compressor measures input level
        # "peak"     : instantaneous sample magnitude
        # "rms"      : windowed RMS (window length = attack time)
        # "envelope" : hybrid peak + exponential smoothing (classic default)
        self.detection_mode = "peak"

        # V5.11: Auto-release (dual-stage program-dependent release)
        # When enabled, fast release (50 ms) and slow release (500 ms) are
        # blended based on the crest factor of 50 ms analysis windows.
        self.auto_release = False

    # ------------------------------------------------------------------
    # Detection-mode level computation (NumPy-based offline helpers)
    # ------------------------------------------------------------------

    def detect_level(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute per-sample detection level according to *self.detection_mode*.

        Args:
            audio: 1-D mono signal (or single channel).
            sr: Sample rate in Hz.

        Returns:
            1-D array of detection levels (linear amplitude, same length as *audio*).
        """
        if self.detection_mode == "peak":
            return np.abs(audio)

        elif self.detection_mode == "rms":
            # Windowed RMS with window length equal to attack time
            window_ms = max(1.0, self.attack)
            win_samples = max(1, int(round(window_ms * sr / 1000.0)))
            # Cumulative sum trick for efficient moving average of x^2
            sq = audio.astype(np.float64) ** 2
            cumsum = np.concatenate([[0.0], np.cumsum(sq)])
            # Sliding window mean of squared values
            rms_sq = np.empty_like(sq)
            for i in range(len(sq)):
                lo = max(0, i + 1 - win_samples)
                rms_sq[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
            return np.sqrt(np.maximum(rms_sq, 0.0)).astype(audio.dtype)

        else:
            # "envelope" — hybrid peak + exponential smoothing
            att_coeff = math.exp(-1.0 / max(1.0, self.attack * sr / 1000.0))
            rel_coeff = math.exp(-1.0 / max(1.0, self.release * sr / 1000.0))
            env = np.empty(len(audio), dtype=np.float64)
            prev = 0.0
            abs_audio = np.abs(audio)
            for i in range(len(audio)):
                inp = float(abs_audio[i])
                if inp > prev:
                    prev = att_coeff * prev + (1.0 - att_coeff) * inp
                else:
                    prev = rel_coeff * prev + (1.0 - rel_coeff) * inp
                env[i] = prev
            return env.astype(audio.dtype)

    def compute_auto_release(self, audio: np.ndarray, sr: int) -> float:
        """Compute a program-dependent release time using dual-stage crest analysis.

        Analyses the crest factor (peak / RMS) over 50 ms windows.
        High crest (transient-heavy) → fast release (50 ms).
        Low crest (sustained/dense) → slow release (500 ms).

        Returns:
            Blended release time in milliseconds.
        """
        fast_release = 50.0   # ms
        slow_release = 500.0  # ms

        win_samples = max(1, int(round(0.050 * sr)))  # 50 ms window
        num_windows = max(1, len(audio) // win_samples)

        crest_values: List[float] = []
        for w in range(num_windows):
            start = w * win_samples
            end = start + win_samples
            block = audio[start:end]
            peak = float(np.max(np.abs(block))) + 1e-12
            rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) + 1e-12
            crest_values.append(peak / rms)

        avg_crest = float(np.mean(crest_values)) if crest_values else 1.0

        # Typical crest factor range: 1.0 (square wave) to ~10+ (very transient)
        # Map 1-6 range to slow-fast blend (clamped)
        t = max(0.0, min(1.0, (avg_crest - 1.0) / 5.0))
        blended = slow_release * (1.0 - t) + fast_release * t
        return blended

    def to_ffmpeg_filter(self, intensity: float = 1.0) -> Optional[str]:
        """Generate FFmpeg acompressor filter string."""
        if not self.enabled:
            return None

        # Scale parameters by intensity
        threshold = self.threshold * intensity
        ratio_scaled = 1.0 + (self.ratio - 1.0) * intensity
        makeup = self.makeup * intensity

        if ratio_scaled <= 1.05:
            return None  # Essentially no compression

        # NOTE: FFmpeg acompressor filter syntax for threshold parameter:
        # - Recent FFmpeg versions (4.3+) accept dB suffix: threshold=X.XdB
        # - Older versions may expect ratio format (0-1 scale) without suffix
        # - The dB suffix is backward compatible with most modern FFmpeg builds
        # - If threshold parsing fails, verify FFmpeg version supports dB syntax
        parts = [
            f"acompressor=threshold={threshold:.1f}dB",
            f"ratio={ratio_scaled:.1f}",
            f"attack={self.attack:.1f}",
            f"release={self.release:.1f}",
            f"makeup={makeup:.1f}dB",
            f"knee={self.knee:.1f}dB",
        ]

        comp_filter = ":".join(parts)

        # Parallel mix: scale makeup gain to simulate dry/wet blend
        # Full wet (1.0) = full compression, lower mix = less makeup = gentler effect
        if self.parallel_mix < 0.99:
            # Scale the makeup gain by mix amount — simple and compatible with -af
            scaled_makeup = makeup * self.parallel_mix
            parts[4] = f"makeup={scaled_makeup:.1f}dB"
            comp_filter = ":".join(parts)

        return comp_filter

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "low_freq": self.low_freq,
            "high_freq": self.high_freq,
            "threshold": self.threshold,
            "ratio": self.ratio,
            "attack": self.attack,
            "release": self.release,
            "makeup": self.makeup,
            "knee": self.knee,
            "sidechain_hpf": self.sidechain_hpf,
            "parallel_mix": self.parallel_mix,
            "detection_mode": self.detection_mode,
            "auto_release": self.auto_release,
        }

    @classmethod
    def from_dict(cls, d: dict):
        band = cls(
            name=d.get("name", "Full"),
            low_freq=d.get("low_freq", 20),
            high_freq=d.get("high_freq", 20000),
        )
        for key in ["enabled", "threshold", "ratio", "attack", "release",
                     "makeup", "knee", "sidechain_hpf", "parallel_mix",
                     "detection_mode", "auto_release"]:
            if key in d:
                setattr(band, key, d[key])
        return band


class Dynamics:
    """Dynamics processor with single-band and multiband modes."""

    def __init__(self):
        self.enabled = True
        self.multiband = False
        self.preset_name = "Standard Master"

        # Single-band mode: one compressor for full spectrum
        self.single_band = CompressorBand("Full", 20, 20000)

        # Multiband mode: 3 bands with crossover frequencies
        self.crossover_low = 200    # Hz (Low/Mid split)
        self.crossover_high = 4000  # Hz (Mid/High split)
        self.bands = [
            CompressorBand("Low", 20, 200),
            CompressorBand("Mid", 200, 4000),
            CompressorBand("High", 4000, 20000),
        ]

        # V5.10.5: Post-mastering scale factor (0.0-1.0) for intensity adjustment
        self.post_scale = 1.0

        # V5.10.5: Per-band gain reduction tracking for meter panels
        self.last_band_gr = {"low": 0.0, "mid": 0.0, "high": 0.0}

    def set_threshold(self, value: float):
        self.single_band.threshold = value

    def set_ratio(self, value: float):
        self.single_band.ratio = value

    def set_attack(self, value: float):
        self.single_band.attack = value

    def set_release(self, value: float):
        self.single_band.release = value

    def set_makeup_gain(self, value: float):
        self.single_band.makeup = value

    def load_preset(self, preset_name: str):
        """Load a dynamics preset."""
        if preset_name not in DYNAMICS_PRESETS:
            return

        self.preset_name = preset_name
        preset = DYNAMICS_PRESETS[preset_name]

        # Apply to single band
        self.single_band.threshold = preset["threshold"]
        self.single_band.ratio = preset["ratio"]
        self.single_band.attack = preset["attack"]
        self.single_band.release = preset["release"]
        self.single_band.makeup = preset["makeup"]
        self.single_band.knee = preset["knee"]
        self.single_band.sidechain_hpf = preset.get("sidechain_hpf", 0)
        self.single_band.parallel_mix = preset.get("parallel_mix", 1.0)

    def load_genre_preset(self, genre_name: str):
        """Load dynamics from genre profile."""
        profile = get_genre_profile(genre_name)
        comp = profile.get("compressor", {})

        self.single_band.threshold = comp.get("threshold", -16)
        self.single_band.ratio = comp.get("ratio", 2.5)
        self.single_band.attack = comp.get("attack", 10)
        self.single_band.release = comp.get("release", 100)
        self.single_band.makeup = comp.get("makeup", 2.0)

    def get_ffmpeg_filters(self, intensity: float = 1.0) -> list:
        """Generate FFmpeg filter chain."""
        if not self.enabled:
            return []

        filters = []

        if not self.multiband:
            # Single-band mode
            band = self.single_band

            # Optional sidechain HPF
            if band.sidechain_hpf > 0:
                filters.append(f"highpass=f={band.sidechain_hpf}:p=1")

            comp_filter = band.to_ffmpeg_filter(intensity)
            if comp_filter:
                filters.append(comp_filter)
        else:
            # Multiband mode: use crossover filters to split then compress
            # FFmpeg approach: split → compress each → merge
            # This is complex in a single filter chain, so we use a simplified approach
            # with frequency-dependent compressor behavior

            # Low band: apply compression with HPF sidechain
            low = self.bands[0]
            if low.enabled:
                comp = low.to_ffmpeg_filter(intensity)
                if comp:
                    # Note: True multiband requires complex filtergraph
                    # For offline processing, we apply sequential band-aware compression
                    filters.append(comp)

            # Mid band
            mid = self.bands[1]
            if mid.enabled:
                comp = mid.to_ffmpeg_filter(intensity)
                if comp:
                    filters.append(comp)

            # High band
            high = self.bands[2]
            if high.enabled:
                comp = high.to_ffmpeg_filter(intensity)
                if comp:
                    filters.append(comp)

        return filters

    def get_multiband_complex_filter(self, intensity: float = 1.0) -> Optional[str]:
        """
        Generate FFmpeg complex filtergraph for true multiband compression.
        Returns a complex filter string for use with -filter_complex.
        This is used when multiband mode is enabled and we need proper band splitting.
        """
        if not self.enabled or not self.multiband:
            return None

        low = self.bands[0]
        mid = self.bands[1]
        high = self.bands[2]

        low_comp = low.to_ffmpeg_filter(intensity) or "anull"
        mid_comp = mid.to_ffmpeg_filter(intensity) or "anull"
        high_comp = high.to_ffmpeg_filter(intensity) or "anull"

        # Complex filtergraph: split into 3 bands, compress each, merge
        # V5.11.0 FIX (BUG-DSP-005): Use 4th-order Linkwitz-Riley crossover (p=4)
        # for flatter reconstruction, and normalize=1 with volume=3 compensation
        # to prevent wildly varying output levels depending on spectral content
        cf = (
            f"[0:a]asplit=3[low][mid][high];"
            f"[low]lowpass=f={self.crossover_low}:p=4,{low_comp}[lo];"
            f"[mid]highpass=f={self.crossover_low}:p=4,"
            f"lowpass=f={self.crossover_high}:p=4,{mid_comp}[mi];"
            f"[high]highpass=f={self.crossover_high}:p=4,{high_comp}[hi];"
            f"[lo][mi][hi]amix=inputs=3:duration=first:normalize=1,volume=3.0"
        )
        return cf

    def get_settings_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "multiband": self.multiband,
            "preset_name": self.preset_name,
            "single_band": self.single_band.to_dict(),
            "crossover_low": self.crossover_low,
            "crossover_high": self.crossover_high,
            "bands": [b.to_dict() for b in self.bands],
        }

    def load_settings_dict(self, d: dict):
        self.enabled = d.get("enabled", True)
        self.multiband = d.get("multiband", False)
        self.preset_name = d.get("preset_name", "Standard Master")
        if "single_band" in d:
            self.single_band = CompressorBand.from_dict(d["single_band"])
        self.crossover_low = d.get("crossover_low", 200)
        self.crossover_high = d.get("crossover_high", 4000)
        if "bands" in d:
            self.bands = [CompressorBand.from_dict(b) for b in d["bands"]]

    def __repr__(self):
        mode = "Multiband" if self.multiband else "Single"
        return f"Dynamics(mode={mode}, preset={self.preset_name})"
