"""
LongPlay Studio V5.0 — Imager Module
Inspired by iZotope Ozone 12 Imager

Features:
- Stereo Width control (0% mono → 100% original → 200% super-wide)
- Per-band width control (Low/Mid/High bands)
- Stereoize: Add width to mono-ish content
- Balance: L/R balance adjustment
- Mono Bass: Keep low frequencies centered
- Correlation display info

Uses FFmpeg: stereotools, pan (for M/S encoding), crossover filters
"""

import math
from typing import List, Optional

import numpy as np

from .genre_profiles import get_genre_profile


# Imager Presets
IMAGER_PRESETS = {
    "Bypass": {
        "description": "No stereo processing",
        "width": 100,
        "mono_bass_freq": 0,
    },
    "Subtle Wide": {
        "description": "Slightly wider stereo image",
        "width": 120,
        "mono_bass_freq": 0,
    },
    "Wide Master": {
        "description": "Noticeably wider with controlled low-end",
        "width": 140,
        "mono_bass_freq": 120,
    },
    "Super Wide": {
        "description": "Maximum width for electronic/ambient",
        "width": 170,
        "mono_bass_freq": 150,
    },
    "Mono Bass Wide Top": {
        "description": "Tight mono bass with wide mids/highs",
        "width": 130,
        "mono_bass_freq": 200,
    },
    "Narrow (Focused)": {
        "description": "Narrower image for focused sound",
        "width": 80,
        "mono_bass_freq": 0,
    },
    "Mono": {
        "description": "Full mono collapse",
        "width": 0,
        "mono_bass_freq": 0,
    },
}


class ImagerBand:
    """Single imager band with width control."""

    def __init__(self, name="Full", low_freq=20, high_freq=20000, width=100):
        self.name = name
        self.enabled = True
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.width = width  # 0=mono, 100=original, 200=max_wide

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "low_freq": self.low_freq,
            "high_freq": self.high_freq,
            "width": self.width,
        }

    @classmethod
    def from_dict(cls, d: dict):
        band = cls(
            name=d.get("name", "Full"),
            low_freq=d.get("low_freq", 20),
            high_freq=d.get("high_freq", 20000),
            width=d.get("width", 100),
        )
        band.enabled = d.get("enabled", True)
        return band


class Imager:
    """Stereo Width controller with per-band control."""

    def __init__(self):
        self.enabled = True
        self.multiband = False
        self.preset_name = "Bypass"

        # Single-band mode
        self.width = 100            # 0-200 (100 = original)
        self.balance = 0.0          # -1.0 (full left) to +1.0 (full right)
        self.mono_bass_freq = 0     # Hz (0 = disabled, >0 = mono below this freq)

        # Correlation safety (V5.11 — auto-reduce width on low correlation)
        self.correlation_safety = True
        self.min_correlation = 0.2

        # Stereoize modes (V5.11 — delay/phase-based stereo decorrelation)
        self.stereoize_mode = "off"   # "off", "I", "II"
        self.stereoize_amount = 0     # 0-100

        # Multiband mode
        self.crossover_low = 200
        self.crossover_high = 4000
        self.bands = [
            ImagerBand("Low", 20, 200, 80),      # Narrower bass
            ImagerBand("Mid", 200, 4000, 110),    # Slightly wider mids
            ImagerBand("High", 4000, 20000, 130), # Wider highs
        ]

    def set_width(self, width_pct: int):
        """Set stereo width. 0=mono, 100=original, 200=super-wide."""
        self.width = max(0, min(200, width_pct))

    def load_preset(self, preset_name: str):
        """Load an imager preset."""
        if preset_name not in IMAGER_PRESETS:
            return
        self.preset_name = preset_name
        preset = IMAGER_PRESETS[preset_name]
        self.width = preset["width"]
        self.mono_bass_freq = preset.get("mono_bass_freq", 0)

    def load_genre_preset(self, genre_name: str):
        """Load stereo width from genre profile."""
        profile = get_genre_profile(genre_name)
        self.width = profile.get("stereo_width", 100)

    # ------------------------------------------------------------------
    # Stereoize I / II — NumPy-based decorrelation processors
    # ------------------------------------------------------------------

    @staticmethod
    def _allpass_cascade(signal: np.ndarray, coeffs: List[float]) -> np.ndarray:
        """First-order allpass cascade: y[n] = coeff*x[n] + x[n-1] - coeff*y[n-1]
        Applies each coefficient sequentially for deeper phase rotation."""
        out = signal.copy()
        for c in coeffs:
            x_prev = 0.0
            y_prev = 0.0
            buf = np.empty_like(out)
            for i in range(len(out)):
                buf[i] = c * out[i] + x_prev - c * y_prev
                x_prev = out[i]
                y_prev = buf[i]
            out = buf
        return out

    def stereoize_i(self, audio: np.ndarray, amount: float, sr: int) -> np.ndarray:
        """Stereoize I — delay-based decorrelation.

        Applies a short delay (0.1 – 5 ms scaled by *amount*) to one channel
        combined with an allpass filter for frequency-dependent phase shift.

        Args:
            audio: 2-D NumPy array, shape (num_samples, 2).
            amount: 0 – 100 (percent).
            sr: Sample rate in Hz.

        Returns:
            Processed stereo audio (same shape).
        """
        if audio.ndim != 2 or audio.shape[1] != 2:
            return audio
        amount = max(0.0, min(100.0, float(amount)))
        if amount == 0:
            return audio

        # Delay 0.1 ms – 5 ms mapped from amount 0-100
        delay_ms = 0.1 + (amount / 100.0) * 4.9
        delay_samples = int(round(delay_ms * sr / 1000.0))

        out = audio.copy()

        # Delay the right channel
        if delay_samples > 0:
            out[delay_samples:, 1] = audio[:-delay_samples, 1]
            out[:delay_samples, 1] = 0.0

        # Allpass filter on right channel for frequency-dependent phase shift
        # Coefficient derived from amount (higher amount → more phase rotation)
        ap_coeff = 0.3 + 0.5 * (amount / 100.0)  # 0.3 – 0.8
        out[:, 1] = self._allpass_cascade(out[:, 1], [ap_coeff])

        # Wet/dry blend based on amount
        blend = amount / 100.0
        result = audio * (1.0 - blend) + out * blend
        return result

    def stereoize_ii(self, audio: np.ndarray, amount: float, sr: int) -> np.ndarray:
        """Stereoize II — phase-based decorrelation (complementary allpass networks).

        Uses complementary allpass filter networks on L and R to create
        approximately 90-degree phase differences at certain frequencies.
        More mono-compatible than Stereoize I because no delay offset is added.

        Args:
            audio: 2-D NumPy array, shape (num_samples, 2).
            amount: 0 – 100 (percent).
            sr: Sample rate in Hz.

        Returns:
            Processed stereo audio (same shape).
        """
        if audio.ndim != 2 or audio.shape[1] != 2:
            return audio
        amount = max(0.0, min(100.0, float(amount)))
        if amount == 0:
            return audio

        # Complementary allpass coefficient sets chosen to approximate
        # a 90-degree phase split (Hilbert-like) at mid-range frequencies.
        # More coefficients → broader frequency coverage.
        base = 0.4 + 0.4 * (amount / 100.0)  # 0.4 – 0.8
        coeffs_l = [base, base * 0.6, base * 0.35]
        coeffs_r = [-base, -base * 0.6, -base * 0.35]

        out = audio.copy()
        out[:, 0] = self._allpass_cascade(audio[:, 0], coeffs_l)
        out[:, 1] = self._allpass_cascade(audio[:, 1], coeffs_r)

        # Wet/dry blend
        blend = amount / 100.0
        result = audio * (1.0 - blend) + out * blend
        return result

    def apply_stereoize(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply the currently configured stereoize mode to *audio*.

        Called as part of the imager process chain when stereoize is enabled.
        """
        if self.stereoize_mode == "I":
            return self.stereoize_i(audio, self.stereoize_amount, sr)
        elif self.stereoize_mode == "II":
            return self.stereoize_ii(audio, self.stereoize_amount, sr)
        return audio

    def apply_correlation_safety(self, audio: np.ndarray, sr: int,
                                  min_correlation: float = 0.2) -> np.ndarray:
        """Auto-reduce stereo width if correlation drops too low.

        AI-generated music sometimes has weird phase issues.
        This prevents mono compatibility problems.

        Args:
            audio: 2-D NumPy array, shape (num_samples, 2).
            sr: Sample rate in Hz.
            min_correlation: Minimum acceptable correlation (default 0.2).

        Returns:
            Processed audio with safe stereo field (same shape).
        """
        if audio.ndim != 2 or audio.shape[1] != 2:
            return audio

        # Use instance setting if not overridden
        if min_correlation is None:
            min_correlation = self.min_correlation

        n_samples = audio.shape[0]

        # Step 1: Compute inter-channel correlation over 50ms windows
        window_size = int(0.050 * sr)  # 50ms
        if window_size < 1:
            window_size = 1
        hop = window_size // 2
        if hop < 1:
            hop = 1

        n_windows = max(1, (n_samples - window_size) // hop)

        # Per-sample width multiplier (1.0 = keep, lower = narrower)
        width_envelope = np.ones(n_samples, dtype=np.float64)

        for i in range(n_windows):
            start = i * hop
            end = start + window_size
            if end > n_samples:
                break

            left = audio[start:end, 0]
            right = audio[start:end, 1]

            # Compute Pearson correlation
            l_mean = np.mean(left)
            r_mean = np.mean(right)
            l_centered = left - l_mean
            r_centered = right - r_mean
            numerator = np.sum(l_centered * r_centered)
            denominator = np.sqrt(np.sum(l_centered ** 2) * np.sum(r_centered ** 2))

            if denominator > 1e-12:
                corr = numerator / denominator
            else:
                corr = 1.0  # silence -> treat as fine

            # Step 2: If correlation < min_correlation, reduce width for that segment
            if corr < min_correlation:
                # Scale width reduction by how far below threshold we are
                # At corr=min_correlation -> factor=1.0, at corr=-1 -> factor=0.0
                factor = max(0.0, (corr + 1.0) / (min_correlation + 1.0))
                width_envelope[start:end] = np.minimum(
                    width_envelope[start:end], factor
                )

        # Step 3: Smooth the width changes to avoid artifacts
        # Use a 10ms smoothing window
        smooth_size = max(1, int(0.010 * sr))
        kernel = np.ones(smooth_size, dtype=np.float64) / smooth_size
        width_envelope = np.convolve(width_envelope, kernel, mode='same')

        # Apply: blend toward mono based on width_envelope
        mid = (audio[:, 0] + audio[:, 1]) * 0.5
        side = (audio[:, 0] - audio[:, 1]) * 0.5

        # Scale side signal by envelope
        side_scaled = side * width_envelope

        result = np.column_stack([
            mid + side_scaled,
            mid - side_scaled,
        ])

        return result

    def _width_to_stereotools_param(self, width_pct: int) -> float:
        """
        Convert width percentage to FFmpeg stereotools 'widening' parameter.
        0% → 0.0 (mono), 100% → 1.0 (original), 200% → 2.0 (max wide)
        """
        return width_pct / 100.0

    def get_ffmpeg_filters(self, intensity: float = 1.0) -> list:
        """Generate FFmpeg filter chain for stereo imaging.

        Uses valid FFmpeg stereotools parameters:
        - stereotools: softclip, mutel, muter, phase, mode, slev, sbal,
                       mlev, mpan, base, delay, sclevel, phase, bmode_in, bmode_out
        NOTE: FFmpeg stereotools DOES support 'mlev' (mid level) and 'slev' (side level)
        as of FFmpeg 4.0+. These are valid M/S matrix parameters.
        See: https://ffmpeg.org/ffmpeg-filters.html#stereotools
        """
        if not self.enabled:
            return []

        filters = []

        if not self.multiband:
            # Single-band stereo width
            effective_width = 100 + (self.width - 100) * intensity
            widening = self._width_to_stereotools_param(int(effective_width))

            if abs(widening - 1.0) > 0.01:
                # stereotools M/S matrix:
                # mlev = mid level (center channel gain)
                # slev = side level (side channel gain)
                # widening < 1.0 → narrower (reduce side level)
                # widening > 1.0 → wider (boost side level)
                mlev = 1.0
                slev = widening

                if effective_width == 0:
                    # Full mono: sum L+R equally
                    filters.append("pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1")
                else:
                    # Use extrastereo for reliable width control across all FFmpeg versions
                    # extrastereo: m = multiplier for stereo difference signal
                    # m=0 → mono, m=1 → original, m=2 → double width
                    filters.append(f"extrastereo=m={slev:.3f}")

            # Mono bass: sum low frequencies to mono using lowpass split
            if self.mono_bass_freq > 0:
                # Use pan filter to mono-ize bass below cutoff
                # This creates a simple bass mono effect without invalid filters
                # We use a lowpass + pan approach via stereotools with base parameter
                # 'base' widens/narrows the stereo base (0.0 = mono, 1.0 = natural)
                base_val = max(0.0, 1.0 - (self.mono_bass_freq / 300.0))
                filters.append(f"stereotools=base={base_val:.3f}")

            # Balance
            if abs(self.balance) > 0.01:
                # Pan balance: -1.0 = left, 0 = center, +1.0 = right
                left_gain = 1.0 - max(0, self.balance)
                right_gain = 1.0 + min(0, self.balance)
                filters.append(
                    f"pan=stereo|c0={left_gain:.2f}*c0|c1={right_gain:.2f}*c1"
                )

        else:
            # Multiband mode — apply different widths per band
            # Simplified approach: use weighted average for offline processing
            total_width = 0
            count = 0
            for band in self.bands:
                if band.enabled:
                    total_width += band.width
                    count += 1
            avg_width = total_width / max(count, 1)
            effective_width = 100 + (avg_width - 100) * intensity
            widening = self._width_to_stereotools_param(int(effective_width))

            if abs(widening - 1.0) > 0.01:
                # extrastereo is universally supported across FFmpeg versions
                filters.append(f"extrastereo=m={widening:.3f}")

        return filters

    def get_multiband_complex_filter(self, intensity: float = 1.0) -> Optional[str]:
        """
        Generate true multiband stereo width using complex filtergraph.
        Returns string for -filter_complex flag.
        """
        if not self.enabled or not self.multiband:
            return None

        low_w = self._width_to_stereotools_param(
            int(100 + (self.bands[0].width - 100) * intensity)
        )
        mid_w = self._width_to_stereotools_param(
            int(100 + (self.bands[1].width - 100) * intensity)
        )
        high_w = self._width_to_stereotools_param(
            int(100 + (self.bands[2].width - 100) * intensity)
        )

        cf = (
            f"[0:a]asplit=3[low][mid][high];"
            f"[low]lowpass=f={self.crossover_low}:p=2,"
            f"extrastereo=m={low_w:.3f}[lo];"
            f"[mid]highpass=f={self.crossover_low}:p=2,"
            f"lowpass=f={self.crossover_high}:p=2,"
            f"extrastereo=m={mid_w:.3f}[mi];"
            f"[high]highpass=f={self.crossover_high}:p=2,"
            f"extrastereo=m={high_w:.3f}[hi];"
            f"[lo][mi][hi]amix=inputs=3:duration=first:normalize=0"
        )
        return cf

    def get_settings_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "multiband": self.multiband,
            "preset_name": self.preset_name,
            "width": self.width,
            "balance": self.balance,
            "mono_bass_freq": self.mono_bass_freq,
            "stereoize_mode": self.stereoize_mode,
            "stereoize_amount": self.stereoize_amount,
            "crossover_low": self.crossover_low,
            "crossover_high": self.crossover_high,
            "bands": [b.to_dict() for b in self.bands],
        }

    def load_settings_dict(self, d: dict):
        for key in ["enabled", "multiband", "preset_name", "width",
                     "balance", "mono_bass_freq", "stereoize_mode",
                     "stereoize_amount", "crossover_low", "crossover_high"]:
            if key in d:
                setattr(self, key, d[key])
        if "bands" in d:
            self.bands = [ImagerBand.from_dict(b) for b in d["bands"]]

    def __repr__(self):
        mode = "Multiband" if self.multiband else "Single"
        return f"Imager(mode={mode}, width={self.width}%, preset={self.preset_name})"
