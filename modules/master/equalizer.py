"""
LongPlay Studio V5.0 — Equalizer Module
Inspired by iZotope Ozone 12 EQ

Features:
- 8-band Parametric EQ (frequency, gain, Q/bandwidth)
- Genre Preset Mode: Auto-load EQ curve from genre profile
- Manual Mode: Adjust each band freely
- Band types: Peak, Low Shelf, High Shelf, Low Pass, High Pass
- Tone Presets: Quick tonal adjustments (Warm, Bright, Bass Boost, etc.)
- Dynamic EQ: Per-band dynamic gain modulation via sidechain envelope follower
- Linear-phase mode: FIR-based EQ via frequency sampling (zero phase distortion)
- Analog mode: Harmonic saturation + proportional-Q behavior

Uses FFmpeg: equalizer, lowshelf, highshelf, highpass, lowpass filters
"""

import math
import numpy as np
from typing import List, Dict, Optional

try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .genre_profiles import get_genre_profile


# Quick EQ Tone Presets (simple one-click tonal adjustments)
EQ_TONE_PRESETS = {
    "Flat": {
        "description": "No EQ changes — bypass",
        "bands": [],
    },
    "Warm": {
        "description": "Boost low-mids, gentle high rolloff",
        "bands": [
            {"freq": 200, "gain": 2.0, "width": 1.0, "type": "lowshelf"},
            {"freq": 8000, "gain": -1.5, "width": 1.0, "type": "highshelf"},
        ],
    },
    "Bright": {
        "description": "Boost presence and air frequencies",
        "bands": [
            {"freq": 3000, "gain": 1.5, "width": 2.0, "type": "equalizer"},
            {"freq": 10000, "gain": 2.5, "width": 1.0, "type": "highshelf"},
        ],
    },
    "Bass Boost": {
        "description": "Heavy low-end enhancement",
        "bands": [
            {"freq": 60, "gain": 4.0, "width": 0.7, "type": "equalizer"},
            {"freq": 120, "gain": 2.0, "width": 1.0, "type": "equalizer"},
        ],
    },
    "Vocal Presence": {
        "description": "Enhance vocal clarity and presence",
        "bands": [
            {"freq": 200, "gain": -1.5, "width": 1.5, "type": "equalizer"},
            {"freq": 2500, "gain": 2.0, "width": 1.5, "type": "equalizer"},
            {"freq": 5000, "gain": 1.5, "width": 2.0, "type": "equalizer"},
        ],
    },
    "De-Mud": {
        "description": "Clean up muddy low-mids",
        "bands": [
            {"freq": 250, "gain": -3.0, "width": 1.0, "type": "equalizer"},
            {"freq": 500, "gain": -1.5, "width": 1.5, "type": "equalizer"},
        ],
    },
    "Air": {
        "description": "Add sparkle and airiness",
        "bands": [
            {"freq": 10000, "gain": 2.0, "width": 2.0, "type": "equalizer"},
            {"freq": 15000, "gain": 3.0, "width": 1.0, "type": "highshelf"},
        ],
    },
    "Scooped (V-Shape)": {
        "description": "Boost lows and highs, cut mids — classic V-shape",
        "bands": [
            {"freq": 80, "gain": 3.0, "width": 0.7, "type": "equalizer"},
            {"freq": 500, "gain": -2.0, "width": 1.5, "type": "equalizer"},
            {"freq": 1000, "gain": -2.0, "width": 1.5, "type": "equalizer"},
            {"freq": 8000, "gain": 2.5, "width": 1.0, "type": "highshelf"},
        ],
    },
    "Loudness Enhance": {
        "description": "Perceptual loudness boost (Fletcher-Munson curve)",
        "bands": [
            {"freq": 60, "gain": 3.0, "width": 0.6, "type": "equalizer"},
            {"freq": 3500, "gain": 2.0, "width": 2.0, "type": "equalizer"},
            {"freq": 10000, "gain": 1.5, "width": 1.0, "type": "highshelf"},
        ],
    },
    "Tape/Analog": {
        "description": "Simulate analog tape frequency response",
        "bands": [
            {"freq": 30, "gain": -3.0, "width": 1.0, "type": "highpass"},
            {"freq": 100, "gain": 1.0, "width": 1.0, "type": "equalizer"},
            {"freq": 14000, "gain": -2.0, "width": 1.0, "type": "highshelf"},
        ],
    },
}


class EQBand:
    """Single EQ band with frequency, gain, Q, and type."""

    TYPES = ["equalizer", "lowshelf", "highshelf", "highpass", "lowpass"]

    def __init__(self, freq=1000, gain=0.0, width=1.0, band_type="equalizer", enabled=True):
        self.freq = freq          # Hz
        self.gain = gain          # dB (-12 to +12)
        self.width = width        # Q/octave width (0.1 to 10)
        self.band_type = band_type
        self.enabled = enabled

        # --- Dynamic EQ parameters ---
        self.dynamic_enabled = False       # Enable dynamic behavior for this band
        self.dynamic_threshold = -20.0     # dB threshold for sidechain detection
        self.dynamic_ratio = 2.0           # Compression/expansion ratio (1.0 = off)
        self.dynamic_attack = 10.0         # Attack time in ms
        self.dynamic_release = 100.0       # Release time in ms
        self._envelope_state = 0.0         # Internal envelope follower state

    def to_ffmpeg_filter(self, intensity: float = 1.0) -> Optional[str]:
        """Convert band to FFmpeg filter string."""
        if not self.enabled:
            return None

        gain = self.gain * intensity

        if self.band_type == "equalizer":
            if abs(gain) < 0.1:
                return None
            return f"equalizer=f={self.freq}:width_type=o:w={self.width:.2f}:g={gain:.1f}"

        elif self.band_type == "lowshelf":
            if abs(gain) < 0.1:
                return None
            return f"lowshelf=f={self.freq}:g={gain:.1f}:t=s:w=0.5"

        elif self.band_type == "highshelf":
            if abs(gain) < 0.1:
                return None
            return f"highshelf=f={self.freq}:g={gain:.1f}:t=s:w=0.5"

        elif self.band_type == "highpass":
            return f"highpass=f={self.freq}:p=2"

        elif self.band_type == "lowpass":
            return f"lowpass=f={self.freq}:p=2"

        return None

    def compute_dynamic_gain(self, sidechain_level_db: float) -> float:
        """Compute effective gain based on sidechain level relative to threshold.

        When dynamic is enabled, the band gain is modulated:
        - Below threshold: gain is reduced proportionally
        - Above threshold: full gain is applied
        - Ratio controls how aggressively gain scales

        Args:
            sidechain_level_db: RMS level of the bandpass-filtered sidechain in dB.

        Returns:
            Effective gain in dB for this band.
        """
        if not self.dynamic_enabled:
            return self.gain

        excess = sidechain_level_db - self.dynamic_threshold
        if excess <= 0:
            # Below threshold: reduce gain. At -inf, gain approaches 0
            # Scale factor: 0..1 based on how close we are to threshold
            # Use a soft knee: gain_factor = max(0, 1 + excess / (abs(threshold) + 1))
            range_db = max(abs(self.dynamic_threshold), 1.0)
            gain_factor = max(0.0, 1.0 + excess / range_db)
            return self.gain * gain_factor
        else:
            # Above threshold: apply full gain, optionally compressed by ratio
            if self.dynamic_ratio > 1.0:
                compressed_excess = excess / self.dynamic_ratio
                gain_factor = min(1.0, (self.dynamic_threshold + compressed_excess) /
                                  (sidechain_level_db if sidechain_level_db != 0 else -0.001))
                # Clamp to full gain
                gain_factor = min(1.0, max(0.0, gain_factor + (1.0 - gain_factor) * 0.5))
            else:
                gain_factor = 1.0
            return self.gain * gain_factor

    def update_envelope(self, input_rms: float, sample_rate: float = 48000.0) -> float:
        """Update the envelope follower state for dynamic EQ sidechain detection.

        Uses a simple peak/RMS envelope follower with attack/release smoothing.

        Args:
            input_rms: Current RMS amplitude of the bandpass-filtered signal.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Current envelope level in dB.
        """
        # Convert ms to per-sample coefficients
        attack_coeff = math.exp(-1.0 / (self.dynamic_attack * 0.001 * sample_rate + 1e-12))
        release_coeff = math.exp(-1.0 / (self.dynamic_release * 0.001 * sample_rate + 1e-12))

        if input_rms > self._envelope_state:
            self._envelope_state = attack_coeff * self._envelope_state + (1.0 - attack_coeff) * input_rms
        else:
            self._envelope_state = release_coeff * self._envelope_state + (1.0 - release_coeff) * input_rms

        # Convert to dB (with floor to avoid log(0))
        level_db = 20.0 * math.log10(max(self._envelope_state, 1e-10))
        return level_db

    def to_dict(self) -> dict:
        return {
            "freq": self.freq,
            "gain": self.gain,
            "width": self.width,
            "type": self.band_type,
            "enabled": self.enabled,
            "dynamic_enabled": self.dynamic_enabled,
            "dynamic_threshold": self.dynamic_threshold,
            "dynamic_ratio": self.dynamic_ratio,
            "dynamic_attack": self.dynamic_attack,
            "dynamic_release": self.dynamic_release,
        }

    @classmethod
    def from_dict(cls, d: dict):
        band = cls(
            freq=d.get("freq", 1000),
            gain=d.get("gain", 0.0),
            width=d.get("width", 1.0),
            band_type=d.get("type", "equalizer"),
            enabled=d.get("enabled", True),
        )
        band.dynamic_enabled = d.get("dynamic_enabled", False)
        band.dynamic_threshold = d.get("dynamic_threshold", -20.0)
        band.dynamic_ratio = d.get("dynamic_ratio", 2.0)
        band.dynamic_attack = d.get("dynamic_attack", 10.0)
        band.dynamic_release = d.get("dynamic_release", 100.0)
        return band


class Equalizer:
    """8-band Parametric EQ with genre presets and manual mode."""

    NUM_BANDS = 8

    # Default band frequencies (spread across spectrum)
    DEFAULT_FREQS = [32, 64, 125, 250, 1000, 4000, 8000, 16000]
    DEFAULT_TYPES = [
        "highpass", "lowshelf", "equalizer", "equalizer",
        "equalizer", "equalizer", "highshelf", "lowpass",
    ]

    def __init__(self):
        self.enabled = True
        self.preset_mode = True     # True = use preset, False = manual
        self.current_preset = "Flat"
        self.bands = self._create_default_bands()

        # --- Linear-phase mode ---
        self.linear_phase = False       # When True, use FIR convolution instead of IIR biquads
        self.linear_phase_taps = 4095   # FIR length (odd, good to ~20Hz at 48kHz)
        self._fir_kernel = None         # Cached FIR kernel (recomputed when bands change)
        self._fir_dirty = True          # Flag to trigger FIR recomputation

        # --- Analog mode ---
        self.analog_mode = False        # When True, add harmonic saturation + proportional-Q

    def _create_default_bands(self) -> List[EQBand]:
        """Create 8 default bands spread across spectrum."""
        bands = []
        for i in range(self.NUM_BANDS):
            bands.append(EQBand(
                freq=self.DEFAULT_FREQS[i],
                gain=0.0,
                width=1.0,
                band_type=self.DEFAULT_TYPES[i],
                enabled=True,
            ))
        return bands

    def load_genre_preset(self, genre_name: str):
        """Load EQ curve from genre profile."""
        profile = get_genre_profile(genre_name)
        eq_data = profile.get("eq", {})
        eq_bands = eq_data.get("bands", [])

        # Reset all bands to flat
        self.bands = self._create_default_bands()

        # Apply genre bands to our 8 bands
        # Match by closest frequency
        for genre_band in eq_bands:
            freq = genre_band["freq"]
            gain = genre_band.get("gain", 0)
            width = genre_band.get("width", 1.0)
            band_type = genre_band.get("type", "equalizer")

            # Find closest existing band
            best_idx = 0
            best_dist = abs(self.bands[0].freq - freq)
            for i, band in enumerate(self.bands):
                dist = abs(band.freq - freq)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            # Apply to that band
            self.bands[best_idx].freq = freq
            self.bands[best_idx].gain = gain
            self.bands[best_idx].width = width
            if band_type in EQBand.TYPES:
                self.bands[best_idx].band_type = band_type

        self.preset_mode = True

    def load_tone_preset(self, preset_name: str):
        """Load a quick tone preset."""
        if preset_name not in EQ_TONE_PRESETS:
            return

        preset = EQ_TONE_PRESETS[preset_name]
        self.current_preset = preset_name

        # Reset bands
        self.bands = self._create_default_bands()

        # Apply preset bands
        for i, band_data in enumerate(preset.get("bands", [])):
            if i < self.NUM_BANDS:
                self.bands[i] = EQBand.from_dict(band_data)

        self.preset_mode = True

    def set_band(self, index: int, freq=None, gain=None, width=None, band_type=None):
        """Manually set a band's parameters."""
        if 0 <= index < self.NUM_BANDS:
            band = self.bands[index]
            if freq is not None:
                band.freq = max(20, min(20000, freq))
            if gain is not None:
                band.gain = max(-12, min(12, gain))
            if width is not None:
                band.width = max(0.1, min(10, width))
            if band_type is not None and band_type in EQBand.TYPES:
                band.band_type = band_type
            self.preset_mode = False

    def get_ffmpeg_filters(self, intensity: float = 1.0) -> list:
        """Generate FFmpeg filter strings for all active bands."""
        if not self.enabled:
            return []

        filters = []
        for band in self.bands:
            f = band.to_ffmpeg_filter(intensity)
            if f:
                filters.append(f)
        return filters

    def get_settings_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "preset_mode": self.preset_mode,
            "current_preset": self.current_preset,
            "linear_phase": self.linear_phase,
            "linear_phase_taps": self.linear_phase_taps,
            "analog_mode": self.analog_mode,
            "bands": [b.to_dict() for b in self.bands],
        }

    def load_settings_dict(self, d: dict):
        self.enabled = d.get("enabled", True)
        self.preset_mode = d.get("preset_mode", True)
        self.current_preset = d.get("current_preset", "Flat")
        self.linear_phase = d.get("linear_phase", False)
        self.linear_phase_taps = d.get("linear_phase_taps", 4096)
        self.analog_mode = d.get("analog_mode", False)
        bands_data = d.get("bands", [])
        self.bands = [EQBand.from_dict(b) for b in bands_data]
        while len(self.bands) < self.NUM_BANDS:
            self.bands.append(EQBand())
        self.invalidate_fir()

    # ---------------------------------------------------------------
    # Linear-phase mode: FIR-based EQ via frequency sampling
    # ---------------------------------------------------------------

    def _biquad_coeffs(self, band: EQBand, sample_rate: float = 48000.0):
        """Compute biquad (second-order IIR) coefficients for a single EQ band.

        Returns (b, a) coefficient arrays suitable for scipy.signal.
        Uses the Audio EQ Cookbook formulas (Robert Bristow-Johnson).
        """
        f0 = band.freq
        gain_db = band.gain
        Q = band.width  # Q factor

        # Analog mode: proportional-Q (Q narrows slightly as gain increases)
        if self.analog_mode and band.band_type == "equalizer":
            gain_factor = abs(gain_db) / 12.0  # Normalize to 0..1 range
            Q = Q * (1.0 - 0.15 * gain_factor)  # Up to 15% narrower at max gain
            Q = max(0.1, Q)

        w0 = 2.0 * math.pi * f0 / sample_rate
        alpha = math.sin(w0) / (2.0 * Q)
        A = 10.0 ** (gain_db / 40.0)  # sqrt of linear gain

        if band.band_type == "equalizer":
            b0 = 1.0 + alpha * A
            b1 = -2.0 * math.cos(w0)
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * math.cos(w0)
            a2 = 1.0 - alpha / A
        elif band.band_type == "lowshelf":
            sq = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1) - (A - 1) * math.cos(w0) + sq)
            b1 = 2.0 * A * ((A - 1) - (A + 1) * math.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * math.cos(w0) - sq)
            a0 = (A + 1) + (A - 1) * math.cos(w0) + sq
            a1 = -2.0 * ((A - 1) + (A + 1) * math.cos(w0))
            a2 = (A + 1) + (A - 1) * math.cos(w0) - sq
        elif band.band_type == "highshelf":
            sq = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1) + (A - 1) * math.cos(w0) + sq)
            b1 = -2.0 * A * ((A - 1) + (A + 1) * math.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * math.cos(w0) - sq)
            a0 = (A + 1) - (A - 1) * math.cos(w0) + sq
            a1 = 2.0 * ((A - 1) - (A + 1) * math.cos(w0))
            a2 = (A + 1) - (A - 1) * math.cos(w0) - sq
        elif band.band_type == "highpass":
            b0 = (1.0 + math.cos(w0)) / 2.0
            b1 = -(1.0 + math.cos(w0))
            b2 = (1.0 + math.cos(w0)) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * math.cos(w0)
            a2 = 1.0 - alpha
        elif band.band_type == "lowpass":
            b0 = (1.0 - math.cos(w0)) / 2.0
            b1 = 1.0 - math.cos(w0)
            b2 = (1.0 - math.cos(w0)) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * math.cos(w0)
            a2 = 1.0 - alpha
        else:
            return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

        b = np.array([b0 / a0, b1 / a0, b2 / a0])
        a = np.array([1.0, a1 / a0, a2 / a0])
        return b, a

    def _compute_combined_response(self, sample_rate: float = 48000.0, n_points: int = 2048):
        """Compute the combined frequency response of all enabled bands.

        Returns:
            (freqs, H): frequency array and complex frequency response.
        """
        freqs = np.linspace(0, sample_rate / 2, n_points)
        H = np.ones(n_points, dtype=complex)

        for band in self.bands:
            if not band.enabled:
                continue
            if band.band_type == "equalizer" and abs(band.gain) < 0.1:
                continue

            b, a = self._biquad_coeffs(band, sample_rate)
            _, h = scipy_signal.freqz(b, a, worN=freqs, fs=sample_rate)
            H *= h

        return freqs, H

    def build_fir_kernel(self, sample_rate: float = 48000.0) -> np.ndarray:
        """Build a linear-phase FIR kernel from the combined IIR EQ response.

        Uses scipy.signal.firwin2 to design an FIR filter that matches the
        magnitude response of the cascaded biquad EQ, with linear (zero) phase.

        Args:
            sample_rate: Audio sample rate in Hz.

        Returns:
            FIR filter kernel as numpy array (self.linear_phase_taps length).
        """
        if not HAS_SCIPY:
            raise RuntimeError("scipy is required for linear-phase EQ mode")

        n_taps = self.linear_phase_taps
        n_points = n_taps // 2 + 1

        freqs, H = self._compute_combined_response(sample_rate, n_points)

        # Extract magnitude response (discard phase — we want linear phase)
        mag = np.abs(H)

        # Normalize frequency to [0, 1] for firwin2 (Nyquist = 1)
        nyquist = sample_rate / 2.0
        norm_freqs = freqs / nyquist

        # Ensure endpoints are exactly 0 and 1
        norm_freqs[0] = 0.0
        norm_freqs[-1] = 1.0

        # Design the FIR filter
        fir_kernel = scipy_signal.firwin2(n_taps, norm_freqs, mag)

        self._fir_kernel = fir_kernel
        self._fir_dirty = False
        return fir_kernel

    def process_linear_phase(self, audio: np.ndarray, sample_rate: float = 48000.0) -> np.ndarray:
        """Process audio using linear-phase FIR convolution.

        Args:
            audio: Input audio array (1D mono or 2D multichannel, shape [samples] or [channels, samples]).
            sample_rate: Audio sample rate in Hz.

        Returns:
            Processed audio array (same shape as input, may be longer due to FIR tail).
        """
        if not HAS_SCIPY:
            raise RuntimeError("scipy is required for linear-phase EQ mode")

        if self._fir_dirty or self._fir_kernel is None:
            self.build_fir_kernel(sample_rate)

        kernel = self._fir_kernel

        if audio.ndim == 1:
            return scipy_signal.fftconvolve(audio, kernel, mode='same')
        elif audio.ndim == 2:
            # Process each channel
            out = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                out[ch] = scipy_signal.fftconvolve(audio[ch], kernel, mode='same')
            return out
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

    # ---------------------------------------------------------------
    # Dynamic EQ processing
    # ---------------------------------------------------------------

    def process_dynamic_eq(self, audio: np.ndarray, sample_rate: float = 48000.0) -> np.ndarray:
        """Process audio with dynamic EQ — per-band gain modulated by sidechain level.

        For each band with dynamic_enabled=True:
        1. Bandpass filter the input at the band's center frequency to get sidechain signal
        2. Measure RMS level via envelope follower
        3. Compute effective gain based on threshold/ratio
        4. Apply the band's EQ with the modulated gain

        Args:
            audio: Input audio (1D mono or 2D [channels, samples]).
            sample_rate: Audio sample rate in Hz.

        Returns:
            Processed audio.
        """
        if not HAS_SCIPY:
            raise RuntimeError("scipy is required for dynamic EQ processing")

        # Work in 2D [channels, samples]
        mono = audio.ndim == 1
        if mono:
            data = audio[np.newaxis, :]
        else:
            data = audio.copy()

        output = data.copy()

        for band in self.bands:
            if not band.enabled or not band.dynamic_enabled:
                continue
            if abs(band.gain) < 0.1:
                continue

            # Design bandpass filter for sidechain detection
            low = band.freq / (2.0 ** (band.width / 2.0))
            high = band.freq * (2.0 ** (band.width / 2.0))
            low = max(20.0, low)
            high = min(sample_rate / 2.0 - 1.0, high)

            if low >= high:
                continue

            sos_bp = scipy_signal.butter(2, [low, high], btype='band', fs=sample_rate, output='sos')

            for ch in range(data.shape[0]):
                # Sidechain: bandpass filtered signal
                sidechain = scipy_signal.sosfilt(sos_bp, data[ch])
                rms = np.sqrt(np.mean(sidechain ** 2) + 1e-20)

                # Update envelope and get level in dB
                level_db = band.update_envelope(rms, sample_rate)

                # Compute dynamic gain
                effective_gain = band.compute_dynamic_gain(level_db)

                if abs(effective_gain) < 0.1:
                    continue

                # Create a temporary band with the modulated gain and apply it
                b, a = self._biquad_coeffs(
                    EQBand(freq=band.freq, gain=effective_gain,
                           width=band.width, band_type=band.band_type),
                    sample_rate
                )
                output[ch] = scipy_signal.lfilter(b, a, output[ch])

        return output[0] if mono else output

    # ---------------------------------------------------------------
    # Analog mode: harmonic saturation
    # ---------------------------------------------------------------

    @staticmethod
    def analog_saturate(audio: np.ndarray) -> np.ndarray:
        """Apply subtle analog-style harmonic saturation.

        Uses tanh soft clipping: y = tanh(x * drive) / tanh(drive)
        where drive = 1.02 for very subtle harmonic coloring.

        Args:
            audio: Input audio array.

        Returns:
            Saturated audio array (same shape).
        """
        drive = 1.02
        tanh_drive = math.tanh(drive)
        return np.tanh(audio * drive) / tanh_drive

    def process_audio(self, audio: np.ndarray, sample_rate: float = 48000.0) -> np.ndarray:
        """Full EQ processing pipeline with all modes.

        Signal flow:
        1. If linear_phase: use FIR convolution for all static bands
           Else: use IIR biquads (existing behavior via lfilter)
        2. Apply dynamic EQ for bands with dynamic_enabled
        3. If analog_mode: apply harmonic saturation

        Args:
            audio: Input audio (1D mono or 2D [channels, samples]).
            sample_rate: Audio sample rate in Hz.

        Returns:
            Processed audio.
        """
        if not self.enabled or not HAS_SCIPY:
            return audio

        result = audio.copy()

        # Step 1: Static EQ (linear-phase FIR or IIR biquads)
        has_dynamic = any(b.dynamic_enabled and b.enabled for b in self.bands)

        if self.linear_phase:
            # Linear-phase mode: apply all bands as FIR convolution
            self._fir_dirty = True  # Recompute to include current band settings
            result = self.process_linear_phase(result, sample_rate)
        else:
            # IIR mode: cascade biquads for non-dynamic bands
            mono = result.ndim == 1
            if mono:
                data = result[np.newaxis, :]
            else:
                data = result

            for band in self.bands:
                if not band.enabled:
                    continue
                if band.dynamic_enabled:
                    continue  # Dynamic bands processed separately
                if band.band_type == "equalizer" and abs(band.gain) < 0.1:
                    continue

                b, a = self._biquad_coeffs(band, sample_rate)
                for ch in range(data.shape[0]):
                    data[ch] = scipy_signal.lfilter(b, a, data[ch])

            result = data[0] if mono else data

        # Step 2: Dynamic EQ processing
        if has_dynamic:
            result = self.process_dynamic_eq(result, sample_rate)

        # Step 3: Analog saturation
        if self.analog_mode:
            result = self.analog_saturate(result)

        return result

    def invalidate_fir(self):
        """Mark the FIR kernel as dirty so it is recomputed on next process call."""
        self._fir_dirty = True
        self._fir_kernel = None

    def get_frequency_response(self, sample_rate: float = 48000.0, n_points: int = 512):
        """Get the magnitude frequency response curve for visualization.

        Returns:
            (freqs, magnitudes_db): Arrays of frequencies (Hz) and magnitude (dB).
        """
        if not HAS_SCIPY:
            return np.array([]), np.array([])

        freqs, H = self._compute_combined_response(sample_rate, n_points)
        mag_db = 20.0 * np.log10(np.abs(H) + 1e-10)
        return freqs, mag_db

    def __repr__(self):
        active = sum(1 for b in self.bands if b.enabled and abs(b.gain) > 0.1)
        dynamic = sum(1 for b in self.bands if b.dynamic_enabled)
        mode = self.current_preset if self.preset_mode else "Manual"
        phase = "Linear" if self.linear_phase else "Minimum"
        analog = "+Analog" if self.analog_mode else ""
        return (f"Equalizer(mode={mode}, active_bands={active}, "
                f"dynamic={dynamic}, phase={phase}{analog})")
