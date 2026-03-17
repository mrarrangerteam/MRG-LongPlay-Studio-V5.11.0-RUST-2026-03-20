"""
LongPlay Studio V5.0 — Equalizer Module
Inspired by iZotope Ozone 12 EQ

Features:
- 8-band Parametric EQ (frequency, gain, Q/bandwidth)
- Genre Preset Mode: Auto-load EQ curve from genre profile
- Manual Mode: Adjust each band freely
- Band types: Peak, Low Shelf, High Shelf, Low Pass, High Pass
- Tone Presets: Quick tonal adjustments (Warm, Bright, Bass Boost, etc.)

Uses FFmpeg: equalizer, lowshelf, highshelf, highpass, lowpass filters
"""

from typing import List, Dict, Optional
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

    def to_dict(self) -> dict:
        return {
            "freq": self.freq,
            "gain": self.gain,
            "width": self.width,
            "type": self.band_type,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            freq=d.get("freq", 1000),
            gain=d.get("gain", 0.0),
            width=d.get("width", 1.0),
            band_type=d.get("type", "equalizer"),
            enabled=d.get("enabled", True),
        )


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
            "bands": [b.to_dict() for b in self.bands],
        }

    def load_settings_dict(self, d: dict):
        self.enabled = d.get("enabled", True)
        self.preset_mode = d.get("preset_mode", True)
        self.current_preset = d.get("current_preset", "Flat")
        bands_data = d.get("bands", [])
        self.bands = [EQBand.from_dict(b) for b in bands_data]
        while len(self.bands) < self.NUM_BANDS:
            self.bands.append(EQBand())

    def __repr__(self):
        active = sum(1 for b in self.bands if b.enabled and abs(b.gain) > 0.1)
        mode = self.current_preset if self.preset_mode else "Manual"
        return f"Equalizer(mode={mode}, active_bands={active})"
