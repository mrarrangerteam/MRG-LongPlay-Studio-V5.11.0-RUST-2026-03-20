"""
LongPlay Studio V5.9 — Tonal Balance Control

Inspired by iZotope Tonal Balance Control 2.
Analyzes audio spectrum and compares against genre-specific target curves
to visualize whether frequency balance is within acceptable range.

Features:
    - Genre-specific target curves (Bass, Mid, Presence, High)
    - Real-time spectral comparison
    - 4-band energy analysis (Low, Low-Mid, Hi-Mid, High)
    - Target zone visualization (min/max acceptable range)
    - Overall tonal balance score (0-100%)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Target curves: per-genre spectral balance targets (dB relative)
# Each defines 4 frequency bands with min/max acceptable energy ranges
# ---------------------------------------------------------------------------

# Band definitions (Hz)
BAND_RANGES = {
    "Low":       (20, 200),
    "Low-Mid":   (200, 2000),
    "Hi-Mid":    (2000, 8000),
    "High":      (8000, 20000),
}

BAND_NAMES = list(BAND_RANGES.keys())

# Target curves: (center_db, tolerance_db) per band
# center_db = ideal level relative to full-spectrum average
# tolerance_db = acceptable deviation (+/-)
TARGET_CURVES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Modern": {
        "Low":     (2.0, 4.0),
        "Low-Mid": (0.0, 3.0),
        "Hi-Mid":  (-1.0, 3.5),
        "High":    (-3.0, 4.0),
    },
    "Bass Heavy": {
        "Low":     (5.0, 3.5),
        "Low-Mid": (1.0, 3.0),
        "Hi-Mid":  (-2.0, 3.0),
        "High":    (-4.0, 4.0),
    },
    "Bright": {
        "Low":     (-1.0, 3.5),
        "Low-Mid": (0.0, 3.0),
        "Hi-Mid":  (1.5, 3.0),
        "High":    (1.0, 4.0),
    },
    "Warm": {
        "Low":     (3.0, 3.5),
        "Low-Mid": (1.0, 3.0),
        "Hi-Mid":  (-1.5, 3.0),
        "High":    (-4.0, 4.0),
    },
    "Orchestral": {
        "Low":     (1.0, 4.0),
        "Low-Mid": (2.0, 3.5),
        "Hi-Mid":  (0.0, 3.0),
        "High":    (-2.0, 4.0),
    },
    "EDM": {
        "Low":     (4.0, 3.0),
        "Low-Mid": (-1.0, 3.0),
        "Hi-Mid":  (1.0, 3.5),
        "High":    (-1.0, 4.0),
    },
    "Hip-Hop": {
        "Low":     (5.0, 3.0),
        "Low-Mid": (0.0, 3.0),
        "Hi-Mid":  (0.0, 3.5),
        "High":    (-2.0, 4.0),
    },
    "Rock": {
        "Low":     (1.0, 3.5),
        "Low-Mid": (1.0, 3.0),
        "Hi-Mid":  (1.0, 3.0),
        "High":    (-1.0, 4.0),
    },
    "Pop": {
        "Low":     (1.0, 3.5),
        "Low-Mid": (0.0, 3.0),
        "Hi-Mid":  (0.5, 3.0),
        "High":    (-1.0, 4.0),
    },
    "Jazz": {
        "Low":     (1.0, 4.0),
        "Low-Mid": (2.0, 3.0),
        "Hi-Mid":  (-1.0, 3.5),
        "High":    (-3.0, 4.0),
    },
    "Classical": {
        "Low":     (0.0, 4.0),
        "Low-Mid": (1.0, 3.5),
        "Hi-Mid":  (0.0, 3.0),
        "High":    (-2.0, 4.5),
    },
    "R&B / Soul": {
        "Low":     (3.0, 3.5),
        "Low-Mid": (1.0, 3.0),
        "Hi-Mid":  (0.0, 3.0),
        "High":    (-2.0, 4.0),
    },
}

# Map common genre names to target curve keys
GENRE_TO_CURVE = {
    "EDM": "EDM", "House": "EDM", "Techno": "EDM", "Dubstep": "Bass Heavy",
    "Future Bass": "EDM", "Drum & Bass": "EDM", "Trance": "Bright",
    "Hip-Hop": "Hip-Hop", "Trap": "Bass Heavy", "Drill": "Bass Heavy",
    "Classic Hip-Hop": "Hip-Hop", "Lo-Fi Hip Hop": "Warm",
    "Pop": "Pop", "Dance Pop": "Modern", "K-Pop": "Bright",
    "Latin Pop": "Modern", "Electropop": "Bright",
    "Rock": "Rock", "Indie Rock": "Rock", "Alternative Rock": "Rock",
    "Metal": "Rock", "Punk": "Rock", "Post-Rock": "Warm",
    "R&B": "R&B / Soul", "Soul": "R&B / Soul", "Neo-Soul": "Warm",
    "Jazz": "Jazz", "Blues": "Jazz", "Bossa Nova": "Jazz",
    "Country": "Warm", "Folk": "Warm", "Acoustic": "Warm",
    "Ambient": "Warm", "Classical": "Classical",
    "Orchestral": "Orchestral", "Film Score": "Orchestral",
    "Afrobeats": "Bass Heavy", "Reggaeton": "Bass Heavy",
}


# ---------------------------------------------------------------------------
# TonalBalanceAnalyzer
# ---------------------------------------------------------------------------
class TonalBalanceAnalyzer:
    """
    Analyzes audio spectral balance and compares to genre-specific targets.

    Usage:
        tba = TonalBalanceAnalyzer()
        tba.set_target("Modern")
        tba.analyze(audio_samples, sample_rate)
        score = tba.score          # 0-100%
        bands = tba.band_energies  # per-band dB values
    """

    FFT_SIZE = 4096

    def __init__(self) -> None:
        self._target_key: str = "Modern"
        self._band_energies: Optional[np.ndarray] = None  # 4 dB values
        self._score: float = 0.0
        self._band_scores: List[float] = [0.0, 0.0, 0.0, 0.0]
        self._in_range: List[bool] = [True, True, True, True]

    # -- Properties --

    @property
    def target_key(self) -> str:
        return self._target_key

    @property
    def target_curve(self) -> Dict[str, Tuple[float, float]]:
        return TARGET_CURVES.get(self._target_key, TARGET_CURVES["Modern"])

    @property
    def band_energies(self) -> Optional[np.ndarray]:
        return self._band_energies.copy() if self._band_energies is not None else None

    @property
    def score(self) -> float:
        return self._score

    @property
    def band_scores(self) -> List[float]:
        return list(self._band_scores)

    @property
    def in_range(self) -> List[bool]:
        return list(self._in_range)

    @property
    def available_targets(self) -> List[str]:
        return sorted(TARGET_CURVES.keys())

    # -- Configuration --

    def set_target(self, target_name: str) -> None:
        if target_name in TARGET_CURVES:
            self._target_key = target_name
            if self._band_energies is not None:
                self._compute_scores()

    def set_target_for_genre(self, genre: str) -> str:
        curve_key = GENRE_TO_CURVE.get(genre, "Modern")
        self.set_target(curve_key)
        return curve_key

    # -- Analysis --

    def analyze(self, samples: np.ndarray, sr: int = 44100) -> bool:
        """Analyze audio samples and compute per-band energy."""
        if len(samples) < self.FFT_SIZE:
            return False

        # Convert to mono if stereo
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)

        # Compute average spectrum
        hop = self.FFT_SIZE // 2
        window = np.hanning(self.FFT_SIZE)
        n_chunks = max(1, (len(samples) - self.FFT_SIZE) // hop)
        acc = np.zeros(self.FFT_SIZE // 2 + 1, dtype=np.float64)

        for i in range(min(n_chunks, 500)):  # cap at 500 chunks for speed
            start = i * hop
            chunk = samples[start:start + self.FFT_SIZE]
            if len(chunk) < self.FFT_SIZE:
                break
            windowed = chunk * window
            mag = np.abs(np.fft.rfft(windowed)) / self.FFT_SIZE
            acc += mag

        acc /= max(n_chunks, 1)
        acc = np.maximum(acc, 1e-12)
        spectrum_db = 20.0 * np.log10(acc)

        # Frequency axis
        freqs = np.fft.rfftfreq(self.FFT_SIZE, d=1.0 / sr)

        # Compute energy per band
        band_energies = np.zeros(4, dtype=np.float64)
        for i, band_name in enumerate(BAND_NAMES):
            lo, hi = BAND_RANGES[band_name]
            mask = (freqs >= lo) & (freqs <= hi)
            if np.any(mask):
                band_energies[i] = np.mean(spectrum_db[mask])
            else:
                band_energies[i] = -60.0

        # Normalize: express each band relative to the overall average
        overall_avg = np.mean(band_energies)
        self._band_energies = band_energies - overall_avg

        self._compute_scores()
        return True

    def _compute_scores(self) -> None:
        """Compute per-band and overall scores based on target curve."""
        if self._band_energies is None:
            return

        target = self.target_curve
        scores = []
        in_range = []

        for i, band_name in enumerate(BAND_NAMES):
            center, tolerance = target[band_name]
            actual = float(self._band_energies[i])
            deviation = abs(actual - center)

            # In range if within tolerance
            is_in = deviation <= tolerance
            in_range.append(is_in)

            # Score: 100% at center, 0% at 2x tolerance
            band_score = max(0.0, 1.0 - deviation / (tolerance * 2.0)) * 100.0
            scores.append(band_score)

        self._band_scores = scores
        self._in_range = in_range
        self._score = sum(scores) / len(scores) if scores else 0.0

    def get_target_range(self, band_index: int) -> Tuple[float, float]:
        """Return (min_db, max_db) for a band's acceptable range."""
        band_name = BAND_NAMES[band_index]
        center, tolerance = self.target_curve[band_name]
        return (center - tolerance, center + tolerance)

    def get_report(self) -> Dict:
        """Return analysis report as dict."""
        report = {
            "target": self._target_key,
            "overall_score": round(self._score, 1),
            "bands": {},
        }
        for i, band_name in enumerate(BAND_NAMES):
            energy = float(self._band_energies[i]) if self._band_energies is not None else 0.0
            lo, hi = self.get_target_range(i)
            report["bands"][band_name] = {
                "energy_db": round(energy, 1),
                "target_min_db": round(lo, 1),
                "target_max_db": round(hi, 1),
                "in_range": self._in_range[i],
                "score": round(self._band_scores[i], 1),
            }
        return report
