"""
Playlist-Aware Spectral Compressor for LongPlay Studio V5
==========================================================

Analyzes the spectral average of an entire playlist (20+ tracks),
then per-track compresses only frequency bands that exceed the
playlist average. This ensures tonal consistency across a full
LongPlay album master without flattening individual track character.

Signal Flow:
    1. analyze_playlist() — FFT all tracks, compute per-band average energy
    2. process_track()    — per-band dynamic compression against playlist average
    3. get_report()       — text summary of adjustments per track

Frequency Bands (8):
    Sub (20-60 Hz), Bass (60-250 Hz), Low-Mid (250-500 Hz),
    Mid (500-2k Hz), Hi-Mid (2k-4k Hz), Presence (4k-6k Hz),
    Brilliance (6k-12k Hz), Air (12k-20k Hz)

Dependencies: numpy, scipy, soundfile
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:
    from scipy.fft import rfft, rfftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BAND_EDGES: list[tuple[str, float, float]] = [
    ("Sub",        20.0,    60.0),
    ("Bass",       60.0,   250.0),
    ("Low-Mid",   250.0,   500.0),
    ("Mid",       500.0,  2000.0),
    ("Hi-Mid",   2000.0,  4000.0),
    ("Presence",  4000.0,  6000.0),
    ("Brilliance",6000.0, 12000.0),
    ("Air",      12000.0, 20000.0),
]

NUM_BANDS = len(BAND_EDGES)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BandConfig:
    """Per-band compressor configuration."""
    ratio: float = 1.5
    attack_ms: float = 10.0
    release_ms: float = 50.0
    makeup_db: float = 0.0


@dataclass
class TrackReport:
    """Per-track compression report."""
    filename: str = ""
    band_gains_db: dict[str, float] = field(default_factory=dict)
    band_reductions_db: dict[str, float] = field(default_factory=dict)
    peak_before: float = 0.0
    peak_after: float = 0.0


# ---------------------------------------------------------------------------
# PlaylistSpectralCompressor
# ---------------------------------------------------------------------------

class PlaylistSpectralCompressor:
    """Playlist-aware spectral compressor for consistent album mastering.

    Parameters
    ----------
    ratio : float
        Default compression ratio for all bands (1.5:1).
    attack_ms : float
        Envelope attack time in milliseconds.
    release_ms : float
        Envelope release time in milliseconds.
    makeup_db : float
        Makeup gain applied after compression (dB).
    fft_size : int
        FFT window size for spectral analysis.
    """

    def __init__(
        self,
        ratio: float = 1.5,
        attack_ms: float = 10.0,
        release_ms: float = 50.0,
        makeup_db: float = 0.0,
        fft_size: int = 4096,
    ) -> None:
        if not HAS_SCIPY:
            raise ImportError(
                "scipy is required for PlaylistSpectralCompressor. "
                "Install with: pip install scipy"
            )

        self.fft_size = fft_size

        # Per-band configs (can be overridden individually)
        self.band_configs: list[BandConfig] = [
            BandConfig(
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
                makeup_db=makeup_db,
            )
            for _ in range(NUM_BANDS)
        ]

        # State
        self._analysis: dict[str, Any] | None = None
        self._reports: list[TrackReport] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_band_ratio(self, band_index: int, ratio: float) -> None:
        """Override ratio for a specific band (0-7)."""
        self.band_configs[band_index].ratio = max(1.0, ratio)

    def set_band_makeup(self, band_index: int, makeup_db: float) -> None:
        """Override makeup gain for a specific band (0-7)."""
        self.band_configs[band_index].makeup_db = makeup_db

    # ------------------------------------------------------------------
    # Spectral helpers
    # ------------------------------------------------------------------

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert to mono by averaging channels if needed."""
        if audio.ndim == 1:
            return audio
        return np.mean(audio, axis=1)

    def _compute_band_energies(
        self, audio: np.ndarray, sr: int
    ) -> np.ndarray:
        """Compute RMS energy per frequency band for a mono signal.

        Returns
        -------
        np.ndarray
            Array of shape (NUM_BANDS,) with RMS energy per band.
        """
        mono = self._to_mono(audio)
        n = len(mono)

        # Windowed FFT (overlap-add style averaging)
        hop = self.fft_size // 2
        num_frames = max(1, (n - self.fft_size) // hop + 1)
        window = np.hanning(self.fft_size)
        freqs = rfftfreq(self.fft_size, d=1.0 / sr)

        # Pre-compute band masks
        band_masks: list[np.ndarray] = []
        for _, lo, hi in BAND_EDGES:
            mask = (freqs >= lo) & (freqs < hi)
            band_masks.append(mask)

        accumulator = np.zeros(NUM_BANDS, dtype=np.float64)

        for i in range(num_frames):
            start = i * hop
            segment = mono[start : start + self.fft_size]
            if len(segment) < self.fft_size:
                segment = np.pad(segment, (0, self.fft_size - len(segment)))
            spectrum = np.abs(rfft(segment * window))

            for b, mask in enumerate(band_masks):
                if np.any(mask):
                    accumulator[b] += np.mean(spectrum[mask] ** 2)

        # RMS per band
        energies = np.sqrt(accumulator / max(num_frames, 1))
        return energies

    # ------------------------------------------------------------------
    # Playlist analysis
    # ------------------------------------------------------------------

    def analyze_playlist(self, file_paths: list[str]) -> dict[str, Any]:
        """Load all tracks and compute playlist-average spectral envelope.

        Parameters
        ----------
        file_paths : list[str]
            Paths to audio files (WAV, FLAC, AIFF, etc.).

        Returns
        -------
        dict
            Analysis dictionary with keys:
            - ``band_names``: list of band name strings
            - ``band_edges``: list of (name, lo_hz, hi_hz) tuples
            - ``avg_energies``: np.ndarray of average RMS per band
            - ``per_track_energies``: dict mapping filename to band energies
            - ``num_tracks``: int
            - ``sample_rate``: int (of the first track loaded)
        """
        if not HAS_SOUNDFILE:
            raise ImportError(
                "soundfile is required for loading audio files. "
                "Install with: pip install soundfile"
            )

        all_energies: list[np.ndarray] = []
        per_track: dict[str, np.ndarray] = {}
        sample_rate: int = 44100

        for path in file_paths:
            try:
                audio, sr = sf.read(path, dtype="float32")
                sample_rate = sr
                energies = self._compute_band_energies(audio, sr)
                all_energies.append(energies)
                per_track[path] = energies
                logger.info("Analyzed: %s", path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path, exc)

        if not all_energies:
            raise ValueError("No tracks could be loaded for analysis.")

        avg_energies = np.mean(np.stack(all_energies), axis=0)

        self._analysis = {
            "band_names": [name for name, _, _ in BAND_EDGES],
            "band_edges": list(BAND_EDGES),
            "avg_energies": avg_energies,
            "per_track_energies": per_track,
            "num_tracks": len(all_energies),
            "sample_rate": sample_rate,
        }
        self._reports.clear()
        return self._analysis

    def analyze_playlist_from_arrays(
        self,
        tracks: list[tuple[str, np.ndarray, int]],
    ) -> dict[str, Any]:
        """Analyze from in-memory arrays instead of file paths.

        Parameters
        ----------
        tracks : list of (name, audio_array, sample_rate)
        """
        all_energies: list[np.ndarray] = []
        per_track: dict[str, np.ndarray] = {}
        sample_rate: int = 44100

        for name, audio, sr in tracks:
            sample_rate = sr
            energies = self._compute_band_energies(audio, sr)
            all_energies.append(energies)
            per_track[name] = energies

        if not all_energies:
            raise ValueError("No tracks provided for analysis.")

        avg_energies = np.mean(np.stack(all_energies), axis=0)

        self._analysis = {
            "band_names": [name for name, _, _ in BAND_EDGES],
            "band_edges": list(BAND_EDGES),
            "avg_energies": avg_energies,
            "per_track_energies": per_track,
            "num_tracks": len(all_energies),
            "sample_rate": sample_rate,
        }
        self._reports.clear()
        return self._analysis

    # ------------------------------------------------------------------
    # Per-track processing
    # ------------------------------------------------------------------

    def process_track(
        self,
        audio: np.ndarray,
        sr: int,
        analysis: dict[str, Any],
        track_name: str = "untitled",
    ) -> np.ndarray:
        """Spectral-compress one track against the playlist average.

        Only bands whose energy *exceeds* the playlist average are
        compressed. Bands at or below average pass through unchanged.

        Parameters
        ----------
        audio : np.ndarray
            Audio samples, shape (N,) mono or (N, C) multi-channel.
        sr : int
            Sample rate in Hz.
        analysis : dict
            Analysis dict returned by ``analyze_playlist()``.
        track_name : str
            Label for the report.

        Returns
        -------
        np.ndarray
            Processed audio (same shape as input).
        """
        avg_energies: np.ndarray = analysis["avg_energies"]
        freqs = rfftfreq(self.fft_size, d=1.0 / sr)

        # Build per-band masks
        band_masks: list[np.ndarray] = []
        for _, lo, hi in BAND_EDGES:
            band_masks.append((freqs >= lo) & (freqs < hi))

        # Determine original shape
        original_shape = audio.shape
        is_stereo = audio.ndim == 2
        channels: list[np.ndarray] = (
            [audio[:, c] for c in range(audio.shape[1])]
            if is_stereo
            else [audio.copy()]
        )

        report = TrackReport(
            filename=track_name,
            peak_before=float(np.max(np.abs(audio))),
        )

        # Track-level band energies (mono)
        track_energies = self._compute_band_energies(audio, sr)

        processed_channels: list[np.ndarray] = []

        for ch_data in channels:
            n = len(ch_data)
            hop = self.fft_size // 2
            window = np.hanning(self.fft_size)
            num_frames = max(1, (n - self.fft_size) // hop + 1)

            output = np.zeros(n, dtype=np.float64)
            window_sum = np.zeros(n, dtype=np.float64)

            for i in range(num_frames):
                start = i * hop
                end = start + self.fft_size
                segment = ch_data[start:end]
                if len(segment) < self.fft_size:
                    segment = np.pad(segment, (0, self.fft_size - len(segment)))

                spectrum = rfft(segment * window)
                magnitude = np.abs(spectrum)
                phase = np.angle(spectrum)

                # Per-band compression
                gain = np.ones_like(magnitude)

                for b, mask in enumerate(band_masks):
                    if not np.any(mask):
                        continue

                    cfg = self.band_configs[b]
                    threshold = avg_energies[b]
                    band_level = track_energies[b]

                    if band_level > threshold and threshold > 1e-12:
                        # Excess in dB
                        excess_db = 20.0 * np.log10(
                            band_level / threshold + 1e-12
                        )
                        # Compressed excess
                        compressed_db = excess_db / cfg.ratio
                        # Gain reduction in dB
                        reduction_db = excess_db - compressed_db
                        # Apply makeup
                        net_db = -reduction_db + cfg.makeup_db
                        band_gain = 10.0 ** (net_db / 20.0)
                        gain[mask] = band_gain

                        report.band_reductions_db[BAND_EDGES[b][0]] = (
                            round(reduction_db, 2)
                        )
                    else:
                        # Below threshold — apply makeup only
                        if cfg.makeup_db != 0.0:
                            band_gain = 10.0 ** (cfg.makeup_db / 20.0)
                            gain[mask] = band_gain
                        report.band_reductions_db[BAND_EDGES[b][0]] = 0.0

                # Reconstruct
                from scipy.fft import irfft

                compressed_spectrum = magnitude * gain * np.exp(1j * phase)
                frame_out = np.real(irfft(compressed_spectrum, n=self.fft_size))

                # Overlap-add
                out_end = min(start + self.fft_size, n)
                length = out_end - start
                output[start:out_end] += (frame_out[:length] * window[:length])
                window_sum[start:out_end] += window[:length] ** 2

            # Normalize by window sum
            nonzero = window_sum > 1e-8
            output[nonzero] /= window_sum[nonzero]

            processed_channels.append(output.astype(np.float32))

        # Reassemble
        if is_stereo:
            result = np.stack(processed_channels, axis=1)
        else:
            result = processed_channels[0]

        report.peak_after = float(np.max(np.abs(result)))

        # Store gain report per band
        for b in range(NUM_BANDS):
            name = BAND_EDGES[b][0]
            reduction = report.band_reductions_db.get(name, 0.0)
            makeup = self.band_configs[b].makeup_db
            report.band_gains_db[name] = round(-reduction + makeup, 2)

        self._reports.append(report)
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> str:
        """Generate a text summary of compression adjustments per track.

        Returns
        -------
        str
            Human-readable multi-line report.
        """
        if not self._reports:
            return "No tracks processed yet."

        lines: list[str] = [
            "=" * 64,
            "  Playlist Spectral Compressor — Summary Report",
            "=" * 64,
            "",
        ]

        if self._analysis:
            lines.append(
                f"  Playlist: {self._analysis['num_tracks']} tracks analyzed"
            )
            lines.append(
                f"  Sample rate: {self._analysis['sample_rate']} Hz"
            )
            lines.append("")

            lines.append("  Playlist Average Energies (RMS):")
            for b, (name, lo, hi) in enumerate(BAND_EDGES):
                e = self._analysis["avg_energies"][b]
                db = 20.0 * np.log10(e + 1e-12)
                lines.append(
                    f"    {name:12s} ({lo:5.0f}-{hi:5.0f} Hz): "
                    f"{db:+6.1f} dB"
                )
            lines.append("")

        for report in self._reports:
            lines.append(f"  Track: {report.filename}")
            lines.append(
                f"    Peak: {report.peak_before:.4f} -> "
                f"{report.peak_after:.4f}"
            )
            for name, lo, hi in BAND_EDGES:
                red = report.band_reductions_db.get(name, 0.0)
                gain = report.band_gains_db.get(name, 0.0)
                if abs(red) > 0.01:
                    lines.append(
                        f"    {name:12s}: -{red:.1f} dB reduction, "
                        f"net {gain:+.1f} dB"
                    )
                else:
                    lines.append(f"    {name:12s}: no compression needed")
            lines.append("")

        lines.append("=" * 64)
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear analysis and reports."""
        self._analysis = None
        self._reports.clear()


# ---------------------------------------------------------------------------
# CLI test with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if not HAS_SCIPY:
        print("ERROR: scipy is required. pip install scipy")
        raise SystemExit(1)

    print("Playlist Spectral Compressor — Synthetic Test")
    print("-" * 50)

    sr = 44100
    duration = 2.0  # seconds
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    # Synthesize 5 test tracks with different spectral profiles
    def make_track(freqs_and_amps: list[tuple[float, float]]) -> np.ndarray:
        """Generate a test signal from frequency/amplitude pairs."""
        sig = np.zeros(n_samples, dtype=np.float32)
        for freq, amp in freqs_and_amps:
            sig += amp * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
        # Normalize to -6 dBFS
        peak = np.max(np.abs(sig))
        if peak > 1e-6:
            sig = sig / peak * 0.5
        return sig

    tracks = [
        ("Track01_balanced", make_track(
            [(40, 0.3), (150, 0.5), (1000, 0.7), (5000, 0.4), (15000, 0.2)]
        )),
        ("Track02_bass_heavy", make_track(
            [(40, 0.9), (100, 0.8), (200, 0.6), (1000, 0.3), (5000, 0.1)]
        )),
        ("Track03_bright", make_track(
            [(40, 0.1), (150, 0.2), (1000, 0.3), (5000, 0.8), (15000, 0.9)]
        )),
        ("Track04_mid_focused", make_track(
            [(40, 0.2), (150, 0.3), (800, 0.9), (3000, 0.7), (10000, 0.2)]
        )),
        ("Track05_scooped", make_track(
            [(40, 0.7), (150, 0.6), (1000, 0.1), (5000, 0.6), (15000, 0.5)]
        )),
    ]

    # Create compressor
    comp = PlaylistSpectralCompressor(
        ratio=1.5,
        attack_ms=10.0,
        release_ms=50.0,
        makeup_db=0.0,
        fft_size=4096,
    )

    # Analyze playlist from arrays
    track_tuples = [(name, audio, sr) for name, audio in tracks]
    analysis = comp.analyze_playlist_from_arrays(track_tuples)

    print(f"\nAnalyzed {analysis['num_tracks']} tracks")
    print(f"Bands: {analysis['band_names']}")
    print()

    # Process each track
    for name, audio in tracks:
        processed = comp.process_track(audio, sr, analysis, track_name=name)
        assert processed.shape == audio.shape, (
            f"Shape mismatch: {processed.shape} != {audio.shape}"
        )

    # Print report
    print(comp.get_report())
    print("\nAll tests passed.")
