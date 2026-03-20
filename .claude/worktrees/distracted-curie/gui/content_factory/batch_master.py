"""
Content Factory — Batch Mastering Lite

Masters multiple songs with ONLY:
  - Dynamics (compression) — even out dynamic range
  - Maximizer (loudness) — consistent LUFS across all songs
  - Imager (stereo width) — consistent stereo image

Does NOT apply EQ because different songs have different tones.
Applying one EQ curve to 20+ songs would destroy tonal balance.

Backend priority:
  1. Rust (PyO3) — longplay.batch_master() with rayon parallel processing
     → Handles unlimited songs, uses all CPU cores
  2. Python fallback — _RealAudioProcessor from chain.py (sequential)
     → Works without compiled Rust backend
"""

from __future__ import annotations

import os
import logging
import threading
from typing import List, Optional, Callable
from pathlib import Path

import numpy as np

from .models import SongEntry, BatchMasterConfig

logger = logging.getLogger(__name__)

# ── Rust backend (preferred — parallel processing via rayon) ──
try:
    from longplay import (
        batch_master as rust_batch_master,
        PyBatchMasterConfig,
        get_parallelism,
    )
    HAS_RUST_BACKEND = True
    logger.info(
        f"[BatchMaster] Rust backend available ({get_parallelism()} threads)"
    )
except ImportError:
    HAS_RUST_BACKEND = False

# Audio I/O
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

# Loudness normalization
try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False

# DSP modules — these are CONFIG HOLDERS, not processors
# Audio processing is done by _RealAudioProcessor static methods
try:
    from modules.master.dynamics import Dynamics
    from modules.master.imager import Imager
    from modules.master.maximizer import Maximizer
    from modules.master.limiter import LookAheadLimiterFast
    from modules.master.chain import _RealAudioProcessor
    HAS_DSP = True
except ImportError:
    HAS_DSP = False


class BatchMasterLite:
    """
    Masters a batch of songs using Dynamics + Loudness + Imager only.

    Automatically uses the Rust backend (longplay.batch_master) when available
    for parallel processing across all CPU cores. Falls back to sequential
    Python processing via _RealAudioProcessor from chain.py.

    Usage:
        bm = BatchMasterLite(config)
        bm.master_batch(songs, output_dir, progress_callback)
    """

    def __init__(self, config: Optional[BatchMasterConfig] = None):
        self._config = config or BatchMasterConfig()
        self._dynamics: Optional[Dynamics] = None
        self._imager: Optional[Imager] = None
        self._maximizer: Optional[Maximizer] = None
        self._cancelled = threading.Event()

        if HAS_DSP:
            self._dynamics = Dynamics()
            self._imager = Imager()
            self._maximizer = Maximizer()

    # ─── Public API ──────────────────────────────────────────────

    def cancel(self) -> None:
        """Signal cancellation of current batch."""
        self._cancelled.set()

    @staticmethod
    def has_rust_backend() -> bool:
        """Check if Rust parallel backend is available."""
        return HAS_RUST_BACKEND

    def master_batch(
        self,
        songs: List[SongEntry],
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """
        Master all songs in batch.

        Uses Rust parallel backend when available, otherwise falls back
        to sequential Python processing.

        Args:
            songs: List of SongEntry to master.
            output_dir: Directory to write mastered files.
            progress_callback: fn(current_index, total, status_msg)

        Returns:
            Number of successfully mastered songs.
        """
        if not self._config.enabled:
            return len(songs)

        os.makedirs(output_dir, exist_ok=True)
        self._cancelled.clear()

        # Try Rust parallel backend first
        if HAS_RUST_BACKEND:
            return self._master_batch_rust(songs, output_dir, progress_callback)

        # Fall back to Python sequential processing
        return self._master_batch_python(songs, output_dir, progress_callback)

    # ─── Rust Backend (parallel via rayon) ────────────────────────

    def _master_batch_rust(
        self,
        songs: List[SongEntry],
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """Master batch using Rust + rayon parallel processing."""
        total = len(songs)
        threads = get_parallelism()
        logger.info(
            f"[BatchMaster] Rust backend: {total} songs × {threads} threads"
        )

        # Build input/output path lists
        input_paths = []
        output_paths = []
        for song in songs:
            src = song.effective_path
            stem = Path(src).stem
            out = os.path.join(output_dir, f"{stem}_mastered.wav")
            input_paths.append(src)
            output_paths.append(out)

        # Build Rust config
        rust_config = PyBatchMasterConfig(
            dynamics_enabled=self._config.dynamics_enabled,
            imager_enabled=self._config.imager_enabled,
            maximizer_enabled=self._config.maximizer_enabled,
            target_lufs=self._config.target_lufs,
            true_peak_limit=self._config.true_peak_limit,
            dynamics_threshold=self._config.dynamics_threshold,
            dynamics_ratio=self._config.dynamics_ratio,
            imager_width=float(self._config.imager_width),
            maximizer_ceiling=self._config.maximizer_ceiling,
        )

        # Progress callback wrapper
        def rust_progress(completed: int, total_count: int, current_file: str):
            if progress_callback:
                filename = Path(current_file).stem
                progress_callback(
                    completed, total_count, f"Mastering: {filename}"
                )

        # Call Rust batch_master (releases GIL, processes in parallel)
        try:
            results = rust_batch_master(
                input_paths, output_paths, rust_config, rust_progress
            )
        except Exception as e:
            logger.error(f"[BatchMaster] Rust batch failed: {e}, falling back to Python")
            return self._master_batch_python(songs, output_dir, progress_callback)

        # Map results back to songs
        success_count = 0
        for song, result in zip(songs, results):
            if result.success:
                song.mastered_path = result.output_path
                success_count += 1
            else:
                logger.error(
                    f"[BatchMaster] Failed: {song.title} — {result.error}"
                )

        if progress_callback:
            progress_callback(total, total, "Mastering complete")

        logger.info(
            f"[BatchMaster] Rust: {success_count}/{total} mastered "
            f"({threads} threads)"
        )
        return success_count

    # ─── Python Backend (sequential fallback) ─────────────────────

    def _master_batch_python(
        self,
        songs: List[SongEntry],
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """Master batch using sequential Python DSP (fallback)."""
        success_count = 0
        total = len(songs)

        logger.info(f"[BatchMaster] Python backend: {total} songs (sequential)")

        for i, song in enumerate(songs):
            if self._cancelled.is_set():
                logger.info("[BatchMaster] Cancelled by user")
                break

            if progress_callback:
                progress_callback(i, total, f"Mastering: {song.title}")

            try:
                out_path = self._master_song_python(song, output_dir)
                if out_path:
                    song.mastered_path = out_path
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to master {song.title}: {e}")

        if progress_callback:
            progress_callback(total, total, "Mastering complete")

        return success_count

    def master_song(self, song: SongEntry, output_dir: str) -> Optional[str]:
        """
        Master a single song (Python backend).

        Returns output file path, or None on failure.
        """
        if not self._config.enabled:
            return song.file_path

        if HAS_RUST_BACKEND:
            return self._master_song_rust(song, output_dir)

        return self._master_song_python(song, output_dir)

    def _master_song_rust(self, song: SongEntry, output_dir: str) -> Optional[str]:
        """Master single song via Rust backend."""
        src = song.effective_path
        stem = Path(src).stem
        out = os.path.join(output_dir, f"{stem}_mastered.wav")

        rust_config = PyBatchMasterConfig(
            dynamics_enabled=self._config.dynamics_enabled,
            imager_enabled=self._config.imager_enabled,
            maximizer_enabled=self._config.maximizer_enabled,
            target_lufs=self._config.target_lufs,
            true_peak_limit=self._config.true_peak_limit,
            dynamics_threshold=self._config.dynamics_threshold,
            dynamics_ratio=self._config.dynamics_ratio,
            imager_width=float(self._config.imager_width),
            maximizer_ceiling=self._config.maximizer_ceiling,
        )

        try:
            results = rust_batch_master([src], [out], rust_config)
            if results and results[0].success:
                return results[0].output_path
            return None
        except Exception as e:
            logger.warning(f"Rust master_song failed: {e}, trying Python")
            return self._master_song_python(song, output_dir)

    def _master_song_python(self, song: SongEntry, output_dir: str) -> Optional[str]:
        """Master a single song using Python DSP."""
        if not HAS_SOUNDFILE:
            logger.warning("soundfile not available — skipping mastering")
            return song.file_path

        # Read audio
        try:
            data, sr = sf.read(song.file_path)
        except Exception as e:
            logger.error(f"Cannot read {song.file_path}: {e}")
            return None

        # Ensure stereo
        if data.ndim == 1:
            data = np.column_stack([data, data])

        logger.info(f"[BatchMaster] Processing: {song.title} "
                     f"({len(data)/sr:.1f}s @ {sr}Hz)")

        # ── Processing pipeline (NO EQ) ──

        # 1. Dynamics (compression)
        if self._config.dynamics_enabled and self._dynamics:
            data = self._apply_dynamics(data, sr)

        # 2. Imager (stereo width)
        if self._config.imager_enabled and self._imager:
            data = self._apply_imager(data, sr)

        # 3. Maximizer (gain push + ceiling)
        if self._config.maximizer_enabled and self._maximizer:
            data = self._apply_maximizer(data, sr)

        # 4. Loudness normalization (LUFS target)
        data = self._normalize_loudness(data, sr)

        # 5. True peak limiting
        data = self._true_peak_limit(data, sr)

        # 6. Final clip
        data = np.clip(data, -1.0, 1.0).astype(np.float32)

        # Write output
        stem = Path(song.file_path).stem
        out_path = os.path.join(output_dir, f"{stem}_mastered.wav")

        try:
            sf.write(out_path, data, sr, subtype='PCM_24')
            logger.info(f"[BatchMaster] Written: {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Cannot write {out_path}: {e}")
            return None

    # ─── DSP Steps (using _RealAudioProcessor from chain.py) ─────

    def _apply_dynamics(self, data: np.ndarray, sr: int) -> np.ndarray:
        """Apply gentle compression via _RealAudioProcessor.process_dynamics()."""
        if not self._dynamics or not HAS_DSP:
            return data

        try:
            # Configure Dynamics config object (single-band for batch)
            self._dynamics.enabled = True
            self._dynamics.multiband = False
            sb = self._dynamics.single_band
            sb.threshold = self._config.dynamics_threshold  # -18 dB
            sb.ratio = self._config.dynamics_ratio          # 2.5
            sb.attack = 10.0                                # ms
            sb.release = 100.0                              # ms
            sb.knee = 6.0                                   # dB (soft knee)

            # Use _RealAudioProcessor static method (NOT .process())
            return _RealAudioProcessor.process_dynamics(
                data, sr, self._dynamics, intensity=1.0
            )
        except Exception as e:
            logger.warning(f"Dynamics processing failed: {e}")
            return data

    def _apply_imager(self, data: np.ndarray, sr: int) -> np.ndarray:
        """Apply stereo width via _RealAudioProcessor.process_imager()."""
        if not self._imager or not HAS_DSP or data.ndim < 2:
            return data

        try:
            # Configure Imager config object
            # Imager.width uses 0-200 integer scale (100 = original)
            self._imager.enabled = True
            self._imager.width = self._config.imager_width  # 0-200 int

            # Use _RealAudioProcessor static method
            return _RealAudioProcessor.process_imager(
                data, sr, self._imager, intensity=1.0
            )
        except Exception as e:
            logger.warning(f"Imager processing failed: {e}")
            return data

    def _apply_maximizer(self, data: np.ndarray, sr: int) -> np.ndarray:
        """Apply gain maximization via _RealAudioProcessor.process_maximizer()."""
        if not self._maximizer or not HAS_DSP:
            return data

        try:
            # Configure Maximizer config object
            self._maximizer.enabled = True
            self._maximizer.ceiling = self._config.maximizer_ceiling  # -0.3 dB

            # Use _RealAudioProcessor static method
            return _RealAudioProcessor.process_maximizer(
                data, sr, self._maximizer, intensity=1.0
            )
        except Exception as e:
            logger.warning(f"Maximizer processing failed: {e}")
            return data

    def _normalize_loudness(self, data: np.ndarray, sr: int) -> np.ndarray:
        """Normalize to target LUFS using pyloudnorm."""
        if not HAS_PYLOUDNORM:
            return data

        try:
            meter = pyln.Meter(sr)
            current_lufs = meter.integrated_loudness(data)

            if np.isinf(current_lufs) or np.isnan(current_lufs):
                return data

            target = self._config.target_lufs
            return pyln.normalize.loudness(data, current_lufs, target)
        except Exception as e:
            logger.warning(f"Loudness normalization failed: {e}")
            return data

    def _true_peak_limit(self, data: np.ndarray, sr: int) -> np.ndarray:
        """Apply true peak brickwall limiting."""
        if not HAS_DSP:
            # Fallback: simple hard clip at ceiling
            ceiling_linear = 10 ** (self._config.true_peak_limit / 20.0)
            peak = np.max(np.abs(data))
            if peak > ceiling_linear:
                data = data * (ceiling_linear / peak)
            return data

        try:
            limiter = LookAheadLimiterFast(
                ceiling_db=self._config.true_peak_limit,
                lookahead_ms=1.5,
                release_ms=50.0,
            )
            return limiter.process(data, sr)
        except Exception as e:
            logger.warning(f"True peak limiting failed: {e}")
            ceiling_linear = 10 ** (self._config.true_peak_limit / 20.0)
            peak = np.max(np.abs(data))
            if peak > ceiling_linear:
                data = data * (ceiling_linear / peak)
            return data
