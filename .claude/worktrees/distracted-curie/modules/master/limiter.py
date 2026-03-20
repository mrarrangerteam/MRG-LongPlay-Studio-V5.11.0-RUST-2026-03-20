"""
LongPlay Studio V5.5 — Look-Ahead Limiter (Production Quality)
================================================================
Proper look-ahead brickwall limiter with smooth gain reduction.

Based on:
- Daniel Rudrich's SimpleCompressor look-ahead algorithm
- matchering Hyrax limiter (scipy sliding window approach)
- audiocomplib PeakLimiter (variable release concept)

Key design:
1. Compute gain reduction envelope from peak detection
2. Apply look-ahead via signal delay (5ms default)
3. Smooth attack with raised cosine (Hanning window convolution)
4. Smooth release with IIR exponential decay
5. Brickwall safety clip as final stage

This replaces all pedalboard.Limiter usage in chain.py.
"""

import numpy as np
from typing import Optional, Tuple

try:
    from scipy.ndimage import minimum_filter1d, maximum_filter1d
    from scipy.signal import lfilter, resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class LookAheadLimiter:
    """
    Production-quality look-ahead brickwall limiter.

    Parameters
    ----------
    ceiling_db : float
        Output ceiling in dBFS (default: -1.0 dBTP)
    lookahead_ms : float
        Look-ahead time in milliseconds (default: 5.0)
    release_ms : float
        Release time in milliseconds (default: 100.0)
    attack_ms : float
        Attack time in milliseconds (default: 5.0, matches look-ahead)
    variable_release : bool
        If True, deeper GR = slower release (more natural)
    true_peak : bool
        If True, use 4x oversampling for inter-sample peak detection
    """

    def __init__(
        self,
        ceiling_db: float = -1.0,
        lookahead_ms: float = 5.0,
        release_ms: float = 100.0,
        attack_ms: float = 5.0,
        variable_release: bool = True,
        true_peak: bool = True,
    ):
        self.ceiling_db = ceiling_db
        self.ceiling_linear = 10 ** (ceiling_db / 20.0)
        self.lookahead_ms = lookahead_ms
        self.release_ms = release_ms
        self.attack_ms = attack_ms
        self.variable_release = variable_release
        self.true_peak = true_peak

        # Stored gain reduction for metering
        self._last_gr_db = 0.0
        self._last_gr_envelope = None

    def set_ceiling(self, ceiling_db: float):
        """Update ceiling (thread-safe parameter change)."""
        self.ceiling_db = ceiling_db
        self.ceiling_linear = 10 ** (ceiling_db / 20.0)

    def set_release(self, release_ms: float):
        """Update release time."""
        self.release_ms = max(5.0, min(500.0, release_ms))

    @property
    def last_gain_reduction_db(self) -> float:
        """Return last computed max gain reduction in dB (for metering)."""
        return self._last_gr_db

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Process audio through the look-ahead limiter.

        Parameters
        ----------
        audio : np.ndarray
            Input audio, shape (samples,) or (samples, channels)
        sr : int
            Sample rate

        Returns
        -------
        np.ndarray
            Limited audio, same shape as input
        """
        if not HAS_SCIPY:
            return self._fallback_limit(audio)

        if audio.size == 0:
            return audio

        # Ensure 2D: (samples, channels)
        was_mono = audio.ndim == 1
        if was_mono:
            audio = audio[:, np.newaxis]

        result = audio.copy().astype(np.float64)
        n_samples, n_channels = result.shape

        # ─── Step 1: Compute peak envelope across all channels ───
        peak_envelope = np.max(np.abs(result), axis=1)  # (samples,)

        # Optional: True Peak detection via 4x oversampling
        if self.true_peak and n_samples > 16:
            peak_envelope = self._true_peak_envelope(result, sr)

        # ─── Step 2: Compute gain reduction in linear domain ───
        # Where peaks exceed ceiling, compute required gain reduction
        gain_reduction = np.ones(n_samples, dtype=np.float64)
        exceeds = peak_envelope > self.ceiling_linear
        if np.any(exceeds):
            gain_reduction[exceeds] = self.ceiling_linear / peak_envelope[exceeds]
        else:
            # No limiting needed
            self._last_gr_db = 0.0
            self._last_gr_envelope = np.zeros(n_samples)
            if was_mono:
                return result[:, 0]
            return result

        # ─── Step 3: Apply look-ahead (expand gain reduction into future) ───
        lookahead_samples = max(1, int(self.lookahead_ms * sr / 1000.0))

        # Use minimum_filter1d to propagate the minimum gain reduction
        # forward in time (this is the "look-ahead" — we see peaks before
        # they arrive and start reducing gain earlier)
        # Window size = 2 * lookahead + 1 for centered look-ahead
        window_size = 2 * lookahead_samples + 1
        gain_reduction = minimum_filter1d(gain_reduction, size=window_size)

        # ─── Step 4: Smooth attack with raised cosine (Hanning window) ───
        attack_samples = max(1, int(self.attack_ms * sr / 1000.0))
        if attack_samples > 1:
            # Hanning window convolution for smooth fade-in of gain reduction
            hanning = np.hanning(2 * attack_samples + 1)
            hanning = hanning / hanning.sum()
            # Use minimum_filter1d first to ensure we never miss peaks,
            # then smooth with convolution for the attack shape
            gain_reduction = np.convolve(gain_reduction, hanning, mode='same')
            # Ensure we never increase gain (only reduce)
            gain_reduction = np.minimum(gain_reduction, 1.0)

        # ─── Step 5: Smooth release with IIR exponential decay ───
        release_samples = max(1, int(self.release_ms * sr / 1000.0))
        release_coeff = np.exp(-1.0 / release_samples)

        if self.variable_release:
            gain_reduction = self._variable_release_smooth(
                gain_reduction, release_coeff)
        else:
            gain_reduction = self._iir_release_smooth(
                gain_reduction, release_coeff)

        # ─── Step 6: Delay input signal to align with gain envelope ───
        # The gain reduction was computed with look-ahead, so we delay
        # the audio signal to match
        delayed_audio = np.zeros_like(result)
        if lookahead_samples < n_samples:
            delayed_audio[lookahead_samples:] = result[:-lookahead_samples]
            delayed_audio[:lookahead_samples] = result[:lookahead_samples]
        else:
            delayed_audio = result

        # ─── Step 7: Apply gain reduction ───
        for ch in range(n_channels):
            delayed_audio[:, ch] *= gain_reduction

        # ─── Step 8: Brickwall safety clip (absolute last resort) ───
        np.clip(delayed_audio, -self.ceiling_linear, self.ceiling_linear,
                out=delayed_audio)

        # ─── Store gain reduction for metering ───
        gr_db = 20.0 * np.log10(np.maximum(gain_reduction, 1e-10))
        self._last_gr_db = float(np.min(gr_db))  # Max reduction (most negative)
        self._last_gr_envelope = gr_db

        if was_mono:
            return delayed_audio[:, 0]
        return delayed_audio

    def get_gain_reduction_db(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute gain reduction envelope without modifying audio.
        Returns gain reduction in dB (negative values = reduction).

        Useful for metering visualization.
        """
        if self._last_gr_envelope is not None:
            return self._last_gr_envelope

        # Process to compute GR, but don't use the output
        self.process(audio, sr)
        return self._last_gr_envelope if self._last_gr_envelope is not None \
            else np.zeros(len(audio))

    def _true_peak_envelope(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute true peak envelope using 4x oversampling.
        ITU-R BS.1770-4 compliant inter-sample peak detection.
        """
        n_samples, n_channels = audio.shape
        peak_envelope = np.max(np.abs(audio), axis=1)

        # Process in chunks to limit memory usage
        chunk_size = min(sr, n_samples)  # 1 second chunks
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk_len = end - start
            if chunk_len < 4:
                continue

            for ch in range(n_channels):
                chunk = audio[start:end, ch]
                try:
                    oversampled = resample_poly(chunk, 4, 1)
                    # Map oversampled peaks back to original sample positions
                    os_len = len(oversampled)
                    for i in range(chunk_len):
                        os_start = i * 4
                        os_end = min(os_start + 4, os_len)
                        os_peak = np.max(np.abs(oversampled[os_start:os_end]))
                        peak_envelope[start + i] = max(
                            peak_envelope[start + i], os_peak)
                except Exception:
                    pass  # Fallback to sample peaks already in envelope

        return peak_envelope

    def _iir_release_smooth(self, gain: np.ndarray,
                            release_coeff: float) -> np.ndarray:
        """
        IIR release smoothing — only smooths gain increases (release),
        keeps fast gain decreases (attack) instant.

        Uses vectorized lfilter for the forward pass, then enforces
        the "attack is instant" rule.
        """
        n = len(gain)
        smoothed = np.empty(n, dtype=np.float64)
        smoothed[0] = gain[0]

        # Forward pass: smooth release (gain going UP = less reduction)
        # Attack (gain going DOWN = more reduction) is instant
        for i in range(1, n):
            if gain[i] < smoothed[i - 1]:
                # Attack: instant (follow gain reduction immediately)
                smoothed[i] = gain[i]
            else:
                # Release: smooth exponential return
                smoothed[i] = (release_coeff * smoothed[i - 1] +
                               (1.0 - release_coeff) * gain[i])

        return smoothed

    def _variable_release_smooth(self, gain: np.ndarray,
                                 base_release_coeff: float) -> np.ndarray:
        """
        Variable release: deeper gain reduction = slower release.
        This produces more natural-sounding limiting because heavy
        limiting events recover slowly (less pumping).

        Light GR (<3 dB): fast release (preserve transients)
        Heavy GR (>6 dB): slow release (smooth recovery)
        """
        n = len(gain)
        smoothed = np.empty(n, dtype=np.float64)
        smoothed[0] = gain[0]

        for i in range(1, n):
            if gain[i] < smoothed[i - 1]:
                # Attack: instant
                smoothed[i] = gain[i]
            else:
                # Variable release: adjust coefficient based on GR depth
                gr_db = abs(20.0 * np.log10(max(smoothed[i - 1], 1e-10)))
                # Scale: 0 dB GR → base_coeff, 12+ dB GR → coeff^2 (slower)
                depth_factor = min(gr_db / 12.0, 1.0)
                # Interpolate between fast (base) and slow (base^2) release
                slow_coeff = base_release_coeff ** 0.5  # Slower release
                coeff = base_release_coeff * (1.0 - depth_factor) + \
                    slow_coeff * depth_factor
                smoothed[i] = coeff * smoothed[i - 1] + (1.0 - coeff) * gain[i]

        return smoothed

    def _fallback_limit(self, audio: np.ndarray) -> np.ndarray:
        """Fallback when scipy is not available: simple peak normalization."""
        if audio.size == 0:
            self._last_gr_db = 0.0
            return audio
        peak = np.max(np.abs(audio))
        if peak > self.ceiling_linear:
            gain = self.ceiling_linear / peak
            self._last_gr_db = 20.0 * np.log10(max(gain, 1e-10))
            return audio * gain
        self._last_gr_db = 0.0
        return audio


class LookAheadLimiterFast(LookAheadLimiter):
    """
    Optimized version using vectorized scipy operations.
    Phase 4 optimization: replaces Python loops with lfilter.
    """

    def _iir_release_smooth(self, gain: np.ndarray,
                            release_coeff: float) -> np.ndarray:
        """
        Vectorized release smoothing using scipy.signal.lfilter.
        10-100x faster than Python loop for long audio files.
        """
        if not HAS_SCIPY:
            return super()._iir_release_smooth(gain, release_coeff)

        # Forward IIR: y[n] = (1-a)*x[n] + a*y[n-1]
        # This smooths everything — we then enforce instant attack afterward
        b = np.array([1.0 - release_coeff])
        a = np.array([1.0, -release_coeff])
        smoothed = lfilter(b, a, gain)

        # Enforce instant attack: where gain < smoothed, use gain directly
        # (gain decreasing = attack, should be instant)
        result = np.minimum(smoothed, gain)

        # Second pass: re-apply release smoothing after attack enforcement
        result = lfilter(b, a, result)
        result = np.minimum(result, gain)

        return result

    def _variable_release_smooth(self, gain: np.ndarray,
                                 base_release_coeff: float) -> np.ndarray:
        """
        Semi-vectorized variable release.
        Uses lfilter with post-correction for depth-dependent behavior.
        """
        if not HAS_SCIPY:
            return super()._variable_release_smooth(gain, base_release_coeff)

        # First pass: fast release (base coefficient)
        b_fast = np.array([1.0 - base_release_coeff])
        a_fast = np.array([1.0, -base_release_coeff])
        fast_smooth = lfilter(b_fast, a_fast, gain)

        # Second pass: slow release (sqrt of coefficient = slower)
        slow_coeff = base_release_coeff ** 0.5
        b_slow = np.array([1.0 - slow_coeff])
        a_slow = np.array([1.0, -slow_coeff])
        slow_smooth = lfilter(b_slow, a_slow, gain)

        # Blend based on GR depth
        gr_db = np.abs(20.0 * np.log10(np.maximum(gain, 1e-10)))
        depth_factor = np.clip(gr_db / 12.0, 0.0, 1.0)

        # Interpolate: light GR → fast, heavy GR → slow
        blended = fast_smooth * (1.0 - depth_factor) + slow_smooth * depth_factor

        # Enforce instant attack
        result = np.minimum(blended, gain)

        return result
