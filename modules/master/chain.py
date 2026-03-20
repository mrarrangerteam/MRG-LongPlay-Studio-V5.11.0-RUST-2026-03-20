"""
LongPlay Studio V5.0 — Master Chain Orchestrator (REAL AUDIO PROCESSING)
==========================================================================
Drop-in replacement for the original chain.py.
Instead of FFmpeg filter chains, this version uses pedalboard + scipy
for professional-grade audio processing (Ozone 12 quality).

Signal Flow:
  Input → EQ → Dynamics → Imager → Maximizer → Loudness Normalize → Output

ALL original API preserved:
  - MasterChain class with same __init__ signature
  - preview(), render(), ai_recommend(), apply_recommendation()
  - save_settings(), load_settings(), reset_all()
  - build_filter_chain(), build_ffmpeg_command() (kept for compatibility)
  - All module attributes: equalizer, dynamics, imager, maximizer
"""

import subprocess
import logging
import os
import json
import time
import shutil
import threading
import numpy as np
from typing import Optional, Dict, List, Callable, Tuple

# Original module imports (unchanged)
from .resonance_suppressor import ResonanceSuppressor
from .equalizer import Equalizer
from .dynamics import Dynamics
from .imager import Imager
from .maximizer import Maximizer
from .loudness import LoudnessMeter, LoudnessAnalysis
from .ai_assist import AIAssist, MasterRecommendation
from .genre_profiles import PLATFORM_TARGETS, IRC_MODES, get_irc_mode
from .limiter import LookAheadLimiter, LookAheadLimiterFast
from .soothe import SootheProcessor

# ─── Audio processing dependencies ───
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy.signal import butter, sosfiltfilt, lfilter, resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pedalboard
    from pedalboard import (
        Pedalboard, Compressor, Limiter, Gain,
        HighpassFilter, LowpassFilter,
        HighShelfFilter, LowShelfFilter, PeakFilter,
    )
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False


# ═══════════════════════════════════════════════════════════════════
#  REAL AUDIO PROCESSING ENGINE (embedded — no external dependency)
# ═══════════════════════════════════════════════════════════════════

class _CrossoverFilter:
    """Linkwitz-Riley crossover filter for multiband splitting."""

    @staticmethod
    def split_2band(data: np.ndarray, sr: int, freq: float) -> Tuple[np.ndarray, np.ndarray]:
        nyq = sr / 2.0
        freq_norm = min(freq / nyq, 0.99)
        if freq_norm <= 0.001:
            return np.zeros_like(data), data
        sos = butter(4, freq_norm, btype='low', output='sos')
        low = sosfiltfilt(sos, data, axis=0)
        high = data - low
        return low, high

    @staticmethod
    def split_3band(data, sr, f1, f2):
        low, rest = _CrossoverFilter.split_2band(data, sr, f1)
        mid, high = _CrossoverFilter.split_2band(rest, sr, f2)
        return low, mid, high

    @staticmethod
    def split_4band(data, sr, f1, f2, f3):
        low, rest = _CrossoverFilter.split_2band(data, sr, f1)
        mid_low, rest2 = _CrossoverFilter.split_2band(rest, sr, f2)
        mid_high, high = _CrossoverFilter.split_2band(rest2, sr, f3)
        return low, mid_low, mid_high, high


def _envelope_follower(signal: np.ndarray, sr: int,
                       attack_ms: float, release_ms: float) -> np.ndarray:
    """Vectorized envelope follower with separate attack/release coefficients."""
    # Use proper separate coefficients for attack and release
    attack_coeff = np.exp(-1.0 / (attack_ms * sr / 1000.0))
    release_coeff = np.exp(-1.0 / (release_ms * sr / 1000.0))
    
    abs_signal = np.abs(signal).astype(np.float64)
    
    # Apply envelope follower with adaptive attack/release
    envelope = np.zeros_like(abs_signal)
    if len(abs_signal) > 0:
        envelope[0] = abs_signal[0]
        for i in range(1, len(abs_signal)):
            # Use attack_coeff when signal rises, release_coeff when it falls
            if abs_signal[i] > envelope[i-1]:
                envelope[i] = attack_coeff * envelope[i-1] + (1.0 - attack_coeff) * abs_signal[i]
            else:
                envelope[i] = release_coeff * envelope[i-1] + (1.0 - release_coeff) * abs_signal[i]
    
    return np.maximum(envelope, 1e-10)


class _RealAudioProcessor:
    """
    Professional audio processing — reads settings from LongPlay modules
    and applies real DSP using pedalboard + scipy.
    """

    # ─── EQ Processing ───
    @staticmethod
    def process_eq(data: np.ndarray, sr: int, equalizer: 'Equalizer',
                   intensity: float) -> np.ndarray:
        """Read EQ settings from Equalizer module and apply real EQ."""
        if not equalizer.enabled:
            return data

        if HAS_PEDALBOARD:
            return _RealAudioProcessor._eq_pedalboard(data, sr, equalizer, intensity)
        elif HAS_SCIPY:
            return _RealAudioProcessor._eq_scipy(data, sr, equalizer, intensity)
        return data

    @staticmethod
    def _eq_pedalboard(data, sr, equalizer, intensity):
        plugins = []
        for band in equalizer.bands:
            # EQBand objects have .enabled, .freq, .gain, .width, .band_type
            if not getattr(band, 'enabled', True):
                continue
            freq = getattr(band, 'freq', 1000)
            gain = getattr(band, 'gain', 0.0) * intensity
            q = getattr(band, 'width', 1.0)  # EQBand uses 'width' for Q
            btype = getattr(band, 'band_type', 'equalizer')  # EQBand uses 'band_type'

            # Map EQBand type names to pedalboard filter names
            # EQBand types: equalizer, lowshelf, highshelf, highpass, lowpass
            if btype == 'equalizer' and abs(gain) < 0.05:
                continue

            if btype == "highpass":
                plugins.append(HighpassFilter(cutoff_frequency_hz=freq))
            elif btype == "lowpass":
                plugins.append(LowpassFilter(cutoff_frequency_hz=freq))
            elif btype == "lowshelf":
                if abs(gain) < 0.05:
                    continue
                plugins.append(LowShelfFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))
            elif btype == "highshelf":
                if abs(gain) < 0.05:
                    continue
                plugins.append(HighShelfFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))
            elif btype == "equalizer":
                plugins.append(PeakFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))

        if not plugins:
            return data

        board = Pedalboard(plugins)
        audio = data.T.astype(np.float32) if data.ndim > 1 else data.reshape(1, -1).astype(np.float32)
        processed = board(audio, sr)
        return processed.T

    @staticmethod
    def _eq_scipy(data, sr, equalizer, intensity):
        result = data.copy()
        nyq = sr / 2.0

        for band in equalizer.bands:
            # EQBand objects have .enabled, .freq, .gain, .width, .band_type
            if not getattr(band, 'enabled', True):
                continue
            freq = getattr(band, 'freq', 1000)
            gain = getattr(band, 'gain', 0.0) * intensity
            q = getattr(band, 'width', 1.0)  # EQBand uses 'width' for Q
            btype = getattr(band, 'band_type', 'equalizer')  # EQBand uses 'band_type'
            freq_norm = min(freq / nyq, 0.99)

            if btype == "highpass":
                sos = butter(2, freq_norm, btype='high', output='sos')
            elif btype == "lowpass":
                sos = butter(2, freq_norm, btype='low', output='sos')
            elif btype in ["equalizer", "lowshelf", "highshelf"]:
                if abs(gain) < 0.05:
                    continue
                A = 10 ** (gain / 40.0)
                w0 = 2 * np.pi * freq / sr
                alpha = np.sin(w0) / (2 * q)

                if btype == "equalizer":  # Peak EQ
                    b0 = 1 + alpha * A
                    b1 = -2 * np.cos(w0)
                    b2 = 1 - alpha * A
                    a0 = 1 + alpha / A
                    a1 = -2 * np.cos(w0)
                    a2 = 1 - alpha / A
                elif btype == "lowshelf":
                    b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
                    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
                    b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
                    a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
                    a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
                    a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
                else:  # highshelf
                    b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
                    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
                    b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
                    a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
                    a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
                    a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

                b_coef = np.array([b0 / a0, b1 / a0, b2 / a0])
                a_coef = np.array([1.0, a1 / a0, a2 / a0])
                sos = np.array([[b_coef[0], b_coef[1], b_coef[2], 1.0, a_coef[1], a_coef[2]]])
            else:
                continue

            if result.ndim > 1:
                for ch in range(result.shape[1]):
                    result[:, ch] = sosfiltfilt(sos, result[:, ch])
            else:
                result = sosfiltfilt(sos, result)

        return result

    # ─── Dynamics Processing (Multiband Compressor) ───
    @staticmethod
    def process_dynamics(data: np.ndarray, sr: int, dynamics: 'Dynamics',
                         intensity: float) -> np.ndarray:
        """Read Dynamics settings and apply real compression."""
        if not dynamics.enabled:
            return data

        if HAS_PEDALBOARD:
            return _RealAudioProcessor._dynamics_pedalboard(data, sr, dynamics, intensity)
        elif HAS_SCIPY:
            return _RealAudioProcessor._dynamics_scipy(data, sr, dynamics, intensity)
        return data

    @staticmethod
    def _dynamics_pedalboard(data, sr, dynamics, intensity):
        """Multiband or single-band compression using pedalboard."""
        if dynamics.multiband:
            # Multiband compression — use dynamics.crossover_low / crossover_high
            cl = getattr(dynamics, 'crossover_low', 200)
            ch = getattr(dynamics, 'crossover_high', 4000)
            low, mid, high = _CrossoverFilter.split_3band(data, sr, cl, ch)
            audio_bands = [low, mid, high]
            result = np.zeros_like(data)

            # Get per-band CompressorBand settings from dynamics module
            band_settings = dynamics.bands if hasattr(dynamics, 'bands') else []

            for i, band_data in enumerate(audio_bands):
                if i < len(band_settings):
                    bs = band_settings[i]
                    thresh = getattr(bs, 'threshold', -20) * intensity
                    ratio = getattr(bs, 'ratio', 4.0)
                    attack = getattr(bs, 'attack', 10.0)  # CompressorBand uses .attack
                    release = getattr(bs, 'release', 100.0)  # CompressorBand uses .release
                else:
                    thresh = -20 * intensity
                    ratio = 4.0
                    attack = 10.0
                    release = 100.0

                board = Pedalboard([
                    Compressor(
                        threshold_db=thresh,
                        ratio=max(1.0, ratio),
                        attack_ms=max(0.1, attack),
                        release_ms=max(1.0, release),
                    )
                ])
                audio = band_data.T.astype(np.float32)
                processed = board(audio, sr).T
                result += processed

            peak = np.max(np.abs(result))
            if peak > 0.99:
                result *= 0.99 / peak
            return result
        else:
            # Single-band compression
            sb = dynamics.single_band if hasattr(dynamics, 'single_band') else None
            if sb:
                thresh = getattr(sb, 'threshold', -20) * intensity
                ratio = getattr(sb, 'ratio', 4.0)
                attack = getattr(sb, 'attack', 10.0)  # CompressorBand uses .attack
                release = getattr(sb, 'release', 100.0)  # CompressorBand uses .release
            else:
                thresh = -20 * intensity
                ratio = 4.0
                attack = 10.0
                release = 100.0

            board = Pedalboard([
                Compressor(
                    threshold_db=thresh,
                    ratio=max(1.0, ratio),
                    attack_ms=max(0.1, attack),
                    release_ms=max(1.0, release),
                )
            ])
            audio = data.T.astype(np.float32) if data.ndim > 1 else data.reshape(1, -1).astype(np.float32)
            result = board(audio, sr).T
            return result

    @staticmethod
    def _dynamics_scipy(data, sr, dynamics, intensity):
        """Fallback dynamics using envelope-based compression."""
        if not HAS_SCIPY:
            return data

        result = data.copy()
        sb = dynamics.single_band if hasattr(dynamics, 'single_band') else None
        thresh_db = getattr(sb, 'threshold', -20) * intensity if sb else -20 * intensity
        ratio = getattr(sb, 'ratio', 4.0) if sb else 4.0
        attack_ms = getattr(sb, 'attack', 10.0) if sb else 10.0
        release_ms = getattr(sb, 'release', 100.0) if sb else 100.0

        for ch in range(data.shape[1] if data.ndim > 1 else 1):
            signal = data[:, ch] if data.ndim > 1 else data
            envelope = _envelope_follower(signal, sr, attack_ms, release_ms)
            env_db = 20 * np.log10(np.maximum(envelope, 1e-10))

            # Gain reduction above threshold
            gain_db = np.where(
                env_db > thresh_db,
                (thresh_db - env_db) * (1.0 - 1.0 / ratio),
                0.0
            )
            gain_linear = 10 ** (gain_db / 20.0)

            if data.ndim > 1:
                result[:, ch] = signal * gain_linear
            else:
                result = signal * gain_linear

        return result

    # ─── Imager Processing (Stereo Width) ───
    @staticmethod
    def process_imager(data: np.ndarray, sr: int, imager: 'Imager',
                       intensity: float) -> np.ndarray:
        """Read Imager settings and apply real stereo processing."""
        if not imager.enabled or data.ndim < 2 or data.shape[1] < 2:
            return data

        if imager.multiband:
            return _RealAudioProcessor._imager_multiband(data, sr, imager, intensity)
        else:
            return _RealAudioProcessor._imager_simple(data, sr, imager, intensity)

    @staticmethod
    def _imager_multiband(data, sr, imager, intensity):
        """Multiband stereo imaging using Mid/Side processing."""
        # Get per-band width from ImagerBand objects in imager.bands
        img_bands = getattr(imager, 'bands', [])
        low_width_pct = img_bands[0].width if len(img_bands) > 0 else 80
        mid_width_pct = img_bands[1].width if len(img_bands) > 1 else 110
        high_width_pct = img_bands[2].width if len(img_bands) > 2 else 130

        low_width = low_width_pct / 100.0
        mid_width = mid_width_pct / 100.0
        high_width = high_width_pct / 100.0

        # Apply intensity interpolation
        low_width = 1.0 + (low_width - 1.0) * intensity
        mid_width = 1.0 + (mid_width - 1.0) * intensity
        high_width = 1.0 + (high_width - 1.0) * intensity

        # Split into 3 bands using imager crossover frequencies
        cl = getattr(imager, 'crossover_low', 200)
        ch = getattr(imager, 'crossover_high', 4000)
        low, mid, high = _CrossoverFilter.split_3band(data, sr, cl, ch)

        result = np.zeros_like(data)
        for band_data, width in zip([low, mid, high], [low_width, mid_width, high_width]):
            left = band_data[:, 0]
            right = band_data[:, 1]
            m = (left + right) * 0.5
            s = (left - right) * 0.5
            s_scaled = s * width
            result[:, 0] += m + s_scaled
            result[:, 1] += m - s_scaled

        # Mono bass
        mono_bass_freq = getattr(imager, 'mono_bass_freq', 0)
        if mono_bass_freq > 0:
            low_band, rest = _CrossoverFilter.split_2band(result, sr, mono_bass_freq)
            mono = (low_band[:, 0] + low_band[:, 1]) * 0.5
            low_band[:, 0] = mono
            low_band[:, 1] = mono
            result = low_band + rest

        peak = np.max(np.abs(result))
        if peak > 0.99:
            result *= 0.99 / peak

        return result

    @staticmethod
    def _imager_simple(data, sr, imager, intensity):
        """Simple stereo width adjustment."""
        width = getattr(imager, 'width', 100) / 100.0
        width = 1.0 + (width - 1.0) * intensity

        left = data[:, 0]
        right = data[:, 1]
        m = (left + right) * 0.5
        s = (left - right) * 0.5
        s_scaled = s * width

        result = np.zeros_like(data)
        result[:, 0] = m + s_scaled
        result[:, 1] = m - s_scaled

        # Balance
        balance = getattr(imager, 'balance', 0.0)
        if abs(balance) > 0.01:
            bal = balance * intensity
            result[:, 0] *= (1.0 - max(0, bal))
            result[:, 1] *= (1.0 + min(0, bal))

        peak = np.max(np.abs(result))
        if peak > 0.99:
            result *= 0.99 / peak

        return result

    # ─── Maximizer Processing (IRC4-style) ───
    @staticmethod
    def process_maximizer(data: np.ndarray, sr: int, maximizer: 'Maximizer',
                          intensity: float) -> np.ndarray:
        """
        Read Maximizer settings and apply real limiting.
        This is the key module — makes the loudness push work!
        """
        if not maximizer.enabled:
            return data

        result = data.copy().astype(np.float64)

        # ── Step 1: Upward Compression ──
        upward_db = getattr(maximizer, 'upward_compress_db', 0)
        if upward_db > 0.1:
            result = _RealAudioProcessor._upward_compress(
                result, sr, amount_db=upward_db * intensity)

        # ── Step 2: Soft Clip ──
        # V5.5 FIX: Default ON at 70% — Suno tracks need soft clip protection
        # BEFORE the limiter to reduce peak-to-loudness ratio
        soft_clip_on = getattr(maximizer, 'soft_clip_enabled', True)
        soft_clip_pct = getattr(maximizer, 'soft_clip_pct', 70)
        if soft_clip_on and soft_clip_pct > 0:
            amount = soft_clip_pct * intensity / 100.0
            drive = 1.0 + amount * 2.0
            result = np.tanh(result * drive) / np.tanh(np.array([drive]))

        # ── Step 3: Gain Push (Ozone 12 style) ──
        # NOTE: gain_db is NOT scaled by intensity — user explicitly sets this value
        # via the GAIN knob. The gain push drives audio into the limiter for loudness.
        gain_db = max(0.0, getattr(maximizer, 'gain_db', 0.0))
        if gain_db > 0.1:
            gain_linear = 10 ** (gain_db / 20.0)
            result = result * gain_linear

        # ── Step 4: Transient Emphasis ──
        transient_pct = getattr(maximizer, 'transient_emphasis_pct', 0)
        if transient_pct > 1.0:
            amount = transient_pct * intensity / 100.0
            for ch in range(result.shape[1] if result.ndim > 1 else 1):
                signal = result[:, ch] if result.ndim > 1 else result
                fast_env = _envelope_follower(signal, sr, 0.5, 10.0)
                slow_env = _envelope_follower(signal, sr, 20.0, 100.0)
                transient = np.maximum(fast_env - slow_env, 0)
                gain = 1.0 + transient * amount * 5.0
                gain = np.clip(gain, 1.0, 3.0)
                if result.ndim > 1:
                    result[:, ch] = signal * gain
                else:
                    result = signal * gain

        # ── Step 5: IRC-style Limiting (uses actual IRC mode parameters) ──
        ceiling_db = getattr(maximizer, 'ceiling', -1.0)
        character = getattr(maximizer, 'character', 5.0)
        irc_mode = getattr(maximizer, 'irc_mode', 'IRC 4')
        irc_sub_mode = getattr(maximizer, 'irc_sub_mode', 'Classic')

        # V5.5 FIX: Look up actual IRC parameters from genre_profiles
        irc_key = f"{irc_mode} - {irc_sub_mode}" if irc_sub_mode else irc_mode
        irc_params = get_irc_mode(irc_key)

        if HAS_PEDALBOARD:
            result = _RealAudioProcessor._irc_limit_pedalboard(
                result, sr, ceiling_db, character, irc_mode, irc_params)
        else:
            ceiling_linear = 10 ** (ceiling_db / 20.0)
            peak = np.max(np.abs(result))
            if peak > ceiling_linear:
                result = result * (ceiling_linear / peak)

        # ── Step 6: True Peak Limiting ──
        true_peak = getattr(maximizer, 'true_peak', True)
        if true_peak and HAS_SCIPY:
            result = _RealAudioProcessor._true_peak_limit(result, sr, ceiling_db)

        return result.astype(np.float64)

    @staticmethod
    def _upward_compress(data, sr, amount_db, threshold_db=-40.0):
        """Upward compression — brings up quiet parts."""
        result = data.copy()
        for ch in range(data.shape[1] if data.ndim > 1 else 1):
            signal = data[:, ch] if data.ndim > 1 else data
            envelope = _envelope_follower(signal, sr, 50.0, 200.0)
            env_db = 20 * np.log10(np.maximum(envelope, 1e-10))

            # Guard against division by zero when threshold_db == -10.0
            upper_bound = -10.0
            denom = upper_bound - threshold_db if abs(upper_bound - threshold_db) > 1e-6 else 1e-6
            gain_db = np.where(
                env_db < threshold_db, 0.0,
                np.where(env_db < upper_bound,
                         amount_db * (1.0 - (env_db - threshold_db) / denom),
                         0.0))
            gain_db = np.clip(gain_db, 0, amount_db)
            gain_linear = 10 ** (gain_db / 20.0)

            smooth_n = int(0.01 * sr)
            if smooth_n > 1:
                kernel = np.ones(smooth_n) / smooth_n
                gain_linear = np.convolve(gain_linear, kernel, mode='same')

            if data.ndim > 1:
                result[:, ch] = signal * gain_linear
            else:
                result = signal * gain_linear
        return result

    @staticmethod
    def _irc_limit_pedalboard(data, sr, ceiling_db, character, irc_mode,
                               irc_params=None):
        """IRC limiting using LookAheadLimiter with IRC mode parameters.

        V5.5 REWRITE: Replaced pedalboard.Limiter (no look-ahead → hard clipping)
        with LookAheadLimiter (5ms look-ahead → clean limiting).

        Parameters used from irc_params:
        - attack: base attack time in ms (0.1 to 3.0)
        - release: base release time in ms (15 to 200)
        - knee: limiter knee in dB (affects threshold offset per band)
        - level_in: input gain multiplier (drives harder into limiter)
        """
        if irc_params is None:
            irc_params = {"attack": 0.5, "release": 40, "knee": 3.0, "level_in": 1.1}

        # Read IRC-specific parameters
        base_attack = irc_params.get("attack", 0.5)
        base_release = irc_params.get("release", 40)
        knee_db = irc_params.get("knee", 3.0)
        level_in = irc_params.get("level_in", 1.0)

        # Character modifies the IRC base parameters (0=smoother, 10=more aggressive)
        char_factor = character / 10.0  # 0.0 to 1.0
        if char_factor < 0.5:
            # Smoother: slower attack, longer release
            smooth = (0.5 - char_factor) * 2.0  # 0.0 to 1.0
            attack_ms = base_attack * (1.0 + smooth * 2.0)
            release_ms = base_release * (1.0 + smooth * 1.5)
        elif char_factor > 0.5:
            # More aggressive: faster attack, shorter release
            aggro = (char_factor - 0.5) * 2.0  # 0.0 to 1.0
            attack_ms = base_attack * (1.0 - aggro * 0.5)
            release_ms = base_release * (1.0 - aggro * 0.4)
        else:
            attack_ms = base_attack
            release_ms = base_release

        release_ms = max(5.0, min(500.0, release_ms))
        attack_ms = max(1.0, min(20.0, attack_ms))

        # Apply IRC level_in boost (drives harder into limiter)
        result = data.copy()
        if level_in > 1.001:
            result = result * level_in

        if "4" in str(irc_mode):
            # IRC 4: Multiband limiting (4-band crossover)
            bands = _CrossoverFilter.split_4band(result, sr, 150, 1000, 5000)
            limited_bands = []
            for i, band_data in enumerate(bands):
                ceiling_linear = 10 ** (ceiling_db / 20.0)
                band_peak = np.max(np.abs(band_data))
                if band_peak > ceiling_linear:
                    # Per-band threshold: offset by knee for multiband character
                    band_ceiling = ceiling_db + knee_db
                    band_limiter = LookAheadLimiterFast(
                        ceiling_db=band_ceiling,
                        lookahead_ms=5.0,
                        release_ms=release_ms,
                        attack_ms=max(attack_ms, 5.0),
                        variable_release=True,
                        true_peak=False,  # True peak only on final stage
                    )
                    limited_bands.append(band_limiter.process(band_data, sr))
                else:
                    limited_bands.append(band_data)
            result = sum(limited_bands)

            # Final brickwall at exact ceiling
            final_limiter = LookAheadLimiterFast(
                ceiling_db=ceiling_db,
                lookahead_ms=5.0,
                release_ms=max(5.0, release_ms * 0.5),
                attack_ms=5.0,
                variable_release=True,
                true_peak=True,
            )
            return final_limiter.process(result, sr)
        else:
            # IRC 1/2/3/5/LL: Single-band limiting with mode-specific timing
            limiter = LookAheadLimiterFast(
                ceiling_db=ceiling_db,
                lookahead_ms=5.0,
                release_ms=release_ms,
                attack_ms=max(attack_ms, 5.0),
                variable_release=(release_ms > 30),
                true_peak=True,
            )
            return limiter.process(result, sr)

    @staticmethod
    def _true_peak_limit(data, sr, ceiling_db):
        """True Peak Limiter (ITU-R BS.1770-4) with 4x oversampling ISP detection.

        Per-chunk: upsample 4x → detect peaks → compute gain reduction →
        apply with 1ms look-ahead smoothing → downsample → hard clip safety.
        """
        try:
            ceiling_linear = 10 ** (ceiling_db / 20.0)
            result = data.copy().astype(np.float64)
            chunk_size = sr  # process 1 second at a time

            for ch in range(data.shape[1] if data.ndim > 1 else 1):
                signal = data[:, ch].astype(np.float64) if data.ndim > 1 else data.astype(np.float64)

                for start in range(0, len(signal), chunk_size):
                    end = min(start + chunk_size, len(signal))
                    chunk = signal[start:end]
                    if len(chunk) < 8:
                        continue

                    # 4x upsample
                    x_4x = resample_poly(chunk, 4, 1)
                    peaks = np.abs(x_4x)

                    # Look-ahead window (1ms at 4x rate)
                    lookahead = max(4, int(sr * 4 * 0.001))

                    # Compute gain envelope at 4x rate
                    gain_4x = np.ones_like(x_4x)
                    for i in range(len(x_4x)):
                        if peaks[i] > ceiling_linear:
                            reduction = ceiling_linear / peaks[i]
                            la_start = max(0, i - lookahead)
                            la_end = min(len(x_4x), i + lookahead)
                            for j in range(la_start, la_end):
                                dist = abs(j - i) / max(1, lookahead)
                                smooth = 1.0 - (1.0 - reduction) * (1.0 - dist * dist)
                                gain_4x[j] = min(gain_4x[j], smooth)

                    # Apply gain at 4x rate
                    x_4x *= gain_4x

                    # Downsample back
                    x_limited = resample_poly(x_4x, 1, 4)[:len(chunk)]

                    # Hard clip safety
                    x_limited = np.clip(x_limited, -ceiling_linear, ceiling_linear)

                    if data.ndim > 1:
                        result[start:end, ch] = x_limited
                    else:
                        result[start:end] = x_limited

            return result.astype(np.float32)
        except Exception:
            ceiling_linear = 10 ** (ceiling_db / 20.0)
            return np.clip(data, -ceiling_linear, ceiling_linear)

    # ─── Final True Peak Limiter ───
    @staticmethod
    def final_true_peak_limit(data: np.ndarray, sr: int,
                              ceiling_db: float = -1.0) -> np.ndarray:
        """
        Final brickwall True Peak limiter — the ABSOLUTE LAST processing step.

        V5.5 REWRITE: Uses LookAheadLimiter with 5ms look-ahead and
        4x oversampled true peak detection. Replaces pedalboard.Limiter
        which had NO look-ahead and caused hard clipping artifacts.

        True Peak (dBTP) measures inter-sample peaks using 4x oversampling,
        which can be up to ~3 dB higher than sample peaks.
        """
        ceiling_linear = 10 ** (ceiling_db / 20.0)
        result = data.copy()

        # Primary: LookAheadLimiter with true peak detection
        final_limiter = LookAheadLimiterFast(
            ceiling_db=ceiling_db,
            lookahead_ms=5.0,
            release_ms=50.0,
            attack_ms=5.0,
            variable_release=True,
            true_peak=True,
        )
        result = final_limiter.process(result, sr)

        # Verification: measure actual True Peak using 4x oversampling
        if HAS_SCIPY:
            max_true_peak = 0.0
            chunk_size = sr  # Process 1-second chunks
            for ch in range(result.shape[1] if result.ndim > 1 else 1):
                signal = result[:, ch] if result.ndim > 1 else result
                for start in range(0, len(signal), chunk_size):
                    end = min(start + chunk_size, len(signal))
                    chunk = signal[start:end]
                    if len(chunk) < 4:
                        continue
                    oversampled = resample_poly(chunk, 4, 1)
                    chunk_tp = np.max(np.abs(oversampled))
                    max_true_peak = max(max_true_peak, chunk_tp)

            # If True Peak still exceeds ceiling (safety net), scale down globally
            if max_true_peak > ceiling_linear:
                reduction = ceiling_linear / max_true_peak
                result = result * reduction
                final_tp_db = 20 * np.log10(ceiling_linear) if ceiling_linear > 0 else -60.0
            else:
                final_tp_db = 20 * np.log10(max(max_true_peak, 1e-10))

            # Verification logging
            tp_error_db = final_tp_db - ceiling_db
            print(f"[TRUE PEAK] Target: {ceiling_db:.2f} dBTP | "
                  f"Actual: {final_tp_db:.2f} dBTP | "
                  f"Error: {tp_error_db:+.2f} dB | "
                  f"{'PASS' if abs(tp_error_db) <= 0.3 else 'FAIL'}")
        else:
            # Fallback: sample-peak with exact ceiling
            peak = np.max(np.abs(result))
            if peak > ceiling_linear:
                result = result * (ceiling_linear / peak)

        return result

    # ─── Loudness Normalization ───
    @staticmethod
    def process_loudness_norm(data, sr, target_lufs=-14.0, target_tp=-1.0,
                              gain_offset_db=0.0):
        """
        Normalize loudness to target LUFS, with gain offset from Maximizer.

        The gain_offset_db shifts the normalization target so the Maximizer's
        gain push is preserved. E.g., if target=-14 and gain_offset=+6,
        the effective target becomes -8 LUFS (louder output).

        The True Peak ceiling is always enforced regardless of gain offset.
        """
        if not HAS_PYLOUDNORM:
            return data

        if data.ndim == 1:
            data = np.column_stack([data, data])

        meter = pyln.Meter(sr)
        current_lufs = meter.integrated_loudness(data)

        if current_lufs <= -70:
            return data

        # Apply gain offset: user's GAIN knob shifts the target louder
        effective_target = target_lufs + gain_offset_db
        # V5.5.1: Extended clamp to allow up to -2 LUFS for aggressive mastering
        # (was -5.0 max, now -2.0 to support 20 dB gain push)
        effective_target = max(-23.0, min(-2.0, effective_target))

        normalized = pyln.normalize.loudness(data, current_lufs, effective_target)

        # FIX V5.4: REMOVED sample-peak limiting here.
        # Reason: This used sample peaks (not True Peaks) which can differ
        # by 0.5-3 dB, causing double-limiting with final_true_peak_limit().
        # True Peak limiting is now handled EXCLUSIVELY by final_true_peak_limit()
        # which uses proper 4x-oversampled ISP detection (ITU-R BS.1770).

        return normalized


# ═══════════════════════════════════════════════════════════════════
#  MASTER CHAIN (drop-in replacement — same API as original)
# ═══════════════════════════════════════════════════════════════════

class MasterChain:
    """
    Master Chain — orchestrates all mastering modules.

    This version uses REAL AUDIO PROCESSING (pedalboard + scipy)
    instead of FFmpeg filter chains.

    Usage (unchanged from original):
        chain = MasterChain()
        chain.load_audio("/path/to/audio.wav")

        # Option A: AI Assist (auto-settings)
        rec = chain.ai_recommend(genre="EDM", platform="YouTube")
        chain.apply_recommendation(rec)

        # Option B: Manual settings
        chain.equalizer.load_tone_preset("Bright")
        chain.maximizer.set_ceiling(-1.0)
        chain.maximizer.set_irc_mode("IRC 4", "Classic")

        # Preview
        chain.preview(callback=progress_fn)

        # Full render
        chain.render(callback=progress_fn)
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

        # Audio paths
        self.input_path = None
        self.output_path = None
        self.preview_path = None

        # Modules (signal flow order) — ResSup → EQ → DYN → IMG → Soothe → MAX
        self.resonance_suppressor = ResonanceSuppressor()
        self.equalizer = Equalizer()
        self.dynamics = Dynamics()
        self.imager = Imager()
        self.soothe = SootheProcessor()
        self.maximizer = Maximizer()

        # Loudness tools
        self.loudness_meter = LoudnessMeter(ffmpeg_path)
        self.ai_assist = AIAssist(ffmpeg_path)

        # Chain settings
        self.intensity = 50          # 0-100% global intensity
        self.target_lufs = -14.0     # Target integrated LUFS
        self.target_tp = -1.0        # Target True Peak
        self.normalize_loudness = True  # Apply loudnorm as final step
        self.platform = "YouTube"

        # State
        self.input_analysis = None   # LoudnessAnalysis of input
        self.output_analysis = None  # LoudnessAnalysis of output
        self.recommendation = None   # Last AI recommendation
        self.is_processing = False
        self._processing_lock = threading.Lock()

        # Real-time meter callback
        self._meter_callback = None
        self._progress_callback = None
        self._meter_lock = threading.Lock()

        # Check capabilities
        self._use_real_processing = HAS_SOUNDFILE and (HAS_PEDALBOARD or HAS_SCIPY)
        if self._use_real_processing:
            print("[CHAIN] ✓ Real Audio Processing enabled "
                  f"(pedalboard={HAS_PEDALBOARD}, scipy={HAS_SCIPY}, "
                  f"pyloudnorm={HAS_PYLOUDNORM})")
        else:
            print("[CHAIN] ⚠ Falling back to FFmpeg processing "
                  f"(soundfile={HAS_SOUNDFILE}, pedalboard={HAS_PEDALBOARD})")

    def set_meter_callback(self, callback):
        """Set real-time meter callback for GUI."""
        self._meter_callback = callback

    def set_progress_callback(self, callback):
        """Set progress callback for GUI."""
        self._progress_callback = callback

    # ─── Audio Loading (unchanged) ───

    def load_audio(self, path: str) -> bool:
        """Set input audio path."""
        if not os.path.exists(path):
            print(f"[CHAIN] File not found: {path}")
            return False
        self.input_path = path
        base, ext = os.path.splitext(path)
        # Always output as WAV for soundfile compatibility; non-WAV formats need FFmpeg
        safe_ext = ext if ext.lower() in ['.wav', '.wave', '.flac'] else '.wav'
        self.output_path = f"{base}_mastered{safe_ext}"
        self.preview_path = f"{base}_preview.wav"
        return True

    # ─── Platform Settings (unchanged) ───

    def set_platform(self, platform: str):
        """Set target platform (affects loudness target)."""
        if platform in PLATFORM_TARGETS:
            self.platform = platform
            target = PLATFORM_TARGETS[platform]
            self.target_lufs = target["target_lufs"]
            self.target_tp = target["true_peak"]
            self.maximizer.set_ceiling(self.target_tp)

    # ─── AI Recommend (unchanged) ───

    def ai_recommend(
        self,
        genre: str = "All-Purpose Mastering",
        platform: str = "YouTube",
        intensity: int = 50,
    ) -> Optional[MasterRecommendation]:
        """Run AI analysis and get recommendations."""
        if not self.input_path:
            print("[CHAIN] No input audio loaded")
            return None
        if not os.path.exists(self.input_path):
            print(f"[CHAIN] Input file not found: {self.input_path}")
            return None
        try:
            self.set_platform(platform)
            self.intensity = intensity
            rec = self.ai_assist.analyze_and_recommend(
                self.input_path, genre, platform, intensity
            )
            if rec:
                self.recommendation = rec
            return rec
        except Exception as e:
            print(f"[CHAIN] ai_recommend error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def apply_recommendation(self, rec: MasterRecommendation):
        """Apply AI recommendation to all modules."""
        try:
            if hasattr(rec, 'equalizer') and rec.equalizer:
                self.equalizer.load_settings_dict(rec.equalizer.get_settings_dict())
            if hasattr(rec, 'dynamics') and rec.dynamics:
                self.dynamics.load_settings_dict(rec.dynamics.get_settings_dict())
            if hasattr(rec, 'imager') and rec.imager:
                self.imager.load_settings_dict(rec.imager.get_settings_dict())
            if hasattr(rec, 'maximizer') and rec.maximizer:
                self.maximizer.load_settings_dict(rec.maximizer.get_settings_dict())
            self.intensity = getattr(rec, 'intensity', 50)
            self.recommendation = rec
            print(f"[CHAIN] Applied AI recommendation for {getattr(rec, 'genre', 'Unknown')}")
        except Exception as e:
            print(f"[CHAIN] apply_recommendation error: {e}")
            import traceback
            traceback.print_exc()
            raise

    # ─── FFmpeg compatibility (kept for fallback) ───

    def build_filter_chain(self) -> str:
        """Build complete FFmpeg audio filter chain string."""
        intensity = self.intensity / 100.0
        all_filters = []
        eq_filters = self.equalizer.get_ffmpeg_filters(intensity)
        all_filters.extend(eq_filters)
        dyn_filters = self.dynamics.get_ffmpeg_filters(intensity)
        all_filters.extend(dyn_filters)
        img_filters = self.imager.get_ffmpeg_filters(intensity)
        all_filters.extend(img_filters)
        max_filters = self.maximizer.get_ffmpeg_filters(intensity)
        all_filters.extend(max_filters)
        if not all_filters:
            return "anull"
        return ",".join(all_filters)

    def build_ffmpeg_command(
        self,
        input_path: str,
        output_path: str,
        preview: bool = False,
        preview_start: float = 0,
        preview_duration: float = 30,
    ) -> List[str]:
        """Build complete FFmpeg command for mastering (fallback)."""
        filter_chain = self.build_filter_chain()
        cmd = [self.ffmpeg_path, "-y"]
        if preview:
            cmd.extend(["-ss", str(preview_start)])
        cmd.extend(["-i", input_path])
        if preview:
            cmd.extend(["-t", str(preview_duration)])
        cmd.extend(["-af", filter_chain])
        ext = os.path.splitext(output_path)[1].lower()
        if ext in [".wav", ".wave"]:
            cmd.extend(["-c:a", "pcm_s24le"])
        elif ext in [".flac"]:
            cmd.extend(["-c:a", "flac", "-compression_level", "8"])
        elif ext in [".mp3"]:
            cmd.extend(["-c:a", "libmp3lame", "-b:a", "320k"])
        elif ext in [".aac", ".m4a"]:
            cmd.extend(["-c:a", "aac", "-b:a", "256k"])
        else:
            cmd.extend(["-c:a", "pcm_s24le"])
        cmd.append(output_path)
        return cmd

    # ═══════════════════════════════════════════════════════
    #  REAL AUDIO PROCESSING — Preview & Render
    # ═══════════════════════════════════════════════════════

    def _process_audio_real(self, data: np.ndarray, sr: int,
                            callback: Optional[Callable] = None) -> np.ndarray:
        """
        Process audio through the complete mastering chain using REAL DSP.
        Signal flow: EQ → Dynamics → Imager → Maximizer → [Loudness Norm]
        """
        intensity = self.intensity / 100.0

        # Ensure stereo
        if data.ndim == 1:
            data = np.column_stack([data, data])

        result = data.astype(np.float64)

        # Step 0: Pre-gain headroom (-3 dB)
        # V5.5 FIX: Suno/Udio tracks are typically -8.6 dB RMS (very hot).
        # Adding -3 dB headroom prevents clipping before the chain even starts.
        # The Maximizer gain push will compensate for final loudness.
        pre_gain_linear = 10 ** (-3.0 / 20.0)  # = 0.7079
        result = result * pre_gain_linear

        # V5.8: Send pre-chain meter data (input signal BEFORE any processing)
        self._send_meter(result, sr, "pre_chain")

        # Step 0.5: Resonance Suppressor (before EQ — clean harsh resonances first)
        if callback:
            callback(7, "Suppressing resonances...")
        if self.resonance_suppressor.enabled:
            result = self.resonance_suppressor.process(result.astype(np.float32)).astype(np.float64)
            result = np.nan_to_num(result, nan=0.0, posinf=0.99, neginf=-0.99)
        self._send_meter(result, sr, "post_resonance")

        # Step 1: EQ
        if callback:
            callback(10, "Applying EQ...")
        result = _RealAudioProcessor.process_eq(result, sr, self.equalizer, intensity)
        result = np.nan_to_num(result, nan=0.0, posinf=0.99, neginf=-0.99)
        self._send_meter(result, sr, "post_eq")

        # Step 2: Dynamics
        if callback:
            callback(25, "Applying Dynamics...")
        result = _RealAudioProcessor.process_dynamics(result, sr, self.dynamics, intensity)
        result = np.nan_to_num(result, nan=0.0, posinf=0.99, neginf=-0.99)
        self._send_meter(result, sr, "post_dynamics")

        # Step 3: Imager
        if callback:
            callback(40, "Applying Stereo Imager...")
        result = _RealAudioProcessor.process_imager(result, sr, self.imager, intensity)
        result = np.nan_to_num(result, nan=0.0, posinf=0.99, neginf=-0.99)
        self._send_meter(result, sr, "post_imager")

        # Step 3.5: Soothe (Dynamic Resonance Suppression)
        if self.soothe.enabled and self.soothe.amount > 0:
            if callback:
                callback(48, "Applying Soothe (Resonance Suppression)...")
            result = self.soothe.process(result)
            result = np.nan_to_num(result, nan=0.0, posinf=0.99, neginf=-0.99)
            self._send_meter(result, sr, "post_soothe")

        # Step 4: Maximizer (THE KEY MODULE — loudness push!)
        if callback:
            callback(55, "Applying Maximizer (IRC Limiting)...")
        result = _RealAudioProcessor.process_maximizer(result, sr, self.maximizer, intensity)
        result = np.nan_to_num(result, nan=0.0, posinf=0.99, neginf=-0.99)
        self._send_meter(result, sr, "post_maximizer")

        # Step 5: Loudness Normalization (optional)
        # Pass the Maximizer's gain_db as offset so the gain push is preserved
        # in the final loudness target. Without this, loudnorm would undo the
        # Maximizer's work by normalizing everything back to target_lufs.
        if self.normalize_loudness:
            if callback:
                callback(75, "Normalizing Loudness...")
            gain_offset = max(0.0, getattr(self.maximizer, 'gain_db', 0.0))
            result = _RealAudioProcessor.process_loudness_norm(
                result, sr, self.target_lufs, self.target_tp,
                gain_offset_db=gain_offset)
            self._send_meter(result, sr, "post_loudnorm")

        # Step 6: Final True Peak Brickwall Limiter
        # This is the ABSOLUTE LAST step — enforces ceiling no matter what.
        # Uses pedalboard Limiter which handles inter-sample peaks properly.
        if callback:
            callback(90, "Enforcing True Peak ceiling...")
        result = _RealAudioProcessor.final_true_peak_limit(
            result, sr, self.target_tp)
        self._send_meter(result, sr, "final")

        return result

    def _send_meter(self, data: np.ndarray, sr: int, stage: str):
        """Send real-time meter data to GUI (thread-safe).

        V5.5 FIX: Sends all keys expected by MetersPanel.update_live_levels():
        - lufs_integrated, lufs_momentary, lufs_short_term (not just "lufs")
        - gain_reduction_db (estimated from peak reduction)
        """
        if self._meter_callback:
            try:
                chunk = data[-4096:] if len(data) > 4096 else data
                if chunk.ndim > 1:
                    left_rms = np.sqrt(np.mean(chunk[:, 0] ** 2))
                    right_rms = np.sqrt(np.mean(chunk[:, 1] ** 2))
                    peak_l = np.max(np.abs(chunk[:, 0]))
                    peak_r = np.max(np.abs(chunk[:, 1]))
                else:
                    left_rms = right_rms = np.sqrt(np.mean(chunk ** 2))
                    peak_l = peak_r = np.max(np.abs(chunk))

                left_peak_db = 20 * np.log10(max(peak_l, 1e-10))
                right_peak_db = 20 * np.log10(max(peak_r, 1e-10))

                levels = {
                    "stage": stage,
                    "left_rms_db": 20 * np.log10(max(left_rms, 1e-10)),
                    "right_rms_db": 20 * np.log10(max(right_rms, 1e-10)),
                    "left_peak_db": left_peak_db,
                    "right_peak_db": right_peak_db,
                }

                # V5.5: Estimate gain reduction (how much the limiter is working)
                max_peak_db = max(left_peak_db, right_peak_db)
                ceiling_db = getattr(self, 'target_tp', -1.0)
                if max_peak_db > ceiling_db:
                    levels["gain_reduction_db"] = ceiling_db - max_peak_db
                else:
                    levels["gain_reduction_db"] = 0.0

                # V5.5: LUFS measurement with correct key names
                # V5.10 FIX: Integrated uses FULL audio, LRA computed properly
                if HAS_PYLOUDNORM:
                    try:
                        meter = pyln.Meter(sr)

                        def _ensure_stereo(buf):
                            if buf.ndim == 1:
                                return np.column_stack([buf, buf])
                            return buf

                        def _safe_lufs(buf):
                            val = meter.integrated_loudness(buf)
                            if val == float('-inf') or val < -70:
                                return -70.0
                            return val

                        # Momentary LUFS (last 400ms)
                        mom_samples = min(len(data), int(sr * 0.4))
                        if mom_samples > 1024:
                            lufs_mom = _safe_lufs(_ensure_stereo(data[-mom_samples:]))
                            levels["lufs_momentary"] = lufs_mom

                        # Short-term LUFS (last 3 seconds)
                        short_samples = min(len(data), int(sr * 3.0))
                        if short_samples > sr:
                            lufs_short = _safe_lufs(_ensure_stereo(data[-short_samples:]))
                            levels["lufs_short_term"] = lufs_short

                        # Integrated LUFS — FULL audio (not just last 10s)
                        if len(data) > sr:
                            full_stereo = _ensure_stereo(data)
                            lufs_int = _safe_lufs(full_stereo)
                            levels["lufs_integrated"] = lufs_int
                            levels["lufs"] = lufs_int  # backward compat

                        # V5.10: Loudness Range (LRA) — ITU-R BS.1770-4
                        # Divide into 3s blocks stepping every 1s, compute per-block LUFS,
                        # apply absolute + relative gates, LRA = P95 - P10
                        block_len = int(sr * 3.0)
                        step_len = int(sr * 1.0)
                        if len(data) >= block_len:
                            full_stereo = _ensure_stereo(data)
                            st_values = []
                            for start in range(0, len(full_stereo) - block_len + 1, step_len):
                                blk = full_stereo[start:start + block_len]
                                val = meter.integrated_loudness(blk)
                                if val != float('-inf') and val > -70:
                                    st_values.append(val)
                            if len(st_values) >= 2:
                                # Absolute gate: discard blocks < -70 LUFS (already done above)
                                # Relative gate: mean of ungated, then discard < mean - 20 LU
                                ungated_mean = np.mean(st_values)
                                gated = [v for v in st_values if v >= ungated_mean - 20.0]
                                if len(gated) >= 2:
                                    gated_sorted = sorted(gated)
                                    p10 = np.percentile(gated_sorted, 10)
                                    p95 = np.percentile(gated_sorted, 95)
                                    levels["lu_range"] = max(0.0, p95 - p10)
                                else:
                                    levels["lu_range"] = 0.0
                            else:
                                levels["lu_range"] = 0.0
                    except Exception:
                        levels["lufs_momentary"] = -70.0
                        levels["lufs_short_term"] = -70.0
                        levels["lufs_integrated"] = -70.0
                        levels["lufs"] = -70.0
                        levels["lu_range"] = 0.0

                with self._meter_lock:
                    self._meter_callback(levels)
            except Exception as e:
                print(f"[METER] ❌ _send_meter ERROR: {e}")
                import traceback
                traceback.print_exc()

    # ─── Preview (REAL processing) ───

    def preview(
        self,
        start_sec: float = 0,
        duration_sec: float = 30,
        callback: Optional[Callable] = None,
    ) -> Optional[str]:
        """
        Render a short preview using REAL audio processing.
        Falls back to FFmpeg if dependencies not available.
        """
        if not self.input_path:
            print("[CHAIN] No input loaded")
            return None

        with self._processing_lock:
            if self.is_processing:
                print("[CHAIN] Warning: was still processing, forcing reset")
                self.is_processing = False
            self.is_processing = True

        try:
            if callback:
                callback(0, "Generating preview...")

            import tempfile
            preview_path = self.preview_path or os.path.join(tempfile.gettempdir(), "longplay_preview.wav")

            if self._use_real_processing:
                return self._preview_real(preview_path, start_sec, duration_sec, callback)
            else:
                return self._preview_ffmpeg(preview_path, start_sec, duration_sec, callback)

        except Exception as e:
            print(f"[CHAIN] Preview error: {e}")
            import traceback
            traceback.print_exc()
            if callback:
                callback(-1, f"Preview error: {e}")
            return None
        finally:
            with self._processing_lock:
                self.is_processing = False

    def _preview_real(self, preview_path, start_sec, duration_sec, callback):
        """Preview using real audio processing."""
        if callback:
            callback(5, "Loading audio...")

        info = sf.info(self.input_path)
        sr = info.samplerate
        start_sample = max(0, int(start_sec * sr))
        end_sample = int((start_sec + duration_sec) * sr)
        data, sr = sf.read(self.input_path, start=start_sample, stop=end_sample)
        if data.ndim == 1:
            data = np.column_stack([data, data])
        preview_data = data

        if callback:
            callback(10, "Processing preview...")

        # Process through mastering chain
        result = self._process_audio_real(preview_data, sr, callback)

        if callback:
            callback(90, "Writing preview...")

        result = np.clip(result, -1.0, 1.0).astype(np.float32)

        # Final safety: ensure output doesn't exceed ceiling
        peak = np.max(np.abs(result))
        if peak > 1.0:
            output_logger = logging.getLogger(__name__)
            result = result / peak * 0.999
            output_logger.warning(f"Output peak {peak:.4f} exceeded 1.0, normalized to 0.999")
        sf.write(preview_path, result, sr, subtype='PCM_24')

        if callback:
            callback(100, "Preview ready!")

        print(f"[CHAIN] Preview saved: {preview_path}")
        return preview_path

    def _preview_ffmpeg(self, preview_path, start_sec, duration_sec, callback):
        """Fallback preview using FFmpeg."""
        cmd = self.build_ffmpeg_command(
            self.input_path, preview_path,
            preview=True,
            preview_start=start_sec,
            preview_duration=duration_sec,
        )
        print(f"[CHAIN] Preview command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[CHAIN] Preview error: {result.stderr}")
            if callback:
                callback(-1, f"Preview failed: {result.stderr[:200]}")
            return None
        if callback:
            callback(100, "Preview ready!")
        return preview_path

    # ─── Render (REAL processing) ───

    def render(
        self,
        output_path: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> Optional[str]:
        """
        Render full mastered audio using REAL audio processing.
        Falls back to FFmpeg if dependencies not available.
        """
        if not self.input_path:
            print("[CHAIN] No input loaded")
            return None

        with self._processing_lock:
            if self.is_processing:
                print("[CHAIN] Warning: was still processing, forcing reset for render")
                self.is_processing = False
            self.is_processing = True

        out_path = output_path or self.output_path

        try:
            if callback:
                callback(0, "Starting mastering...")

            if self._use_real_processing:
                return self._render_real(out_path, callback)
            else:
                if self.normalize_loudness:
                    return self._render_with_loudnorm(out_path, callback)
                else:
                    return self._render_single_pass(out_path, callback)

        except Exception as e:
            print(f"[CHAIN] Render error: {e}")
            import traceback
            traceback.print_exc()
            if callback:
                callback(-1, f"Render error: {e}")
            return None
        finally:
            with self._processing_lock:
                self.is_processing = False

    def _render_real(self, output_path, callback):
        """Full render using real audio processing."""
        if callback:
            callback(5, "Loading audio...")

        data, sr = sf.read(self.input_path)
        if data.ndim == 1:
            data = np.column_stack([data, data])

        print(f"[CHAIN] Processing {len(data)/sr:.1f}s audio at {sr}Hz "
              f"(real DSP, pedalboard={HAS_PEDALBOARD})")

        # Process through mastering chain
        result = self._process_audio_real(data, sr, callback)

        if callback:
            callback(85, "Writing output...")

        result = np.clip(result, -1.0, 1.0).astype(np.float32)

        # Final safety: ensure output doesn't exceed ceiling
        peak = np.max(np.abs(result))
        if peak > 1.0:
            output_logger = logging.getLogger(__name__)
            result = result / peak * 0.999
            output_logger.warning(f"Output peak {peak:.4f} exceeded 1.0, normalized to 0.999")

        # Write output — only WAV and FLAC supported by soundfile
        ext = os.path.splitext(output_path)[1].lower()
        if ext in [".flac"]:
            sf.write(output_path, result, sr, format='FLAC')
        elif ext in [".wav", ".wave"]:
            sf.write(output_path, result, sr, subtype='PCM_24')
        else:
            # Unsupported format: write as WAV, then convert via FFmpeg
            wav_path = output_path.rsplit('.', 1)[0] + '_temp.wav'
            sf.write(wav_path, result, sr, subtype='PCM_24')
            try:
                conv_cmd = [self.ffmpeg_path, "-y", "-i", wav_path]
                if ext in [".mp3"]:
                    conv_cmd.extend(["-c:a", "libmp3lame", "-b:a", "320k"])
                elif ext in [".aac", ".m4a"]:
                    conv_cmd.extend(["-c:a", "aac", "-b:a", "256k"])
                elif ext in [".ogg"]:
                    conv_cmd.extend(["-c:a", "libvorbis", "-q:a", "8"])
                else:
                    conv_cmd.extend(["-c:a", "pcm_s24le"])
                conv_cmd.append(output_path)
                subprocess.run(conv_cmd, capture_output=True, timeout=300)
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)

        # Analyze output
        if callback:
            callback(90, "Analyzing output...")

        self.output_analysis = self.loudness_meter.analyze(output_path)
        self.input_analysis = self.loudness_meter.analyze(self.input_path)

        if self.output_analysis:
            print(f"[CHAIN] Output: {self.output_analysis}")

        if callback:
            callback(100, "Mastering complete!")

        print(f"[CHAIN] Mastered output saved: {output_path}")
        return output_path

    # ─── FFmpeg fallback methods (unchanged from original) ───

    def _render_single_pass(self, output_path, callback):
        """Single-pass render (no loudness normalization) — FFmpeg fallback."""
        if callback:
            callback(10, "Processing audio...")
        cmd = self.build_ffmpeg_command(self.input_path, output_path)
        print(f"[CHAIN] Render command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"[CHAIN] Render error: {result.stderr}")
            if callback:
                callback(-1, "Render failed")
            return None
        if callback:
            callback(100, "Mastering complete!")
        return output_path

    def _render_with_loudnorm(self, output_path, callback):
        """Two-pass render with loudness normalization — FFmpeg fallback."""
        import tempfile
        temp_dir = os.path.dirname(output_path) or tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, "_longplay_master_temp.wav")
        try:
            if callback:
                callback(10, "Pass 1: Applying mastering chain...")
            cmd1 = self.build_ffmpeg_command(self.input_path, temp_file)
            result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=600)
            if result1.returncode != 0:
                print(f"[CHAIN] Pass 1 error: {result1.stderr}")
                if callback:
                    callback(-1, "Pass 1 failed")
                return None

            if callback:
                callback(40, "Pass 2: Analyzing loudness...")
            analysis = self.loudness_meter.analyze(temp_file)
            if not analysis:
                shutil.move(temp_file, output_path)
                if callback:
                    callback(100, "Mastering complete (no loudnorm)")
                return output_path

            if callback:
                callback(60, "Pass 2: Normalizing loudness...")
            # Apply gain offset so Maximizer's gain push is preserved
            gain_offset = max(0.0, getattr(self.maximizer, 'gain_db', 0.0))
            effective_lufs = max(-23.0, min(-5.0, self.target_lufs + gain_offset))
            loudnorm_filter = self.loudness_meter.get_loudnorm_filter(
                analysis, target_lufs=effective_lufs, target_tp=self.target_tp)

            cmd2 = [self.ffmpeg_path, "-y", "-i", temp_file, "-af", loudnorm_filter]
            ext = os.path.splitext(output_path)[1].lower()
            if ext in [".wav", ".wave"]:
                cmd2.extend(["-c:a", "pcm_s24le"])
            elif ext in [".flac"]:
                cmd2.extend(["-c:a", "flac"])
            elif ext in [".mp3"]:
                cmd2.extend(["-c:a", "libmp3lame", "-b:a", "320k"])
            elif ext in [".aac", ".m4a"]:
                cmd2.extend(["-c:a", "aac", "-b:a", "256k"])
            elif ext in [".ogg"]:
                cmd2.extend(["-c:a", "libvorbis", "-q:a", "8"])
            else:
                cmd2.extend(["-c:a", "pcm_s24le"])
            cmd2.append(output_path)

            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
            if result2.returncode != 0:
                shutil.move(temp_file, output_path)
                if callback:
                    callback(100, "Mastering complete (loudnorm skipped)")
                return output_path

            if callback:
                callback(90, "Verifying output...")
            self.output_analysis = self.loudness_meter.analyze(output_path)
            self.input_analysis = self.loudness_meter.analyze(self.input_path)

            if callback:
                callback(100, "Mastering complete!")
            return output_path

        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    # ─── A/B Comparison ───

    def get_ab_comparison(self) -> Optional[dict]:
        """
        Get Before/After comparison data.
        Returns dict with before/after LUFS, TP, LRA or None if not available.

        V5.3 FIX: Added missing method that was called by UI but didn't exist.
        """
        if not self.input_analysis or not self.output_analysis:
            return None

        try:
            before = {
                "lufs": self.input_analysis.integrated_lufs,
                "true_peak": self.input_analysis.true_peak_dbtp,
                "lra": self.input_analysis.lra,
            }
            after = {
                "lufs": self.output_analysis.integrated_lufs,
                "true_peak": self.output_analysis.true_peak_dbtp,
                "lra": self.output_analysis.lra,
            }
            delta = {
                "lufs": after["lufs"] - before["lufs"],
                "true_peak": after["true_peak"] - before["true_peak"],
                "lra": after["lra"] - before["lra"],
            }
            return {"before": before, "after": after, "delta": delta}
        except Exception as e:
            print(f"[CHAIN] A/B comparison error: {e}")
            return None

    # ─── Settings Save/Load (unchanged) ───

    def save_settings(self, filepath: str):
        """Save all module settings to JSON file."""
        data = {
            "version": "5.9",
            "chain": {
                "intensity": self.intensity,
                "target_lufs": self.target_lufs,
                "target_tp": self.target_tp,
                "normalize_loudness": self.normalize_loudness,
                "platform": self.platform,
            },
            "resonance_suppressor": self.resonance_suppressor.get_settings_dict(),
            "equalizer": self.equalizer.get_settings_dict(),
            "dynamics": self.dynamics.get_settings_dict(),
            "imager": self.imager.get_settings_dict(),
            "maximizer": self.maximizer.get_settings_dict(),
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[CHAIN] Settings saved to {filepath}")
        except (OSError, IOError) as e:
            print(f"[CHAIN] Failed to save settings: {e}")

    def load_settings(self, filepath: str) -> bool:
        """Load settings from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chain_data = data.get("chain", {})
            self.intensity = chain_data.get("intensity", 50)
            self.target_lufs = chain_data.get("target_lufs", -14.0)
            self.target_tp = chain_data.get("target_tp", -1.0)
            self.normalize_loudness = chain_data.get("normalize_loudness", True)
            self.platform = chain_data.get("platform", "YouTube")
            if "equalizer" in data:
                self.equalizer.load_settings_dict(data["equalizer"])
            if "resonance_suppressor" in data:
                self.resonance_suppressor.load_settings_dict(data["resonance_suppressor"])
            if "dynamics" in data:
                self.dynamics.load_settings_dict(data["dynamics"])
            if "imager" in data:
                self.imager.load_settings_dict(data["imager"])
            if "maximizer" in data:
                self.maximizer.load_settings_dict(data["maximizer"])
            print(f"[CHAIN] Settings loaded from {filepath}")
            return True
        except Exception as e:
            print(f"[CHAIN] Load error: {e}")
            return False

    def reset_all(self):
        """Reset all modules to default settings."""
        self.resonance_suppressor = ResonanceSuppressor()
        self.equalizer = Equalizer()
        self.dynamics = Dynamics()
        self.imager = Imager()
        self.maximizer = Maximizer()
        self.intensity = 50
        self.normalize_loudness = True
        self.recommendation = None

    def get_chain_summary(self) -> str:
        """Get human-readable summary of current chain settings."""
        engine_type = "Real DSP" if self._use_real_processing else "FFmpeg"
        lines = [
            f"=== Master Chain ({engine_type}) ===",
            f"Platform: {self.platform} (Target: {self.target_lufs} LUFS, {self.target_tp} dBTP)",
            f"Intensity: {self.intensity}%",
            f"",
            f"EQ: {self.equalizer}",
            f"Dynamics: {self.dynamics}",
            f"Imager: {self.imager}",
            f"Maximizer: {self.maximizer}",
            f"Loudness Normalize: {'ON' if self.normalize_loudness else 'OFF'}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        engine = "RealDSP" if self._use_real_processing else "FFmpeg"
        return (
            f"MasterChain(engine={engine}, platform={self.platform}, "
            f"intensity={self.intensity}%, "
            f"modules=[EQ={'ON' if self.equalizer.enabled else 'OFF'}, "
            f"Dyn={'ON' if self.dynamics.enabled else 'OFF'}, "
            f"Img={'ON' if self.imager.enabled else 'OFF'}, "
            f"Max={'ON' if self.maximizer.enabled else 'OFF'}])"
        )
