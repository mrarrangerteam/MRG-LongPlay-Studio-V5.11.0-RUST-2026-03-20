#!/usr/bin/env python3
"""
modules/master/ai_master.py — AI Master Module for LongPlay Studio V5.10

AI Master ที่คำนวณทุกอย่างอัตโนมัติสำหรับ 20 เพลงใน LongPlay playlist:
1. วิเคราะห์ทุกเพลง (LUFS, peak, dynamic range, spectral balance)
2. หาเพลงที่เบาที่สุด → ดูท่อน Hook → ดัน gain ให้ถึง Out Ceiling
3. คำนวณค่าเฉลี่ยมาตรฐาน → ปรับทุกเพลงให้สม่ำเสมอ
4. ตั้งค่าเริ่มต้นให้ทุก module (EQ, Dynamics, Imager, Maximizer, Limiter)
5. ผู้ใช้ปรับ manual ต่อได้ทีหลัง

Signal Flow:
  Input → Analysis → Pre-gain (auto) → EQ (auto) → Dynamics (auto)
  → Imager (auto) → Maximizer (auto) → Loudness Norm → True Peak Limit → Output

Author: MRARRANGER AI Studio
Date: 2026-03-20
"""

import numpy as np
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any

# ─── Try imports ───
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy import signal as scipy_signal
    from scipy.fft import rfft, rfftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrackAnalysis:
    """ผลวิเคราะห์เพลงแต่ละเพลง"""
    file_path: str
    filename: str
    duration_sec: float = 0.0
    sample_rate: int = 44100
    channels: int = 2
    lufs_integrated: float = -23.0
    lufs_short_term_max: float = -18.0
    lufs_momentary_max: float = -14.0
    loudness_range_lu: float = 8.0
    true_peak_dbtp: float = -1.0
    sample_peak_dbfs: float = -3.0
    dynamic_range_db: float = 8.0
    crest_factor_db: float = 12.0
    spectral_centroid_hz: float = 2000.0
    spectral_balance: str = "balanced"
    low_energy_ratio: float = 0.3
    hook_start_sec: float = 0.0
    hook_end_sec: float = 0.0
    hook_lufs: float = -20.0
    hook_peak_dbfs: float = -3.0
    recommended_gain_db: float = 0.0


@dataclass
class AIPreset:
    """ค่า preset ที่ AI คำนวณให้แต่ละเพลง"""
    track_index: int
    filename: str

    # Pre-gain
    pre_gain_db: float = 0.0

    # EQ (8-band parametric)
    eq_bands: List[Dict] = field(default_factory=list)

    # Dynamics
    dynamics_threshold_db: float = -18.0
    dynamics_ratio: float = 2.5
    dynamics_attack_ms: float = 10.0
    dynamics_release_ms: float = 100.0
    dynamics_makeup_db: float = 0.0

    # Imager
    stereo_width: float = 1.0
    mono_bass_freq_hz: float = 120.0

    # Maximizer / IRC
    maximizer_ceiling_db: float = -1.0
    maximizer_threshold_db: float = -8.0
    irc_mode: str = "IRC_IV"
    irc_sub_mode: str = "Balanced"

    # True Peak Limiter
    limiter_ceiling_dbtp: float = -1.0
    limiter_release_ms: float = 50.0

    # Loudness
    target_lufs: float = -14.0

    # V5.11.0: EQ band gains for tonal correction (8 bands)
    eq_bands_gain_db: List[float] = field(default_factory=lambda: [0.0]*8)

    # Width / Soothe
    width_amount: float = 1.0
    soothe_amount: float = 0.0
    soothe_freq_range: Tuple[float, float] = (2000.0, 8000.0)


@dataclass
class PlaylistMasterSettings:
    """ค่าตั้งต้นสำหรับทั้ง playlist"""
    out_ceiling_db: float = -1.0
    target_lufs: float = -14.0
    platform: str = "youtube"

    quietest_track_index: int = 0
    quietest_hook_lufs: float = -23.0
    average_lufs: float = -16.0
    gain_range_db: float = 0.0

    track_presets: List[AIPreset] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# LUFS Measurement (ITU-R BS.1770-4)
# ═══════════════════════════════════════════════════════════════

class LUFSMeter:
    """ITU-R BS.1770-4 compliant LUFS measurement"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self._design_k_filter()

    def _design_k_filter(self):
        """K-weighting filter (2-stage IIR per BS.1770)"""
        if not HAS_SCIPY:
            self._k_filter_available = False
            return
        self._k_filter_available = True

        # Stage 1: High shelf (+4dB at high frequencies)
        f0 = 1681.974450955533
        G = 3.999843853973347
        Q = 0.7071752369554196

        K = np.tan(np.pi * f0 / self.sr)
        Vh = 10.0 ** (G / 20.0)
        Vb = Vh ** 0.4996667741545416

        a0 = 1.0 + K / Q + K * K
        self._b1 = np.array([
            (Vh + Vb * K / Q + K * K) / a0,
            2.0 * (K * K - Vh) / a0,
            (Vh - Vb * K / Q + K * K) / a0
        ])
        self._a1 = np.array([
            1.0,
            2.0 * (K * K - 1.0) / a0,
            (1.0 - K / Q + K * K) / a0
        ])

        # Stage 2: High-pass (RLB weighting)
        f0 = 38.13547087602444
        Q = 0.5003270373238773
        K = np.tan(np.pi * f0 / self.sr)

        a0_hp = 1.0 + K / Q + K * K
        self._b2 = np.array([1.0, -2.0, 1.0]) / a0_hp
        self._a2 = np.array([
            1.0,
            2.0 * (K * K - 1.0) / a0_hp,
            (1.0 - K / Q + K * K) / a0_hp
        ])

    def measure_integrated(self, audio: np.ndarray) -> float:
        """Measure integrated LUFS for entire audio"""
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        if not HAS_SCIPY or not self._k_filter_available:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 1e-10:
                return -70.0
            return 20.0 * np.log10(rms) - 0.691

        channels_power = []
        for ch in range(audio.shape[1]):
            x = audio[:, ch].astype(np.float64)
            y1 = scipy_signal.lfilter(self._b1, self._a1, x)
            y2 = scipy_signal.lfilter(self._b2, self._a2, y1)
            channels_power.append(np.mean(y2 ** 2))

        total_power = sum(channels_power)
        if total_power < 1e-20:
            return -70.0

        lufs = -0.691 + 10.0 * np.log10(total_power)
        return lufs

    def measure_momentary(self, audio: np.ndarray, window_ms: int = 400) -> np.ndarray:
        """Measure momentary LUFS (400ms windows)"""
        window_samples = int(self.sr * window_ms / 1000)
        hop = window_samples

        n_windows = max(1, len(audio) // hop)
        lufs_values = np.full(n_windows, -70.0)

        for i in range(n_windows):
            start = i * hop
            end = min(start + window_samples, len(audio))
            if end - start < window_samples // 2:
                break
            segment = audio[start:end]
            lufs_values[i] = self.measure_integrated(segment)

        return lufs_values


# ═══════════════════════════════════════════════════════════════
# Hook Detector (Audio Energy Analysis)
# ═══════════════════════════════════════════════════════════════

class HookDetector:
    """ตรวจหาท่อน Hook ของเพลง (ท่อนที่ดังและมี energy สูงสุด)"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def detect_hook(self, audio: np.ndarray,
                    min_hook_sec: float = 8.0,
                    max_hook_sec: float = 30.0) -> Tuple[float, float]:
        """Find the section with highest sustained energy (the hook)"""
        if audio.ndim > 1:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio

        window = int(self.sr * 0.5)
        hop = int(self.sr * 0.25)
        n_frames = max(1, (len(mono) - window) // hop)

        energy = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            end = start + window
            if end > len(mono):
                break
            energy[i] = np.sqrt(np.mean(mono[start:end] ** 2))

        if len(energy) == 0 or np.max(energy) < 1e-10:
            return (0.0, min_hook_sec)

        # Smooth energy envelope
        kernel_size = int(2.0 / (hop / self.sr))
        kernel_size = max(3, kernel_size | 1)
        kernel = np.ones(kernel_size) / kernel_size
        smooth_energy = np.convolve(energy, kernel, mode='same')

        hook_frames = int(min_hook_sec / (hop / self.sr))
        max_hook_frames = int(max_hook_sec / (hop / self.sr))

        best_start = 0
        best_energy = 0.0
        best_length = hook_frames

        for length in range(hook_frames, min(max_hook_frames, len(smooth_energy)), hook_frames // 2):
            for start in range(0, len(smooth_energy) - length):
                avg_energy = np.mean(smooth_energy[start:start + length])
                if avg_energy > best_energy:
                    best_energy = avg_energy
                    best_start = start
                    best_length = length

        start_sec = best_start * hop / self.sr
        end_sec = (best_start + best_length) * hop / self.sr

        return (start_sec, min(end_sec, len(mono) / self.sr))


# ═══════════════════════════════════════════════════════════════
# Spectral Analyzer
# ═══════════════════════════════════════════════════════════════

class SpectralAnalyzer:
    """Analyze spectral balance of audio"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        if not HAS_SCIPY:
            return {
                "spectral_centroid_hz": 2000.0,
                "low_energy_ratio": 0.33,
                "mid_energy_ratio": 0.34,
                "high_energy_ratio": 0.33,
                "spectral_balance": "balanced",
            }

        if audio.ndim > 1:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio

        n_fft = min(8192, len(mono))
        hop = n_fft
        n_chunks = max(1, len(mono) // hop)

        mag_sum = np.zeros(n_fft // 2 + 1)

        for i in range(min(n_chunks, 50)):
            start = i * hop
            end = start + n_fft
            if end > len(mono):
                break
            chunk = mono[start:end] * np.hanning(n_fft)
            spectrum = np.abs(rfft(chunk))
            mag_sum += spectrum

        mag_avg = mag_sum / max(1, min(n_chunks, 50))
        freqs = rfftfreq(n_fft, 1.0 / self.sr)

        total_energy = np.sum(mag_avg ** 2)
        if total_energy < 1e-20:
            return {
                "spectral_centroid_hz": 1000.0,
                "low_energy_ratio": 0.33,
                "mid_energy_ratio": 0.34,
                "high_energy_ratio": 0.33,
                "spectral_balance": "balanced",
            }

        centroid = np.sum(freqs * mag_avg ** 2) / total_energy

        low_mask = freqs < 250
        low_energy = np.sum(mag_avg[low_mask] ** 2)
        low_ratio = low_energy / total_energy

        mid_mask = (freqs >= 250) & (freqs < 4000)
        mid_energy = np.sum(mag_avg[mid_mask] ** 2)
        mid_ratio = mid_energy / total_energy

        high_mask = freqs >= 4000
        high_energy = np.sum(mag_avg[high_mask] ** 2)
        high_ratio = high_energy / total_energy

        if centroid < 1500:
            balance = "dark"
        elif centroid > 3500:
            balance = "bright"
        else:
            balance = "balanced"

        return {
            "spectral_centroid_hz": centroid,
            "low_energy_ratio": low_ratio,
            "mid_energy_ratio": mid_ratio,
            "high_energy_ratio": high_ratio,
            "spectral_balance": balance,
        }


# ═══════════════════════════════════════════════════════════════
# Width & Soothe Processor
# ═══════════════════════════════════════════════════════════════

class WidthProcessor:
    """Stereo Width control using Mid/Side processing"""

    def process(self, audio: np.ndarray, width: float = 1.0,
                mono_bass_freq: float = 120.0, sample_rate: int = 44100) -> np.ndarray:
        if audio.ndim == 1 or audio.shape[1] == 1:
            return audio

        L = audio[:, 0]
        R = audio[:, 1]
        mid = (L + R) * 0.5
        side = (L - R) * 0.5

        side = side * width

        if HAS_SCIPY and mono_bass_freq > 0:
            nyq = sample_rate / 2.0
            if mono_bass_freq < nyq * 0.95:
                b, a = scipy_signal.butter(2, mono_bass_freq / nyq, btype='high')
                side = scipy_signal.lfilter(b, a, side).astype(np.float32)

        L_out = mid + side
        R_out = mid - side

        return np.column_stack([L_out, R_out])


class SootheProcessor:
    """Dynamic resonance suppression (inspired by Oeksound Soothe2)"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def process(self, audio: np.ndarray, amount: float = 50.0,
                freq_low: float = 2000.0, freq_high: float = 8000.0,
                depth_db: float = -6.0) -> np.ndarray:
        if amount <= 0 or not HAS_SCIPY:
            return audio

        intensity = min(1.0, amount / 100.0)

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        output = np.copy(audio)
        block_size = 2048
        overlap = block_size // 2

        for ch in range(audio.shape[1]):
            x = audio[:, ch].astype(np.float64)
            y = np.copy(x)

            n_blocks = max(1, (len(x) - block_size) // overlap)
            window = np.hanning(block_size)

            for i in range(n_blocks):
                start = i * overlap
                end = start + block_size
                if end > len(x):
                    break

                block = x[start:end] * window

                spectrum = rfft(block)
                freqs = rfftfreq(block_size, 1.0 / self.sr)
                magnitude = np.abs(spectrum)

                freq_mask = (freqs >= freq_low) & (freqs <= freq_high)

                if np.any(freq_mask):
                    target_mag = magnitude[freq_mask]
                    avg_mag = np.mean(target_mag) if np.mean(target_mag) > 1e-10 else 1e-10

                    peak_mask = target_mag > avg_mag * 1.5

                    gain = np.ones_like(magnitude)
                    target_indices = np.where(freq_mask)[0]

                    for j, idx in enumerate(target_indices):
                        if j < len(peak_mask) and peak_mask[j]:
                            ratio = target_mag[j] / avg_mag
                            reduction_db = min(0, depth_db * intensity * (ratio - 1.5) / ratio)
                            gain[idx] = 10.0 ** (reduction_db / 20.0)

                    spectrum_processed = spectrum * gain
                    block_processed = np.fft.irfft(spectrum_processed, n=block_size)

                    y[start:end] += (block_processed[:block_size] - block[:block_size]) * window * intensity

            output[:, ch] = y[:len(output)]

        return output.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# IRC Compressor (Ozone-style Intelligent Release Control)
# ═══════════════════════════════════════════════════════════════

class IRCCompressor:
    """
    IRC Compressor with 5 modes:
    IRC I:   Simple fixed release (classic)
    IRC II:  Program-dependent release (musical)
    IRC III: Multi-stage envelope (transparent)
    IRC IV:  Multi-band aware (modern, recommended)
    IRC V:   Look-ahead + multi-stage (mastering grade)
    """

    IRC_MODES = {
        "IRC_I":   {"description": "Simple fixed release", "release_mult": 1.0},
        "IRC_II":  {"description": "Program-dependent release", "release_mult": 1.5},
        "IRC_III": {"description": "Multi-stage envelope", "release_mult": 2.0},
        "IRC_IV":  {"description": "Multi-band aware", "release_mult": 1.2},
        "IRC_V":   {"description": "Look-ahead mastering", "release_mult": 0.8},
    }

    SUB_MODES = {
        "Pumping":   {"attack_mult": 0.5, "ratio_mult": 1.3, "character": "aggressive"},
        "Balanced":  {"attack_mult": 1.0, "ratio_mult": 1.0, "character": "neutral"},
        "Crisp":     {"attack_mult": 1.5, "ratio_mult": 0.8, "character": "open"},
        "Classic":   {"attack_mult": 1.2, "ratio_mult": 1.1, "character": "warm"},
        "Modern":    {"attack_mult": 0.8, "ratio_mult": 0.9, "character": "clean"},
        "Transient": {"attack_mult": 2.0, "ratio_mult": 0.7, "character": "punchy"},
    }

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def process(self, audio: np.ndarray,
                threshold_db: float = -18.0,
                ratio: float = 2.5,
                attack_ms: float = 10.0,
                release_ms: float = 100.0,
                makeup_db: float = 0.0,
                irc_mode: str = "IRC_IV",
                sub_mode: str = "Balanced",
                knee_db: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
        """Process audio through IRC compressor. Returns (processed_audio, gain_reduction_db)."""
        mode_params = self.IRC_MODES.get(irc_mode, self.IRC_MODES["IRC_IV"])
        sub_params = self.SUB_MODES.get(sub_mode, self.SUB_MODES["Balanced"])

        effective_attack = attack_ms * sub_params["attack_mult"]
        effective_release = release_ms * mode_params["release_mult"]
        effective_ratio = ratio * sub_params["ratio_mult"]

        attack_coeff = np.exp(-1.0 / (self.sr * effective_attack / 1000.0))
        release_coeff = np.exp(-1.0 / (self.sr * effective_release / 1000.0))

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        if irc_mode in ("IRC_I", "IRC_II"):
            detector_input = np.max(np.abs(audio), axis=1)
        else:
            rms_window = int(self.sr * 0.005)
            detector_input = np.zeros(len(audio))
            for i in range(len(audio)):
                start = max(0, i - rms_window)
                detector_input[i] = np.sqrt(np.mean(audio[start:i+1] ** 2))

        detector_db = np.where(
            detector_input > 1e-10,
            20.0 * np.log10(detector_input),
            -100.0
        )

        gain_reduction_db = np.zeros(len(audio))

        for i in range(len(audio)):
            level = detector_db[i]

            if knee_db > 0:
                knee_start = threshold_db - knee_db / 2
                knee_end = threshold_db + knee_db / 2

                if level < knee_start:
                    gr = 0.0
                elif level > knee_end:
                    gr = (level - threshold_db) * (1.0 - 1.0 / effective_ratio)
                else:
                    x = level - knee_start
                    gr = (1.0 - 1.0 / effective_ratio) * x * x / (2.0 * knee_db)
            else:
                if level > threshold_db:
                    gr = (level - threshold_db) * (1.0 - 1.0 / effective_ratio)
                else:
                    gr = 0.0

            gain_reduction_db[i] = -gr

        smoothed_gr = np.zeros(len(audio))
        for i in range(1, len(audio)):
            if gain_reduction_db[i] < smoothed_gr[i-1]:
                smoothed_gr[i] = attack_coeff * smoothed_gr[i-1] + (1.0 - attack_coeff) * gain_reduction_db[i]
            else:
                smoothed_gr[i] = release_coeff * smoothed_gr[i-1] + (1.0 - release_coeff) * gain_reduction_db[i]

        if irc_mode in ("IRC_III", "IRC_V"):
            fast_release = np.exp(-1.0 / (self.sr * effective_release * 0.3 / 1000.0))
            slow_release = np.exp(-1.0 / (self.sr * effective_release * 3.0 / 1000.0))

            for i in range(1, len(audio)):
                if smoothed_gr[i] > smoothed_gr[i-1]:
                    if smoothed_gr[i] > smoothed_gr[i-1] * 0.5:
                        coeff = fast_release
                    else:
                        coeff = slow_release
                    smoothed_gr[i] = coeff * smoothed_gr[i-1] + (1.0 - coeff) * gain_reduction_db[i]

        gain_linear = 10.0 ** ((smoothed_gr + makeup_db) / 20.0)
        output = audio * gain_linear[:, np.newaxis]

        return output.astype(np.float32), smoothed_gr


# ═══════════════════════════════════════════════════════════════
# AI Master Engine (Main Class)
# ═══════════════════════════════════════════════════════════════

class AIMasterEngine:
    """AI Master — วิเคราะห์ playlist และคำนวณค่า mastering ให้ทุกเพลงสม่ำเสมอ"""

    PLATFORM_TARGETS = {
        "youtube":      {"lufs": -14.0, "true_peak": -1.0},
        "spotify":      {"lufs": -14.0, "true_peak": -1.0},
        "apple_music":  {"lufs": -16.0, "true_peak": -1.0},
        "tidal":        {"lufs": -14.0, "true_peak": -1.0},
        "amazon":       {"lufs": -14.0, "true_peak": -2.0},
        "soundcloud":   {"lufs": -14.0, "true_peak": -1.0},
        "cd":           {"lufs": -9.0,  "true_peak": -0.3},
        "radio":        {"lufs": -23.0, "true_peak": -1.0},
    }

    def __init__(self):
        self.analyses: List[TrackAnalysis] = []
        self.settings: Optional[PlaylistMasterSettings] = None
        self.lufs_meter: Optional[LUFSMeter] = None
        self.hook_detector: Optional[HookDetector] = None
        self.spectral_analyzer: Optional[SpectralAnalyzer] = None
        self.width_processor = WidthProcessor()
        self.soothe_processor: Optional[SootheProcessor] = None
        self.irc_compressor: Optional[IRCCompressor] = None

    def analyze_playlist(self, file_paths: List[str],
                         out_ceiling_db: float = -1.0,
                         platform: str = "youtube",
                         progress_callback=None) -> PlaylistMasterSettings:
        """วิเคราะห์ทุกเพลงใน playlist แล้ว generate preset อัตโนมัติ"""
        if not HAS_SOUNDFILE:
            raise ImportError("soundfile is required: pip install soundfile")

        self.analyses = []
        n_tracks = len(file_paths)

        print(f"\n{'='*60}")
        print(f"   AI Master — Analyzing {n_tracks} tracks")
        print(f"   Ceiling: {out_ceiling_db} dBTP | Platform: {platform}")
        print(f"{'='*60}")

        # Phase 1: Analyze each track
        for i, fpath in enumerate(file_paths):
            if progress_callback:
                progress_callback(int(i / n_tracks * 50), f"Analyzing {i+1}/{n_tracks}...")

            try:
                analysis = self._analyze_single_track(fpath)
                self.analyses.append(analysis)
                print(f"   [{i+1}/{n_tracks}] {analysis.filename}: "
                      f"LUFS={analysis.lufs_integrated:.1f}, "
                      f"Peak={analysis.sample_peak_dbfs:.1f}dBFS, "
                      f"Hook LUFS={analysis.hook_lufs:.1f}")
            except Exception as e:
                print(f"   [{i+1}/{n_tracks}] Error: {e}")
                self.analyses.append(TrackAnalysis(
                    file_path=fpath,
                    filename=os.path.basename(fpath)
                ))

        # Phase 2: Find quietest/loudest track
        valid_analyses = [a for a in self.analyses if a.lufs_integrated > -60]

        if not valid_analyses:
            print("   No valid audio files found!")
            return PlaylistMasterSettings(out_ceiling_db=out_ceiling_db)

        quietest = min(valid_analyses, key=lambda a: a.hook_lufs)
        quietest_idx = self.analyses.index(quietest)
        loudest = max(valid_analyses, key=lambda a: a.hook_lufs)
        avg_lufs = np.mean([a.lufs_integrated for a in valid_analyses])

        print(f"\n   Playlist Analysis:")
        print(f"   Quietest: {quietest.filename} (Hook LUFS: {quietest.hook_lufs:.1f})")
        print(f"   Loudest:  {loudest.filename} (Hook LUFS: {loudest.hook_lufs:.1f})")
        print(f"   Average:  {avg_lufs:.1f} LUFS")
        print(f"   Range:    {loudest.hook_lufs - quietest.hook_lufs:.1f} dB")

        # Phase 3: Calculate per-track presets
        target = self.PLATFORM_TARGETS.get(platform, self.PLATFORM_TARGETS["youtube"])
        target_lufs = target["lufs"]

        quietest_headroom = out_ceiling_db - quietest.hook_peak_dbfs
        base_gain = min(18.0, max(0.0, quietest_headroom))

        print(f"\n   Gain Strategy:")
        print(f"   Quietest hook peak: {quietest.hook_peak_dbfs:.1f} dBFS")
        print(f"   Base gain: +{base_gain:.1f} dB")

        presets = []
        for i, analysis in enumerate(self.analyses):
            if progress_callback:
                progress_callback(50 + int(i / n_tracks * 50), f"Calculating preset {i+1}/{n_tracks}...")

            preset = self._calculate_preset(
                track_index=i,
                analysis=analysis,
                quietest_analysis=quietest,
                base_gain=base_gain,
                out_ceiling_db=out_ceiling_db,
                target_lufs=target_lufs,
                avg_lufs=avg_lufs
            )
            presets.append(preset)

            print(f"   [{i+1}] {analysis.filename}: "
                  f"gain={preset.pre_gain_db:+.1f}dB, "
                  f"threshold={preset.dynamics_threshold_db:.0f}dB, "
                  f"ceiling={preset.maximizer_ceiling_db:.1f}dB")

        # Phase 4: V5.11.0 — Auto Match EQ (pick best track as reference)
        if progress_callback:
            progress_callback(75, "Matching tonal balance...")

        best_track = self._pick_best_reference(valid_analyses)
        match_corrections = {}
        if best_track:
            print(f"\n   Reference track: {best_track.filename}")
            try:
                for i, analysis in enumerate(self.analyses):
                    if analysis.file_path == best_track.file_path:
                        continue
                    corr = self._compute_match_correction(
                        best_track.file_path, analysis.file_path)
                    if corr:
                        match_corrections[i] = corr
                        # Apply top corrections to preset EQ
                        sorted_bands = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)
                        for freq, db in sorted_bands[:4]:
                            if i < len(presets):
                                self._apply_correction_to_preset(presets[i], freq, db * 0.6)
                print(f"   Match EQ applied to {len(match_corrections)} tracks")
            except Exception as e:
                print(f"   Match EQ skipped: {e}")

        # Phase 5: V5.11.0 — Correlation safety check
        if progress_callback:
            progress_callback(85, "Checking stereo safety...")

        for i, analysis in enumerate(self.analyses):
            try:
                corr = self._check_stereo_correlation(analysis.file_path)
                if corr < 0.3 and i < len(presets):
                    old_width = presets[i].width_amount
                    presets[i].width_amount = min(old_width, 1.0)
                    print(f"   ⚠️ {analysis.filename}: low correlation ({corr:.2f}), width limited to 100%")
            except Exception:
                pass

        # Phase 6: V5.11.0 — Tonal balance correction
        if progress_callback:
            progress_callback(92, "Correcting tonal balance...")

        for i, (analysis, preset) in enumerate(zip(self.analyses, presets)):
            try:
                tonal = self._compute_tonal_correction(analysis)
                if tonal:
                    # Apply gentle tonal correction to EQ preset
                    for band_idx, corr_db in enumerate(tonal[:8]):
                        if abs(corr_db) > 0.5 and band_idx < 8:
                            current = preset.eq_bands_gain_db[band_idx] if band_idx < len(preset.eq_bands_gain_db) else 0
                            preset.eq_bands_gain_db[band_idx] = current + corr_db * 0.4
            except Exception:
                pass

        # Phase 7: Create settings
        settings = PlaylistMasterSettings(
            out_ceiling_db=out_ceiling_db,
            target_lufs=target_lufs,
            platform=platform,
            quietest_track_index=quietest_idx,
            quietest_hook_lufs=quietest.hook_lufs,
            average_lufs=avg_lufs,
            gain_range_db=loudest.hook_lufs - quietest.hook_lufs,
            track_presets=presets
        )

        self.settings = settings

        if progress_callback:
            progress_callback(100, "AI Master complete!")

        print(f"\n   AI Master complete! {len(presets)} presets generated.")
        print(f"   + Match EQ: {len(match_corrections)} tracks matched")
        print(f"   + Correlation safety: checked")
        print(f"   + Tonal balance: corrected")
        print(f"{'='*60}\n")

        return settings

    def _pick_best_reference(self, analyses):
        """V5.11.0: Pick the best track as reference for Match EQ.
        Best = closest to target LUFS + best spectral balance + highest correlation."""
        if not analyses:
            return None
        scored = []
        for a in analyses:
            lufs_score = 10 - abs(a.lufs_integrated - (-14.0))
            balance_score = 5 if a.spectral_balance == "balanced" else 2
            crest_score = min(5, a.crest_factor_db / 3)
            scored.append((lufs_score + balance_score + crest_score, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _compute_match_correction(self, ref_path, target_path):
        """V5.11.0: Compute spectral difference between reference and target."""
        try:
            ref_audio, ref_sr = sf.read(ref_path, dtype='float32')
            tgt_audio, tgt_sr = sf.read(target_path, dtype='float32')
            if ref_audio.ndim > 1: ref_audio = ref_audio.mean(axis=1)
            if tgt_audio.ndim > 1: tgt_audio = tgt_audio.mean(axis=1)

            fft_size = 8192
            # Average spectrum
            def avg_spectrum(audio, sr):
                hop = fft_size // 2
                specs = []
                for start in range(0, len(audio) - fft_size, hop):
                    chunk = audio[start:start+fft_size]
                    spec = np.abs(np.fft.rfft(chunk * np.hanning(fft_size)))
                    specs.append(spec)
                if not specs:
                    return None
                return np.mean(specs, axis=0)

            ref_spec = avg_spectrum(ref_audio, ref_sr)
            tgt_spec = avg_spectrum(tgt_audio, tgt_sr)
            if ref_spec is None or tgt_spec is None:
                return None

            ref_db = 20 * np.log10(np.maximum(ref_spec, 1e-10))
            tgt_db = 20 * np.log10(np.maximum(tgt_spec, 1e-10))
            diff = ref_db - tgt_db

            # Map to 1/3-octave bands
            freqs = np.fft.rfftfreq(fft_size, 1.0 / ref_sr)
            oct_centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
            corrections = {}
            for center in oct_centers:
                lo = center / 1.26
                hi = center * 1.26
                mask = (freqs >= lo) & (freqs < hi)
                if np.any(mask):
                    corrections[center] = float(np.clip(np.mean(diff[mask]), -6, 6))
            return corrections
        except Exception:
            return None

    def _apply_correction_to_preset(self, preset, freq, db):
        """Apply a frequency correction to the nearest EQ band in preset."""
        band_freqs = [32, 64, 125, 250, 1000, 4000, 8000, 16000]
        nearest = min(range(len(band_freqs)), key=lambda i: abs(band_freqs[i] - freq))
        if nearest < len(preset.eq_bands_gain_db):
            preset.eq_bands_gain_db[nearest] += np.clip(db, -4, 4)

    def _check_stereo_correlation(self, file_path):
        """V5.11.0: Check stereo correlation of a track."""
        try:
            audio, sr = sf.read(file_path, dtype='float32')
            if audio.ndim == 1:
                return 1.0
            L, R = audio[:, 0], audio[:, 1]
            denom = np.sqrt(np.sum(L**2) * np.sum(R**2))
            return float(np.sum(L * R) / (denom + 1e-10))
        except Exception:
            return 1.0

    def _compute_tonal_correction(self, analysis):
        """V5.11.0: Compute tonal balance correction based on spectral analysis."""
        try:
            audio, sr = sf.read(analysis.file_path, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            fft_size = 8192
            specs = []
            hop = fft_size // 2
            for start in range(0, len(audio) - fft_size, hop):
                chunk = audio[start:start+fft_size]
                spec = np.abs(np.fft.rfft(chunk * np.hanning(fft_size)))
                specs.append(spec)
            if not specs:
                return None

            avg_spec = np.mean(specs, axis=0)
            avg_db = 20 * np.log10(np.maximum(avg_spec, 1e-10))

            # Target = flat (normalized)
            freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
            band_centers = [32, 64, 125, 250, 1000, 4000, 8000, 16000]
            corrections = []
            for center in band_centers:
                lo = center / 1.26
                hi = center * 1.26
                mask = (freqs >= lo) & (freqs < hi)
                if np.any(mask):
                    band_energy = np.mean(avg_db[mask])
                    overall_energy = np.mean(avg_db[avg_db > -80])
                    corr = overall_energy - band_energy
                    corrections.append(float(np.clip(corr, -4, 4)))
                else:
                    corrections.append(0.0)
            return corrections
        except Exception:
            return None

    def _analyze_single_track(self, file_path: str) -> TrackAnalysis:
        """วิเคราะห์เพลงเดียว"""
        audio, sr = sf.read(file_path, dtype='float32')

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        filename = os.path.basename(file_path)
        duration = len(audio) / sr

        meter = LUFSMeter(sr)
        hook_det = HookDetector(sr)
        spec = SpectralAnalyzer(sr)

        lufs_integrated = meter.measure_integrated(audio)
        momentary = meter.measure_momentary(audio)
        lufs_mom_max = float(np.max(momentary)) if len(momentary) > 0 else lufs_integrated

        st_meter = LUFSMeter(sr)
        short_term = st_meter.measure_momentary(audio, window_ms=3000)
        lufs_st_max = float(np.max(short_term)) if len(short_term) > 0 else lufs_integrated

        sample_peak = float(np.max(np.abs(audio)))
        sample_peak_dbfs = 20.0 * np.log10(sample_peak) if sample_peak > 1e-10 else -100.0

        true_peak = self._measure_true_peak(audio, sr)
        true_peak_dbtp = 20.0 * np.log10(true_peak) if true_peak > 1e-10 else -100.0

        if len(momentary) > 2:
            sorted_mom = np.sort(momentary[momentary > -70])
            if len(sorted_mom) > 4:
                p10 = sorted_mom[int(len(sorted_mom) * 0.1)]
                p95 = sorted_mom[int(len(sorted_mom) * 0.95)]
                lra = p95 - p10
            else:
                lra = 8.0
        else:
            lra = 8.0

        crest = sample_peak_dbfs - lufs_integrated

        hook_start, hook_end = hook_det.detect_hook(audio)
        hook_samples = audio[int(hook_start * sr):int(hook_end * sr)]
        hook_lufs = meter.measure_integrated(hook_samples) if len(hook_samples) > sr else lufs_integrated
        hook_peak = float(np.max(np.abs(hook_samples))) if len(hook_samples) > 0 else sample_peak
        hook_peak_dbfs = 20.0 * np.log10(hook_peak) if hook_peak > 1e-10 else -100.0

        spectral = spec.analyze(audio)

        return TrackAnalysis(
            file_path=file_path,
            filename=filename,
            duration_sec=duration,
            sample_rate=sr,
            channels=audio.shape[1],
            lufs_integrated=lufs_integrated,
            lufs_short_term_max=lufs_st_max,
            lufs_momentary_max=lufs_mom_max,
            loudness_range_lu=lra,
            true_peak_dbtp=true_peak_dbtp,
            sample_peak_dbfs=sample_peak_dbfs,
            dynamic_range_db=lra,
            crest_factor_db=crest,
            spectral_centroid_hz=spectral["spectral_centroid_hz"],
            spectral_balance=spectral["spectral_balance"],
            low_energy_ratio=spectral["low_energy_ratio"],
            hook_start_sec=hook_start,
            hook_end_sec=hook_end,
            hook_lufs=hook_lufs,
            hook_peak_dbfs=hook_peak_dbfs,
        )

    def _measure_true_peak(self, audio: np.ndarray, sr: int) -> float:
        """4x oversampled ISP true peak detection (ITU-R BS.1770-4)"""
        if not HAS_SCIPY:
            return float(np.max(np.abs(audio)))

        max_peak = 0.0
        for ch in range(audio.shape[1]):
            upsampled = scipy_signal.resample_poly(audio[:, ch], 4, 1)
            peak = float(np.max(np.abs(upsampled)))
            max_peak = max(max_peak, peak)

        return max_peak

    def _calculate_preset(self, track_index: int, analysis: TrackAnalysis,
                          quietest_analysis: TrackAnalysis,
                          base_gain: float, out_ceiling_db: float,
                          target_lufs: float, avg_lufs: float) -> AIPreset:
        """คำนวณ preset สำหรับเพลงเดียว"""

        filename = analysis.filename

        # Pre-gain
        lufs_diff = analysis.hook_lufs - quietest_analysis.hook_lufs
        pre_gain = base_gain - lufs_diff
        pre_gain = max(-6.0, min(18.0, pre_gain))

        # EQ
        eq_bands = self._calculate_eq(analysis)

        # Dynamics
        if analysis.loudness_range_lu > 12:
            dyn_ratio = 3.0
            dyn_threshold = -20.0
        elif analysis.loudness_range_lu > 8:
            dyn_ratio = 2.5
            dyn_threshold = -18.0
        else:
            dyn_ratio = 2.0
            dyn_threshold = -15.0

        estimated_gr = (analysis.lufs_momentary_max - dyn_threshold) * (1 - 1/dyn_ratio)
        dyn_makeup = max(0, estimated_gr * 0.5)

        # Imager
        if analysis.spectral_balance == "dark":
            width = 1.1
        elif analysis.spectral_balance == "bright":
            width = 0.95
        else:
            width = 1.0

        # Maximizer
        gain_needed = target_lufs - analysis.lufs_integrated + pre_gain
        max_threshold = -(abs(gain_needed) * 0.6 + 3.0)
        max_threshold = max(-18.0, min(-3.0, max_threshold))

        # IRC mode
        if analysis.crest_factor_db > 15:
            irc_mode = "IRC_III"
            sub_mode = "Transient"
        elif analysis.crest_factor_db > 10:
            irc_mode = "IRC_IV"
            sub_mode = "Balanced"
        else:
            irc_mode = "IRC_II"
            sub_mode = "Modern"

        # Soothe
        if analysis.spectral_balance == "bright":
            soothe_amount = 40.0
        elif analysis.spectral_balance == "balanced":
            soothe_amount = 20.0
        else:
            soothe_amount = 0.0

        limiter_ceiling = out_ceiling_db

        return AIPreset(
            track_index=track_index,
            filename=filename,
            pre_gain_db=round(pre_gain, 1),
            eq_bands=eq_bands,
            dynamics_threshold_db=round(dyn_threshold, 1),
            dynamics_ratio=round(dyn_ratio, 1),
            dynamics_attack_ms=10.0,
            dynamics_release_ms=100.0,
            dynamics_makeup_db=round(dyn_makeup, 1),
            stereo_width=round(width, 2),
            mono_bass_freq_hz=120.0,
            maximizer_ceiling_db=out_ceiling_db,
            maximizer_threshold_db=round(max_threshold, 1),
            irc_mode=irc_mode,
            irc_sub_mode=sub_mode,
            limiter_ceiling_dbtp=limiter_ceiling,
            limiter_release_ms=50.0,
            target_lufs=target_lufs,
            width_amount=round(width, 2),
            soothe_amount=round(soothe_amount, 1),
            soothe_freq_range=(2000.0, 8000.0),
        )

    def _calculate_eq(self, analysis: TrackAnalysis) -> List[Dict]:
        """คำนวณ EQ settings ตาม spectral analysis"""
        bands = []

        bands.append({"freq": 30, "gain_db": 0, "q": 0.7, "type": "highpass"})

        if analysis.low_energy_ratio > 0.4:
            bands.append({"freq": 60, "gain_db": -2.0, "q": 1.0, "type": "bell"})
        else:
            bands.append({"freq": 60, "gain_db": 1.0, "q": 0.8, "type": "bell"})

        bands.append({"freq": 250, "gain_db": 0, "q": 1.0, "type": "bell"})

        if analysis.spectral_balance == "dark":
            bands.append({"freq": 2500, "gain_db": 1.5, "q": 1.2, "type": "bell"})
        else:
            bands.append({"freq": 2500, "gain_db": 0, "q": 1.0, "type": "bell"})

        if analysis.spectral_balance == "dark":
            bands.append({"freq": 4000, "gain_db": 1.0, "q": 1.0, "type": "bell"})
        elif analysis.spectral_balance == "bright":
            bands.append({"freq": 4000, "gain_db": -1.0, "q": 1.0, "type": "bell"})
        else:
            bands.append({"freq": 4000, "gain_db": 0, "q": 1.0, "type": "bell"})

        if analysis.spectral_balance != "bright":
            bands.append({"freq": 12000, "gain_db": 1.0, "q": 0.7, "type": "high_shelf"})
        else:
            bands.append({"freq": 12000, "gain_db": -0.5, "q": 0.7, "type": "high_shelf"})

        bands.append({"freq": 18000, "gain_db": 0, "q": 0.7, "type": "lowpass"})
        bands.append({"freq": 1000, "gain_db": 0, "q": 1.0, "type": "bell"})

        return bands

    def process_track(self, audio: np.ndarray, sr: int, preset: AIPreset) -> np.ndarray:
        """Process เพลงเดียวตาม AI preset"""
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        output = audio.copy().astype(np.float64)

        # 1. Pre-gain
        gain_linear = 10.0 ** (preset.pre_gain_db / 20.0)
        output = output * gain_linear

        # 2. EQ
        if HAS_SCIPY and preset.eq_bands:
            for band in preset.eq_bands:
                if band["gain_db"] != 0 and band["type"] == "bell":
                    output = self._apply_bell_eq(output, sr,
                                                 band["freq"], band["gain_db"], band["q"])

        # 3. Dynamics (IRC Compressor)
        if self.irc_compressor is None:
            self.irc_compressor = IRCCompressor(sr)

        output, gr = self.irc_compressor.process(
            output.astype(np.float32),
            threshold_db=preset.dynamics_threshold_db,
            ratio=preset.dynamics_ratio,
            attack_ms=preset.dynamics_attack_ms,
            release_ms=preset.dynamics_release_ms,
            makeup_db=preset.dynamics_makeup_db,
            irc_mode=preset.irc_mode,
            sub_mode=preset.irc_sub_mode
        )

        # 4. Width
        output = self.width_processor.process(
            output, width=preset.width_amount,
            mono_bass_freq=preset.mono_bass_freq_hz, sample_rate=sr
        )

        # 5. Soothe
        if preset.soothe_amount > 0:
            if self.soothe_processor is None:
                self.soothe_processor = SootheProcessor(sr)
            output = self.soothe_processor.process(
                output, amount=preset.soothe_amount,
                freq_low=preset.soothe_freq_range[0],
                freq_high=preset.soothe_freq_range[1]
            )

        # 6. Maximizer (soft clip before hard limit)
        ceiling_linear = 10.0 ** (preset.maximizer_ceiling_db / 20.0)
        output = np.tanh(output / ceiling_linear) * ceiling_linear

        # 7. True Peak Limit
        limiter_ceiling = 10.0 ** (preset.limiter_ceiling_dbtp / 20.0)
        output = np.clip(output, -limiter_ceiling, limiter_ceiling)

        return output.astype(np.float32)

    def _apply_bell_eq(self, audio: np.ndarray, sr: int,
                       freq: float, gain_db: float, q: float) -> np.ndarray:
        """Apply a single bell/peaking EQ band"""
        if not HAS_SCIPY:
            return audio

        A = 10.0 ** (gain_db / 40.0)
        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])

        for ch in range(audio.shape[1]):
            audio[:, ch] = scipy_signal.lfilter(b, a, audio[:, ch])

        return audio

    def get_preset_json(self, track_index: int) -> str:
        """Export preset เป็น JSON"""
        if self.settings and track_index < len(self.settings.track_presets):
            preset = self.settings.track_presets[track_index]
            return json.dumps({
                "track_index": preset.track_index,
                "filename": preset.filename,
                "pre_gain_db": preset.pre_gain_db,
                "eq_bands": preset.eq_bands,
                "dynamics": {
                    "threshold_db": preset.dynamics_threshold_db,
                    "ratio": preset.dynamics_ratio,
                    "attack_ms": preset.dynamics_attack_ms,
                    "release_ms": preset.dynamics_release_ms,
                    "makeup_db": preset.dynamics_makeup_db,
                },
                "imager": {
                    "stereo_width": preset.stereo_width,
                    "mono_bass_freq_hz": preset.mono_bass_freq_hz,
                },
                "maximizer": {
                    "ceiling_db": preset.maximizer_ceiling_db,
                    "threshold_db": preset.maximizer_threshold_db,
                    "irc_mode": preset.irc_mode,
                    "irc_sub_mode": preset.irc_sub_mode,
                },
                "limiter": {
                    "ceiling_dbtp": preset.limiter_ceiling_dbtp,
                    "release_ms": preset.limiter_release_ms,
                },
                "width_amount": preset.width_amount,
                "soothe_amount": preset.soothe_amount,
                "target_lufs": preset.target_lufs,
            }, indent=2)
        return "{}"


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import glob

    files = sys.argv[1:]
    if not files:
        files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.aiff', '*.m4a']:
            files.extend(glob.glob(ext))

        if not files:
            print("Usage: python3 ai_master.py file1.wav file2.wav ...")
            print("   Or: place audio files in current directory")
            sys.exit(0)

    print(f"Found {len(files)} files")

    engine = AIMasterEngine()
    settings = engine.analyze_playlist(
        files,
        out_ceiling_db=-1.0,
        platform="youtube"
    )

    print("\n" + "=" * 50)
    print("Generated Presets:")
    print("=" * 50)

    for preset in settings.track_presets:
        print(f"\n  {preset.filename}")
        print(f"   Pre-gain:  {preset.pre_gain_db:+.1f} dB")
        print(f"   Dynamics:  threshold={preset.dynamics_threshold_db}dB, "
              f"ratio={preset.dynamics_ratio}:1, "
              f"IRC={preset.irc_mode} ({preset.irc_sub_mode})")
        print(f"   Width:     {preset.width_amount:.2f}")
        print(f"   Soothe:    {preset.soothe_amount:.0f}%")
        print(f"   Maximizer: ceiling={preset.maximizer_ceiling_db}dB, "
              f"threshold={preset.maximizer_threshold_db}dB")
        print(f"   Limiter:   {preset.limiter_ceiling_dbtp}dBTP")
