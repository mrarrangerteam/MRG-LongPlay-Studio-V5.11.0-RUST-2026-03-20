"""
MRG LongPlay Studio V5.11.0 — Soothe2-Quality Resonance Suppressor

Per-bin spectral dynamic EQ that automatically detects and suppresses
harsh resonances. Modeled after oeksound Soothe2's algorithm:

Algorithm:
  Input → STFT (4096, 75% overlap, Hann) → Spectral Envelope →
  Resonance Detection (per-bin adaptive threshold) →
  Per-Bin Gain Calculation → Temporal Smoothing (attack/release) →
  Spectral Smoothing (sharpness) → Phase-Preserving Reconstruction →
  Overlap-Add → Output

Key features vs basic resonance suppressor:
  - Per-bin processing (1000+ frequency bins, not just a few bands)
  - Adaptive threshold from local spectral neighborhood (context-aware)
  - Temporal smoothing per-bin (attack/release controlled by "speed")
  - Spectral gain smoothing (controlled by "sharpness")
  - Phase preservation (transparent, no artifacts)
  - Sensitivity curve (user-adjustable per-frequency sensitivity)
  - Delta mode (hear only what's removed)
  - Mid/Side processing option

Author: MRARRANGER AI Studio
"""

import numpy as np
try:
    from scipy import signal as sig
    from scipy.ndimage import uniform_filter1d, gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ResonanceSuppressor:
    """Soothe2-quality per-bin spectral dynamic resonance suppressor."""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.enabled = True

        # ═══ User Parameters (Soothe2-style) ═══
        self.depth = 5.0            # Max reduction dB (0-20)
        self.sharpness = 4.0        # Spectral smoothing (1=broad, 10=surgical)
        self.selectivity = 3.5      # Detection threshold (1=everything, 10=only worst)
        self.speed_attack = 5.0     # Attack ms (0.5-50)
        self.speed_release = 50.0   # Release ms (5-500)
        self.mode = "soft"          # "soft" or "hard"
        self.mix = 1.0              # Wet/dry (0-1)
        self.trim = 0.0             # Output trim dB (-12 to +12)
        self.delta = False          # Delta mode (hear only removed signal)
        self.freq_low = 200.0       # Low frequency limit Hz
        self.freq_high = 12000.0    # High frequency limit Hz
        self.mid_side = "stereo"    # "stereo", "mid", "side"

        # ═══ Sensitivity curve (user-adjustable nodes) ═══
        # List of (freq_hz, sensitivity) tuples — sensitivity 0-2 (1=normal)
        self._sensitivity_nodes = [
            (200, 0.5), (1000, 1.0), (3000, 1.2), (5000, 1.3),
            (8000, 1.0), (12000, 0.7)
        ]

        # ═══ STFT Configuration ═══
        self._fft_size = 4096
        self._overlap = 4           # 75% overlap (4x)
        self._hop = self._fft_size // self._overlap
        self._window = np.hanning(self._fft_size).astype(np.float64)
        self._n_bins = self._fft_size // 2 + 1

        # ═══ Internal State (per-bin) ═══
        self._prev_gains = np.ones(self._n_bins, dtype=np.float64)
        self._reduction_db = np.zeros(self._n_bins, dtype=np.float64)
        self._overlap_buf = None    # Overlap-add buffers per channel
        self._freqs = np.fft.rfftfreq(self._fft_size, 1.0 / self.sr)

        # Pre-compute sensitivity curve at bin frequencies
        self._sensitivity_per_bin = self._interpolate_sensitivity()

    # ═══ Parameter Setters ═══

    def set_depth(self, db: float):
        self.depth = np.clip(db, 0.0, 20.0)

    def set_sharpness(self, val: float):
        self.sharpness = np.clip(val, 1.0, 10.0)

    def set_selectivity(self, val: float):
        self.selectivity = np.clip(val, 1.0, 10.0)

    def set_speed(self, attack_ms: float, release_ms: float):
        self.speed_attack = max(0.5, attack_ms)
        self.speed_release = max(5.0, release_ms)

    def set_mode(self, mode: str):
        self.mode = mode if mode in ("soft", "hard") else "soft"

    def set_trim(self, db: float):
        self.trim = np.clip(db, -12.0, 12.0)

    def set_delta(self, enabled: bool):
        self.delta = enabled

    def set_freq_range(self, low_hz: float, high_hz: float):
        self.freq_low = max(20, low_hz)
        self.freq_high = min(20000, high_hz)

    def set_sensitivity_nodes(self, nodes: list):
        """Set sensitivity curve nodes: [(freq_hz, sensitivity), ...]"""
        self._sensitivity_nodes = nodes
        self._sensitivity_per_bin = self._interpolate_sensitivity()

    # ═══ Sensitivity Curve Interpolation ═══

    def _interpolate_sensitivity(self) -> np.ndarray:
        """Interpolate user sensitivity nodes to per-bin curve."""
        if not self._sensitivity_nodes:
            return np.ones(self._n_bins)

        nodes = sorted(self._sensitivity_nodes, key=lambda x: x[0])
        node_freqs = [n[0] for n in nodes]
        node_vals = [n[1] for n in nodes]

        sensitivity = np.interp(
            self._freqs, node_freqs, node_vals,
            left=node_vals[0], right=node_vals[-1]
        )
        return np.clip(sensitivity, 0.0, 2.0)

    # ═══ Core: Resonance Detection (Per-Bin Adaptive Threshold) ═══

    def _detect_resonances_perbin(self, magnitude: np.ndarray) -> np.ndarray:
        """Soothe2-style per-bin resonance detection.

        For each bin:
          1. Compute local spectral average (neighborhood)
          2. Compare bin magnitude to local average
          3. If exceeds threshold → compute gain reduction
          4. Scale by sensitivity curve and depth

        Returns: gain_linear per bin (0-1, where <1 = reduction)
        """
        n = len(magnitude)
        mag_db = 20 * np.log10(np.maximum(magnitude, 1e-10))

        # Local spectral envelope — neighborhood size from sharpness
        # sharpness 1 → very wide neighborhood (broad detection)
        # sharpness 10 → narrow neighborhood (surgical detection)
        neighborhood = max(3, int(n * 0.05 * (11 - self.sharpness) / 10.0))
        if neighborhood % 2 == 0:
            neighborhood += 1

        if HAS_SCIPY:
            envelope = uniform_filter1d(mag_db, size=neighborhood)
        else:
            kernel = np.ones(neighborhood) / neighborhood
            envelope = np.convolve(mag_db, kernel, mode='same')

        # Adaptive threshold — selectivity controls how much excess triggers reduction
        # selectivity 1 → threshold 1 dB (everything is resonance)
        # selectivity 10 → threshold 10 dB (only extreme peaks)
        threshold_db = self.selectivity

        # Excess above local average
        excess_db = np.maximum(0, mag_db - envelope - threshold_db)

        # Scale by sensitivity curve (user-adjustable per frequency)
        excess_db *= self._sensitivity_per_bin

        # Frequency range masking — only process within freq_low..freq_high
        freq_mask = (self._freqs >= self.freq_low) & (self._freqs <= self.freq_high)
        excess_db *= freq_mask

        # Compute gain reduction
        max_red = self.depth
        if self.mode == "hard":
            max_red *= 1.5

        reduction_db = np.minimum(excess_db * (max_red / 6.0), max_red)

        # Spectral smoothing of the gain curve (sharpness)
        # Low sharpness → smooth the reduction curve (broad gentle cuts)
        # High sharpness → keep it sharp (surgical narrow cuts)
        if HAS_SCIPY:
            smooth_sigma = max(0.5, (10 - self.sharpness) * 2.0)
            reduction_db = gaussian_filter1d(reduction_db, sigma=smooth_sigma)
        else:
            smooth_n = max(1, int((10 - self.sharpness) * 3))
            if smooth_n > 1:
                kernel = np.ones(smooth_n) / smooth_n
                reduction_db = np.convolve(reduction_db, kernel, mode='same')

        # Convert to linear gain
        gain_linear = 10 ** (-reduction_db / 20.0)

        return gain_linear

    # ═══ Core: Process Audio ═══

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through Soothe2-style resonance suppressor.

        Uses STFT with 75% overlap, per-bin detection, phase preservation,
        and overlap-add reconstruction.

        audio: (n_samples,) or (n_samples, 2)
        Returns: processed audio, same shape
        """
        if not self.enabled or self.depth < 0.1:
            return audio

        mono = audio.ndim == 1
        if mono:
            audio = np.column_stack([audio, audio])

        n_samples, n_channels = audio.shape
        output = np.zeros_like(audio, dtype=np.float64)

        # Attack/release coefficients (per-bin temporal smoothing)
        att_coeff = np.exp(-1.0 / max(1, self.speed_attack * self.sr / 1000.0))
        rel_coeff = np.exp(-1.0 / max(1, self.speed_release * self.sr / 1000.0))

        # Mid/Side encoding if needed
        if self.mid_side == "mid":
            mid = (audio[:, 0] + audio[:, 1]) * 0.5
            side = (audio[:, 0] - audio[:, 1]) * 0.5
            process_signal = np.column_stack([mid, mid])  # Process mid only
            bypass_signal = np.column_stack([side, side])
        elif self.mid_side == "side":
            mid = (audio[:, 0] + audio[:, 1]) * 0.5
            side = (audio[:, 0] - audio[:, 1]) * 0.5
            process_signal = np.column_stack([side, side])
            bypass_signal = np.column_stack([mid, mid])
        else:
            process_signal = audio
            bypass_signal = None

        # Process in overlapping STFT frames
        pos = 0
        while pos + self._fft_size <= n_samples:
            # Windowed frames
            frames = []
            specs = []
            for ch in range(n_channels):
                frame = process_signal[pos:pos + self._fft_size, ch].astype(np.float64)
                frame *= self._window
                spec = np.fft.rfft(frame)
                frames.append(frame)
                specs.append(spec)

            # Detect resonances from combined magnitude (stereo-linked detection)
            combined_mag = np.zeros(self._n_bins)
            for spec in specs:
                combined_mag += np.abs(spec)
            combined_mag /= n_channels

            # Per-bin gain curve
            gain_curve = self._detect_resonances_perbin(combined_mag)

            # Temporal smoothing (per-bin attack/release)
            for i in range(self._n_bins):
                if gain_curve[i] < self._prev_gains[i]:
                    # Attack (gain going down = more reduction)
                    self._prev_gains[i] = (att_coeff * self._prev_gains[i] +
                                           (1 - att_coeff) * gain_curve[i])
                else:
                    # Release (gain going back up)
                    self._prev_gains[i] = (rel_coeff * self._prev_gains[i] +
                                           (1 - rel_coeff) * gain_curve[i])

            # Store reduction for UI display
            self._reduction_db = 20 * np.log10(np.maximum(self._prev_gains, 1e-10))

            # Apply gain to each channel (phase-preserving)
            for ch in range(n_channels):
                modified_spec = specs[ch] * self._prev_gains
                out_frame = np.fft.irfft(modified_spec, n=self._fft_size)
                out_frame *= self._window

                # Overlap-add
                output[pos:pos + self._fft_size, ch] += out_frame

            pos += self._hop

        # Handle remaining samples (pass through)
        if pos < n_samples:
            output[pos:] = process_signal[pos:]

        # Normalize for COLA (Constant Overlap-Add)
        # For Hann window with 4x overlap: normalization factor
        cola_norm = self._hop / (self._fft_size * 0.5)
        output *= cola_norm

        # Trim
        if abs(self.trim) > 0.01:
            output *= 10 ** (self.trim / 20.0)

        # Delta mode: output only the removed signal
        if self.delta:
            delta_signal = process_signal.astype(np.float64) - output
            return delta_signal[:, 0] if mono else delta_signal

        # Wet/dry mix
        if self.mix < 1.0:
            output = self.mix * output + (1.0 - self.mix) * process_signal.astype(np.float64)

        # Mid/Side decoding
        if self.mid_side == "mid":
            # Reconstruct from processed mid + original side
            mid_out = output[:, 0]
            side_orig = (audio[:, 0] - audio[:, 1]) * 0.5
            output[:, 0] = mid_out + side_orig
            output[:, 1] = mid_out - side_orig
        elif self.mid_side == "side":
            mid_orig = (audio[:, 0] + audio[:, 1]) * 0.5
            side_out = output[:, 0]
            output[:, 0] = mid_orig + side_out
            output[:, 1] = mid_orig - side_out

        result = output.astype(audio.dtype)
        return result[:, 0] if mono else result

    # ═══ UI Data ═══

    def get_reduction_curve(self) -> np.ndarray:
        """Get current per-bin gain reduction in dB for UI display.

        Returns array of dB values (negative = reduction).
        Can be downsampled to 64 bins for display.
        """
        return self._reduction_db.copy()

    def get_reduction_spectrum(self) -> np.ndarray:
        """Get reduction curve downsampled to 64 bins for popup meter display."""
        full = self._reduction_db
        if len(full) < 64:
            return full
        # Downsample by averaging bins
        indices = np.linspace(0, len(full) - 1, 64, dtype=int)
        return full[indices]

    def get_frequencies(self) -> np.ndarray:
        """Get frequency array matching the reduction curve."""
        return self._freqs.copy()

    def get_display_data(self) -> dict:
        """Get all display data for Soothe2-style meter panel."""
        return {
            "reduction_db": self._reduction_db.copy(),
            "freqs": self._freqs.copy(),
            "sensitivity": self._sensitivity_per_bin.copy(),
            "depth": self.depth,
            "freq_low": self.freq_low,
            "freq_high": self.freq_high,
            "mode": self.mode,
            "delta": self.delta,
        }

    def reset(self):
        """Reset internal state."""
        self._prev_gains = np.ones(self._n_bins, dtype=np.float64)
        self._reduction_db = np.zeros(self._n_bins, dtype=np.float64)

    # ═══ Settings Persistence ═══

    def get_settings_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "depth": self.depth,
            "sharpness": self.sharpness,
            "selectivity": self.selectivity,
            "speed_attack": self.speed_attack,
            "speed_release": self.speed_release,
            "mode": self.mode,
            "mix": self.mix,
            "trim": self.trim,
            "delta": self.delta,
            "freq_low": self.freq_low,
            "freq_high": self.freq_high,
            "mid_side": self.mid_side,
        }

    def load_settings_dict(self, d: dict):
        self.enabled = d.get("enabled", True)
        self.depth = d.get("depth", 5.0)
        self.sharpness = d.get("sharpness", 4.0)
        self.selectivity = d.get("selectivity", 3.5)
        self.speed_attack = d.get("speed_attack", 5.0)
        self.speed_release = d.get("speed_release", 50.0)
        self.mode = d.get("mode", "soft")
        self.mix = d.get("mix", 1.0)
        self.trim = d.get("trim", 0.0)
        self.delta = d.get("delta", False)
        self.freq_low = d.get("freq_low", 200.0)
        self.freq_high = d.get("freq_high", 12000.0)
        self.mid_side = d.get("mid_side", "stereo")

    def get_ffmpeg_filters(self, intensity: float = 1.0) -> list:
        """No FFmpeg equivalent — spectral processing is real-DSP only."""
        return []
