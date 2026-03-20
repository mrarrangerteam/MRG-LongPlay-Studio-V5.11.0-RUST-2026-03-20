"""
MRG LongPlay Studio — Auto Resonance Suppressor (Soothe2-style)

Automatic detection and suppression of harsh resonances using
spectral analysis + dynamic multi-band notch filtering.

For beginners: just enable it and it automatically tames harshness.

Signal flow:
  Input → FFT analysis → Detect resonance peaks → Dynamic EQ attenuation → Output

Parameters:
  - depth: How much to cut resonances (0-20 dB)
  - sharpness: How narrow the cuts are (Q factor)
  - selectivity: How picky about what counts as resonance
  - speed: Attack/release for dynamic behavior
  - mode: soft (gentle) / hard (aggressive)
"""

import numpy as np
try:
    from scipy import signal as sig
except ImportError:
    sig = None


class ResonanceSuppressor:
    """Auto resonance suppressor — detects and attenuates harsh frequencies."""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.enabled = True

        # Parameters (Soothe2-style)
        self.depth = 5.0          # dB of max reduction (0-20)
        self.sharpness = 4.0      # Q factor for notch filters (1-10)
        self.selectivity = 3.5    # How selective (1=everything, 10=only worst)
        self.speed_attack = 5.0   # ms
        self.speed_release = 50.0 # ms
        self.mode = "soft"        # "soft" or "hard"
        self.mix = 1.0            # wet/dry (0-1)
        self.trim = 0.0           # output trim dB (-12 to +12)
        self.delta = False        # delta mode (hear only removed signal)

        # Internal state
        self._fft_size = 2048
        self._hop = self._fft_size // 4
        self._window = np.hanning(self._fft_size).astype(np.float32)
        self._prev_gains = None  # smoothed gain curve
        self._overlap_buf_l = np.zeros(self._fft_size, dtype=np.float32)
        self._overlap_buf_r = np.zeros(self._fft_size, dtype=np.float32)

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

    def _detect_resonances(self, spectrum: np.ndarray) -> np.ndarray:
        """Detect resonance peaks in magnitude spectrum.

        Returns a gain curve (0-1) where resonances are attenuated.
        """
        n_bins = len(spectrum)
        mag_db = 20 * np.log10(np.maximum(spectrum, 1e-10))

        # Compute local spectral envelope (smoothed version)
        kernel_size = max(3, int(n_bins * 0.02 * (11 - self.sharpness)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        envelope = np.convolve(mag_db, np.ones(kernel_size) / kernel_size, mode='same')

        # Resonances = where spectrum exceeds envelope by threshold
        threshold = 12.0 - self.selectivity  # selectivity 1→11dB, 10→2dB
        excess = np.maximum(0, mag_db - envelope - threshold)

        # Convert excess to gain reduction
        max_reduction = self.depth
        if self.mode == "hard":
            max_reduction *= 1.5

        reduction_db = np.minimum(excess * (max_reduction / 10.0), max_reduction)
        gain_linear = 10 ** (-reduction_db / 20.0)

        return gain_linear

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process stereo audio through resonance suppressor.

        audio: numpy array shape (n_samples,) or (n_samples, 2)
        Returns: processed audio same shape
        """
        if not self.enabled or self.depth < 0.1:
            return audio

        mono = audio.ndim == 1
        if mono:
            audio = np.column_stack([audio, audio])

        n_samples = len(audio)
        output = np.zeros_like(audio, dtype=np.float32)

        # Attack/release coefficients
        attack_coeff = np.exp(-1.0 / (self.speed_attack * self.sr / 1000.0))
        release_coeff = np.exp(-1.0 / (self.speed_release * self.sr / 1000.0))

        # Process in overlapping frames
        pos = 0
        while pos + self._fft_size <= n_samples:
            frame_l = audio[pos:pos + self._fft_size, 0] * self._window
            frame_r = audio[pos:pos + self._fft_size, 1] * self._window

            # FFT
            spec_l = np.fft.rfft(frame_l)
            spec_r = np.fft.rfft(frame_r)

            # Detect resonances from mid signal (L+R)
            mid_spec = np.abs(spec_l) + np.abs(spec_r)
            gain_curve = self._detect_resonances(mid_spec)

            # Smooth gain over time (attack/release)
            if self._prev_gains is None:
                self._prev_gains = gain_curve.copy()
            else:
                for i in range(len(gain_curve)):
                    if gain_curve[i] < self._prev_gains[i]:
                        # Attack (reducing)
                        self._prev_gains[i] = attack_coeff * self._prev_gains[i] + (1 - attack_coeff) * gain_curve[i]
                    else:
                        # Release (recovering)
                        self._prev_gains[i] = release_coeff * self._prev_gains[i] + (1 - release_coeff) * gain_curve[i]

            # Apply gain curve
            spec_l *= self._prev_gains
            spec_r *= self._prev_gains

            # IFFT
            out_l = np.fft.irfft(spec_l, n=self._fft_size).astype(np.float32) * self._window
            out_r = np.fft.irfft(spec_r, n=self._fft_size).astype(np.float32) * self._window

            # Overlap-add
            output[pos:pos + self._fft_size, 0] += out_l
            output[pos:pos + self._fft_size, 1] += out_r

            pos += self._hop

        # Handle remaining samples
        if pos < n_samples:
            output[pos:, :] = audio[pos:, :]

        # Normalize overlap-add gain (Hann window OLA factor)
        ola_gain = self._hop / (0.5 * self._fft_size)
        output *= ola_gain

        # Trim
        if abs(self.trim) > 0.01:
            output *= 10 ** (self.trim / 20.0)

        # Delta mode: output only the removed signal
        if self.delta:
            delta_signal = audio - output
            return delta_signal[:, 0] if mono else delta_signal

        # Wet/dry mix
        if self.mix < 1.0:
            output = self.mix * output + (1.0 - self.mix) * audio

        return output[:, 0] if mono else output

    def get_reduction_curve(self) -> np.ndarray:
        """Get current gain reduction curve for UI display."""
        if self._prev_gains is not None:
            return 20 * np.log10(np.maximum(self._prev_gains, 1e-10))
        return np.zeros(self._fft_size // 2 + 1)

    def reset(self):
        """Reset internal state."""
        self._prev_gains = None

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

    def get_ffmpeg_filters(self, intensity: float = 1.0) -> list:
        """No FFmpeg equivalent — spectral processing is real-DSP only."""
        return []


class AutoDynamicProcessor:
    """Automatic dynamic processor — analyzes audio and applies optimal compression.

    For beginners: just enable it and it automatically balances dynamics.
    Analyzes the audio's dynamic range and applies gentle compression
    to make quiet parts louder and loud parts controlled.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.enabled = True

        # Auto-detected parameters (filled by analyze())
        self.threshold = -20.0     # dBFS
        self.ratio = 2.0           # :1
        self.attack = 10.0         # ms
        self.release = 100.0       # ms
        self.makeup_gain = 0.0     # dB
        self.knee = 6.0            # dB (soft knee)

        # Internal
        self._envelope = 0.0
        self._target = "balanced"  # "gentle", "balanced", "aggressive"

    def set_target(self, target: str):
        self._target = target if target in ("gentle", "balanced", "aggressive") else "balanced"

    def analyze(self, audio: np.ndarray, sr: int = None):
        """Analyze audio and auto-set optimal compression parameters.

        Call this once before processing to detect dynamics.
        """
        if sr:
            self.sr = sr

        if audio.ndim > 1:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio

        # Compute RMS envelope in 50ms windows
        win_samples = int(self.sr * 0.05)
        n_windows = len(mono) // win_samples
        if n_windows < 2:
            return

        rms_values = []
        for i in range(n_windows):
            chunk = mono[i * win_samples:(i + 1) * win_samples]
            rms = np.sqrt(np.mean(chunk ** 2))
            if rms > 1e-10:
                rms_values.append(20 * np.log10(rms))

        if not rms_values:
            return

        rms_arr = np.array(rms_values)

        # Analyze dynamics
        peak_db = np.max(rms_arr)
        rms_mean = np.mean(rms_arr[rms_arr > -60])  # ignore silence
        rms_std = np.std(rms_arr[rms_arr > -60])
        dynamic_range = np.percentile(rms_arr[rms_arr > -60], 95) - np.percentile(rms_arr[rms_arr > -60], 5)

        # Auto-set parameters based on analysis and target
        if self._target == "gentle":
            self.threshold = rms_mean + rms_std * 0.5
            self.ratio = 1.5
            self.attack = 20.0
            self.release = 200.0
            self.knee = 10.0
        elif self._target == "balanced":
            self.threshold = rms_mean
            self.ratio = 2.5
            self.attack = 10.0
            self.release = 100.0
            self.knee = 6.0
        elif self._target == "aggressive":
            self.threshold = rms_mean - rms_std * 0.5
            self.ratio = 4.0
            self.attack = 5.0
            self.release = 50.0
            self.knee = 3.0

        # Makeup gain to compensate for compression
        avg_reduction = max(0, rms_mean - self.threshold) * (1 - 1 / self.ratio)
        self.makeup_gain = avg_reduction * 0.7  # 70% compensation

        print(f"[AUTO DYNAMICS] Analyzed: DR={dynamic_range:.1f}dB, "
              f"threshold={self.threshold:.1f}, ratio={self.ratio:.1f}:1, "
              f"makeup={self.makeup_gain:.1f}dB ({self._target})")

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply automatic compression to audio.

        audio: numpy array shape (n_samples,) or (n_samples, 2)
        """
        if not self.enabled:
            return audio

        mono_input = audio.ndim == 1
        if mono_input:
            audio = np.column_stack([audio, audio])

        output = audio.copy().astype(np.float64)
        n_samples = len(audio)

        # Envelope follower coefficients
        att_coeff = np.exp(-1.0 / (self.attack * self.sr / 1000.0))
        rel_coeff = np.exp(-1.0 / (self.release * self.sr / 1000.0))

        # Knee parameters
        half_knee = self.knee / 2.0

        envelope = self._envelope

        for i in range(n_samples):
            # Peak detection (stereo linked)
            sample_abs = max(abs(audio[i, 0]), abs(audio[i, 1]))
            if sample_abs < 1e-10:
                input_db = -100.0
            else:
                input_db = 20 * np.log10(sample_abs)

            # Envelope follower
            if input_db > envelope:
                envelope = att_coeff * envelope + (1 - att_coeff) * input_db
            else:
                envelope = rel_coeff * envelope + (1 - rel_coeff) * input_db

            # Gain computation with soft knee
            if envelope < self.threshold - half_knee:
                gain_db = 0.0
            elif envelope > self.threshold + half_knee:
                gain_db = (self.threshold + (envelope - self.threshold) / self.ratio) - envelope
            else:
                # Soft knee region
                x = envelope - self.threshold + half_knee
                gain_db = ((1 / self.ratio - 1) * x * x) / (2 * self.knee) if self.knee > 0 else 0.0

            # Apply gain + makeup
            total_gain = gain_db + self.makeup_gain
            gain_linear = 10 ** (total_gain / 20.0)

            output[i, 0] *= gain_linear
            output[i, 1] *= gain_linear

        self._envelope = envelope

        result = output.astype(np.float32)
        return result[:, 0] if mono_input else result

    def reset(self):
        self._envelope = 0.0
