"""
modules/master/soothe.py — Spectral Dynamic Resonance Suppressor
Soothe 2-style algorithm by Oeksound

Key Parameters:
- Amount (0-100): resonance suppression intensity
- Speed (0-100): reaction speed (fast=reactive, slow=smooth)
- Smoothing (0-100): spectral smoothing of gain curve
- Low/Mid/High bands (0-200%): per-band sensitivity
- Shape/Cut mode: Shape=reduce peaks only, Cut=reduce entire band
- Tame Transients: reduce transient harshness
- Delta mode: hear only what's removed
"""

import numpy as np

try:
    from scipy.fft import rfft, irfft, rfftfreq
    from scipy.signal.windows import hann
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SootheProcessor:
    """Backward-compatible wrapper — delegates to Soothe2Processor."""

    def __init__(self, sample_rate=44100):
        self._impl = Soothe2Processor(sample_rate)
        self.sr = sample_rate
        self.enabled = False
        self.amount = 0.0
        self.freq_low = 2000.0
        self.freq_high = 8000.0
        self.depth_db = -6.0
        self.sensitivity = 1.5

    def set_params(self, amount=None, freq_low=None, freq_high=None, depth_db=None):
        if amount is not None:
            self.amount = max(0.0, min(100.0, amount))
            self._impl.amount = self.amount
        if freq_low is not None:
            self.freq_low = max(100, min(20000, freq_low))
            self._impl.crossover_low = int(self.freq_low)
        if freq_high is not None:
            self.freq_high = max(self.freq_low, min(20000, freq_high))
            self._impl.crossover_high = int(self.freq_high)

    def process(self, audio):
        if not self.enabled or self.amount <= 0:
            return audio
        self._impl.amount = self.amount
        return self._impl.process(audio)

    def get_settings_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "amount": self.amount,
            "freq_low": self.freq_low,
            "freq_high": self.freq_high,
            "depth_db": self.depth_db,
            "sensitivity": self.sensitivity,
        }

    def load_settings_dict(self, d: dict):
        for key in ["enabled", "amount", "freq_low", "freq_high",
                     "depth_db", "sensitivity"]:
            if key in d:
                setattr(self, key, d[key])

    def __repr__(self):
        return (f"SootheProcessor(amount={self.amount:.0f}%, "
                f"range={self.freq_low:.0f}-{self.freq_high:.0f}Hz)")


class Soothe2Processor:
    """Spectral Dynamic Resonance Suppressor (Soothe 2-style algorithm).

    Two-pass STFT: compute average spectrum, then dynamically reduce
    frequencies that exceed the average (resonances).
    """

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

        # Main controls
        self.amount = 25.0
        self.speed = 50.0
        self.smoothing = 50.0
        self.mode = "shape"       # "shape" or "cut"
        self.tame_transients = False
        self.delta = False

        # 3-band sensitivity
        self.band_sensitivity = {"low": 100, "mid": 100, "high": 100}
        self.crossover_low = 500
        self.crossover_high = 5000

        # Internal
        self._fft_size = 4096
        self._hop = self._fft_size // 4
        self._window = None
        self._prev_gain = None

    def set_params(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            elif key in self.band_sensitivity:
                self.band_sensitivity[key] = max(0, min(200, val))

    def process(self, audio):
        """Process audio through Soothe 2-style spectral dynamic algorithm."""
        if not HAS_SCIPY or self.amount <= 0:
            return audio

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        output = np.zeros_like(audio, dtype=np.float64)

        fft_size = self._fft_size
        hop = self._hop

        if self._window is None:
            self._window = hann(fft_size, sym=False).astype(np.float64)

        window = self._window
        freqs = rfftfreq(fft_size, 1.0 / self.sr)
        n_bins = len(freqs)

        speed_coeff = 0.05 + (self.speed / 100.0) * 0.9
        smooth_bins = max(1, int((1.0 - self.smoothing / 100.0) * 20) + 1)

        # Band sensitivity masks
        low_mask = freqs < self.crossover_low
        mid_mask = (freqs >= self.crossover_low) & (freqs < self.crossover_high)
        high_mask = freqs >= self.crossover_high

        sensitivity = np.ones(n_bins)
        sensitivity[low_mask] = self.band_sensitivity["low"] / 100.0
        sensitivity[mid_mask] = self.band_sensitivity["mid"] / 100.0
        sensitivity[high_mask] = self.band_sensitivity["high"] / 100.0

        intensity = self.amount / 100.0
        max_reduction_db = -12.0 * intensity

        for ch in range(audio.shape[1]):
            x = audio[:, ch].astype(np.float64)
            y = np.zeros(len(x) + fft_size, dtype=np.float64)
            d = np.zeros(len(x) + fft_size, dtype=np.float64)
            norm = np.zeros(len(x) + fft_size, dtype=np.float64)

            prev_gain = np.ones(n_bins) if self._prev_gain is None else self._prev_gain.copy()

            # Pass 1: compute average spectrum
            avg_mag = np.zeros(n_bins)
            avg_count = 0
            for start in range(0, len(x) - fft_size, hop * 4):
                block = x[start:start + fft_size] * window
                mag = np.abs(rfft(block))
                avg_mag += mag
                avg_count += 1

            if avg_count > 0:
                avg_mag /= avg_count
            avg_mag = np.maximum(avg_mag, 1e-10)

            if smooth_bins > 1:
                kernel = np.ones(smooth_bins) / smooth_bins
                avg_mag = np.convolve(avg_mag, kernel, mode='same')

            # Pass 2: dynamic processing
            for start in range(0, len(x) - fft_size, hop):
                block = x[start:start + fft_size] * window
                spectrum = rfft(block)
                mag = np.abs(spectrum)
                phase = np.angle(spectrum)

                ratio = mag / np.maximum(avg_mag, 1e-10)

                target_gain = np.ones(n_bins)

                if self.mode == "shape":
                    exceed = (ratio > 1.0) & (sensitivity > 0)
                    excess = np.clip(ratio - 1.0, 0, None)
                    reduction_db = max_reduction_db * np.minimum(excess, 1.0) * sensitivity
                    target_gain[exceed] = 10.0 ** (reduction_db[exceed] / 20.0)
                else:
                    exceed = (ratio > 0.8) & (sensitivity > 0)
                    excess = np.clip(ratio - 0.8, 0, None)
                    reduction_db = max_reduction_db * np.minimum(excess, 1.0) * sensitivity
                    target_gain[exceed] = 10.0 ** (reduction_db[exceed] / 20.0)

                if self.tame_transients:
                    high_energy = np.mean(mag[high_mask] ** 2)
                    high_avg = np.mean(avg_mag[high_mask] ** 2)
                    if high_energy > high_avg * 2.0:
                        tr = min(1.0, (high_energy / high_avg - 2.0) * 0.3)
                        target_gain[high_mask] *= (1.0 - tr * intensity)

                gain = prev_gain * (1.0 - speed_coeff) + target_gain * speed_coeff

                if smooth_bins > 1:
                    gain = np.convolve(gain, np.ones(smooth_bins) / smooth_bins, mode='same')

                gain = np.clip(gain, 10.0 ** (max_reduction_db / 20.0), 1.0)
                prev_gain = gain.copy()

                processed_spectrum = mag * gain * np.exp(1j * phase)
                processed_block = irfft(processed_spectrum, n=fft_size)
                delta_block = block - processed_block[:fft_size]

                y[start:start + fft_size] += processed_block[:fft_size] * window
                d[start:start + fft_size] += delta_block * window
                norm[start:start + fft_size] += window ** 2

            norm = np.maximum(norm, 1e-10)

            if self.delta:
                output[:, ch] = (d[:len(x)] / norm[:len(x)])
            else:
                output[:, ch] = (y[:len(x)] / norm[:len(x)])

            self._prev_gain = prev_gain

        return output.astype(np.float32)

    def get_reduction_spectrum(self):
        """Return current gain reduction per frequency for metering."""
        if self._prev_gain is not None:
            return 20.0 * np.log10(np.maximum(self._prev_gain, 1e-10))
        return np.zeros(self._fft_size // 2 + 1)

    def get_settings_dict(self) -> dict:
        return {
            "amount": self.amount, "speed": self.speed,
            "smoothing": self.smoothing, "mode": self.mode,
            "tame_transients": self.tame_transients, "delta": self.delta,
            "band_sensitivity": dict(self.band_sensitivity),
            "crossover_low": self.crossover_low,
            "crossover_high": self.crossover_high,
        }

    def load_settings_dict(self, d: dict):
        for key in ["amount", "speed", "smoothing", "mode",
                     "tame_transients", "delta", "crossover_low", "crossover_high"]:
            if key in d:
                setattr(self, key, d[key])
        if "band_sensitivity" in d:
            self.band_sensitivity.update(d["band_sensitivity"])
