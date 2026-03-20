"""
modules/master/soothe.py — Dynamic Resonance Suppression
Inspired by Oeksound Soothe2

Detects harsh resonance peaks in the 2-8kHz range and dynamically
reduces them without affecting the rest of the spectrum.
"""

import numpy as np

try:
    from scipy.fft import rfft, irfft, rfftfreq
    from scipy.signal.windows import hann
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SootheProcessor:
    """Dynamic resonance suppression processor."""

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.enabled = False
        self.amount = 0.0         # 0-100
        self.freq_low = 2000.0    # Hz
        self.freq_high = 8000.0   # Hz
        self.depth_db = -6.0      # max reduction per resonance
        self.sensitivity = 1.5    # threshold = avg * sensitivity

    def set_params(self, amount=None, freq_low=None, freq_high=None, depth_db=None):
        """Update parameters with validation."""
        if amount is not None:
            self.amount = max(0.0, min(100.0, amount))
        if freq_low is not None:
            self.freq_low = max(100, min(20000, freq_low))
        if freq_high is not None:
            self.freq_high = max(self.freq_low, min(20000, freq_high))
        if depth_db is not None:
            self.depth_db = max(-12, min(0, depth_db))

    def process(self, audio):
        """Process audio array (numpy float32, stereo).

        Args:
            audio: numpy array, shape (samples, channels) or (samples,)

        Returns:
            Processed audio with resonances suppressed.
        """
        if self.amount <= 0 or not self.enabled or not HAS_SCIPY:
            return audio

        intensity = self.amount / 100.0
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        output = np.copy(audio)
        block_size = 2048
        hop = block_size // 2
        window = hann(block_size, sym=False).astype(np.float64)

        for ch in range(audio.shape[1]):
            x = audio[:, ch].astype(np.float64)
            y = np.zeros_like(x)
            norm = np.zeros_like(x)

            for start in range(0, len(x) - block_size, hop):
                block = x[start:start + block_size] * window
                spectrum = rfft(block)
                freqs = rfftfreq(block_size, 1.0 / self.sr)
                magnitude = np.abs(spectrum)

                # Target frequency range
                mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)

                gain = np.ones_like(magnitude)
                if np.any(mask):
                    target_mag = magnitude[mask]
                    avg = np.mean(target_mag)
                    if avg > 1e-10:
                        indices = np.where(mask)[0]
                        for j, idx in enumerate(indices):
                            if magnitude[idx] > avg * self.sensitivity:
                                ratio = magnitude[idx] / avg
                                reduction = self.depth_db * intensity * min(
                                    1.0, (ratio - self.sensitivity) / ratio)
                                gain[idx] = 10.0 ** (reduction / 20.0)

                processed_block = irfft(spectrum * gain, n=block_size)
                y[start:start + block_size] += processed_block * window
                norm[start:start + block_size] += window * window

            # Normalize overlap-add
            norm = np.maximum(norm, 1e-10)
            output[:, ch] = (y / norm).astype(np.float32)

        return output

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
                f"range={self.freq_low:.0f}-{self.freq_high:.0f}Hz, "
                f"depth={self.depth_db:.0f}dB)")
