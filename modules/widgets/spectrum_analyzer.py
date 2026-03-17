"""
SpectrumAnalyzerWidget — FFT spectrum display with log frequency axis.
"""

import math
import numpy as np
try:
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtCore import Qt, QPointF
    from PyQt6.QtGui import (
        QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
        QLinearGradient,
    )
except ImportError:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtGui import (
        QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
        QLinearGradient,
    )


class SpectrumAnalyzerWidget(QWidget):
    """
    4096-FFT spectrum analyzer with Hann window, log frequency, teal fill.

    - Log frequency: 20 Hz – 20 kHz
    - dB scale: -80 to 0 dB
    - Teal filled polygon with peak hold
    """

    FREQ_MIN = 20.0
    FREQ_MAX = 20000.0
    DB_MIN = -80.0
    DB_MAX = 0.0
    FFT_SIZE = 4096

    FREQ_LABELS = [
        (20, "20"), (50, "50"), (100, "100"), (200, "200"), (500, "500"),
        (1000, "1k"), (2000, "2k"), (5000, "5k"), (10000, "10k"), (20000, "20k"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(400, 180)
        self._magnitudes = None  # dB values per bin
        self._freqs = None       # frequency per bin
        self._peak_hold = None   # peak hold dB values
        self._peak_decay = 0.5   # dB per frame decay
        self._sample_rate = 44100

    def set_audio_data(self, samples: np.ndarray, sample_rate: int = 44100):
        """Compute FFT from audio samples and update display."""
        self._sample_rate = sample_rate

        if samples is None or len(samples) == 0:
            return

        # Mono mix if stereo
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Take last FFT_SIZE samples
        n = min(len(samples), self.FFT_SIZE)
        chunk = samples[-n:]

        # Hann window
        window = np.hanning(n)
        windowed = chunk * window

        # FFT
        fft_result = np.fft.rfft(windowed, n=self.FFT_SIZE)
        magnitudes = np.abs(fft_result)

        # Convert to dB
        magnitudes = 20.0 * np.log10(np.maximum(magnitudes, 1e-10))

        # Normalize (max should be ~0 dB for full-scale signal)
        magnitudes -= np.max(magnitudes)  # relative to peak

        self._freqs = np.fft.rfftfreq(self.FFT_SIZE, 1.0 / sample_rate)
        self._magnitudes = magnitudes

        # Peak hold
        if self._peak_hold is None or len(self._peak_hold) != len(magnitudes):
            self._peak_hold = magnitudes.copy()
        else:
            for i in range(len(magnitudes)):
                if magnitudes[i] > self._peak_hold[i]:
                    self._peak_hold[i] = magnitudes[i]
                else:
                    self._peak_hold[i] -= self._peak_decay

        self.update()

    def reset(self):
        self._magnitudes = None
        self._peak_hold = None
        self.update()

    def _freq_to_x(self, freq, plot_x, plot_w):
        if freq <= 0:
            freq = self.FREQ_MIN
        log_min = math.log10(self.FREQ_MIN)
        log_max = math.log10(self.FREQ_MAX)
        log_f = math.log10(max(self.FREQ_MIN, min(self.FREQ_MAX, freq)))
        ratio = (log_f - log_min) / (log_max - log_min)
        return plot_x + int(ratio * plot_w)

    def _db_to_y(self, db, plot_y, plot_h):
        db = max(self.DB_MIN, min(self.DB_MAX, db))
        ratio = (db - self.DB_MIN) / (self.DB_MAX - self.DB_MIN)
        return plot_y + plot_h - int(ratio * plot_h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        margin_left = 28
        margin_bottom = 18
        margin_top = 4
        margin_right = 4
        plot_x = margin_left
        plot_y = margin_top
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        # Background
        p.fillRect(0, 0, w, h, QColor("#0A0A0C"))
        p.fillRect(plot_x, plot_y, plot_w, plot_h, QColor("#111114"))

        # Grid lines
        p.setPen(QPen(QColor("#1E1E22"), 1, Qt.PenStyle.DotLine))
        for db in [-72, -60, -48, -36, -24, -12, -6, 0]:
            y = self._db_to_y(db, plot_y, plot_h)
            p.drawLine(plot_x, y, plot_x + plot_w, y)

        for freq, _ in self.FREQ_LABELS:
            x = self._freq_to_x(freq, plot_x, plot_w)
            p.drawLine(x, plot_y, x, plot_y + plot_h)

        # Axis labels
        p.setFont(QFont("Menlo", 6))
        p.setPen(QColor("#8E8A82"))
        for db in [-60, -48, -36, -24, -12, 0]:
            y = self._db_to_y(db, plot_y, plot_h)
            p.drawText(2, y + 3, f"{db:>3}")

        for freq, label in self.FREQ_LABELS:
            x = self._freq_to_x(freq, plot_x, plot_w)
            p.drawText(x - 6, h - 2, label)

        if self._magnitudes is None or self._freqs is None:
            p.end()
            return

        # Build spectrum path (only bins within display range)
        path = QPainterPath()
        bottom_y = plot_y + plot_h
        first = True

        for i in range(len(self._freqs)):
            freq = self._freqs[i]
            if freq < self.FREQ_MIN or freq > self.FREQ_MAX:
                continue
            db = self._magnitudes[i]
            x = self._freq_to_x(freq, plot_x, plot_w)
            y = self._db_to_y(db, plot_y, plot_h)
            if first:
                path.moveTo(x, bottom_y)
                path.lineTo(x, y)
                first = False
            else:
                path.lineTo(x, y)

        # Close path for fill
        if not first:
            path.lineTo(self._freq_to_x(self.FREQ_MAX, plot_x, plot_w), bottom_y)
            path.closeSubpath()

        # Teal filled spectrum
        fill_grad = QLinearGradient(0, plot_y, 0, bottom_y)
        fill_grad.setColorAt(0.0, QColor(0, 180, 216, 100))
        fill_grad.setColorAt(1.0, QColor(0, 180, 216, 20))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(fill_grad))
        p.drawPath(path)

        # Outline
        p.setPen(QPen(QColor("#00B4D8"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Peak hold trace
        if self._peak_hold is not None:
            peak_path = QPainterPath()
            peak_first = True
            for i in range(len(self._freqs)):
                freq = self._freqs[i]
                if freq < self.FREQ_MIN or freq > self.FREQ_MAX:
                    continue
                db = self._peak_hold[i]
                x = self._freq_to_x(freq, plot_x, plot_w)
                y = self._db_to_y(db, plot_y, plot_h)
                if peak_first:
                    peak_path.moveTo(x, y)
                    peak_first = False
                else:
                    peak_path.lineTo(x, y)
            p.setPen(QPen(QColor(255, 255, 255, 80), 1))
            p.drawPath(peak_path)

        # Border
        p.setPen(QPen(QColor("#1E1E22"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(plot_x, plot_y, plot_w, plot_h)

        p.end()
