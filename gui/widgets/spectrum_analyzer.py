"""
Real-time spectrum analyzer overlay for EQ display.

Story 4.1 — Epic 4: Pro Mastering.

Features:
    - FFT-based spectrum display (4096-point)
    - 30fps update during playback
    - Logarithmic frequency axis (20 Hz - 20 kHz)
    - dB scale (-60 to 0 dB)
    - Color gradient (cool blue → warm orange)
    - Overlaid on EQ band display
    - Pre/Post EQ toggle
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from gui.utils.compat import (
    QWidget, QTimer, QPainter, QPen, QColor, QFont, QLinearGradient,
    Qt, QRect, QRectF, QPointF, QPainterPath, QBrush, QSizePolicy,
    pyqtSignal,
)
from gui.styles import Colors


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FFT_SIZE = 4096
SAMPLE_RATE_DEFAULT = 44100
FREQ_MIN = 20.0
FREQ_MAX = 20000.0
DB_MIN = -60.0
DB_MAX = 0.0
NUM_DISPLAY_BINS = 256          # downsampled for smooth drawing
UPDATE_FPS = 30
SMOOTHING_FACTOR = 0.7          # exponential smoothing for display


def _log_freq_to_x(freq: float, width: float) -> float:
    """Map a frequency to an x-pixel using logarithmic scale."""
    if freq <= FREQ_MIN:
        return 0.0
    log_min = math.log10(FREQ_MIN)
    log_max = math.log10(FREQ_MAX)
    return (math.log10(freq) - log_min) / (log_max - log_min) * width


def _db_to_y(db: float, height: float) -> float:
    """Map a dB value to a y-pixel (top = 0 dB, bottom = DB_MIN)."""
    clamped = max(DB_MIN, min(DB_MAX, db))
    return (1.0 - (clamped - DB_MIN) / (DB_MAX - DB_MIN)) * height


# ---------------------------------------------------------------------------
# SpectrumAnalyzerWidget
# ---------------------------------------------------------------------------
class SpectrumAnalyzerWidget(QWidget):
    """
    Professional real-time spectrum analyzer inspired by iZotope Ozone.

    Accepts raw PCM samples via ``feed_samples()`` and renders a smooth
    logarithmic spectrum overlaid on the EQ display area.
    """

    # Signal emitted when pre/post toggle changes (True = post-EQ)
    pre_post_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(180)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # spectrum data
        self._magnitude_db: np.ndarray = np.full(NUM_DISPLAY_BINS, DB_MIN, dtype=np.float64)
        self._smoothed_db: np.ndarray = np.full(NUM_DISPLAY_BINS, DB_MIN, dtype=np.float64)
        self._peak_db: np.ndarray = np.full(NUM_DISPLAY_BINS, DB_MIN, dtype=np.float64)
        self._sample_rate: int = SAMPLE_RATE_DEFAULT
        self._is_post_eq: bool = False  # False = pre-EQ, True = post-EQ

        # FFT window
        self._window: np.ndarray = np.hanning(FFT_SIZE)

        # display update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        # peak decay
        self._peak_decay = 0.005  # dB per tick

        # gradient colours for the spectrum fill
        self._grad_colors: List[Tuple[float, QColor]] = [
            (0.0, QColor(0, 100, 200, 120)),      # cool blue (low freq)
            (0.5, QColor(0, 180, 220, 140)),       # teal (mid)
            (0.8, QColor(255, 180, 50, 160)),      # warm orange (high)
            (1.0, QColor(255, 100, 50, 180)),      # red-orange (very high)
        ]

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        """Start the 30 fps display refresh."""
        self._timer.start(1000 // UPDATE_FPS)

    def stop(self) -> None:
        """Stop updating and clear the display."""
        self._timer.stop()
        self._magnitude_db[:] = DB_MIN
        self._smoothed_db[:] = DB_MIN
        self._peak_db[:] = DB_MIN
        self.update()

    @property
    def is_post_eq(self) -> bool:
        return self._is_post_eq

    @is_post_eq.setter
    def is_post_eq(self, value: bool) -> None:
        self._is_post_eq = value
        self.pre_post_changed.emit(value)

    def toggle_pre_post(self) -> None:
        """Toggle between pre-EQ and post-EQ spectrum display."""
        self.is_post_eq = not self._is_post_eq

    def feed_samples(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE_DEFAULT) -> None:
        """
        Feed a chunk of audio samples for FFT analysis.

        Args:
            samples: 1-D float array (mono or left channel), length >= FFT_SIZE.
            sample_rate: Sample rate of the audio data.
        """
        self._sample_rate = sample_rate
        if len(samples) < FFT_SIZE:
            padded = np.zeros(FFT_SIZE, dtype=np.float64)
            padded[:len(samples)] = samples
            samples = padded

        # take last FFT_SIZE samples
        chunk = samples[-FFT_SIZE:].astype(np.float64)
        windowed = chunk * self._window

        # compute FFT magnitude
        spectrum = np.abs(np.fft.rfft(windowed)) / FFT_SIZE
        spectrum = np.maximum(spectrum, 1e-12)
        mag_db = 20.0 * np.log10(spectrum)

        # resample to NUM_DISPLAY_BINS on log-frequency axis
        freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sample_rate)
        log_bins = np.logspace(
            math.log10(FREQ_MIN), math.log10(FREQ_MAX), NUM_DISPLAY_BINS
        )
        self._magnitude_db = np.interp(log_bins, freqs, mag_db)

    # -- internal ----------------------------------------------------------

    def _on_tick(self) -> None:
        """Timer tick — smooth & repaint."""
        # exponential smoothing
        self._smoothed_db = (
            SMOOTHING_FACTOR * self._smoothed_db
            + (1.0 - SMOOTHING_FACTOR) * self._magnitude_db
        )
        # peak hold with slow decay
        above = self._smoothed_db > self._peak_db
        self._peak_db[above] = self._smoothed_db[above]
        self._peak_db[~above] -= self._peak_decay
        np.clip(self._peak_db, DB_MIN, DB_MAX, out=self._peak_db)

        self.update()

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            w = float(self.width())
            h = float(self.height())
            if w < 10 or h < 10:
                return

            margin_left = 40.0
            margin_right = 10.0
            margin_top = 20.0
            margin_bottom = 25.0
            plot_w = w - margin_left - margin_right
            plot_h = h - margin_top - margin_bottom

            # background
            painter.fillRect(self.rect(), QColor(Colors.BG_PRIMARY))

            # grid lines — frequency
            painter.setPen(QPen(QColor(Colors.BORDER), 1, Qt.PenStyle.DotLine))
            freq_labels = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            painter.setFont(QFont("Inter", 7))
            for freq in freq_labels:
                x = margin_left + _log_freq_to_x(freq, plot_w)
                painter.drawLine(QPointF(x, margin_top), QPointF(x, margin_top + plot_h))
                label = f"{freq // 1000}k" if freq >= 1000 else str(freq)
                painter.setPen(QColor(Colors.TEXT_TERTIARY))
                painter.drawText(QRectF(x - 15, margin_top + plot_h + 3, 30, 15),
                                 Qt.AlignmentFlag.AlignCenter, label)
                painter.setPen(QPen(QColor(Colors.BORDER), 1, Qt.PenStyle.DotLine))

            # grid lines — dB
            db_labels = [0, -6, -12, -18, -24, -36, -48, -60]
            for db in db_labels:
                y = margin_top + _db_to_y(db, plot_h)
                painter.setPen(QPen(QColor(Colors.BORDER), 1, Qt.PenStyle.DotLine))
                painter.drawLine(QPointF(margin_left, y), QPointF(margin_left + plot_w, y))
                painter.setPen(QColor(Colors.TEXT_TERTIARY))
                painter.drawText(QRectF(0, y - 7, margin_left - 4, 14),
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                                 f"{db}")

            # build spectrum path
            path = QPainterPath()
            fill_path = QPainterPath()
            bottom_y = margin_top + plot_h

            log_bins = np.logspace(math.log10(FREQ_MIN), math.log10(FREQ_MAX), NUM_DISPLAY_BINS)
            first = True
            for i in range(NUM_DISPLAY_BINS):
                x = margin_left + _log_freq_to_x(log_bins[i], plot_w)
                y = margin_top + _db_to_y(self._smoothed_db[i], plot_h)
                pt = QPointF(x, y)
                if first:
                    path.moveTo(pt)
                    fill_path.moveTo(QPointF(x, bottom_y))
                    fill_path.lineTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
                    fill_path.lineTo(pt)

            # close fill path
            last_x = margin_left + _log_freq_to_x(log_bins[-1], plot_w)
            fill_path.lineTo(QPointF(last_x, bottom_y))
            fill_path.closeSubpath()

            # gradient fill
            grad = QLinearGradient(margin_left, 0, margin_left + plot_w, 0)
            for pos, color in self._grad_colors:
                grad.setColorAt(pos, color)
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(fill_path)

            # spectrum line
            line_grad = QLinearGradient(margin_left, 0, margin_left + plot_w, 0)
            line_grad.setColorAt(0.0, QColor(0, 150, 255, 220))
            line_grad.setColorAt(0.5, QColor(0, 220, 220, 240))
            line_grad.setColorAt(1.0, QColor(255, 160, 60, 240))
            painter.setPen(QPen(QBrush(line_grad), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            # peak line
            peak_path = QPainterPath()
            first = True
            for i in range(NUM_DISPLAY_BINS):
                x = margin_left + _log_freq_to_x(log_bins[i], plot_w)
                y = margin_top + _db_to_y(self._peak_db[i], plot_h)
                pt = QPointF(x, y)
                if first:
                    peak_path.moveTo(pt)
                    first = False
                else:
                    peak_path.lineTo(pt)
            painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
            painter.drawPath(peak_path)

            # Pre/Post label
            painter.setFont(QFont("Inter", 9, QFont.Weight.Bold))
            label = "POST EQ" if self._is_post_eq else "PRE EQ"
            label_color = QColor(Colors.TEAL) if self._is_post_eq else QColor(Colors.LED_AMBER)
            painter.setPen(label_color)
            painter.drawText(QRectF(w - 80, 2, 75, 16),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             label)
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Click to toggle pre/post EQ."""
        if event.button() == Qt.MouseButton.LeftButton:
            # check if click is in the pre/post label area
            if event.position().x() > self.width() - 85 and event.position().y() < 20:
                self.toggle_pre_post()
                return
        super().mousePressEvent(event)
