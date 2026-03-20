"""
Waves WLM Plus Exact Clone — professional loudness meter widget.

Story 4.2 — Epic 4: Pro Mastering.

Features:
    - Teal-to-dark gradient background
    - Green/yellow/red LED-segment meters
    - Momentary (large), Short-term (medium), Integrated (large)
    - True Peak L/R bars
    - LRA display
    - Loudness histogram
    - Preset targets: ITU-R BS.1770, EBU R128, ATSC A/85, custom
    - Start/Stop/Reset logging controls
    - LED-style 7-segment font for numeric displays
    - Gain reduction timeline history with peak hold
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from gui.utils.compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QFrame, QTimer,
    QPainter, QPen, QColor, QFont, QLinearGradient, QBrush,
    Qt, QRect, QRectF, QPointF, QPainterPath, QSizePolicy,
    pyqtSignal,
)
from gui.styles import Colors


# ---------------------------------------------------------------------------
# WLM Plus design constants
# ---------------------------------------------------------------------------
WLM_BG_DARK = "#0C1A1F"
WLM_BG_TEAL = "#0D2B30"
WLM_BG_PANEL = "#0A1E24"
WLM_BORDER = "#1A4A52"
WLM_TEXT_DIM = "#4A8A92"
WLM_TEXT_BRIGHT = "#80D8E0"
WLM_LED_GREEN = "#00CC66"
WLM_LED_YELLOW = "#CCCC00"
WLM_LED_RED = "#FF3333"
WLM_LED_OFF = "#1A2A2E"
WLM_SEGMENT_GAP = 1

# Loudness standard presets
LOUDNESS_PRESETS: Dict[str, Dict[str, float]] = {
    "ITU-R BS.1770": {"target_lufs": -24.0, "true_peak": -2.0, "lra_max": 20.0},
    "EBU R128": {"target_lufs": -23.0, "true_peak": -1.0, "lra_max": 15.0},
    "ATSC A/85": {"target_lufs": -24.0, "true_peak": -2.0, "lra_max": 20.0},
    "Spotify": {"target_lufs": -14.0, "true_peak": -1.0, "lra_max": 15.0},
    "YouTube": {"target_lufs": -14.0, "true_peak": -1.0, "lra_max": 15.0},
    "Apple Music": {"target_lufs": -16.0, "true_peak": -1.0, "lra_max": 15.0},
    "Tidal": {"target_lufs": -14.0, "true_peak": -1.0, "lra_max": 15.0},
    "CD Master": {"target_lufs": -9.0, "true_peak": -0.3, "lra_max": 12.0},
    "Custom": {"target_lufs": -14.0, "true_peak": -1.0, "lra_max": 15.0},
}

HISTOGRAM_MAX_SAMPLES = 1000


# ---------------------------------------------------------------------------
# 7-Segment LED Display
# ---------------------------------------------------------------------------
class SevenSegmentDisplay(QWidget):
    """WLM Plus style 7-segment LED numeric display."""

    def __init__(self, label: str = "", digits: int = 5, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label = label
        self._digits = digits
        self._value: float = -70.0
        self._unit: str = "LUFS"
        self._accent_color = QColor(WLM_LED_GREEN)
        self.setFixedHeight(65)
        self.setMinimumWidth(90)

    def set_value(self, value: float) -> None:
        self._value = value
        self.update()

    def set_accent(self, color: str) -> None:
        self._accent_color = QColor(color)

    def set_unit(self, unit: str) -> None:
        self._unit = unit

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # background panel
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(WLM_BG_DARK))
        grad.setColorAt(1.0, QColor("#050E12"))
        painter.fillRect(self.rect(), QBrush(grad))

        # border
        painter.setPen(QPen(QColor(WLM_BORDER), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # label
        painter.setFont(QFont("Inter", 7, QFont.Weight.Bold))
        painter.setPen(QColor(WLM_TEXT_DIM))
        painter.drawText(QRectF(4, 2, w - 8, 14), Qt.AlignmentFlag.AlignCenter, self._label.upper())

        # value
        painter.setFont(QFont("Courier New", 22, QFont.Weight.Bold))
        painter.setPen(self._accent_color)
        txt = f"{self._value:.1f}" if self._value > -70 else "---"
        painter.drawText(QRectF(4, 14, w - 8, 30), Qt.AlignmentFlag.AlignCenter, txt)

        # unit
        painter.setFont(QFont("Inter", 7))
        painter.setPen(QColor(WLM_TEXT_DIM))
        painter.drawText(QRectF(4, h - 16, w - 8, 14), Qt.AlignmentFlag.AlignCenter, self._unit)

        painter.end()


# ---------------------------------------------------------------------------
# Segmented LED Meter Bar
# ---------------------------------------------------------------------------
class LEDMeterBar(QWidget):
    """Vertical LED-segment meter bar like WLM Plus."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._level: float = -70.0   # dB
        self._peak: float = -70.0    # peak hold dB
        self._range_min: float = -60.0
        self._range_max: float = 0.0
        self._num_segments: int = 40
        self.setFixedWidth(22)
        self.setMinimumHeight(120)

        self._peak_timer = QTimer(self)
        self._peak_timer.timeout.connect(self._decay_peak)
        self._peak_timer.start(80)

    def set_level(self, db: float) -> None:
        self._level = max(self._range_min, min(self._range_max, db))
        if db > self._peak:
            self._peak = db
        self.update()

    def _decay_peak(self) -> None:
        self._peak = max(self._range_min, self._peak - 0.3)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        w, h = self.width(), self.height()
        seg_h = h // self._num_segments
        gap = WLM_SEGMENT_GAP

        painter.fillRect(self.rect(), QColor(WLM_BG_DARK))

        level_ratio = (self._level - self._range_min) / (self._range_max - self._range_min)
        peak_ratio = (self._peak - self._range_min) / (self._range_max - self._range_min)
        filled = int(level_ratio * self._num_segments)
        peak_seg = int(peak_ratio * self._num_segments)

        for i in range(self._num_segments):
            seg_y = h - (i + 1) * seg_h
            ratio = i / self._num_segments

            if ratio < 0.6:
                on_color = QColor(WLM_LED_GREEN)
            elif ratio < 0.85:
                on_color = QColor(WLM_LED_YELLOW)
            else:
                on_color = QColor(WLM_LED_RED)

            if i < filled:
                painter.fillRect(2, seg_y + gap, w - 4, seg_h - gap, on_color)
            else:
                painter.fillRect(2, seg_y + gap, w - 4, seg_h - gap, QColor(WLM_LED_OFF))

            if i == peak_seg and peak_seg > 0:
                painter.fillRect(2, seg_y + gap, w - 4, seg_h - gap, on_color)

        painter.end()


# ---------------------------------------------------------------------------
# Loudness Histogram
# ---------------------------------------------------------------------------
class LoudnessHistogram(QWidget):
    """Distribution graph showing LUFS values over time."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._data: deque = deque(maxlen=HISTOGRAM_MAX_SAMPLES)
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def add_sample(self, lufs: float) -> None:
        self._data.append(lufs)
        self.update()

    def reset(self) -> None:
        self._data.clear()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # background
        painter.fillRect(self.rect(), QColor(WLM_BG_DARK))
        painter.setPen(QPen(QColor(WLM_BORDER), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        if not self._data:
            painter.setFont(QFont("Inter", 8))
            painter.setPen(QColor(WLM_TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "HISTOGRAM")
            painter.end()
            return

        # build histogram bins (-60 to 0 LUFS, 1 dB bins)
        bins = [0] * 60
        for val in self._data:
            idx = int(max(0, min(59, val + 60)))
            bins[idx] += 1

        max_count = max(bins) if max(bins) > 0 else 1
        bar_w = max(1.0, (w - 4) / 60.0)

        for i, count in enumerate(bins):
            bar_h = (count / max_count) * (h - 4)
            x = 2 + i * bar_w
            y = h - 2 - bar_h

            ratio = i / 60.0
            if ratio < 0.4:
                color = QColor(0, 120, 80, 180)
            elif ratio < 0.7:
                color = QColor(0, 200, 100, 200)
            elif ratio < 0.85:
                color = QColor(200, 200, 0, 200)
            else:
                color = QColor(255, 60, 60, 200)

            painter.fillRect(QRectF(x, y, bar_w - 0.5, bar_h), color)

        painter.end()


# ---------------------------------------------------------------------------
# Gain Reduction History
# ---------------------------------------------------------------------------
class GainReductionGraph(QWidget):
    """Timeline graph showing gain reduction over time."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._data: deque = deque(maxlen=300)
        self._peak_hold: float = 0.0
        self.setMinimumHeight(50)

    def add_sample(self, gr_db: float) -> None:
        self._data.append(gr_db)
        if gr_db < self._peak_hold:
            self._peak_hold = gr_db
        self.update()

    def reset(self) -> None:
        self._data.clear()
        self._peak_hold = 0.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(self.rect(), QColor(WLM_BG_DARK))
        painter.setPen(QPen(QColor(WLM_BORDER), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        if not self._data:
            painter.setFont(QFont("Inter", 7))
            painter.setPen(QColor(WLM_TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "GAIN REDUCTION")
            painter.end()
            return

        # draw GR curve (0 dB at top, -20 dB at bottom)
        gr_min = -20.0
        path = QPainterPath()
        data_list = list(self._data)
        step = max(1.0, w / max(len(data_list), 1))

        for i, val in enumerate(data_list):
            x = i * step
            ratio = max(0.0, min(1.0, val / gr_min))
            y = 2 + ratio * (h - 4)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        painter.setPen(QPen(QColor(255, 120, 0, 200), 1.5))
        painter.drawPath(path)

        # peak hold line
        peak_ratio = max(0.0, min(1.0, self._peak_hold / gr_min))
        peak_y = 2 + peak_ratio * (h - 4)
        painter.setPen(QPen(QColor(WLM_LED_RED), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(0, peak_y), QPointF(w, peak_y))

        # peak label
        painter.setFont(QFont("Courier New", 7))
        painter.setPen(QColor(WLM_LED_RED))
        painter.drawText(QRectF(w - 50, peak_y - 12, 48, 12),
                         Qt.AlignmentFlag.AlignRight, f"{self._peak_hold:.1f} dB")

        painter.end()


# ---------------------------------------------------------------------------
# WavesWLMPlusMeter — Main composite widget
# ---------------------------------------------------------------------------
class WavesWLMPlusMeter(QWidget):
    """
    Complete Waves WLM Plus clone widget.

    Layout matches WLM Plus:
        [Momentary]  [Short-term]  [Integrated]  [TP L] [TP R]
        [LRA]        [Histogram]
        [Controls]   [GR History]
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._is_logging = False
        self._log_start_time: float = 0.0
        self._preset_name: str = "YouTube"

        self._build_ui()
        self._apply_preset(self._preset_name)

        # refresh timer
        self._refresh = QTimer(self)
        self._refresh.timeout.connect(self._on_refresh)

    # -- build UI ----------------------------------------------------------

    def _build_ui(self) -> None:
        self.setStyleSheet(f"background: {WLM_BG_DARK}; color: {WLM_TEXT_BRIGHT};")

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Title bar
        title = QLabel("WLM PLUS")
        title.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {WLM_TEXT_BRIGHT}; padding: 2px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # -- top row: meters -------------------------------------------------
        top = QHBoxLayout()
        top.setSpacing(4)

        # Momentary
        self.momentary_display = SevenSegmentDisplay("Momentary")
        self.momentary_display.set_accent(WLM_LED_RED)
        top.addWidget(self.momentary_display)

        self.momentary_bar = LEDMeterBar()
        top.addWidget(self.momentary_bar)

        # Short-term
        self.short_term_display = SevenSegmentDisplay("Short-term")
        self.short_term_display.set_accent(WLM_LED_YELLOW)
        top.addWidget(self.short_term_display)

        # Integrated
        self.integrated_display = SevenSegmentDisplay("Integrated")
        self.integrated_display.set_accent(WLM_LED_GREEN)
        top.addWidget(self.integrated_display)

        self.integrated_bar = LEDMeterBar()
        top.addWidget(self.integrated_bar)

        # True Peak L/R
        tp_frame = QFrame()
        tp_frame.setStyleSheet(f"background: {WLM_BG_PANEL}; border: 1px solid {WLM_BORDER};")
        tp_layout = QVBoxLayout(tp_frame)
        tp_layout.setContentsMargins(4, 2, 4, 2)
        tp_label = QLabel("TRUE PEAK")
        tp_label.setFont(QFont("Inter", 7, QFont.Weight.Bold))
        tp_label.setStyleSheet(f"color: {WLM_TEXT_DIM}; border: none;")
        tp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tp_layout.addWidget(tp_label)

        tp_bars = QHBoxLayout()
        self.tp_left_bar = LEDMeterBar()
        self.tp_right_bar = LEDMeterBar()
        tp_bars.addWidget(QLabel("L"))
        tp_bars.addWidget(self.tp_left_bar)
        tp_bars.addWidget(QLabel("R"))
        tp_bars.addWidget(self.tp_right_bar)
        tp_layout.addLayout(tp_bars)
        top.addWidget(tp_frame)

        root.addLayout(top)

        # -- middle row: LRA + histogram -------------------------------------
        mid = QHBoxLayout()
        mid.setSpacing(4)

        self.lra_display = SevenSegmentDisplay("LRA")
        self.lra_display.set_accent(WLM_TEXT_BRIGHT)
        self.lra_display.set_unit("LU")
        self.lra_display.setFixedWidth(80)
        mid.addWidget(self.lra_display)

        self.histogram = LoudnessHistogram()
        mid.addWidget(self.histogram, stretch=1)

        root.addLayout(mid)

        # -- gain reduction graph --------------------------------------------
        self.gr_graph = GainReductionGraph()
        root.addWidget(self.gr_graph)

        # -- controls bar ----------------------------------------------------
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(LOUDNESS_PRESETS.keys()))
        self.preset_combo.setCurrentText(self._preset_name)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.preset_combo.setFixedWidth(140)
        self.preset_combo.setStyleSheet(
            f"QComboBox {{ background: {WLM_BG_PANEL}; color: {WLM_TEXT_BRIGHT}; "
            f"border: 1px solid {WLM_BORDER}; padding: 3px; }}"
        )
        ctrl.addWidget(self.preset_combo)

        btn_style = (
            f"QPushButton {{ background: {WLM_BG_PANEL}; color: {WLM_TEXT_BRIGHT}; "
            f"border: 1px solid {WLM_BORDER}; padding: 4px 12px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: {WLM_BG_TEAL}; }}"
        )

        self.btn_start = QPushButton("START")
        self.btn_start.setStyleSheet(btn_style)
        self.btn_start.clicked.connect(self._on_start)
        ctrl.addWidget(self.btn_start)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet(btn_style)
        self.btn_stop.clicked.connect(self._on_stop)
        ctrl.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("RESET")
        self.btn_reset.setStyleSheet(btn_style)
        self.btn_reset.clicked.connect(self._on_reset)
        ctrl.addWidget(self.btn_reset)

        # target display
        self.target_label = QLabel("Target: -14.0 LUFS")
        self.target_label.setFont(QFont("Inter", 8))
        self.target_label.setStyleSheet(f"color: {WLM_TEXT_DIM};")
        ctrl.addWidget(self.target_label)
        ctrl.addStretch()

        root.addLayout(ctrl)

    # -- public API --------------------------------------------------------

    def update_meters(
        self,
        momentary: float = -70.0,
        short_term: float = -70.0,
        integrated: float = -70.0,
        true_peak_l: float = -70.0,
        true_peak_r: float = -70.0,
        lra: float = 0.0,
        gain_reduction: float = 0.0,
    ) -> None:
        """Update all meter values at once."""
        self.momentary_display.set_value(momentary)
        self.momentary_bar.set_level(momentary)
        self.short_term_display.set_value(short_term)
        self.integrated_display.set_value(integrated)
        self.integrated_bar.set_level(integrated)
        self.tp_left_bar.set_level(true_peak_l)
        self.tp_right_bar.set_level(true_peak_r)
        self.lra_display.set_value(lra)

        if self._is_logging:
            self.histogram.add_sample(momentary)
            self.gr_graph.add_sample(gain_reduction)

    def start_logging(self) -> None:
        """Start loudness logging."""
        self._is_logging = True
        self._log_start_time = time.time()
        self._refresh.start(33)  # ~30fps

    def stop_logging(self) -> None:
        """Stop loudness logging."""
        self._is_logging = False
        self._refresh.stop()

    def reset_logging(self) -> None:
        """Reset all logging data."""
        self._is_logging = False
        self._refresh.stop()
        self.histogram.reset()
        self.gr_graph.reset()
        self.momentary_display.set_value(-70.0)
        self.short_term_display.set_value(-70.0)
        self.integrated_display.set_value(-70.0)
        self.tp_left_bar.set_level(-70.0)
        self.tp_right_bar.set_level(-70.0)
        self.lra_display.set_value(0.0)

    def get_current_preset(self) -> Dict[str, float]:
        """Return the currently selected loudness preset values."""
        return dict(LOUDNESS_PRESETS.get(self._preset_name, LOUDNESS_PRESETS["YouTube"]))

    # -- internal ----------------------------------------------------------

    def _on_preset_changed(self, name: str) -> None:
        self._preset_name = name
        self._apply_preset(name)

    def _apply_preset(self, name: str) -> None:
        preset = LOUDNESS_PRESETS.get(name, LOUDNESS_PRESETS["YouTube"])
        self.target_label.setText(f"Target: {preset['target_lufs']:.1f} LUFS")

    def _on_start(self) -> None:
        self.start_logging()

    def _on_stop(self) -> None:
        self.stop_logging()

    def _on_reset(self) -> None:
        self.reset_logging()

    def _on_refresh(self) -> None:
        self.update()
