"""
Real-time audio level meters and LUFS display.

Classes:
    RealTimeMeter — Waves-style segmented LED bar meter
    LUFSDisplay   — Recessed LCD-style LUFS value display
"""

import math

from gui.utils.compat import (
    QWidget, QTimer, QPainter, QPen, QColor, QFont, QLinearGradient,
    Qt, QRect,
)
from gui.styles import Colors


class RealTimeMeter(QWidget):
    """Professional real-time audio level meter - Waves-style with segmented LED bars"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.left_level = 0.0
        self.right_level = 0.0
        self.peak_left = 0.0
        self.peak_right = 0.0
        self.current_position_ms = 0
        self.is_playing = False
        self.num_segments = 35

        self._audio_engine = None

        self.peak_hold_timer = QTimer()
        self.peak_hold_timer.timeout.connect(self._decay_peaks)
        self.peak_hold_timer.start(50)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)

    def start(self):
        self.is_playing = True
        self.update_timer.start(33)

    def stop(self):
        self.is_playing = False
        self.update_timer.stop()
        self.left_level = 0.0
        self.right_level = 0.0
        self.peak_left = 0.0
        self.peak_right = 0.0
        self.update()

    def setPosition(self, position_ms: int):
        """Update meter based on playback position"""
        self.current_position_ms = position_ms
        if self.is_playing:
            self._generate_levels_for_position(position_ms)

    def set_audio_engine(self, engine):
        """V5.5: Connect real audio analysis engine for actual level metering."""
        self._audio_engine = engine

    def _generate_levels_for_position(self, position_ms: int):
        """V5.5.2: Read REAL audio levels from AudioAnalysisEngine (with gain applied)."""
        if self._audio_engine is not None:
            levels = self._audio_engine.analyze_at_position(position_ms, window_ms=50)
            l_db = max(-48.0, levels["left_rms_db"])
            r_db = max(-48.0, levels["right_rms_db"])
            self.left_level = max(0.01, min(0.99, (l_db + 48.0) / 48.0))
            self.right_level = max(0.01, min(0.99, (r_db + 48.0) / 48.0))
            self._last_levels_db = levels
        else:
            self.left_level = 0.0
            self.right_level = 0.0

        if self.left_level > self.peak_left:
            self.peak_left = self.left_level
        if self.right_level > self.peak_right:
            self.peak_right = self.right_level

    def _update_display(self):
        self.update()

    def _decay_peaks(self):
        self.peak_left = max(0, self.peak_left - 0.02)
        self.peak_right = max(0, self.peak_right - 0.02)

    def _get_segment_color(self, segment_index: int) -> QColor:
        ratio = segment_index / self.num_segments
        if ratio < 0.6:
            return QColor(Colors.METER_GREEN)
        elif ratio < 0.8:
            return QColor(Colors.METER_YELLOW)
        else:
            return QColor(Colors.METER_RED)

    def _draw_meter_bar(self, painter: QPainter, x: int, y: int, width: int, height: int,
                        level: float, peak: float, is_left: bool):
        segment_height = height // self.num_segments
        gap = 2
        segment_h = segment_height - gap

        bezel_pen = QPen(QColor(Colors.CHROME_DARK), 1)
        painter.setPen(bezel_pen)
        painter.drawRect(x, y, width, height)

        highlight_pen = QPen(QColor(Colors.CHROME_HIGHLIGHT), 1)
        painter.setPen(highlight_pen)
        painter.drawLine(x + 1, y + 1, x + width - 2, y + 1)
        painter.drawLine(x + 1, y + 1, x + 1, y + height - 2)

        painter.fillRect(x + 2, y + 2, width - 4, height - 4, QColor(Colors.PANEL_DEEP))

        filled_segments = int(level * self.num_segments)

        for i in range(self.num_segments):
            seg_y = y + height - (i + 1) * segment_height

            if i < filled_segments:
                color = self._get_segment_color(i)
                painter.fillRect(x + 4, seg_y + gap // 2, width - 8, segment_h, color)

                glow_color = QColor(color)
                glow_color.setAlpha(80)
                painter.fillRect(x + 3, seg_y + gap // 2 - 1, width - 6, segment_h + 2, glow_color)
            else:
                painter.fillRect(x + 4, seg_y + gap // 2, width - 8, segment_h,
                                 QColor(Colors.CHROME_DARK))

        if peak > 0:
            peak_segment = int(peak * self.num_segments)
            peak_y = y + height - peak_segment * segment_height

            peak_color = QColor(Colors.METER_RED) if peak > 0.8 else QColor("#ffffff")
            painter.setPen(QPen(peak_color, 3))
            painter.drawLine(x + 2, peak_y, x + width - 2, peak_y)

            peak_glow = QColor(peak_color)
            peak_glow.setAlpha(100)
            glow_pen = QPen(peak_glow, 5)
            painter.setPen(glow_pen)
            painter.drawLine(x + 2, peak_y, x + width - 2, peak_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            w = self.width()
            h = self.height()

            painter.fillRect(self.rect(), QColor(Colors.BG_PRIMARY))

            db_width = 28
            bar_width = (w - db_width - 60) // 2
            meter_height = h - 50
            meter_top = 15

            left_bar_x = db_width + 15
            right_bar_x = left_bar_x + bar_width + 15

            painter.setPen(QColor(Colors.TEXT_TERTIARY))
            painter.setFont(QFont("Inter", 7))
            db_values = [(0, "0"), (-3, "-3"), (-6, "-6"), (-12, "-12"),
                         (-18, "-18"), (-24, "-24"), (-36, "-36"), (-48, "-48")]

            for db_val, db_label in db_values:
                if db_val == 0:
                    ratio = 0.0
                else:
                    ratio = min(1.0, abs(db_val) / 48.0)

                y = meter_top + int(ratio * meter_height)
                painter.drawText(5, y + 3, db_label)
                painter.drawLine(db_width - 3, y, db_width, y)

            self._draw_meter_bar(painter, left_bar_x, meter_top, bar_width, meter_height,
                                 self.left_level, self.peak_left, True)
            self._draw_meter_bar(painter, right_bar_x, meter_top, bar_width, meter_height,
                                 self.right_level, self.peak_right, False)

            painter.setFont(QFont("Inter", 11, QFont.Weight.Bold))
            painter.setPen(QColor(Colors.LED_AMBER))

            left_label_x = left_bar_x + bar_width // 2 - 5
            right_label_x = right_bar_x + bar_width // 2 - 5

            painter.drawText(left_label_x, h - 10, "L")
            painter.drawText(right_label_x, h - 10, "R")
        finally:
            painter.end()


class LUFSDisplay(QWidget):
    """Waves-style LUFS value display with recessed LCD appearance"""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label
        self.value = -14.0
        self.unit_text = "LUFS"
        self.setMinimumWidth(100)
        self.setFixedHeight(80)
        self.setStyleSheet(f"background: {Colors.BG_PRIMARY};")

    def setValue(self, value: float):
        self.value = value
        self.update()

    def _get_accent_color(self) -> str:
        label_upper = self.label.upper()
        if "INTEGRATED" in label_upper:
            return Colors.TEAL_GLOW
        elif "SHORT" in label_upper:
            return Colors.LED_AMBER_GLOW
        elif "MOMENTARY" in label_upper:
            return Colors.METER_RED
        else:
            return Colors.LED_AMBER_GLOW

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            w = self.width()
            h = self.height()

            painter.fillRect(self.rect(), QColor(Colors.BG_PRIMARY))

            frame_margin = 4
            frame_rect = QRect(frame_margin, frame_margin, w - 2 * frame_margin, h - 2 * frame_margin)

            bezel_pen = QPen(QColor(Colors.CHROME_DARK), 1)
            painter.setPen(bezel_pen)
            painter.drawRect(frame_rect)

            highlight_pen = QPen(QColor(Colors.CHROME_HIGHLIGHT), 1)
            painter.setPen(highlight_pen)
            painter.drawLine(frame_margin + 1, frame_margin + 1,
                             w - frame_margin - 2, frame_margin + 1)
            painter.drawLine(frame_margin + 1, frame_margin + 1,
                             frame_margin + 1, h - frame_margin - 2)

            lcd_rect = QRect(frame_margin + 2, frame_margin + 2,
                             w - 2 * frame_margin - 4, h - 2 * frame_margin - 4)

            gradient = QLinearGradient(0, lcd_rect.top(), 0, lcd_rect.bottom())
            gradient.setColorAt(0.0, QColor(Colors.PANEL_DEEP))
            gradient.setColorAt(1.0, QColor("#050507"))

            painter.fillRect(lcd_rect, gradient)

            shadow_pen = QPen(QColor(Colors.CHROME_DARK), 1)
            painter.setPen(shadow_pen)
            painter.drawLine(frame_margin + 2, h - frame_margin - 3,
                             w - frame_margin - 3, h - frame_margin - 3)
            painter.drawLine(w - frame_margin - 3, frame_margin + 2,
                             w - frame_margin - 3, h - frame_margin - 3)

            top_height = 18
            middle_height = h - 2 * frame_margin - 4 - top_height - 14
            bottom_height = 14

            label_rect = QRect(frame_margin + 3, frame_margin + 3,
                               w - 2 * frame_margin - 6, top_height)
            painter.setFont(QFont("Inter", 8, QFont.Weight.Bold))
            painter.setPen(QColor(Colors.LED_AMBER))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, self.label.upper())

            value_rect = QRect(frame_margin + 3, frame_margin + 3 + top_height,
                               w - 2 * frame_margin - 6, middle_height)
            painter.setFont(QFont("Courier New", 28, QFont.Weight.Bold))
            accent_color = self._get_accent_color()
            painter.setPen(QColor(accent_color))
            painter.drawText(value_rect, Qt.AlignmentFlag.AlignCenter, f"{self.value:.1f}")

            unit_rect = QRect(frame_margin + 3,
                              frame_margin + 3 + top_height + middle_height,
                              w - 2 * frame_margin - 6, bottom_height)
            painter.setFont(QFont("Inter", 7))
            painter.setPen(QColor(Colors.TEXT_TERTIARY))
            painter.drawText(unit_rect, Qt.AlignmentFlag.AlignCenter, self.unit_text)
        finally:
            painter.end()
