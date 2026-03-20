"""
TonalBalanceWidget — Vintage hardware-style tonal balance visualization.

Inspired by iZotope Tonal Balance Control 2.
Displays 4-band frequency balance against genre-specific target zones.
"""

import math

try:
    from PyQt6.QtWidgets import QWidget, QSizePolicy
    from PyQt6.QtCore import Qt, QRectF
    from PyQt6.QtGui import (
        QPainter, QPen, QBrush, QColor, QFont,
        QLinearGradient, QPainterPath,
    )
except ImportError:
    from PySide6.QtWidgets import QWidget, QSizePolicy
    from PySide6.QtCore import Qt, QRectF
    from PySide6.QtGui import (
        QPainter, QPen, QBrush, QColor, QFont,
        QLinearGradient, QPainterPath,
    )


class TonalBalanceWidget(QWidget):
    """
    Vintage-styled tonal balance meter.

    Shows 4 frequency bands (Low, Low-Mid, Hi-Mid, High) as vertical bars
    with target zones overlaid. Neve/SSL warm color scheme.
    """

    BAND_NAMES = ["LOW", "LOW-MID", "HI-MID", "HIGH"]
    BAND_COLORS = ["#D4845A", "#D4A84B", "#8BAF6E", "#6BAED6"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Data
        self._band_energies = [0.0, 0.0, 0.0, 0.0]  # dB relative
        self._target_ranges = [(-2, 6), (-3, 3), (-4.5, 2.5), (-7, 1)]  # (min, max)
        self._in_range = [True, True, True, True]
        self._score = 0.0
        self._target_name = "Modern"

    def set_data(self, energies, target_ranges, in_range, score, target_name=""):
        """Update display data."""
        self._band_energies = list(energies) if energies is not None else [0.0] * 4
        self._target_ranges = list(target_ranges)
        self._in_range = list(in_range)
        self._score = score
        if target_name:
            self._target_name = target_name
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        # Background
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0.0, QColor("#1A1510"))
        bg.setColorAt(1.0, QColor("#0E0A06"))
        p.fillRect(0, 0, w, h, QBrush(bg))

        # Border
        p.setPen(QPen(QColor("#382E28"), 1))
        p.drawRect(0, 0, w - 1, h - 1)

        # Layout
        margin_l = 10
        margin_r = 10
        margin_t = 26
        margin_b = 30
        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b

        # Title + Score
        p.setFont(QFont("Georgia", 8, QFont.Weight.Bold))
        p.setPen(QColor("#C89B3C"))
        p.drawText(margin_l, 4, plot_w, 18, Qt.AlignmentFlag.AlignLeft, f"TONAL BALANCE — {self._target_name.upper()}")

        score_color = QColor("#5C9A3C") if self._score >= 70 else QColor("#E8C830") if self._score >= 40 else QColor("#CC3333")
        p.setPen(score_color)
        p.drawText(margin_l, 4, plot_w, 18, Qt.AlignmentFlag.AlignRight, f"{self._score:.0f}%")

        # dB range for plot
        db_min = -12.0
        db_max = 12.0
        db_range = db_max - db_min

        def db_to_y(db_val):
            ratio = (db_val - db_min) / db_range
            return margin_t + plot_h * (1.0 - ratio)

        # Zero line
        zero_y = db_to_y(0)
        p.setPen(QPen(QColor("#4A4238"), 1, Qt.PenStyle.DashLine))
        p.drawLine(margin_l, int(zero_y), margin_l + plot_w, int(zero_y))

        # Grid lines
        p.setFont(QFont("Menlo", 6))
        p.setPen(QColor("#3A3228"))
        for db in [-9, -6, -3, 0, 3, 6, 9]:
            y = db_to_y(db)
            p.drawLine(margin_l, int(y), margin_l + plot_w, int(y))
            if db != 0:
                p.setPen(QColor("#5A5044"))
                p.drawText(0, int(y) - 5, margin_l - 2, 10,
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                           f"{db:+d}")
                p.setPen(QColor("#3A3228"))

        # Draw bands
        band_w = plot_w / 4
        gap = 4

        for i in range(4):
            bx = margin_l + i * band_w
            bar_x = bx + gap
            bar_w = band_w - gap * 2

            # Target zone (shaded region)
            t_min, t_max = self._target_ranges[i]
            zone_y_top = db_to_y(min(t_max, db_max))
            zone_y_bot = db_to_y(max(t_min, db_min))
            zone_h = zone_y_bot - zone_y_top

            zone_color = QColor(self.BAND_COLORS[i])
            zone_color.setAlpha(35)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(zone_color))
            p.drawRect(QRectF(bar_x, zone_y_top, bar_w, zone_h))

            # Target zone border
            border_color = QColor(self.BAND_COLORS[i])
            border_color.setAlpha(80)
            p.setPen(QPen(border_color, 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(bar_x, zone_y_top, bar_w, zone_h))

            # Energy bar
            energy = self._band_energies[i]
            energy_clamped = max(db_min, min(db_max, energy))
            bar_y = db_to_y(energy_clamped)

            bar_color = QColor(self.BAND_COLORS[i])
            if not self._in_range[i]:
                bar_color = QColor("#CC3333")

            # Draw bar from zero to energy
            if energy_clamped >= 0:
                fill_top = bar_y
                fill_h = zero_y - bar_y
            else:
                fill_top = zero_y
                fill_h = bar_y - zero_y

            grad = QLinearGradient(bar_x, fill_top, bar_x, fill_top + fill_h)
            grad.setColorAt(0.0, bar_color)
            bar_dark = QColor(bar_color)
            bar_dark.setAlpha(150)
            grad.setColorAt(1.0, bar_dark)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(QRectF(bar_x + 2, fill_top, bar_w - 4, max(fill_h, 2)), 2, 2)

            # Energy dot indicator
            p.setPen(Qt.PenStyle.NoPen)
            dot_color = QColor(bar_color)
            dot_color.setAlpha(220)
            p.setBrush(QBrush(dot_color))
            p.drawEllipse(QRectF(bar_x + bar_w / 2 - 3, bar_y - 3, 6, 6))

            # Band label
            p.setFont(QFont("Georgia", 6, QFont.Weight.Bold))
            label_color = QColor(self.BAND_COLORS[i]) if self._in_range[i] else QColor("#CC3333")
            p.setPen(label_color)
            p.drawText(int(bx), h - margin_b + 2, int(band_w), 12,
                       Qt.AlignmentFlag.AlignCenter, self.BAND_NAMES[i])

            # Energy value
            p.setFont(QFont("Menlo", 6))
            p.setPen(QColor("#F0E8D8"))
            p.drawText(int(bx), h - margin_b + 14, int(band_w), 10,
                       Qt.AlignmentFlag.AlignCenter, f"{energy:+.1f}dB")

        p.end()
