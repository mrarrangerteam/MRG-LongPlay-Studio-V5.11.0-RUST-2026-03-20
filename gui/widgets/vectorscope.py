"""
Vectorscope display for stereo field visualization.

Story P3-8 — Phase 3: Ozone Clone.

QPainter half-circle display showing:
    - L-R on X axis, L+R on Y axis
    - Fading phosphor-style dots (teal/cyan)
    - Center line = mono, spread = stereo width
    - Correlation meter below (-1 to +1)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from gui.utils.compat import (
    QWidget, QPainter, QPen, QColor, QFont, QBrush, QRectF, QPointF,
    Qt, QSizePolicy, QTimer, QPainterPath, QLinearGradient,
)

OZ_BG = "#1a1a2e"
OZ_GRID = "#2a2a44"
OZ_TEAL = "#00d4aa"
OZ_CYAN = "#00ccff"
OZ_TEXT = "#888899"
MAX_POINTS = 2048
DECAY_ALPHA = 4  # alpha reduction per tick


class Vectorscope(QWidget):
    """Half-circle vectorscope with phosphor-style dot rendering."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._points: deque = deque(maxlen=MAX_POINTS)
        self._correlation: float = 1.0
        self._width_pct: float = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._decay)

    def start(self) -> None:
        self._timer.start(33)

    def stop(self) -> None:
        self._timer.stop()
        self._points.clear()
        self.update()

    def feed_samples(self, left: np.ndarray, right: np.ndarray) -> None:
        """Feed stereo samples for display."""
        n = min(len(left), len(right), 512)
        if n == 0:
            return
        l, r = left[:n], right[:n]
        x = (l - r) * 0.707
        y = (l + r) * 0.707
        for i in range(0, n, 4):  # downsample for perf
            self._points.append((float(x[i]), float(y[i]), 180))

        # correlation
        lr = np.sum(l * r)
        ll = np.sum(l * l)
        rr = np.sum(r * r)
        denom = math.sqrt(max(ll * rr, 1e-20))
        self._correlation = float(lr / denom) if denom > 0 else 1.0
        self._width_pct = (1.0 - self._correlation) * 100.0

    def _decay(self) -> None:
        new_pts = deque(maxlen=MAX_POINTS)
        for x, y, a in self._points:
            na = a - DECAY_ALPHA
            if na > 0:
                new_pts.append((x, y, na))
        self._points = new_pts
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        corr_h = 20
        scope_h = h - corr_h - 4
        cx, cy = w / 2.0, scope_h

        # background
        p.fillRect(self.rect(), QColor(OZ_BG))

        # semi-circle outline
        radius = min(w / 2.0 - 10, scope_h - 10)
        p.setPen(QPen(QColor(OZ_GRID), 1))
        arc_rect = QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius)
        p.drawArc(arc_rect, 0, 180 * 16)

        # grid lines
        p.setPen(QPen(QColor(OZ_GRID), 1, Qt.PenStyle.DotLine))
        p.drawLine(QPointF(cx, cy), QPointF(cx, cy - radius))  # center (mono)
        p.drawLine(QPointF(cx - radius, cy), QPointF(cx + radius, cy))  # baseline

        # L/R labels
        p.setFont(QFont("Inter", 7))
        p.setPen(QColor(OZ_TEXT))
        p.drawText(QRectF(cx - radius - 5, cy - 12, 15, 12), Qt.AlignmentFlag.AlignCenter, "L")
        p.drawText(QRectF(cx + radius - 8, cy - 12, 15, 12), Qt.AlignmentFlag.AlignCenter, "R")

        # points
        for x, y, alpha in self._points:
            px = cx + x * radius * 0.9
            py_val = cy - abs(y) * radius * 0.9
            if py_val < cy - radius:
                continue
            color = QColor(OZ_TEAL)
            color.setAlpha(min(255, alpha))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawEllipse(QPointF(px, py_val), 1.5, 1.5)

        # correlation meter bar
        corr_y = h - corr_h
        bar_w = w - 20
        bar_x = 10

        p.setPen(QPen(QColor(OZ_GRID), 1))
        p.drawRect(QRectF(bar_x, corr_y, bar_w, corr_h - 2))

        # fill based on correlation
        fill_ratio = (self._correlation + 1.0) / 2.0  # -1..1 → 0..1
        fill_w = fill_ratio * bar_w
        if self._correlation > 0:
            color = QColor(OZ_TEAL)
        else:
            color = QColor("#ff4444")
        color.setAlpha(180)
        p.fillRect(QRectF(bar_x, corr_y, fill_w, corr_h - 2), color)

        # correlation label
        p.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        p.setPen(QColor("#ffffff"))
        p.drawText(QRectF(bar_x, corr_y, bar_w, corr_h - 2),
                   Qt.AlignmentFlag.AlignCenter,
                   f"Corr: {self._correlation:.2f}  Width: {self._width_pct:.0f}%")

        p.end()
