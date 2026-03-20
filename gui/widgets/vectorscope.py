"""
Vectorscope display for stereo field visualization.

3D/4D phosphor-style display:
    - Glowing neon dots with radial gradient halos
    - Multi-layer depth: older dots fade + shrink (Z-axis illusion)
    - Radial gradient background (deep space look)
    - Glowing arc with bloom effect
    - Animated correlation meter with gradient fill
    - Grid lines with perspective convergence
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

try:
    from PyQt6.QtGui import QRadialGradient, QConicalGradient
except ImportError:
    from PySide6.QtGui import QRadialGradient, QConicalGradient

# ── Colors ──
BG_CENTER = "#0d0d1a"
BG_EDGE = "#060610"
GRID_DIM = "#1a1a33"
GRID_BRIGHT = "#2a2a55"
NEON_CYAN = "#00e5ff"
NEON_TEAL = "#00d4aa"
NEON_PURPLE = "#b388ff"
NEON_PINK = "#ff4081"
GLOW_CYAN = "#00e5ff"
TEXT_DIM = "#667788"
TEXT_BRIGHT = "#aabbcc"
CORR_GOOD = "#00e5aa"
CORR_BAD = "#ff3355"

MAX_POINTS = 3000
DECAY_ALPHA = 3


class Vectorscope(QWidget):
    """3D phosphor-style vectorscope with glow effects."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._points: deque = deque(maxlen=MAX_POINTS)
        self._correlation: float = 1.0
        self._width_pct: float = 0.0
        self._peak_l: float = 0.0
        self._peak_r: float = 0.0

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
        for i in range(0, n, 3):
            self._points.append((float(x[i]), float(y[i]), 220))

        self._peak_l = float(np.max(np.abs(l)))
        self._peak_r = float(np.max(np.abs(r)))

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
        corr_h = 24
        scope_h = h - corr_h - 6
        cx, cy = w / 2.0, scope_h
        radius = min(w / 2.0 - 12, scope_h - 12)

        # ═══ Background: radial gradient (deep space) ═══
        bg_grad = QRadialGradient(cx, cy * 0.6, max(w, h) * 0.8)
        bg_grad.setColorAt(0.0, QColor(BG_CENTER))
        bg_grad.setColorAt(0.4, QColor("#0a0a18"))
        bg_grad.setColorAt(1.0, QColor(BG_EDGE))
        p.fillRect(self.rect(), QBrush(bg_grad))

        # ═══ Outer glow ring ═══
        for i in range(3):
            glow_color = QColor(NEON_CYAN)
            glow_color.setAlpha(15 - i * 4)
            p.setPen(QPen(glow_color, 3 - i))
            arc_rect = QRectF(cx - radius - i, cy - radius - i,
                              2 * (radius + i), 2 * (radius + i))
            p.drawArc(arc_rect, 0, 180 * 16)

        # ═══ Main arc with gradient stroke ═══
        arc_pen = QPen(QColor(NEON_CYAN), 1.5)
        arc_pen.setColor(QColor("#00b8d4"))
        p.setPen(arc_pen)
        arc_rect = QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius)
        p.drawArc(arc_rect, 0, 180 * 16)

        # ═══ Grid: perspective lines with depth fade ═══
        # Center vertical (mono axis)
        grad_center = QLinearGradient(cx, cy, cx, cy - radius)
        grad_center.setColorAt(0.0, QColor(GRID_BRIGHT))
        grad_center.setColorAt(0.5, QColor(GRID_DIM))
        grad_center.setColorAt(1.0, QColor("#0d0d1a"))
        p.setPen(QPen(QBrush(grad_center), 1))
        p.drawLine(QPointF(cx, cy), QPointF(cx, cy - radius))

        # Baseline
        grad_base = QLinearGradient(cx - radius, cy, cx + radius, cy)
        grad_base.setColorAt(0.0, QColor("#0d0d1a"))
        grad_base.setColorAt(0.3, QColor(GRID_BRIGHT))
        grad_base.setColorAt(0.7, QColor(GRID_BRIGHT))
        grad_base.setColorAt(1.0, QColor("#0d0d1a"))
        p.setPen(QPen(QBrush(grad_base), 1))
        p.drawLine(QPointF(cx - radius, cy), QPointF(cx + radius, cy))

        # 45-degree guide lines (subtle)
        p.setPen(QPen(QColor(GRID_DIM), 0.5, Qt.PenStyle.DotLine))
        for angle_deg in [45, 135]:
            rad = math.radians(angle_deg)
            ex = cx + math.cos(rad) * radius * 0.95
            ey = cy - math.sin(rad) * radius * 0.95
            p.drawLine(QPointF(cx, cy), QPointF(ex, ey))

        # Concentric depth rings (3D illusion)
        for ring in [0.33, 0.66]:
            ring_color = QColor(GRID_DIM)
            ring_color.setAlpha(30)
            p.setPen(QPen(ring_color, 0.5))
            r2 = radius * ring
            p.drawArc(QRectF(cx - r2, cy - r2, 2 * r2, 2 * r2), 0, 180 * 16)

        # ═══ L / R labels with glow ═══
        label_font = QFont("Menlo", 8, QFont.Weight.Bold)
        p.setFont(label_font)
        for text, lx in [("L", cx - radius - 2), ("R", cx + radius - 10)]:
            # Glow behind text
            glow = QColor(NEON_CYAN)
            glow.setAlpha(40)
            p.setPen(glow)
            p.drawText(QRectF(lx - 1, cy - 14, 15, 14), Qt.AlignmentFlag.AlignCenter, text)
            # Main text
            p.setPen(QColor(NEON_CYAN))
            p.drawText(QRectF(lx, cy - 15, 15, 14), Qt.AlignmentFlag.AlignCenter, text)

        # ═══ 3D Phosphor dots with glow halos ═══
        for x, y, alpha in self._points:
            px = cx + x * radius * 0.88
            py_val = cy - abs(y) * radius * 0.88
            if py_val < cy - radius:
                continue

            # Depth factor: newer dots (high alpha) are bigger and brighter
            depth = alpha / 220.0  # 0..1
            dot_size = 1.0 + depth * 2.5

            # Color shifts with depth: cyan → purple for older dots
            if depth > 0.6:
                dot_color = QColor(NEON_CYAN)
            elif depth > 0.3:
                dot_color = QColor(NEON_TEAL)
            else:
                dot_color = QColor(NEON_PURPLE)
            dot_color.setAlpha(min(255, int(alpha * 1.2)))

            # Glow halo (larger, transparent)
            if depth > 0.4:
                halo_color = QColor(dot_color)
                halo_color.setAlpha(int(alpha * 0.15))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(halo_color)
                p.drawEllipse(QPointF(px, py_val), dot_size * 3, dot_size * 3)

            # Core dot
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(dot_color)
            p.drawEllipse(QPointF(px, py_val), dot_size, dot_size)

            # Hot center (white core for newest dots)
            if depth > 0.8:
                core = QColor("#ffffff")
                core.setAlpha(int(alpha * 0.4))
                p.setBrush(core)
                p.drawEllipse(QPointF(px, py_val), dot_size * 0.4, dot_size * 0.4)

        # ═══ Correlation meter (3D bar with gradient) ═══
        corr_y = h - corr_h
        bar_w = w - 16
        bar_x = 8
        bar_h = corr_h - 4

        # Bar background with inset shadow
        bar_bg = QLinearGradient(bar_x, corr_y, bar_x, corr_y + bar_h)
        bar_bg.setColorAt(0.0, QColor("#080810"))
        bar_bg.setColorAt(0.3, QColor("#0c0c18"))
        bar_bg.setColorAt(1.0, QColor("#101020"))
        p.setPen(QPen(QColor("#1a1a33"), 1))
        p.setBrush(QBrush(bar_bg))
        p.drawRoundedRect(QRectF(bar_x, corr_y, bar_w, bar_h), 3, 3)

        # Fill based on correlation
        fill_ratio = (self._correlation + 1.0) / 2.0
        fill_w = max(2, fill_ratio * (bar_w - 4))

        if self._correlation > 0.3:
            fill_grad = QLinearGradient(bar_x + 2, corr_y, bar_x + 2, corr_y + bar_h)
            fill_grad.setColorAt(0.0, QColor("#00ffc8"))
            fill_grad.setColorAt(0.4, QColor(CORR_GOOD))
            fill_grad.setColorAt(1.0, QColor("#009966"))
        elif self._correlation > 0:
            fill_grad = QLinearGradient(bar_x + 2, corr_y, bar_x + 2, corr_y + bar_h)
            fill_grad.setColorAt(0.0, QColor("#ffcc00"))
            fill_grad.setColorAt(1.0, QColor("#cc8800"))
        else:
            fill_grad = QLinearGradient(bar_x + 2, corr_y, bar_x + 2, corr_y + bar_h)
            fill_grad.setColorAt(0.0, QColor("#ff5566"))
            fill_grad.setColorAt(1.0, QColor(CORR_BAD))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(fill_grad))
        p.drawRoundedRect(QRectF(bar_x + 2, corr_y + 2, fill_w, bar_h - 4), 2, 2)

        # Glossy highlight on top of bar
        gloss = QLinearGradient(bar_x, corr_y + 2, bar_x, corr_y + bar_h / 2)
        gloss.setColorAt(0.0, QColor(255, 255, 255, 30))
        gloss.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setBrush(QBrush(gloss))
        p.drawRoundedRect(QRectF(bar_x + 2, corr_y + 2, fill_w, bar_h / 2 - 2), 2, 2)

        # Correlation text with shadow
        text_font = QFont("Menlo", 8, QFont.Weight.Bold)
        p.setFont(text_font)
        label = f"Corr: {self._correlation:.2f}   Width: {self._width_pct:.0f}%"

        # Text shadow
        p.setPen(QColor(0, 0, 0, 120))
        p.drawText(QRectF(bar_x + 1, corr_y + 1, bar_w, bar_h),
                   Qt.AlignmentFlag.AlignCenter, label)
        # Main text
        p.setPen(QColor("#e0f0ff"))
        p.drawText(QRectF(bar_x, corr_y, bar_w, bar_h),
                   Qt.AlignmentFlag.AlignCenter, label)

        p.end()
