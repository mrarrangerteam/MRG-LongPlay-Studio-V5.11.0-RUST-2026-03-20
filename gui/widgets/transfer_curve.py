"""
Transfer curve display for dynamics compression visualization.

Story P3-9 — Phase 3: Ozone Clone.

QPainter widget showing:
    - Input dB on X axis, Output dB on Y axis
    - 1:1 reference line (no compression)
    - Actual compression curve with threshold, ratio, knee
    - Soft-knee interpolation around threshold
"""

from __future__ import annotations

import math
from typing import Optional

from gui.utils.compat import (
    QWidget, QPainter, QPen, QColor, QFont, QBrush, QRectF, QPointF,
    Qt, QSizePolicy, QPainterPath,
)

OZ_BG = "#1a1a2e"
OZ_GRID = "#2a2a44"
OZ_REF = "#333355"
OZ_CURVE = "#00d4aa"
OZ_KNEE = "#00ffcc"
OZ_THRESHOLD = "#ff6644"
OZ_TEXT = "#888899"

DB_MIN = -60.0
DB_MAX = 0.0


def _compute_gain(input_db: float, threshold: float, ratio: float, knee: float) -> float:
    """Compute output dB for a given input dB using compressor transfer function."""
    if knee <= 0:
        # hard knee
        if input_db <= threshold:
            return input_db
        return threshold + (input_db - threshold) / ratio

    # soft knee
    half_knee = knee / 2.0
    if input_db < threshold - half_knee:
        return input_db
    elif input_db > threshold + half_knee:
        return threshold + (input_db - threshold) / ratio
    else:
        # quadratic interpolation in knee region
        x = input_db - threshold + half_knee
        return input_db + ((1.0 / ratio) - 1.0) * (x * x) / (2.0 * knee)


class TransferCurveWidget(QWidget):
    """Dynamics transfer curve display — Ozone 12 style."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._threshold: float = -20.0
        self._ratio: float = 4.0
        self._knee: float = 6.0
        self._makeup: float = 0.0

    def set_params(self, threshold: float = -20.0, ratio: float = 4.0,
                   knee: float = 6.0, makeup: float = 0.0) -> None:
        self._threshold = threshold
        self._ratio = max(ratio, 1.0)
        self._knee = max(knee, 0.0)
        self._makeup = makeup
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin = 30
        pw = w - 2 * margin
        ph = h - 2 * margin

        p.fillRect(self.rect(), QColor(OZ_BG))

        def db_to_x(db: float) -> float:
            return margin + (db - DB_MIN) / (DB_MAX - DB_MIN) * pw

        def db_to_y(db: float) -> float:
            return margin + (1.0 - (db - DB_MIN) / (DB_MAX - DB_MIN)) * ph

        # grid
        p.setPen(QPen(QColor(OZ_GRID), 1, Qt.PenStyle.DotLine))
        for db in range(int(DB_MIN), int(DB_MAX) + 1, 12):
            x = db_to_x(db)
            y = db_to_y(db)
            p.drawLine(QPointF(x, margin), QPointF(x, margin + ph))
            p.drawLine(QPointF(margin, y), QPointF(margin + pw, y))

        # axis labels
        p.setFont(QFont("Inter", 7))
        p.setPen(QColor(OZ_TEXT))
        for db in [-48, -36, -24, -12, 0]:
            x = db_to_x(db)
            y = db_to_y(db)
            p.drawText(QRectF(x - 12, margin + ph + 2, 24, 12),
                       Qt.AlignmentFlag.AlignCenter, str(db))
            p.drawText(QRectF(2, y - 6, margin - 4, 12),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(db))

        # 1:1 reference line
        p.setPen(QPen(QColor(OZ_REF), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(db_to_x(DB_MIN), db_to_y(DB_MIN)),
                   QPointF(db_to_x(DB_MAX), db_to_y(DB_MAX)))

        # threshold line
        p.setPen(QPen(QColor(OZ_THRESHOLD), 1, Qt.PenStyle.DashDotLine))
        tx = db_to_x(self._threshold)
        p.drawLine(QPointF(tx, margin), QPointF(tx, margin + ph))

        # transfer curve
        path = QPainterPath()
        steps = 200
        first = True
        for i in range(steps + 1):
            input_db = DB_MIN + (DB_MAX - DB_MIN) * i / steps
            output_db = _compute_gain(input_db, self._threshold, self._ratio, self._knee)
            output_db += self._makeup
            output_db = max(DB_MIN, min(DB_MAX, output_db))
            x = db_to_x(input_db)
            y = db_to_y(output_db)
            if first:
                path.moveTo(x, y)
                first = False
            else:
                path.lineTo(x, y)

        p.setPen(QPen(QColor(OZ_CURVE), 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # knee region highlight
        if self._knee > 0:
            k_start = self._threshold - self._knee / 2.0
            k_end = self._threshold + self._knee / 2.0
            kx1 = db_to_x(max(DB_MIN, k_start))
            kx2 = db_to_x(min(DB_MAX, k_end))
            knee_color = QColor(OZ_KNEE)
            knee_color.setAlpha(30)
            p.fillRect(QRectF(kx1, margin, kx2 - kx1, ph), knee_color)

        # labels
        p.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        p.setPen(QColor(OZ_TEXT))
        p.drawText(QRectF(margin, h - 14, pw, 12), Qt.AlignmentFlag.AlignCenter, "INPUT (dB)")
        # vertical label would require rotation, skip for simplicity

        # info
        p.setFont(QFont("Courier New", 8))
        p.setPen(QColor("#ffffff"))
        info = f"T:{self._threshold:.0f}  R:{self._ratio:.1f}:1  K:{self._knee:.0f}"
        p.drawText(QRectF(margin + 4, margin + 2, pw, 14),
                   Qt.AlignmentFlag.AlignLeft, info)

        p.end()
