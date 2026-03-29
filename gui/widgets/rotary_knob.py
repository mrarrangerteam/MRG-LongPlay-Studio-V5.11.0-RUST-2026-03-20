"""
Ozone 12 style rotary knob widget.

Story P3-7 — Phase 3: Ozone Clone.

QPainter-drawn rotary knob with:
    - Dark circle with arc indicator
    - Value display in center
    - Mouse drag to adjust (vertical drag = value change)
    - Double-click to type exact value
    - Configurable range, default, step, unit suffix
    - Label below knob
"""

from __future__ import annotations

import math
from typing import Optional

from gui.utils.compat import (
    QWidget, QPainter, QPen, QColor, QFont, QBrush, QRectF, QPointF,
    Qt, QSizePolicy, pyqtSignal, QLineEdit, QVBoxLayout,
)


# Ozone 12 palette
OZ_BG = "#1a1a2e"
OZ_RING = "#2a2a44"
OZ_ARC = "#00d4aa"
OZ_ARC_DIM = "#005544"
OZ_TEXT = "#ffffff"
OZ_LABEL = "#888899"
OZ_POINTER = "#00ffcc"

ARC_START = 225    # degrees (bottom-left)
ARC_SPAN = 270     # total arc sweep


class RotaryKnob(QWidget):
    """Ozone 12 style rotary knob with value display."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str = "",
        min_val: float = 0.0,
        max_val: float = 100.0,
        default: float = 0.0,
        step: float = 0.1,
        suffix: str = "",
        decimals: int = 1,
        accent_color: str = OZ_ARC,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._min = min_val
        self._max = max_val
        self._value = default
        self._default = default
        self._step = step
        self._suffix = suffix
        self._decimals = decimals
        self._accent = QColor(accent_color)
        self._dragging = False
        self._drag_start_y = 0.0
        self._drag_start_val = 0.0

        self.setFixedSize(80, 100)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # -- public API --------------------------------------------------------

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, v: float) -> None:
        v = max(self._min, min(self._max, v))
        if v != self._value:
            self._value = v
            self.valueChanged.emit(v)
            self.update()

    def set_value(self, v: float) -> None:
        self.value = v

    def reset(self) -> None:
        self.value = self._default

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        knob_size = min(w, h - 20) - 4
        cx, cy = w / 2.0, knob_size / 2.0 + 2
        r = knob_size / 2.0

        # background ring
        p.setPen(QPen(QColor(OZ_RING), 3))
        p.setBrush(QBrush(QColor(OZ_BG)))
        rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        p.drawEllipse(rect)

        # inactive arc
        p.setPen(QPen(QColor(OZ_ARC_DIM), 3))
        p.drawArc(rect.adjusted(4, 4, -4, -4),
                   int(ARC_START * 16), int(ARC_SPAN * 16))

        # active arc
        ratio = (self._value - self._min) / max(self._max - self._min, 1e-9)
        active_span = ratio * ARC_SPAN
        p.setPen(QPen(self._accent, 4))
        p.drawArc(rect.adjusted(4, 4, -4, -4),
                   int(ARC_START * 16), -int(active_span * 16))

        # pointer line
        angle_deg = ARC_START - active_span
        angle_rad = math.radians(angle_deg)
        pr = r - 10
        px = cx + pr * math.cos(angle_rad)
        py = cy - pr * math.sin(angle_rad)
        p.setPen(QPen(QColor(OZ_POINTER), 2))
        p.drawLine(QPointF(cx, cy), QPointF(px, py))

        # center dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(OZ_BG))
        p.drawEllipse(QPointF(cx, cy), 6, 6)

        # value text
        val_str = f"{self._value:.{self._decimals}f}{self._suffix}"
        p.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        p.setPen(QColor(OZ_TEXT))
        p.drawText(QRectF(0, cy + r - 12, w, 16),
                   Qt.AlignmentFlag.AlignCenter, val_str)

        # label
        if self._label:
            p.setFont(QFont("Inter", 7, QFont.Weight.Bold))
            p.setPen(QColor(OZ_LABEL))
            p.drawText(QRectF(0, h - 14, w, 14),
                       Qt.AlignmentFlag.AlignCenter, self._label.upper())

        p.end()

    # -- mouse interaction -------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_y = event.position().y()
            self._drag_start_val = self._value
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            dy = self._drag_start_y - event.position().y()
            sensitivity = (self._max - self._min) / 200.0
            new_val = self._drag_start_val + dy * sensitivity
            self.value = round(new_val / self._step) * self._step
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.value = self._default

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta > 0:
            self.value = self._value + self._step
        elif delta < 0:
            self.value = self._value - self._step
        super().wheelEvent(event)
