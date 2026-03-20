"""
OzoneRotaryKnob — Vintage hardware-style rotary knob widget (QPainter).

Inspired by Neve 1073, SSL, Chandler, Pultec outboard gear.
Brushed-metal body, chicken-head pointer, engraved markings, warm tones.
"""

import math
try:
    from PyQt6.QtWidgets import QWidget, QSizePolicy
    from PyQt6.QtCore import Qt, pyqtSignal, QPointF
    from PyQt6.QtGui import (
        QPainter, QPen, QBrush, QColor, QFont,
        QConicalGradient, QPainterPath, QRadialGradient, QLinearGradient,
    )
except ImportError:
    from PySide6.QtWidgets import QWidget, QSizePolicy
    from PySide6.QtCore import Qt, Signal as pyqtSignal, QPointF
    from PySide6.QtGui import (
        QPainter, QPen, QBrush, QColor, QFont,
        QConicalGradient, QPainterPath, QRadialGradient, QLinearGradient,
    )


class OzoneRotaryKnob(QWidget):
    """
    Vintage hardware-style rotary knob (Neve/SSL/Pultec aesthetic).

    Features:
    - Brushed metal 3D body with conical gradient
    - Chicken-head style pointer indicator
    - Engraved tick marks around the arc
    - Warm amber/cream color scheme
    - Center value readout with backlit LED look

    Signals:
        valueChanged(float): emitted when value changes
    """

    valueChanged = pyqtSignal(float)

    ARC_START = 225.0
    ARC_SPAN = -270.0

    def __init__(self, name: str = "", min_val: float = 0.0, max_val: float = 100.0,
                 default: float = 0.0, unit: str = "", decimals: int = 1,
                 large: bool = False, parent=None):
        super().__init__(parent)
        self._name = name
        self._min = min_val
        self._max = max_val
        self._default = default
        self._value = default
        self._unit = unit
        self._decimals = decimals

        size = 80 if large else 64
        self.setFixedSize(size, size + 28)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._dragging = False
        self._drag_start_y = 0
        self._drag_start_val = 0.0
        self._fine_mode = False

        # Vintage color palette (Neve/SSL/Pultec inspired)
        self._body_dark = QColor("#2C2520")       # Dark walnut body
        self._body_mid = QColor("#4A403A")         # Brushed metal mid
        self._body_light = QColor("#6B5E54")       # Metal highlight
        self._body_rim = QColor("#1A1510")         # Deep shadow rim
        self._track_bg = QColor("#3A3228")         # Engraved track
        self._arc_color = QColor("#D4A84B")        # Warm gold arc (Neve amber)
        self._arc_glow = QColor("#F5D080")         # Gold glow
        self._text_color = QColor("#F0E8D8")       # Warm cream text
        self._text_dim = QColor("#9C8E78")         # Aged label text
        self._indicator = QColor("#E8A832")        # Amber pointer
        self._pointer_line = QColor("#F5D080")     # Bright pointer line
        self._led_bg = QColor("#1A1208")           # LED readout background
        self._tick_color = QColor("#6B5E54")       # Engraved tick marks

    # ── Properties ──

    def value(self) -> float:
        return self._value

    def setValue(self, val: float):
        val = max(self._min, min(self._max, val))
        if val != self._value:
            self._value = val
            self.valueChanged.emit(self._value)
            self.update()

    def setRange(self, min_val: float, max_val: float):
        self._min = min_val
        self._max = max_val
        self.setValue(self._value)

    def setDefault(self, default: float):
        self._default = default

    # ── Value ↔ angle mapping ──

    def _val_to_ratio(self) -> float:
        rng = self._max - self._min
        if rng == 0:
            return 0.0
        return (self._value - self._min) / rng

    def _ratio_to_val(self, ratio: float) -> float:
        ratio = max(0.0, min(1.0, ratio))
        return self._min + ratio * (self._max - self._min)

    # ── Paint ──

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h_total = self.width(), self.height()
        knob_size = min(w, h_total - 28)
        cx = w / 2
        name_h = 12
        cy = name_h + knob_size / 2
        radius = knob_size / 2 - 4
        ratio = self._val_to_ratio()

        # ── Name label (engraved text above) ──
        if self._name:
            p.setFont(QFont("Georgia", 7, QFont.Weight.Bold))
            p.setPen(self._text_dim)
            p.drawText(0, 0, w, name_h, Qt.AlignmentFlag.AlignCenter, self._name.upper())

        # ── Outer shadow ring (depth effect) ──
        p.setPen(Qt.PenStyle.NoPen)
        shadow_grad = QRadialGradient(cx, cy, radius + 3)
        shadow_grad.setColorAt(0.0, QColor(0, 0, 0, 0))
        shadow_grad.setColorAt(0.7, QColor(0, 0, 0, 0))
        shadow_grad.setColorAt(0.85, QColor(0, 0, 0, 100))
        shadow_grad.setColorAt(1.0, QColor(0, 0, 0, 60))
        p.setBrush(QBrush(shadow_grad))
        p.drawEllipse(QPointF(cx, cy), radius + 3, radius + 3)

        # ── Engraved tick marks (like Neve faceplates) ──
        num_ticks = 11
        for i in range(num_ticks):
            tick_ratio = i / (num_ticks - 1)
            angle_deg = self.ARC_START + self.ARC_SPAN * tick_ratio
            angle_rad = math.radians(angle_deg)
            inner_r = radius + 1
            outer_r = radius + 4
            x1 = cx + inner_r * math.cos(angle_rad)
            y1 = cy - inner_r * math.sin(angle_rad)
            x2 = cx + outer_r * math.cos(angle_rad)
            y2 = cy - outer_r * math.sin(angle_rad)
            tick_pen = QPen(self._tick_color, 1.0)
            p.setPen(tick_pen)
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # ── Knob body (brushed metal with conical gradient — Neve style) ──
        body_grad = QConicalGradient(cx, cy, 135)
        body_grad.setColorAt(0.0, self._body_light)
        body_grad.setColorAt(0.25, self._body_mid)
        body_grad.setColorAt(0.5, self._body_dark)
        body_grad.setColorAt(0.75, self._body_mid)
        body_grad.setColorAt(1.0, self._body_light)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(body_grad))
        p.drawEllipse(QPointF(cx, cy), radius, radius)

        # ── Metal rim (beveled edge) ──
        rim_grad = QConicalGradient(cx, cy, 45)
        rim_grad.setColorAt(0.0, QColor("#8A7E72"))
        rim_grad.setColorAt(0.25, QColor("#5A5048"))
        rim_grad.setColorAt(0.5, QColor("#3A3228"))
        rim_grad.setColorAt(0.75, QColor("#5A5048"))
        rim_grad.setColorAt(1.0, QColor("#8A7E72"))
        p.setPen(QPen(QBrush(rim_grad), 2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), radius, radius)

        # ── Inner groove ring ──
        inner_r = radius - 5
        p.setPen(QPen(QColor("#1A1510"), 1.0))
        p.drawEllipse(QPointF(cx, cy), inner_r, inner_r)

        # ── Value arc (warm gold, active portion) ──
        arc_margin = 6
        arc_r = radius - arc_margin
        arc_rect = (cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2)

        # Track background (engraved groove)
        p.setPen(QPen(self._track_bg, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(int(arc_rect[0]), int(arc_rect[1]),
                  int(arc_rect[2]), int(arc_rect[3]),
                  int(self.ARC_START * 16), int(self.ARC_SPAN * 16))

        # Active gold arc
        if ratio > 0.001:
            val_span = self.ARC_SPAN * ratio
            p.setPen(QPen(self._arc_color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(int(arc_rect[0]), int(arc_rect[1]),
                      int(arc_rect[2]), int(arc_rect[3]),
                      int(self.ARC_START * 16), int(val_span * 16))

        # ── Chicken-head pointer (Neve/Chandler style) ──
        angle_deg = self.ARC_START + self.ARC_SPAN * ratio
        angle_rad = math.radians(angle_deg)

        # Pointer line from center to edge
        ptr_inner = 4
        ptr_outer = radius - 7
        px1 = cx + ptr_inner * math.cos(angle_rad)
        py1 = cy - ptr_inner * math.sin(angle_rad)
        px2 = cx + ptr_outer * math.cos(angle_rad)
        py2 = cy - ptr_outer * math.sin(angle_rad)
        p.setPen(QPen(self._pointer_line, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(px1, py1), QPointF(px2, py2))

        # Pointer tip dot (amber)
        dot_x = cx + (ptr_outer + 1) * math.cos(angle_rad)
        dot_y = cy - (ptr_outer + 1) * math.sin(angle_rad)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._indicator))
        p.drawEllipse(QPointF(dot_x, dot_y), 3.0, 3.0)

        # ── Center hub (metal cap) ──
        hub_r = 5
        hub_grad = QRadialGradient(cx - 1, cy - 1, hub_r)
        hub_grad.setColorAt(0.0, QColor("#8A7E72"))
        hub_grad.setColorAt(0.5, QColor("#5A5048"))
        hub_grad.setColorAt(1.0, QColor("#3A3228"))
        p.setBrush(QBrush(hub_grad))
        p.setPen(QPen(QColor("#1A1510"), 0.5))
        p.drawEllipse(QPointF(cx, cy), hub_r, hub_r)

        # ── LED-style value readout (below knob) ──
        val_text = f"{self._value:.{self._decimals}f}"
        led_y = name_h + knob_size + 1
        led_w = min(w - 4, 50)
        led_h = 13
        led_x = cx - led_w / 2

        # LED background (recessed)
        p.setPen(QPen(QColor("#0A0804"), 1))
        p.setBrush(QBrush(self._led_bg))
        p.drawRoundedRect(int(led_x), int(led_y), int(led_w), led_h, 2, 2)

        # Value text (warm amber glow)
        p.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        p.setPen(self._arc_glow)
        p.drawText(int(led_x), int(led_y), int(led_w), led_h,
                   Qt.AlignmentFlag.AlignCenter, val_text)

        # ── Unit label (engraved below LED) ──
        if self._unit:
            unit_y = led_y + led_h + 1
            p.setFont(QFont("Georgia", 6))
            p.setPen(self._text_dim)
            p.drawText(0, int(unit_y), w, 10, Qt.AlignmentFlag.AlignCenter, self._unit)

        p.end()

    # ── Mouse interaction ──

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_y = event.position().y()
            self._drag_start_val = self._value
            self._fine_mode = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.setCursor(Qt.CursorShape.BlankCursor)

    def mouseMoveEvent(self, event):
        if self._dragging:
            dy = self._drag_start_y - event.position().y()
            sensitivity = 0.1 if self._fine_mode else 1.0
            rng = self._max - self._min
            # 200px drag = full range
            delta = (dy / 200.0) * rng * sensitivity
            self.setValue(self._drag_start_val + delta)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.unsetCursor()

    def mouseDoubleClickEvent(self, event):
        self.setValue(self._default)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        rng = self._max - self._min
        step = rng * 0.01 * (1 if delta > 0 else -1)
        self.setValue(self._value + step)
