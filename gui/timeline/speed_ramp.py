"""
Speed ramp with curve editor — variable playback speed via bezier curves.

Story 3.5 — Epic 3: CapCut Features.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from gui.utils.compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QDoubleSpinBox,
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    Qt, QRectF, QPointF, QSize, QSizePolicy,
    QPainter, QPen, QBrush, QColor, QPainterPath, QFont,
    pyqtSignal,
)


# ---------------------------------------------------------------------------
# Bezier curve helpers (shared logic, kept local to avoid circular imports)
# ---------------------------------------------------------------------------

def _cubic_bezier(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
    u = 1.0 - t
    return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Speed ramp presets
# ---------------------------------------------------------------------------

class SpeedPreset(Enum):
    """Built-in speed ramp curves."""
    NORMAL = auto()
    EASE_IN = auto()
    EASE_OUT = auto()
    FLASH = auto()
    MONTAGE = auto()
    BULLET_TIME = auto()


# Preset definitions: list of (time_normalized, speed_multiplier) control points
# Each preset is a list of BezierCurve control points
_PRESET_CURVES: Dict[SpeedPreset, List[Tuple[float, float]]] = {
    SpeedPreset.NORMAL: [
        (0.0, 1.0), (0.33, 1.0), (0.67, 1.0), (1.0, 1.0),
    ],
    SpeedPreset.EASE_IN: [
        (0.0, 0.3), (0.33, 0.3), (0.67, 1.0), (1.0, 1.0),
    ],
    SpeedPreset.EASE_OUT: [
        (0.0, 1.0), (0.33, 1.0), (0.67, 0.3), (1.0, 0.3),
    ],
    SpeedPreset.FLASH: [
        (0.0, 1.0), (0.2, 4.0), (0.8, 4.0), (1.0, 1.0),
    ],
    SpeedPreset.MONTAGE: [
        (0.0, 2.0), (0.25, 0.5), (0.75, 3.0), (1.0, 1.0),
    ],
    SpeedPreset.BULLET_TIME: [
        (0.0, 1.0), (0.3, 0.1), (0.7, 0.1), (1.0, 1.0),
    ],
}


# ---------------------------------------------------------------------------
# SpeedRamp model
# ---------------------------------------------------------------------------

@dataclass
class SpeedRamp:
    """Variable-speed ramp defined by a bezier curve.

    Attributes:
        id:                   Unique identifier.
        control_points:       List of (normalised_time, speed_multiplier) points.
        audio_pitch_correct:  Whether to pitch-correct audio when speed changes.
        clip_id:              The clip this ramp applies to.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    control_points: List[Tuple[float, float]] = field(
        default_factory=lambda: list(_PRESET_CURVES[SpeedPreset.NORMAL])
    )
    audio_pitch_correct: bool = True
    clip_id: str = ""

    def get_speed_at(self, progress: float) -> float:
        """Evaluate speed multiplier at normalised *progress* (0..1).

        Uses cubic bezier interpolation through the control points.
        Result clamped to [0.1, 10.0].
        """
        progress = _clamp(progress, 0.0, 1.0)
        pts = self.control_points
        if len(pts) < 2:
            return 1.0

        if len(pts) == 4:
            # Cubic bezier on the speed (y) values
            speed = _cubic_bezier(progress, pts[0][1], pts[1][1], pts[2][1], pts[3][1])
        else:
            # Linear interpolation fallback for arbitrary point counts
            speed = self._lerp_points(progress)

        return _clamp(speed, 0.1, 10.0)

    def _lerp_points(self, progress: float) -> float:
        pts = self.control_points
        if not pts:
            return 1.0
        if progress <= pts[0][0]:
            return pts[0][1]
        if progress >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            t0, v0 = pts[i]
            t1, v1 = pts[i + 1]
            if t0 <= progress <= t1:
                span = t1 - t0
                if span <= 0:
                    return v0
                frac = (progress - t0) / span
                return v0 + (v1 - v0) * frac
        return 1.0

    @classmethod
    def from_preset(cls, preset: SpeedPreset, clip_id: str = "") -> "SpeedRamp":
        """Create a SpeedRamp from a built-in preset."""
        pts = list(_PRESET_CURVES.get(preset, _PRESET_CURVES[SpeedPreset.NORMAL]))
        return cls(control_points=pts, clip_id=clip_id)

    # -- FFmpeg filter generation -------------------------------------------

    def to_ffmpeg_filter(self, clip_duration: float = 10.0) -> str:
        """Generate setpts and atempo FFmpeg filters for this speed ramp.

        For variable speed, we approximate with a PTS expression.
        Returns a comma-separated filter string.
        """
        filters: List[str] = []

        # Sample the curve and build a PTS expression
        # For simplicity we use the average speed for setpts
        n_samples = 100
        speeds = [self.get_speed_at(i / n_samples) for i in range(n_samples + 1)]
        avg_speed = sum(speeds) / len(speeds)

        if abs(avg_speed - 1.0) > 0.01:
            # Video: setpts scales presentation timestamps
            pts_factor = 1.0 / avg_speed
            filters.append(f"setpts={pts_factor:.4f}*PTS")

            # Audio: atempo (must be between 0.5 and 100, chain if needed)
            audio_filters = self._build_atempo(avg_speed)
            if audio_filters:
                filters.append(audio_filters)

        return ",".join(filters) if filters else ""

    def _build_atempo(self, speed: float) -> str:
        """Build chained atempo filters for speeds outside [0.5, 2.0]."""
        if not self.audio_pitch_correct:
            return ""
        if speed <= 0:
            return "atempo=1.0"

        parts: List[str] = []
        remaining = speed
        # atempo only accepts 0.5 to 100.0; chain for extremes
        while remaining > 2.0:
            parts.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            parts.append("atempo=0.5")
            remaining /= 0.5
        parts.append(f"atempo={remaining:.4f}")
        return ",".join(parts)


# ---------------------------------------------------------------------------
# SpeedCurveEditor widget
# ---------------------------------------------------------------------------

class SpeedCurveScene(QGraphicsScene):
    """Scene that draws the speed ramp curve with draggable control points."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ramp: Optional[SpeedRamp] = None
        self._width = 500.0
        self._height = 200.0
        self.setSceneRect(0, 0, self._width, self._height)

    def set_ramp(self, ramp: SpeedRamp) -> None:
        self._ramp = ramp
        self._rebuild()

    def _rebuild(self) -> None:
        self.clear()
        if self._ramp is None:
            return

        margin = 25.0
        w = self._width - 2 * margin
        h = self._height - 2 * margin
        speed_min = 0.1
        speed_max = 10.0

        def to_scene(t: float, speed: float) -> QPointF:
            sx = margin + t * w
            # Log scale for better visual
            log_min = math.log10(speed_min)
            log_max = math.log10(speed_max)
            log_s = math.log10(_clamp(speed, speed_min, speed_max))
            sy = margin + h - (log_s - log_min) / (log_max - log_min) * h
            return QPointF(sx, sy)

        # Grid lines
        grid_pen = QPen(QColor("#333333"), 1)
        for spd in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            if speed_min <= spd <= speed_max:
                pt = to_scene(0, spd)
                self.addLine(margin, pt.y(), margin + w, pt.y(), grid_pen)
                label = self.addText(f"{spd}x", QFont("Segoe UI", 7))
                label.setDefaultTextColor(QColor("#666666"))
                label.setPos(2, pt.y() - 8)

        # 1x reference line
        ref_pen = QPen(QColor("#555555"), 1, Qt.PenStyle.DashLine)
        ref_pt = to_scene(0, 1.0)
        self.addLine(margin, ref_pt.y(), margin + w, ref_pt.y(), ref_pen)

        # Draw the curve
        curve_pen = QPen(QColor("#FF6F00"), 2)
        path = QPainterPath()
        steps = max(int(w), 100)
        for i in range(steps + 1):
            progress = i / steps
            speed = self._ramp.get_speed_at(progress)
            pt = to_scene(progress, speed)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        self.addPath(path, curve_pen)

        # Control point handles
        handle_pen = QPen(QColor("#FFFFFF"), 1)
        handle_brush = QBrush(QColor("#FF6F00"))
        for cp_t, cp_s in self._ramp.control_points:
            pt = to_scene(cp_t, cp_s)
            handle = self.addEllipse(pt.x() - 5, pt.y() - 5, 10, 10, handle_pen, handle_brush)
            handle.setToolTip(f"t={cp_t:.2f}  speed={cp_s:.2f}x")
            handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)


class SpeedCurveEditor(QWidget):
    """Widget for editing speed ramp curves."""

    speed_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ramp: Optional[SpeedRamp] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Speed Ramp"))

        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Custom")
        for sp in SpeedPreset:
            self._preset_combo.addItem(sp.name.replace("_", " ").title(), sp)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        toolbar.addWidget(QLabel("Preset:"))
        toolbar.addWidget(self._preset_combo)

        self._pitch_check = QCheckBox("Pitch Correct")
        self._pitch_check.setChecked(True)
        self._pitch_check.stateChanged.connect(self._on_pitch_changed)
        toolbar.addWidget(self._pitch_check)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Curve view
        self._scene = SpeedCurveScene()
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setMinimumHeight(180)
        self._view.setStyleSheet("background: #1A1A1A; border: 1px solid #333;")
        layout.addWidget(self._view)

        # Speed range label
        range_label = QLabel("Range: 0.1x — 10x")
        range_label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(range_label)

    # -- public API ---------------------------------------------------------

    def set_ramp(self, ramp: SpeedRamp) -> None:
        self._ramp = ramp
        self._pitch_check.setChecked(ramp.audio_pitch_correct)
        self._scene.set_ramp(ramp)

    def get_ramp(self) -> Optional[SpeedRamp]:
        return self._ramp

    # -- slots --------------------------------------------------------------

    def _on_preset_changed(self, index: int) -> None:
        if index <= 0 or self._ramp is None:
            return
        preset = self._preset_combo.itemData(index)
        if preset is not None:
            self._ramp.control_points = list(
                _PRESET_CURVES.get(preset, _PRESET_CURVES[SpeedPreset.NORMAL])
            )
            self._scene.set_ramp(self._ramp)
            self.speed_changed.emit()

    def _on_pitch_changed(self, state: int) -> None:
        if self._ramp is not None:
            self._ramp.audio_pitch_correct = bool(state)
            self.speed_changed.emit()
