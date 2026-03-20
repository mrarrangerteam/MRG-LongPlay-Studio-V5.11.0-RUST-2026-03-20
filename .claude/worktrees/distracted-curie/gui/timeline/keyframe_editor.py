"""
Keyframe editor widget — diamond markers on clips with editable curves.

Story 3.1 — Epic 3: CapCut Features.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from gui.utils.compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem,
    Qt, QRectF, QPointF, QSize, QSizePolicy,
    QPainter, QPen, QBrush, QColor, QPainterPath, QFont,
    pyqtSignal,
)

from gui.models.keyframes import (
    KeyframeType, Keyframe, KeyframeTrack, interpolate_value,
)


# ---------------------------------------------------------------------------
# Diamond marker for a single keyframe
# ---------------------------------------------------------------------------

class KeyframeDiamond(QGraphicsRectItem):
    """A small diamond shape representing one keyframe on the timeline."""

    SIZE = 10

    def __init__(self, keyframe: Keyframe, x: float, y: float, parent: Optional[QGraphicsItem] = None) -> None:
        half = self.SIZE / 2.0
        super().__init__(-half, -half, self.SIZE, self.SIZE, parent)
        self.keyframe = keyframe
        self.setPos(x, y)
        self.setRotation(45.0)
        self.setBrush(QBrush(QColor("#FFD740")))
        self.setPen(QPen(QColor("#FF6F00"), 1))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setToolTip(f"t={keyframe.time:.2f}  v={keyframe.value:.2f}")


# ---------------------------------------------------------------------------
# Curve display scene
# ---------------------------------------------------------------------------

class KeyframeCurveScene(QGraphicsScene):
    """Scene that draws the interpolation curve for a KeyframeTrack."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._track: Optional[KeyframeTrack] = None
        self._diamonds: List[KeyframeDiamond] = []
        self._width = 600.0
        self._height = 200.0
        self.setSceneRect(0, 0, self._width, self._height)

    # -- public API ---------------------------------------------------------

    def set_track(self, track: KeyframeTrack) -> None:
        self._track = track
        self._rebuild()

    def refresh(self) -> None:
        self._rebuild()

    # -- internal -----------------------------------------------------------

    def _rebuild(self) -> None:
        self.clear()
        self._diamonds.clear()
        if self._track is None or not self._track.keyframes:
            return

        kfs = self._track.keyframes
        t_min = kfs[0].time
        t_max = kfs[-1].time
        if t_max <= t_min:
            t_max = t_min + 1.0

        v_min = min(kf.value for kf in kfs)
        v_max = max(kf.value for kf in kfs)
        if v_max <= v_min:
            v_min -= 0.5
            v_max += 0.5

        margin = 20.0
        w = self._width - 2 * margin
        h = self._height - 2 * margin

        def to_scene(t: float, v: float) -> QPointF:
            sx = margin + (t - t_min) / (t_max - t_min) * w
            sy = margin + h - (v - v_min) / (v_max - v_min) * h
            return QPointF(sx, sy)

        # Draw interpolation curve
        pen = QPen(QColor("#00E5FF"), 2)
        steps = max(int(w), 100)
        path = QPainterPath()
        for i in range(steps + 1):
            t = t_min + (t_max - t_min) * i / steps
            val = self._track.get_value_at(t)
            if val is None:
                continue
            pt = to_scene(t, val)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        self.addPath(path, pen)

        # Draw diamond markers
        for kf in kfs:
            pt = to_scene(kf.time, kf.value)
            diamond = KeyframeDiamond(kf, pt.x(), pt.y())
            self.addItem(diamond)
            self._diamonds.append(diamond)

        # Axis labels
        font = QFont("Segoe UI", 8)
        for kf in kfs:
            pt = to_scene(kf.time, kf.value)
            label = self.addText(f"{kf.time:.1f}s", font)
            label.setDefaultTextColor(QColor("#AAAAAA"))
            label.setPos(pt.x() - 12, self._height - margin + 2)


# ---------------------------------------------------------------------------
# Main editor widget
# ---------------------------------------------------------------------------

class KeyframeEditor(QWidget):
    """Widget for editing keyframe tracks — curve view + controls."""

    keyframe_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._track: Optional[KeyframeTrack] = None
        self._setup_ui()

    # -- UI setup -----------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Top toolbar
        toolbar = QHBoxLayout()
        self._param_label = QLabel("Parameter: —")
        self._param_label.setStyleSheet("color: #CCCCCC; font-size: 12px;")
        toolbar.addWidget(self._param_label)
        toolbar.addStretch()

        self._interp_combo = QComboBox()
        for kt in KeyframeType:
            self._interp_combo.addItem(kt.name, kt)
        self._interp_combo.currentIndexChanged.connect(self._on_interp_changed)
        toolbar.addWidget(QLabel("Interpolation:"))
        toolbar.addWidget(self._interp_combo)

        self._add_btn = QPushButton("+ Add")
        self._add_btn.clicked.connect(self._on_add_keyframe)
        toolbar.addWidget(self._add_btn)

        self._del_btn = QPushButton("— Remove")
        self._del_btn.clicked.connect(self._on_remove_keyframe)
        toolbar.addWidget(self._del_btn)

        layout.addLayout(toolbar)

        # Curve view
        self._scene = KeyframeCurveScene()
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setMinimumHeight(180)
        self._view.setStyleSheet("background: #1E1E1E; border: 1px solid #333;")
        layout.addWidget(self._view)

    # -- public API ---------------------------------------------------------

    def set_track(self, track: KeyframeTrack) -> None:
        self._track = track
        self._param_label.setText(f"Parameter: {track.parameter_name}")
        self._scene.set_track(track)

    def refresh(self) -> None:
        if self._track:
            self._scene.set_track(self._track)

    # -- slots --------------------------------------------------------------

    def _on_interp_changed(self, index: int) -> None:
        if self._track is None:
            return
        kt = self._interp_combo.currentData()
        if kt is None:
            return
        # Apply to selected diamonds
        for diamond in self._scene._diamonds:
            if diamond.isSelected():
                diamond.keyframe.interpolation = kt
        self._scene.refresh()
        self.keyframe_changed.emit()

    def _on_add_keyframe(self) -> None:
        if self._track is None:
            return
        # Add at midpoint or end
        if self._track.keyframes:
            last = self._track.keyframes[-1]
            t = last.time + 1.0
            v = last.value
        else:
            t = 0.0
            v = 0.0
        kf = Keyframe(time=t, value=v)
        self._track.add_keyframe(kf)
        self._scene.set_track(self._track)
        self.keyframe_changed.emit()

    def _on_remove_keyframe(self) -> None:
        if self._track is None:
            return
        for diamond in list(self._scene._diamonds):
            if diamond.isSelected():
                self._track.remove_keyframe(diamond.keyframe.id)
        self._scene.set_track(self._track)
        self.keyframe_changed.emit()
