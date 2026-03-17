"""
Transitions library — 12+ transition types via FFmpeg xfade filter.

Story 3.3 — Epic 3: CapCut Features.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from gui.utils.compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QComboBox, QGroupBox,
    Qt, QRectF, QPointF, QSize,
    QPainter, QPen, QBrush, QColor, QFont, QIcon,
    QGraphicsRectItem, QGraphicsItem,
    pyqtSignal,
)


# ---------------------------------------------------------------------------
# Transition types
# ---------------------------------------------------------------------------

class TransitionType(Enum):
    """Available transition effects (mapped to FFmpeg xfade names)."""
    CROSSFADE = "fade"
    DISSOLVE = "dissolve"
    WIPE_LEFT = "wipeleft"
    WIPE_RIGHT = "wiperight"
    WIPE_UP = "wipeup"
    WIPE_DOWN = "wipedown"
    ZOOM_IN = "zoomin"
    ZOOM_OUT = "fadefast"
    SLIDE_LEFT = "slideleft"
    SLIDE_RIGHT = "slideright"
    IRIS = "circleopen"
    FADE_TO_BLACK = "fadeblack"


# Human-readable labels
_TRANSITION_LABELS: Dict[TransitionType, str] = {
    TransitionType.CROSSFADE: "Crossfade",
    TransitionType.DISSOLVE: "Dissolve",
    TransitionType.WIPE_LEFT: "Wipe Left",
    TransitionType.WIPE_RIGHT: "Wipe Right",
    TransitionType.WIPE_UP: "Wipe Up",
    TransitionType.WIPE_DOWN: "Wipe Down",
    TransitionType.ZOOM_IN: "Zoom In",
    TransitionType.ZOOM_OUT: "Zoom Out",
    TransitionType.SLIDE_LEFT: "Slide Left",
    TransitionType.SLIDE_RIGHT: "Slide Right",
    TransitionType.IRIS: "Iris (Circle)",
    TransitionType.FADE_TO_BLACK: "Fade to Black",
}

# Icon/colour hints per transition
_TRANSITION_COLORS: Dict[TransitionType, str] = {
    TransitionType.CROSSFADE: "#4FC3F7",
    TransitionType.DISSOLVE: "#81D4FA",
    TransitionType.WIPE_LEFT: "#AED581",
    TransitionType.WIPE_RIGHT: "#AED581",
    TransitionType.WIPE_UP: "#CE93D8",
    TransitionType.WIPE_DOWN: "#CE93D8",
    TransitionType.ZOOM_IN: "#FFB74D",
    TransitionType.ZOOM_OUT: "#FFB74D",
    TransitionType.SLIDE_LEFT: "#4DD0E1",
    TransitionType.SLIDE_RIGHT: "#4DD0E1",
    TransitionType.IRIS: "#F48FB1",
    TransitionType.FADE_TO_BLACK: "#90A4AE",
}


class EasingType(Enum):
    """Easing for transition timing."""
    LINEAR = "linear"
    EASE_IN = "easing_in"
    EASE_OUT = "easing_out"
    EASE_IN_OUT = "easing_in_out"


# ---------------------------------------------------------------------------
# Transition model
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """A transition between two adjacent clips.

    Attributes:
        id:         Unique identifier.
        type:       The transition effect.
        duration:   Length in seconds (0.1–5.0).
        easing:     Easing curve type.
        clip_a_id:  ID of the outgoing clip.
        clip_b_id:  ID of the incoming clip.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: TransitionType = TransitionType.CROSSFADE
    duration: float = 1.0
    easing: EasingType = EasingType.LINEAR
    clip_a_id: str = ""
    clip_b_id: str = ""

    def __post_init__(self) -> None:
        self.duration = max(0.1, min(5.0, self.duration))

    def to_ffmpeg_filter(self, offset: float) -> str:
        """Generate an FFmpeg xfade filter string.

        *offset* is the timeline time (seconds) where the transition begins.
        Returns a filter like ``xfade=transition=fade:duration=1:offset=5``.
        """
        return (
            f"xfade=transition={self.type.value}"
            f":duration={self.duration:.3f}"
            f":offset={offset:.3f}"
        )

    @property
    def label(self) -> str:
        return _TRANSITION_LABELS.get(self.type, self.type.name)

    @property
    def color(self) -> str:
        return _TRANSITION_COLORS.get(self.type, "#888888")


# ---------------------------------------------------------------------------
# TransitionItem — diamond icon between clips on the timeline
# ---------------------------------------------------------------------------

class TransitionItem(QGraphicsRectItem):
    """Visual diamond between two clips representing a transition."""

    SIZE = 16

    def __init__(
        self,
        transition: Transition,
        x: float,
        y: float,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        half = self.SIZE / 2.0
        super().__init__(-half, -half, self.SIZE, self.SIZE, parent)
        self.transition = transition
        self.setPos(x, y)
        self.setRotation(45.0)
        self.setBrush(QBrush(QColor(transition.color)))
        self.setPen(QPen(QColor("#FFFFFF"), 1))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setToolTip(f"{transition.label}  ({transition.duration:.1f}s)")


# ---------------------------------------------------------------------------
# TransitionLibraryPanel — browseable panel of available transitions
# ---------------------------------------------------------------------------

class TransitionLibraryPanel(QWidget):
    """Panel showing available transitions with thumbnails/icons."""

    transition_selected = pyqtSignal(object)  # emits TransitionType

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Transitions")
        header.setStyleSheet("color: #FFFFFF; font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        # Duration control
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Duration:"))
        self._dur_spin = QDoubleSpinBox()
        self._dur_spin.setRange(0.1, 5.0)
        self._dur_spin.setSingleStep(0.1)
        self._dur_spin.setValue(1.0)
        self._dur_spin.setSuffix(" s")
        dur_row.addWidget(self._dur_spin)
        layout.addLayout(dur_row)

        # Easing
        ease_row = QHBoxLayout()
        ease_row.addWidget(QLabel("Easing:"))
        self._ease_combo = QComboBox()
        for et in EasingType:
            self._ease_combo.addItem(et.name.replace("_", " ").title(), et)
        ease_row.addWidget(self._ease_combo)
        layout.addLayout(ease_row)

        # List of transitions
        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget { background: #2A2A2A; border: none; }"
            "QListWidget::item { padding: 8px; color: #DDDDDD; }"
            "QListWidget::item:selected { background: #3A3A3A; }"
        )
        for tt in TransitionType:
            item = QListWidgetItem(f"  {_TRANSITION_LABELS.get(tt, tt.name)}")
            item.setData(Qt.ItemDataRole.UserRole, tt)
            self._list.addItem(item)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._list)

        # Apply button
        apply_btn = QPushButton("Apply Transition")
        apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(apply_btn)

    # -- public API ---------------------------------------------------------

    def get_selected_transition(self) -> Optional[Transition]:
        """Create a Transition from the current selection, or None."""
        item = self._list.currentItem()
        if item is None:
            return None
        tt = item.data(Qt.ItemDataRole.UserRole)
        easing = self._ease_combo.currentData()
        return Transition(
            type=tt,
            duration=self._dur_spin.value(),
            easing=easing if easing is not None else EasingType.LINEAR,
        )

    # -- slots --------------------------------------------------------------

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        tt = item.data(Qt.ItemDataRole.UserRole)
        if tt is not None:
            self.transition_selected.emit(tt)

    def _on_apply(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            tt = item.data(Qt.ItemDataRole.UserRole)
            if tt is not None:
                self.transition_selected.emit(tt)
