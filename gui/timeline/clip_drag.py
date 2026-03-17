"""
Clip drag-and-drop behaviour for the multi-track timeline.

Provides a mixin / helper that adds:
    - Horizontal (time) and vertical (track) dragging of ClipItems
    - Ghost / shadow visual feedback during drag
    - Snap-to-grid during drag
    - Signal emission on drop: clip_moved(clip_id, new_track_id, new_start_time)
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

from gui.utils.compat import (
    QGraphicsRectItem, QGraphicsItem,
    Qt, QPointF, QRectF,
    QColor, QPen, QBrush,
)
from gui.models.track import Track

if TYPE_CHECKING:
    from gui.timeline.multi_track_timeline import MultiTrackTimeline, ClipItem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GHOST_OPACITY = 0.40
RULER_HEIGHT = 28
TRACK_GAP = 2


# ---------------------------------------------------------------------------
# DragState — bookkeeping for an in-progress drag
# ---------------------------------------------------------------------------
class DragState:
    """Transient state for a clip being dragged."""

    __slots__ = (
        "clip_item", "ghost", "original_x", "original_y",
        "original_track_index", "start_offset",
        "original_start_time", "original_track_id",
    )

    def __init__(self, clip_item: "ClipItem") -> None:
        self.clip_item: "ClipItem" = clip_item
        self.ghost: Optional[QGraphicsRectItem] = None
        rect = clip_item.rect()
        self.original_x: float = rect.x()
        self.original_y: float = rect.y()
        self.original_track_index: int = clip_item.track_index
        self.original_start_time: float = clip_item.clip.start_time
        self.original_track_id: str = clip_item.clip.track_id
        self.start_offset: float = 0.0  # cursor offset within clip


# ---------------------------------------------------------------------------
# Helper functions (called from MultiTrackTimeline)
# ---------------------------------------------------------------------------

def _track_y_ranges(tracks: List[Track]) -> List[tuple]:
    """Return list of (y_start, y_end, track) for each track."""
    ranges = []
    y = RULER_HEIGHT
    for track in tracks:
        ranges.append((y, y + track.height, track))
        y += track.height + TRACK_GAP
    return ranges


def begin_drag(
    timeline: "MultiTrackTimeline",
    clip_item: "ClipItem",
    scene_pos: QPointF,
) -> DragState:
    """Start dragging a ClipItem.  Creates a ghost rectangle."""
    state = DragState(clip_item)
    rect = clip_item.rect()
    state.start_offset = scene_pos.x() - rect.x()

    # Create ghost
    ghost = QGraphicsRectItem(rect)
    ghost.setBrush(QBrush(QColor(255, 255, 255, 60)))
    ghost.setPen(QPen(QColor("#FF9500"), 1, Qt.PenStyle.DashLine))
    ghost.setOpacity(GHOST_OPACITY)
    ghost.setZValue(5000)
    timeline._scene.addItem(ghost)
    state.ghost = ghost

    # Dim original
    clip_item.setOpacity(0.35)

    return state


def update_drag(
    timeline: "MultiTrackTimeline",
    state: DragState,
    scene_pos: QPointF,
) -> None:
    """Move ghost to follow cursor, snapping horizontally."""
    if state.ghost is None:
        return

    pps = timeline._pps
    tracks = timeline._project.tracks
    ranges = _track_y_ranges(tracks)

    # --- horizontal position (time) ---
    raw_time = max(0.0, (scene_pos.x() - state.start_offset) / pps)
    snapped_time = timeline.snap_time(raw_time)
    new_x = snapped_time * pps

    # --- vertical position (track) ---
    new_y = state.original_y
    for y_start, y_end, track in ranges:
        if y_start <= scene_pos.y() < y_end:
            new_y = y_start
            break

    rect = state.ghost.rect()
    state.ghost.setRect(new_x, new_y, rect.width(), rect.height())


def end_drag(
    timeline: "MultiTrackTimeline",
    state: DragState,
    scene_pos: QPointF,
) -> None:
    """Finish drag — compute new track & time, emit signal, clean up."""
    pps = timeline._pps
    tracks = timeline._project.tracks
    ranges = _track_y_ranges(tracks)

    # Determine target track
    target_track: Optional[Track] = None
    target_index = state.original_track_index
    for idx, (y_start, y_end, track) in enumerate(ranges):
        if y_start <= scene_pos.y() < y_end:
            target_track = track
            target_index = idx
            break

    if target_track is None:
        # Dropped outside any track — revert
        cancel_drag(timeline, state)
        return

    # Compute new start time
    raw_time = max(0.0, (scene_pos.x() - state.start_offset) / pps)
    new_start = timeline.snap_time(raw_time)

    # Clean up ghost
    if state.ghost is not None:
        timeline._scene.removeItem(state.ghost)
        state.ghost = None

    # Restore opacity
    state.clip_item.setOpacity(0.88)

    # Update model
    clip = state.clip_item.clip
    old_track_id = clip.track_id
    clip.start_time = new_start

    if target_track.id != old_track_id:
        # Move clip between tracks
        for t in tracks:
            if t.id == old_track_id:
                t.remove_clip(clip.id)
                break
        clip.track_id = target_track.id
        target_track.add_clip(clip)

    # Emit signal
    timeline.clip_moved.emit(clip.id, target_track.id, new_start)

    # Rebuild to reflect new layout
    timeline._project._recalculate_duration()
    timeline._rebuild_scene()


def cancel_drag(
    timeline: "MultiTrackTimeline",
    state: DragState,
) -> None:
    """Cancel an in-progress drag, restoring original position."""
    if state.ghost is not None:
        timeline._scene.removeItem(state.ghost)
        state.ghost = None
    state.clip_item.setOpacity(0.88)


# ---------------------------------------------------------------------------
# Wrapper class — delegates to module-level functions
# ---------------------------------------------------------------------------
class ClipDragHandler:
    """Thin OO wrapper around the module-level drag functions."""

    def __init__(self, timeline: "MultiTrackTimeline") -> None:
        self._timeline = timeline
        self._state: Optional[DragState] = None

    @property
    def active(self) -> bool:
        return self._state is not None

    def begin(self, clip_item: "ClipItem", scene_pos: QPointF) -> None:
        self._state = begin_drag(self._timeline, clip_item, scene_pos)

    def update(self, scene_pos: QPointF) -> None:
        if self._state is not None:
            update_drag(self._timeline, self._state, scene_pos)

    def end(self, scene_pos: QPointF) -> None:
        if self._state is not None:
            end_drag(self._timeline, self._state, scene_pos)
            self._state = None

    def cancel(self) -> None:
        if self._state is not None:
            cancel_drag(self._timeline, self._state)
            self._state = None
