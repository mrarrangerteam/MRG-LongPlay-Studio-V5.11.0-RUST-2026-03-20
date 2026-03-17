"""
Clip trimming and splitting logic for the multi-track timeline.

Provides:
    TrimEdge       — Enum: LEFT or RIGHT edge
    TrimState      — Bookkeeping for an in-progress trim drag
    begin_trim     — Start trimming a clip edge
    update_trim    — Move the edge during drag
    end_trim       — Finish trim, update model, emit signal
    split_clip_at  — Split a clip into two at a given time
"""

from __future__ import annotations

import uuid
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from gui.utils.compat import (
    QGraphicsRectItem, Qt, QPointF,
    QColor, QPen, QBrush,
)
from gui.models.track import Clip

if TYPE_CHECKING:
    from gui.timeline.multi_track_timeline import MultiTrackTimeline, ClipItem


# ---------------------------------------------------------------------------
# Edge enum
# ---------------------------------------------------------------------------
class TrimEdge(Enum):
    LEFT = auto()
    RIGHT = auto()


# ---------------------------------------------------------------------------
# Trim handle detection
# ---------------------------------------------------------------------------
HANDLE_WIDTH_PX = 8  # pixels from clip edge to count as a "handle"


def detect_trim_edge(clip_item: "ClipItem", scene_x: float, pps: float) -> Optional[TrimEdge]:
    """Return TrimEdge.LEFT / RIGHT if *scene_x* is within handle zone, else None."""
    rect = clip_item.rect()
    left_x = rect.x()
    right_x = rect.x() + rect.width()

    if abs(scene_x - left_x) <= HANDLE_WIDTH_PX:
        return TrimEdge.LEFT
    if abs(scene_x - right_x) <= HANDLE_WIDTH_PX:
        return TrimEdge.RIGHT
    return None


# ---------------------------------------------------------------------------
# TrimState
# ---------------------------------------------------------------------------
class TrimState:
    """Bookkeeping for an active trim operation."""

    __slots__ = (
        "clip_item", "edge", "original_start", "original_duration",
        "original_in", "original_out", "indicator",
    )

    def __init__(self, clip_item: "ClipItem", edge: TrimEdge) -> None:
        self.clip_item = clip_item
        self.edge = edge
        self.original_start: float = clip_item.clip.start_time
        self.original_duration: float = clip_item.clip.duration
        self.original_in: float = clip_item.clip.in_point
        self.original_out: float = clip_item.clip.out_point
        self.indicator: Optional[QGraphicsRectItem] = None


# ---------------------------------------------------------------------------
# Trim operations
# ---------------------------------------------------------------------------

def begin_trim(
    timeline: "MultiTrackTimeline",
    clip_item: "ClipItem",
    edge: TrimEdge,
) -> TrimState:
    """Start trimming *clip_item* from *edge*."""
    state = TrimState(clip_item, edge)

    # Visual indicator — thin coloured bar on the edge being trimmed
    rect = clip_item.rect()
    if edge == TrimEdge.LEFT:
        ind_x = rect.x()
    else:
        ind_x = rect.x() + rect.width() - 3

    indicator = QGraphicsRectItem(ind_x, rect.y(), 3, rect.height())
    indicator.setBrush(QBrush(QColor("#FF9500")))
    indicator.setPen(QPen(Qt.PenStyle.NoPen))
    indicator.setZValue(6000)
    timeline._scene.addItem(indicator)
    state.indicator = indicator

    return state


def update_trim(
    timeline: "MultiTrackTimeline",
    state: TrimState,
    scene_x: float,
) -> None:
    """Update the clip's edge position during drag."""
    pps = timeline._pps
    clip = state.clip_item.clip
    new_time = max(0.0, scene_x / pps)
    new_time = timeline.snap_time(new_time)

    if state.edge == TrimEdge.LEFT:
        # Trimming left edge: adjusts start_time and in_point
        delta = new_time - state.original_start
        # Don't let trim past the right edge
        max_delta = state.original_duration - 0.05
        delta = min(delta, max_delta)
        clip.start_time = state.original_start + delta
        clip.in_point = state.original_in + delta
        clip.duration = state.original_duration - delta
    else:
        # Trimming right edge: adjusts duration and out_point
        new_end = new_time
        new_dur = max(0.05, new_end - clip.start_time)
        delta = new_dur - state.original_duration
        clip.duration = new_dur
        clip.out_point = state.original_out + delta

    # Update visual indicator
    if state.indicator is not None:
        rect = state.clip_item.rect()
        ci_x = clip.start_time * pps
        ci_w = max(clip.duration * pps, 2)
        if state.edge == TrimEdge.LEFT:
            state.indicator.setRect(ci_x, rect.y(), 3, rect.height())
        else:
            state.indicator.setRect(ci_x + ci_w - 3, rect.y(), 3, rect.height())

    # Refresh clip item geometry
    state.clip_item.refresh_geometry(
        pps,
        state.clip_item.rect().y(),
        state.clip_item.rect().height() + 2,  # +TRACK_GAP was subtracted
    )


def end_trim(
    timeline: "MultiTrackTimeline",
    state: TrimState,
) -> None:
    """Finish trim, clean up indicator, emit signal."""
    if state.indicator is not None:
        timeline._scene.removeItem(state.indicator)
        state.indicator = None

    clip = state.clip_item.clip
    timeline._project._recalculate_duration()
    timeline.clip_trimmed.emit(clip.id, clip.in_point, clip.out_point)


def cancel_trim(
    timeline: "MultiTrackTimeline",
    state: TrimState,
) -> None:
    """Revert trim to original values."""
    clip = state.clip_item.clip
    clip.start_time = state.original_start
    clip.duration = state.original_duration
    clip.in_point = state.original_in
    clip.out_point = state.original_out

    if state.indicator is not None:
        timeline._scene.removeItem(state.indicator)
        state.indicator = None

    timeline._rebuild_scene()


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_clip_at(
    timeline: "MultiTrackTimeline",
    clip_id: str,
    split_time: float,
) -> bool:
    """
    Split the clip identified by *clip_id* at *split_time* into two clips.

    Returns True if the split was performed, False otherwise.
    """
    project = timeline._project
    clip = project.get_clip(clip_id)
    if clip is None:
        return False

    # Ensure split_time is within the clip
    if split_time <= clip.start_time or split_time >= clip.end_time:
        return False

    # Find the track
    track = project.get_track(clip.track_id)
    if track is None:
        return False

    # Duration of left part
    left_dur = split_time - clip.start_time
    right_dur = clip.duration - left_dur

    # Create right clip
    right_clip = Clip(
        id=uuid.uuid4().hex[:12],
        track_id=track.id,
        start_time=split_time,
        duration=right_dur,
        source_path=clip.source_path,
        in_point=clip.in_point + left_dur,
        out_point=clip.out_point,
        name=f"{clip.name} (R)",
        properties=dict(clip.properties),
    )

    # Modify original clip (becomes the left part)
    clip.duration = left_dur
    clip.out_point = clip.in_point + left_dur
    clip.name = f"{clip.name}" if "(R)" not in clip.name else clip.name

    # Add right clip to track
    track.add_clip(right_clip)

    project._recalculate_duration()
    timeline.clip_split.emit(clip_id, split_time)
    timeline._rebuild_scene()

    return True


# ---------------------------------------------------------------------------
# Wrapper class — delegates to module-level functions
# ---------------------------------------------------------------------------
class ClipTrimSplitHandler:
    """Thin OO wrapper around the module-level trim/split functions."""

    def __init__(self, timeline: "MultiTrackTimeline") -> None:
        self._timeline = timeline
        self._state: Optional[TrimState] = None

    @property
    def active(self) -> bool:
        return self._state is not None

    def detect_edge(self, clip_item: "ClipItem", scene_x: float) -> Optional[TrimEdge]:
        return detect_trim_edge(clip_item, scene_x, self._timeline._pps)

    def begin(self, clip_item: "ClipItem", edge: TrimEdge) -> None:
        self._state = begin_trim(self._timeline, clip_item, edge)

    def update(self, scene_x: float) -> None:
        if self._state is not None:
            update_trim(self._timeline, self._state, scene_x)

    def end(self) -> None:
        if self._state is not None:
            end_trim(self._timeline, self._state)
            self._state = None

    def cancel(self) -> None:
        if self._state is not None:
            cancel_trim(self._timeline, self._state)
            self._state = None

    def split(self, clip_id: str, split_time: float) -> bool:
        return split_clip_at(self._timeline, clip_id, split_time)
