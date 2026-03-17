"""Timeline widgets — CapCut-style timeline, canvas, track controls."""
from gui.timeline.canvas import TimelineCanvas, TrackControlButton, TrackControlsPanel
from gui.timeline.capcut_timeline import CapCutTimeline
from gui.timeline.track_list import TrackListItem, DraggableTrackListWidget
from gui.timeline.multi_track_timeline import (
    MultiTrackTimeline, ClipItem, PlayheadItem, TrackHeaderPanel,
)
# Story 3.1 — Keyframe editor
from gui.timeline.keyframe_editor import KeyframeEditor, KeyframeCurveScene, KeyframeDiamond
# Story 3.2 — Text layer
from gui.timeline.text_layer import TextClip, TextClipItem, TextPropertiesPanel, TextAnimation
# Story 3.5 — Speed ramp
from gui.timeline.speed_ramp import SpeedRamp, SpeedPreset, SpeedCurveEditor

__all__ = [
    "TimelineCanvas",
    "TrackControlButton",
    "TrackControlsPanel",
    "CapCutTimeline",
    "TrackListItem",
    "DraggableTrackListWidget",
    "MultiTrackTimeline",
    "ClipItem",
    "PlayheadItem",
    "TrackHeaderPanel",
    # Story 3.1
    "KeyframeEditor",
    "KeyframeCurveScene",
    "KeyframeDiamond",
    # Story 3.2
    "TextClip",
    "TextClipItem",
    "TextPropertiesPanel",
    "TextAnimation",
    # Story 3.5
    "SpeedRamp",
    "SpeedPreset",
    "SpeedCurveEditor",
]
