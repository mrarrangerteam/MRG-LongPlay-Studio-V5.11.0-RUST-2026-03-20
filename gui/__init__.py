"""
gui/ package — Re-exports every public class for backward compatibility.

    from gui import LongPlayStudioV4, Colors, AudioPlayerWidget, ...

All 28 original classes are available at this top level.

Import order (BUG-INT-007):
  1. gui.styles          — pure data (Colors), no Qt widget deps
  2. gui.audio_player    — standalone player classes
  3. gui.widgets.*       — leaf widgets (meter, waveform, drop_zone, collapsible)
  4. gui.video.*         — preview/detached (depends on widgets)
  5. gui.timeline.*      — canvas, tracks (depends on widgets + models)
  6. gui.dialogs.*       — dialog windows (depend on widgets + timeline + video)
  7. gui.main            — main window (depends on everything above)
  8. gui.utils.compat    — lightweight flag, safe anywhere

This ordering ensures each layer only imports from layers above it.
No circular imports exist because gui.main is imported last and no
sub-module imports from gui/__init__.py itself.
"""

# --- Core ---
from gui.styles import Colors
from gui.audio_player import (
    AudioPlayerWidget,
    MediaFile,
    TrackState,
    AudioAnalysisEngine,
)

# --- Widgets ---
from gui.widgets.meter import RealTimeMeter, LUFSDisplay
from gui.widgets.waveform import WaveformCache, ThumbnailCache
from gui.widgets.drop_zone import DropZoneListWidget
from gui.widgets.collapsible import CollapsibleSection

# --- Video ---
from gui.video.preview import VideoPreviewCard, VideoThread
from gui.video.detached import DetachedVideoWindow

# --- Timeline ---
from gui.timeline.canvas import TimelineCanvas, TrackControlButton, TrackControlsPanel
from gui.timeline.capcut_timeline import CapCutTimeline
from gui.timeline.track_list import TrackListItem, DraggableTrackListWidget

# --- Dialogs ---
from gui.dialogs.ai_dj import AIDJDialog
from gui.dialogs.ai_video import AIVideoDialog
from gui.dialogs.youtube_gen import YouTubeGeneratorDialog
from gui.dialogs.hook_extractor import HookExtractorDialog
from gui.dialogs.video_prompt import VideoPromptDialog
from gui.dialogs.timestamp import TimestampDialog
from gui.dialogs.content_factory import ContentFactoryDialog
from gui.dialogs.lipsync_dialog import LipSyncDialog

# --- Main Window ---
from gui.main import LongPlayStudioV4, LicenseDialog, check_and_show_license, main

# --- Qt compat flag ---
from gui.utils.compat import PYQT6

__all__ = [
    # Core
    "Colors",
    "AudioPlayerWidget",
    "MediaFile",
    "TrackState",
    "AudioAnalysisEngine",
    # Widgets
    "RealTimeMeter",
    "LUFSDisplay",
    "WaveformCache",
    "ThumbnailCache",
    "DropZoneListWidget",
    "CollapsibleSection",
    # Video
    "VideoPreviewCard",
    "VideoThread",
    "DetachedVideoWindow",
    # Timeline
    "TimelineCanvas",
    "TrackControlButton",
    "TrackControlsPanel",
    "CapCutTimeline",
    "TrackListItem",
    "DraggableTrackListWidget",
    # Dialogs
    "AIDJDialog",
    "AIVideoDialog",
    "YouTubeGeneratorDialog",
    "HookExtractorDialog",
    "VideoPromptDialog",
    "TimestampDialog",
    "ContentFactoryDialog",
    "LipSyncDialog",
    # Main
    "LongPlayStudioV4",
    "LicenseDialog",
    "check_and_show_license",
    "main",
    # Compat
    "PYQT6",
]
