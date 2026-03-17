"""
Track, Clip, and Project data models for the multi-track timeline.

Classes:
    TrackType  — Enum of track kinds (VIDEO, AUDIO, TEXT, EFFECTS)
    Clip       — Dataclass representing a single clip on a track
    Track      — A named track containing an ordered list of clips
    Project    — Top-level container holding tracks and project settings
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class TrackType(Enum):
    """Supported track types."""
    VIDEO = auto()
    AUDIO = auto()
    TEXT = auto()
    EFFECTS = auto()


@dataclass
class Clip:
    """A single clip on a timeline track."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    track_id: str = ""
    start_time: float = 0.0          # seconds on timeline
    duration: float = 0.0            # seconds
    source_path: str = ""
    in_point: float = 0.0            # source in-point (seconds)
    out_point: float = 0.0           # source out-point (seconds)
    name: str = ""
    properties: Dict[str, object] = field(default_factory=dict)

    # -- derived helpers --------------------------------------------------
    @property
    def end_time(self) -> float:
        """Return the timeline end position of this clip."""
        return self.start_time + self.duration

    def clone(self) -> "Clip":
        """Return a deep-enough copy with a new id."""
        return Clip(
            id=uuid.uuid4().hex[:12],
            track_id=self.track_id,
            start_time=self.start_time,
            duration=self.duration,
            source_path=self.source_path,
            in_point=self.in_point,
            out_point=self.out_point,
            name=self.name,
            properties=dict(self.properties),
        )


# Default visual colours per track type
_DEFAULT_TRACK_COLORS: Dict[TrackType, str] = {
    TrackType.VIDEO: "#00CED1",
    TrackType.AUDIO: "#4FC3F7",
    TrackType.TEXT: "#CE93D8",
    TrackType.EFFECTS: "#FFB340",
}

_DEFAULT_TRACK_HEIGHT = 60


class Track:
    """A single track in the timeline (video, audio, text, or effects)."""

    def __init__(
        self,
        id: str = "",
        name: str = "Untitled",
        type: TrackType = TrackType.VIDEO,
        clips: Optional[List[Clip]] = None,
        muted: bool = False,
        solo: bool = False,
        locked: bool = False,
        height: int = _DEFAULT_TRACK_HEIGHT,
        color: str = "",
    ) -> None:
        self.id: str = id or uuid.uuid4().hex[:12]
        self.name = name
        self.type = type
        self.clips: List[Clip] = clips if clips is not None else []
        self.muted = muted
        self.solo = solo
        self.locked = locked
        self.height = height
        self.color = color or _DEFAULT_TRACK_COLORS.get(type, "#888888")

    # -- clip management --------------------------------------------------
    def add_clip(self, clip: Clip) -> None:
        clip.track_id = self.id
        self.clips.append(clip)
        self.clips.sort(key=lambda c: c.start_time)

    def remove_clip(self, clip_id: str) -> Optional[Clip]:
        for i, clip in enumerate(self.clips):
            if clip.id == clip_id:
                return self.clips.pop(i)
        return None

    def get_clip(self, clip_id: str) -> Optional[Clip]:
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        return None

    def get_clip_at(self, time: float) -> Optional[Clip]:
        """Return the first clip that covers *time*."""
        for clip in self.clips:
            if clip.start_time <= time < clip.end_time:
                return clip
        return None

    @property
    def end_time(self) -> float:
        if not self.clips:
            return 0.0
        return max(c.end_time for c in self.clips)


class Project:
    """Top-level project container for the multi-track timeline."""

    def __init__(
        self,
        tracks: Optional[List[Track]] = None,
        duration: float = 0.0,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (1920, 1080),
    ) -> None:
        self.tracks: List[Track] = tracks if tracks is not None else []
        self.duration = duration
        self.fps = fps
        self.resolution = resolution

    # -- track management -------------------------------------------------
    def add_track(self, track: Track, index: Optional[int] = None) -> None:
        if index is not None:
            self.tracks.insert(index, track)
        else:
            self.tracks.append(track)

    def remove_track(self, track_id: str) -> Optional[Track]:
        for i, track in enumerate(self.tracks):
            if track.id == track_id:
                return self.tracks.pop(i)
        return None

    def get_track(self, track_id: str) -> Optional[Track]:
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None

    # -- clip helpers (project-wide) --------------------------------------
    def add_clip(self, track_id: str, clip: Clip) -> bool:
        """Add *clip* to the track identified by *track_id*."""
        track = self.get_track(track_id)
        if track is None:
            return False
        track.add_clip(clip)
        self._recalculate_duration()
        return True

    def remove_clip(self, clip_id: str) -> Optional[Clip]:
        """Remove a clip by id from whichever track contains it."""
        for track in self.tracks:
            removed = track.remove_clip(clip_id)
            if removed is not None:
                self._recalculate_duration()
                return removed
        return None

    def get_clip(self, clip_id: str) -> Optional[Clip]:
        for track in self.tracks:
            clip = track.get_clip(clip_id)
            if clip is not None:
                return clip
        return None

    def get_clip_at(self, time: float, track_id: Optional[str] = None) -> Optional[Clip]:
        """Return the first clip at *time*, optionally scoped to a track."""
        if track_id is not None:
            track = self.get_track(track_id)
            return track.get_clip_at(time) if track else None
        for track in self.tracks:
            clip = track.get_clip_at(time)
            if clip is not None:
                return clip
        return None

    # -- internal ---------------------------------------------------------
    def _recalculate_duration(self) -> None:
        if self.tracks:
            self.duration = max((t.end_time for t in self.tracks), default=0.0)
        else:
            self.duration = 0.0
