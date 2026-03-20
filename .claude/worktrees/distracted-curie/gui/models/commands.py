"""
Undo / Redo command pattern for the multi-track timeline.

Provides:
    Command          — Abstract base with execute() / undo()
    CommandHistory   — Stack-based manager with undo / redo
    Concrete commands: MoveClipCommand, TrimClipCommand, SplitClipCommand,
                       AddClipCommand, DeleteClipCommand,
                       AddTrackCommand, DeleteTrackCommand
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from gui.models.track import Clip, Track, TrackType, Project


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Command(ABC):
    """Abstract undo-able command."""

    @property
    def description(self) -> str:
        """Human-readable description for the history panel."""
        return self.__class__.__name__

    @abstractmethod
    def execute(self) -> None:
        ...

    @abstractmethod
    def undo(self) -> None:
        ...


# ---------------------------------------------------------------------------
# CommandHistory
# ---------------------------------------------------------------------------
class CommandHistory:
    """
    Stack-based undo / redo manager.

    Signals are *not* emitted directly (no Qt dependency).  Instead, an
    optional ``on_change`` callback is invoked after every execute / undo /
    redo so that UI layers can react.
    """

    def __init__(self, on_change: Optional[Callable[[], None]] = None) -> None:
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        self._on_change = on_change

    # -- public API --------------------------------------------------------
    def execute(self, cmd: Command) -> None:
        """Execute *cmd* and push it onto the undo stack."""
        cmd.execute()
        self._undo_stack.append(cmd)
        self._redo_stack.clear()          # new action invalidates redo
        self._notify()

    def undo(self) -> Optional[Command]:
        if not self._undo_stack:
            return None
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        self._notify()
        return cmd

    def redo(self) -> Optional[Command]:
        if not self._redo_stack:
            return None
        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)
        self._notify()
        return cmd

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_stack(self) -> List[Command]:
        return list(self._undo_stack)

    @property
    def redo_stack(self) -> List[Command]:
        return list(self._redo_stack)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._notify()

    # -- internal ----------------------------------------------------------
    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()


# ---------------------------------------------------------------------------
# Concrete commands
# ---------------------------------------------------------------------------

class MoveClipCommand(Command):
    """Move a clip to a new track and/or start time."""

    def __init__(
        self,
        project: Project,
        clip_id: str,
        new_track_id: str,
        new_start_time: float,
    ) -> None:
        self._project = project
        self._clip_id = clip_id
        self._new_track_id = new_track_id
        self._new_start = new_start_time
        # Saved for undo
        self._old_track_id: str = ""
        self._old_start: float = 0.0

    @property
    def description(self) -> str:
        return "Move clip"

    def execute(self) -> None:
        clip = self._project.get_clip(self._clip_id)
        if clip is None:
            return
        self._old_track_id = clip.track_id
        self._old_start = clip.start_time

        if self._new_track_id != self._old_track_id:
            old_track = self._project.get_track(self._old_track_id)
            new_track = self._project.get_track(self._new_track_id)
            if old_track is not None:
                old_track.remove_clip(self._clip_id)
            if new_track is not None:
                clip.start_time = self._new_start
                new_track.add_clip(clip)
        else:
            clip.start_time = self._new_start

        self._project._recalculate_duration()

    def undo(self) -> None:
        clip = self._project.get_clip(self._clip_id)
        if clip is None:
            return

        if clip.track_id != self._old_track_id:
            cur_track = self._project.get_track(clip.track_id)
            old_track = self._project.get_track(self._old_track_id)
            if cur_track is not None:
                cur_track.remove_clip(self._clip_id)
            if old_track is not None:
                clip.start_time = self._old_start
                old_track.add_clip(clip)
        else:
            clip.start_time = self._old_start

        self._project._recalculate_duration()


class TrimClipCommand(Command):
    """Trim a clip's in/out points and duration."""

    def __init__(
        self,
        project: Project,
        clip_id: str,
        new_start: float,
        new_duration: float,
        new_in: float,
        new_out: float,
    ) -> None:
        self._project = project
        self._clip_id = clip_id
        self._new_start = new_start
        self._new_dur = new_duration
        self._new_in = new_in
        self._new_out = new_out
        # Saved for undo
        self._old_start: float = 0.0
        self._old_dur: float = 0.0
        self._old_in: float = 0.0
        self._old_out: float = 0.0

    @property
    def description(self) -> str:
        return "Trim clip"

    def execute(self) -> None:
        clip = self._project.get_clip(self._clip_id)
        if clip is None:
            return
        self._old_start = clip.start_time
        self._old_dur = clip.duration
        self._old_in = clip.in_point
        self._old_out = clip.out_point

        clip.start_time = self._new_start
        clip.duration = self._new_dur
        clip.in_point = self._new_in
        clip.out_point = self._new_out
        self._project._recalculate_duration()

    def undo(self) -> None:
        clip = self._project.get_clip(self._clip_id)
        if clip is None:
            return
        clip.start_time = self._old_start
        clip.duration = self._old_dur
        clip.in_point = self._old_in
        clip.out_point = self._old_out
        self._project._recalculate_duration()


class SplitClipCommand(Command):
    """Split a clip into two at a given time."""

    def __init__(
        self,
        project: Project,
        clip_id: str,
        split_time: float,
    ) -> None:
        self._project = project
        self._clip_id = clip_id
        self._split_time = split_time
        # Saved for undo
        self._original_duration: float = 0.0
        self._original_out: float = 0.0
        self._original_name: str = ""
        self._right_clip_id: str = ""

    @property
    def description(self) -> str:
        return "Split clip"

    def execute(self) -> None:
        clip = self._project.get_clip(self._clip_id)
        if clip is None:
            return

        track = self._project.get_track(clip.track_id)
        if track is None:
            return

        self._original_duration = clip.duration
        self._original_out = clip.out_point
        self._original_name = clip.name

        left_dur = self._split_time - clip.start_time
        right_dur = clip.duration - left_dur

        right_clip = Clip(
            id=uuid.uuid4().hex[:12],
            track_id=track.id,
            start_time=self._split_time,
            duration=right_dur,
            source_path=clip.source_path,
            in_point=clip.in_point + left_dur,
            out_point=clip.out_point,
            name=f"{clip.name} (R)",
            properties=dict(clip.properties),
        )
        self._right_clip_id = right_clip.id

        clip.duration = left_dur
        clip.out_point = clip.in_point + left_dur

        track.add_clip(right_clip)
        self._project._recalculate_duration()

    def undo(self) -> None:
        clip = self._project.get_clip(self._clip_id)
        if clip is None:
            return

        # Remove right clip
        self._project.remove_clip(self._right_clip_id)

        # Restore original clip
        clip.duration = self._original_duration
        clip.out_point = self._original_out
        clip.name = self._original_name
        self._project._recalculate_duration()


class AddClipCommand(Command):
    """Add a new clip to a track."""

    def __init__(self, project: Project, track_id: str, clip: Clip) -> None:
        self._project = project
        self._track_id = track_id
        self._clip = clip

    @property
    def description(self) -> str:
        return f"Add clip '{self._clip.name}'"

    def execute(self) -> None:
        self._project.add_clip(self._track_id, self._clip)

    def undo(self) -> None:
        self._project.remove_clip(self._clip.id)


class DeleteClipCommand(Command):
    """Delete a clip from the project."""

    def __init__(self, project: Project, clip_id: str) -> None:
        self._project = project
        self._clip_id = clip_id
        self._saved_clip: Optional[Clip] = None
        self._saved_track_id: str = ""

    @property
    def description(self) -> str:
        name = self._saved_clip.name if self._saved_clip else self._clip_id
        return f"Delete clip '{name}'"

    def execute(self) -> None:
        clip = self._project.get_clip(self._clip_id)
        if clip is not None:
            self._saved_track_id = clip.track_id
            self._saved_clip = clip.clone()
            self._saved_clip.id = clip.id  # preserve original id for undo
            self._saved_clip.track_id = clip.track_id
            self._saved_clip.start_time = clip.start_time
            self._saved_clip.duration = clip.duration
            self._saved_clip.in_point = clip.in_point
            self._saved_clip.out_point = clip.out_point
        self._project.remove_clip(self._clip_id)

    def undo(self) -> None:
        if self._saved_clip is not None:
            # Re-create the clip with original values
            restored = Clip(
                id=self._saved_clip.id,
                track_id=self._saved_track_id,
                start_time=self._saved_clip.start_time,
                duration=self._saved_clip.duration,
                source_path=self._saved_clip.source_path,
                in_point=self._saved_clip.in_point,
                out_point=self._saved_clip.out_point,
                name=self._saved_clip.name,
                properties=dict(self._saved_clip.properties),
            )
            self._project.add_clip(self._saved_track_id, restored)


class AddTrackCommand(Command):
    """Add a new track to the project."""

    def __init__(
        self,
        project: Project,
        track: Track,
        index: Optional[int] = None,
    ) -> None:
        self._project = project
        self._track = track
        self._index = index

    @property
    def description(self) -> str:
        return f"Add track '{self._track.name}'"

    def execute(self) -> None:
        self._project.add_track(self._track, self._index)

    def undo(self) -> None:
        self._project.remove_track(self._track.id)


class DeleteTrackCommand(Command):
    """Delete a track from the project."""

    def __init__(self, project: Project, track_id: str) -> None:
        self._project = project
        self._track_id = track_id
        self._saved_track: Optional[Track] = None
        self._saved_index: int = 0

    @property
    def description(self) -> str:
        name = self._saved_track.name if self._saved_track else self._track_id
        return f"Delete track '{name}'"

    def execute(self) -> None:
        for i, t in enumerate(self._project.tracks):
            if t.id == self._track_id:
                self._saved_index = i
                break
        self._saved_track = self._project.remove_track(self._track_id)

    def undo(self) -> None:
        if self._saved_track is not None:
            self._project.add_track(self._saved_track, self._saved_index)
