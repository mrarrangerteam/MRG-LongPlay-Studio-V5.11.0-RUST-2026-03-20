"""
Auto-save and crash recovery system.

Story 5.3 — Epic 5: Polish & Production.

Features:
    - Auto-save project state every 60 seconds
    - Crash recovery from auto-save on next launch
    - Rotated save slots (keep last 3 auto-saves)
    - JSON-serializable project state
    - Lock file to detect improper shutdown
    - Integrates with CommandHistory for undo state preservation
"""

from __future__ import annotations

import json
import os
import time
import threading
from typing import Callable, Dict, List, Optional

from gui.models.track import Clip, Track, TrackType, Project
from gui.models.commands import CommandHistory


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AUTOSAVE_DIR = os.path.join(os.path.expanduser("~"), ".longplay", "autosave")
AUTOSAVE_INTERVAL = 60          # seconds
MAX_AUTOSAVE_SLOTS = 3
LOCK_FILE = "session.lock"
RECOVERY_FILE = "recovery_info.json"


# ---------------------------------------------------------------------------
# Project serialization
# ---------------------------------------------------------------------------
def project_to_dict(project: Project) -> Dict:
    """Serialize a Project to a JSON-compatible dict."""
    tracks = []
    for track in project.tracks:
        clips = []
        for clip in track.clips:
            clips.append({
                "id": clip.id,
                "track_id": clip.track_id,
                "start_time": clip.start_time,
                "duration": clip.duration,
                "source_path": clip.source_path,
                "in_point": clip.in_point,
                "out_point": clip.out_point,
                "name": clip.name,
                "properties": _serialize_properties(clip.properties),
            })
        tracks.append({
            "id": track.id,
            "name": track.name,
            "type": track.type.name,
            "clips": clips,
            "muted": track.muted,
            "solo": track.solo,
            "locked": track.locked,
            "height": track.height,
            "color": track.color,
        })

    return {
        "version": "5.9",
        "tracks": tracks,
        "duration": project.duration,
        "fps": project.fps,
        "resolution": list(project.resolution),
        "saved_at": time.time(),
        "saved_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def dict_to_project(data: Dict) -> Project:
    """Deserialize a Project from a dict."""
    tracks: List[Track] = []
    for td in data.get("tracks", []):
        clips: List[Clip] = []
        for cd in td.get("clips", []):
            clip = Clip(
                id=cd.get("id", ""),
                track_id=cd.get("track_id", ""),
                start_time=cd.get("start_time", 0.0),
                duration=cd.get("duration", 0.0),
                source_path=cd.get("source_path", ""),
                in_point=cd.get("in_point", 0.0),
                out_point=cd.get("out_point", 0.0),
                name=cd.get("name", ""),
                properties=cd.get("properties", {}),
            )
            clips.append(clip)

        track_type = TrackType[td.get("type", "VIDEO")]
        track = Track(
            id=td.get("id", ""),
            name=td.get("name", "Untitled"),
            type=track_type,
            clips=clips,
            muted=td.get("muted", False),
            solo=td.get("solo", False),
            locked=td.get("locked", False),
            height=td.get("height", 60),
            color=td.get("color", ""),
        )
        tracks.append(track)

    return Project(
        tracks=tracks,
        duration=data.get("duration", 0.0),
        fps=data.get("fps", 30.0),
        resolution=tuple(data.get("resolution", [1920, 1080])),
    )


def _serialize_properties(props: Dict) -> Dict:
    """Ensure all property values are JSON-serializable."""
    result = {}
    for k, v in props.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            result[k] = v
        elif isinstance(v, (list, tuple)):
            result[k] = list(v)
        elif isinstance(v, dict):
            result[k] = _serialize_properties(v)
        else:
            result[k] = str(v)
    return result


# ---------------------------------------------------------------------------
# AutoSaveManager
# ---------------------------------------------------------------------------
class AutoSaveManager:
    """
    Automatic project save manager with crash recovery.

    Saves project state periodically and provides recovery
    from improper shutdown.
    """

    def __init__(
        self,
        autosave_dir: str = AUTOSAVE_DIR,
        interval: int = AUTOSAVE_INTERVAL,
    ) -> None:
        self._dir = autosave_dir
        self._interval = interval
        self._project: Optional[Project] = None
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._is_running = False
        self._on_save: Optional[Callable[[str], None]] = None
        self._save_count = 0

        os.makedirs(self._dir, exist_ok=True)

    # -- public API --------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def save_count(self) -> int:
        return self._save_count

    def set_project(self, project: Project) -> None:
        """Set the project to auto-save."""
        with self._lock:
            self._project = project

    def set_save_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback invoked after each auto-save. Receives the save path."""
        self._on_save = callback

    def start(self) -> None:
        """Start auto-save timer and create lock file."""
        if self._is_running:
            return
        self._is_running = True
        self._create_lock()
        self._schedule_next()

    def stop(self) -> None:
        """Stop auto-save timer and remove lock file."""
        self._is_running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._remove_lock()

    def save_now(self) -> Optional[str]:
        """Perform an immediate auto-save. Returns the save path or None."""
        return self._do_save()

    def has_recovery(self) -> bool:
        """Check if there's a recovery file from a previous crash."""
        lock_path = os.path.join(self._dir, LOCK_FILE)
        if os.path.exists(lock_path):
            # lock file exists = improper shutdown
            return self._find_latest_autosave() is not None
        return False

    def recover(self) -> Optional[Project]:
        """
        Recover project from the latest auto-save.

        Returns the recovered Project or None.
        """
        path = self._find_latest_autosave()
        if path is None:
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            project = dict_to_project(data)
            # clean up lock file after successful recovery
            self._remove_lock()
            return project
        except (json.JSONDecodeError, OSError, KeyError) as e:
            print(f"[AUTOSAVE] Recovery failed: {e}")
            return None

    def get_recovery_info(self) -> Optional[Dict]:
        """Get info about available recovery data."""
        path = self._find_latest_autosave()
        if path is None:
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "path": path,
                "saved_at": data.get("saved_at_iso", "Unknown"),
                "tracks": len(data.get("tracks", [])),
                "duration": data.get("duration", 0.0),
            }
        except (json.JSONDecodeError, OSError):
            return None

    def cleanup(self) -> int:
        """Remove all auto-save files. Returns count of deleted files."""
        count = 0
        for filename in os.listdir(self._dir):
            if filename.startswith("autosave_") and filename.endswith(".json"):
                try:
                    os.remove(os.path.join(self._dir, filename))
                    count += 1
                except OSError:
                    pass
        self._remove_lock()
        return count

    # -- internal ----------------------------------------------------------

    def _schedule_next(self) -> None:
        if not self._is_running:
            return
        self._timer = threading.Timer(self._interval, self._on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _on_timer(self) -> None:
        if not self._is_running:
            return
        self._do_save()
        self._schedule_next()

    def _do_save(self) -> Optional[str]:
        with self._lock:
            project = self._project

        if project is None:
            return None

        try:
            data = project_to_dict(project)
            slot = self._save_count % MAX_AUTOSAVE_SLOTS
            filename = f"autosave_{slot}.json"
            path = os.path.join(self._dir, filename)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            self._save_count += 1

            # write recovery info
            recovery = {
                "latest_slot": slot,
                "latest_file": filename,
                "timestamp": time.time(),
            }
            recovery_path = os.path.join(self._dir, RECOVERY_FILE)
            with open(recovery_path, "w", encoding="utf-8") as f:
                json.dump(recovery, f)

            if self._on_save is not None:
                self._on_save(path)

            return path

        except OSError as e:
            print(f"[AUTOSAVE] Save failed: {e}")
            return None

    def _create_lock(self) -> None:
        lock_path = os.path.join(self._dir, LOCK_FILE)
        try:
            with open(lock_path, "w") as f:
                f.write(f"{os.getpid()}\n{time.time()}\n")
        except OSError:
            pass

    def _remove_lock(self) -> None:
        lock_path = os.path.join(self._dir, LOCK_FILE)
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except OSError:
            pass

    def _find_latest_autosave(self) -> Optional[str]:
        """Find the most recently modified auto-save file."""
        # check recovery info first
        recovery_path = os.path.join(self._dir, RECOVERY_FILE)
        if os.path.exists(recovery_path):
            try:
                with open(recovery_path, "r") as f:
                    recovery = json.load(f)
                latest = os.path.join(self._dir, recovery["latest_file"])
                if os.path.exists(latest):
                    return latest
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        # fallback: find newest autosave file
        candidates = []
        for filename in os.listdir(self._dir):
            if filename.startswith("autosave_") and filename.endswith(".json"):
                path = os.path.join(self._dir, filename)
                candidates.append((os.path.getmtime(path), path))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
