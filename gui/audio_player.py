"""
Audio player widget, media file data classes, and audio analysis engine.

Classes:
    AudioPlayerWidget — Simple audio player using QMediaPlayer
    MediaFile         — Dataclass representing a media file
    TrackState        — Dataclass for per-track state (locked, visible, muted)
    AudioAnalysisEngine — Loads audio into memory for real-time level analysis
"""

import os
import math
import subprocess
from dataclasses import dataclass

import numpy as np

from gui.utils.compat import (
    QWidget, QTimer, QUrl, QMediaPlayer, QAudioOutput, pyqtSignal,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MediaFile:
    path: str
    name: str
    duration: float = 0.0
    lufs: float = -14.0
    file_type: str = "audio"  # audio, video, gif
    video_assignment: int = 0  # V4.25.10: Track which video this audio uses

    @property
    def duration_str(self) -> str:
        mins = int(self.duration // 60)
        secs = int(self.duration % 60)
        return f"{mins}:{secs:02d}"


@dataclass
class TrackState:
    """State for each track (locked, visible, muted)"""
    locked: bool = False
    visible: bool = True
    muted: bool = False


# ---------------------------------------------------------------------------
# Audio Analysis Engine
# ---------------------------------------------------------------------------

class AudioAnalysisEngine:
    """V5.5: Loads audio data into memory and provides real-time level analysis
    at any playback position. Used by both Mini Meter (main GUI) and
    Master Module meters to show REAL audio levels during playback.
    """

    def __init__(self):
        self._audio_data = {}   # {file_path: np.ndarray}
        self._sample_rates = {}  # {file_path: int}
        self._current_file = None
        self._current_data = None
        self._current_sr = 44100
        self._gain_linear = 1.0
        self._ceiling_linear = 10 ** (-1.0 / 20.0)  # -1.0 dBTP default
        self._has_soundfile = False
        try:
            import soundfile as sf
            self._sf = sf
            self._has_soundfile = True
            print("[AUDIO ENGINE] soundfile available -- real meter analysis enabled")
        except ImportError:
            print("[AUDIO ENGINE] soundfile not available -- using fallback meter")

    def load_file(self, file_path: str):
        """Load an audio file into memory for real-time analysis."""
        if not self._has_soundfile or not os.path.exists(file_path):
            return False
        if file_path in self._audio_data:
            self._current_file = file_path
            self._current_data = self._audio_data[file_path]
            self._current_sr = self._sample_rates[file_path]
            return True
        try:
            data, sr = self._sf.read(file_path, dtype='float32')
            if data.ndim == 1:
                data = np.column_stack([data, data])
            self._audio_data[file_path] = data
            self._sample_rates[file_path] = sr
            self._current_file = file_path
            self._current_data = data
            self._current_sr = sr
            print(f"[AUDIO ENGINE] Loaded: {os.path.basename(file_path)} "
                  f"({len(data)/sr:.1f}s, {sr}Hz, {data.shape[1]}ch)")
            return True
        except Exception as e:
            print(f"[AUDIO ENGINE] Load error: {e}")
            return False

    def clear_cache(self):
        """Free memory by clearing cached audio data."""
        self._audio_data.clear()
        self._sample_rates.clear()
        self._current_data = None
        self._current_file = None

    def set_gain(self, gain_db: float, ceiling_dbtp: float = -1.0):
        """Set gain and ceiling for accurate metering + preview rendering."""
        self._gain_linear = 10 ** (gain_db / 20.0) if gain_db > 0.01 else 1.0
        self._ceiling_linear = 10 ** (ceiling_dbtp / 20.0)

    def get_gained_audio(self):
        """Return current audio data with gain+ceiling applied.
        Returns (data_np, sample_rate) or (None, None).
        """
        if self._current_data is None:
            return None, None
        data = self._current_data.copy()
        if self._gain_linear > 1.001:
            data = data * self._gain_linear
            data = np.clip(data, -self._ceiling_linear, self._ceiling_linear)
        return data, self._current_sr

    def analyze_at_position(self, position_ms: int, window_ms: int = 100) -> dict:
        """Analyze audio levels at the given playback position.

        Returns dict with: left_peak, right_peak, left_rms, right_rms (0.0-1.0 linear),
        left_peak_db, right_peak_db, left_rms_db, right_rms_db (dBFS).
        """
        if self._current_data is None:
            return self._empty_result()

        sr = self._current_sr
        data = self._current_data
        center_sample = int((position_ms / 1000.0) * sr)
        window_samples = int((window_ms / 1000.0) * sr)
        start = max(0, center_sample - window_samples // 2)
        end = min(len(data), start + window_samples)

        if end <= start or start >= len(data):
            return self._empty_result()

        chunk = data[start:end]
        if self._gain_linear > 1.001:
            chunk = chunk * self._gain_linear
            chunk = np.clip(chunk, -self._ceiling_linear, self._ceiling_linear)
        left = chunk[:, 0]
        right = chunk[:, 1] if chunk.shape[1] > 1 else left

        left_peak = float(np.max(np.abs(left)))
        right_peak = float(np.max(np.abs(right)))
        left_rms = float(np.sqrt(np.mean(left ** 2)))
        right_rms = float(np.sqrt(np.mean(right ** 2)))

        eps = 1e-10
        return {
            "left_peak": left_peak,
            "right_peak": right_peak,
            "left_rms": left_rms,
            "right_rms": right_rms,
            "left_peak_db": 20 * math.log10(max(left_peak, eps)),
            "right_peak_db": 20 * math.log10(max(right_peak, eps)),
            "left_rms_db": 20 * math.log10(max(left_rms, eps)),
            "right_rms_db": 20 * math.log10(max(right_rms, eps)),
        }

    def _empty_result(self):
        return {
            "left_peak": 0.0, "right_peak": 0.0,
            "left_rms": 0.0, "right_rms": 0.0,
            "left_peak_db": -70.0, "right_peak_db": -70.0,
            "left_rms_db": -70.0, "right_rms_db": -70.0,
        }


# ---------------------------------------------------------------------------
# Audio Player Widget
# ---------------------------------------------------------------------------

class AudioPlayerWidget(QWidget):
    """Simple audio player using QMediaPlayer"""
    position_changed = pyqtSignal(int)   # position in ms
    duration_changed = pyqtSignal(int)   # duration in ms
    play_state_changed = pyqtSignal(bool)  # is_playing
    track_changed = pyqtSignal(int, str)   # index, filename

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_playing = False
        self.current_file_index = 0
        self.files = []
        self.durations = []  # duration in ms for each file
        self.crossfade_ms = 5000

        # Media player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)

        # Connect signals
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.player.errorOccurred.connect(self._on_error)

        # Timer for checking position
        self.timer = QTimer()
        self.timer.timeout.connect(self._check_position)
        self.timer.setInterval(100)

    def _on_error(self, error, error_string=""):
        """Handle media player errors"""
        print(f"Audio Player Error: {error} - {error_string}")
        if self.current_file_index < len(self.files) - 1:
            print("Skipping to next track...")
            self._load_file(self.current_file_index + 1)
            if self.is_playing:
                self.player.play()

    def load_files(self, file_paths: list):
        """Load multiple audio files"""
        self.files = file_paths
        self.durations = []
        self.current_file_index = 0

        for path in file_paths:
            try:
                duration_ms = self._get_duration_ffprobe(path)
                self.durations.append(duration_ms)
            except Exception as e:
                print(f"Warning: Could not get duration for {path}: {e}")
                self.durations.append(180000)

        if self.files:
            self._load_file(0)

    def _get_duration_ffprobe(self, path: str) -> int:
        """Get audio duration in ms using ffprobe"""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                path
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout.strip():
                duration_sec = float(result.stdout.strip())
                return int(duration_sec * 1000)
        except Exception as e:
            print(f"ffprobe error for {path}: {e}")

        return 180000  # Default 3 minutes

    def _load_file(self, index: int):
        """Load a specific file by index"""
        if 0 <= index < len(self.files):
            self.current_file_index = index
            file_path = self.files[index]
            filename = os.path.basename(file_path)
            print(f"Loading track {index + 1}/{len(self.files)}: {filename}")
            self.player.setSource(QUrl.fromLocalFile(file_path))
            self.track_changed.emit(index, filename)

    def _on_position_changed(self, position: int):
        """Handle position change"""
        if not self.files or not self.durations:
            return
        audio_global_pos = sum(self.durations[:self.current_file_index]) + position
        timeline_pos = audio_global_pos - (self.current_file_index * self.crossfade_ms)
        self.position_changed.emit(max(0, timeline_pos))

    def _on_duration_changed(self, duration: int):
        """Handle duration change"""
        if len(self.durations) > 1:
            total_duration = sum(self.durations) - (len(self.durations) - 1) * self.crossfade_ms
        else:
            total_duration = sum(self.durations)
        self.duration_changed.emit(max(0, total_duration))

    def _on_media_status_changed(self, status):
        """Handle media status change"""
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if self.current_file_index < len(self.files) - 1:
                self._load_file(self.current_file_index + 1)
                self.player.play()
            else:
                self.is_playing = False
                self.play_state_changed.emit(False)
                self.timer.stop()

    def _check_position(self):
        """Check position for crossfade"""
        pass  # Simplified - no crossfade in preview

    def play(self):
        """Start playback"""
        if not self.files:
            print("[PLAYER] No audio files loaded - cannot play")
            return
        try:
            self.player.play()
            self.is_playing = True
            self.play_state_changed.emit(True)
            self.timer.start()
        except Exception as e:
            print(f"[PLAYER] Play error: {e}")
            self.is_playing = False

    def pause(self):
        """Pause playback"""
        self.player.pause()
        self.is_playing = False
        self.play_state_changed.emit(False)
        self.timer.stop()

    def stop(self):
        """Stop playback"""
        self.player.stop()
        self.is_playing = False
        self.play_state_changed.emit(False)
        self.timer.stop()
        self.current_file_index = 0

    def seek(self, position_ms: int):
        """Seek to position (TIMELINE position with crossfade adjustment)"""
        if not self.files or not self.durations:
            return
        cumulative_timeline = 0
        for i, duration in enumerate(self.durations):
            effective_dur = duration if i == 0 else max(0, duration - self.crossfade_ms)
            if cumulative_timeline + effective_dur > position_ms:
                local_pos = position_ms - cumulative_timeline
                if i != self.current_file_index:
                    self._load_file(i)
                self.player.setPosition(max(0, local_pos))
                return
            cumulative_timeline += effective_dur

    def set_crossfade(self, crossfade_sec: int):
        """Set crossfade duration"""
        self.crossfade_ms = crossfade_sec * 1000
