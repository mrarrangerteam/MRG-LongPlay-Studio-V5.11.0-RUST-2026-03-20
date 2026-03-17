"""
Full CapCut-style timeline widget with controls and scrollbar.

Classes:
    CapCutTimeline — Complete timeline with track controls, canvas, scrollbar, and zoom
"""

from typing import List

from gui.utils.compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QSlider, QTimer, Qt, pyqtSignal,
)
from gui.styles import Colors
from gui.audio_player import MediaFile
from gui.timeline.canvas import TimelineCanvas, TrackControlsPanel


class CapCutTimeline(QWidget):
    """Full CapCut-style timeline with controls and scrollbar"""

    seekRequested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.audio_tracks: List[MediaFile] = []
        self.video_tracks: List[MediaFile] = []
        self.gif_tracks: List[MediaFile] = []
        self.total_duration_ms = 0
        self.playhead_position_ms = 0
        self.crossfade_sec = 5
        self.pixels_per_second = 15
        self.scroll_offset = 0
        self.is_playing = False

        self._setup_ui()

        self.playhead_timer = QTimer()
        self.playhead_timer.timeout.connect(self._update_playhead)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{ background: {Colors.BG_PRIMARY}; border: none; }}
            QScrollBar:vertical {{
                background: {Colors.BG_TERTIARY};
                width: 10px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.ACCENT};
                min-height: 30px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        scroll_content = QWidget()
        timeline_area = QHBoxLayout(scroll_content)
        timeline_area.setSpacing(0)
        timeline_area.setContentsMargins(0, 0, 0, 0)

        self.track_controls = TrackControlsPanel()
        self.track_controls.addRulerSpacer(30)
        timeline_area.addWidget(self.track_controls)

        self.canvas = TimelineCanvas()
        self.canvas.seekRequested.connect(self.seekRequested.emit)
        timeline_area.addWidget(self.canvas, 1)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)

        self.scrollbar = QSlider(Qt.Orientation.Horizontal)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(1000)
        self.scrollbar.setValue(0)
        self.scrollbar.setFixedHeight(16)
        self.scrollbar.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {Colors.BG_TERTIARY};
                height: 12px;
                border-radius: 6px;
            }}
            QSlider::handle:horizontal {{
                background: {Colors.ACCENT};
                width: 80px;
                margin: -2px 0;
                border-radius: 6px;
            }}
            QSlider::sub-page:horizontal {{
                background: {Colors.BG_SECONDARY};
                border-radius: 6px;
            }}
        """)
        self.scrollbar.valueChanged.connect(self._on_scroll)
        main_layout.addWidget(self.scrollbar)

    def zoomIn(self):
        if self.pixels_per_second < 5:
            self.pixels_per_second = min(5, self.pixels_per_second * 1.5)
        else:
            self.pixels_per_second = min(200, self.pixels_per_second + 5)
        self._refresh_canvas()

    def zoomOut(self):
        if self.pixels_per_second <= 5:
            self.pixels_per_second = max(0.5, self.pixels_per_second * 0.7)
        else:
            self.pixels_per_second = max(1, self.pixels_per_second - 5)
        self._refresh_canvas()

    def zoomFit(self):
        canvas_width = self.canvas.width() - 20
        if self.total_duration_ms > 0 and canvas_width > 100:
            total_sec = self.total_duration_ms / 1000.0
            self.pixels_per_second = max(0.5, canvas_width / total_sec)
            print(f"[TIMELINE] Zoom to fit: {self.pixels_per_second:.1f} px/sec for {total_sec:.0f}sec in {canvas_width}px")
        self._refresh_canvas()

    def _refresh_canvas(self):
        self.canvas.setData(
            self.audio_tracks,
            self.video_tracks,
            self.gif_tracks,
            self.total_duration_ms,
            self.crossfade_sec,
            self.pixels_per_second
        )

    def _on_scroll(self, value):
        if self.total_duration_ms > 0:
            max_scroll = max(0, self._get_total_width() - self.canvas.width())
            self.scroll_offset = int((value / 1000) * max_scroll) if max_scroll > 0 else 0
            self.canvas.setScrollOffset(self.scroll_offset)

    def _get_total_width(self):
        return int((self.total_duration_ms / 1000) * self.pixels_per_second) + 200

    def setTracks(self, audio: List[MediaFile], video: List[MediaFile], gif: List[MediaFile] = None):
        self.audio_tracks = audio or []
        self.video_tracks = video or []
        self.gif_tracks = gif or []

        if audio:
            total = 0
            for i, track in enumerate(audio):
                if i == 0:
                    total += track.duration
                else:
                    total += max(0, track.duration - self.crossfade_sec)
            self.total_duration_ms = int(total * 1000)
        else:
            self.total_duration_ms = 0

        self.canvas.setData(
            self.audio_tracks,
            self.video_tracks,
            self.gif_tracks,
            self.total_duration_ms,
            self.crossfade_sec,
            self.pixels_per_second
        )

        self._update_track_controls()

        total_width = self._get_total_width()
        visible_width = self.canvas.width()
        self.scrollbar.setEnabled(total_width > visible_width)

    def _update_track_controls(self):
        self.track_controls.clear()
        self.track_controls.addRulerSpacer(30)

        gif_name = self.gif_tracks[0].name[:8] if self.gif_tracks else "GIF"
        self.track_controls.addTrackRow("gif", gif_name, 0)

        vid_name = self.video_tracks[0].name[:8] if self.video_tracks else "Video"
        self.track_controls.addTrackRow("video", vid_name, 1)

        if len(self.audio_tracks) == 1:
            self.track_controls.addTrackRow("audio", "Master", 2)
        else:
            for i, audio in enumerate(self.audio_tracks):
                label = f"{i+1}.{audio.name[:6]}"
                self.track_controls.addTrackRow("audio", label, i + 2)

    def setCrossfade(self, seconds: int):
        self.crossfade_sec = seconds
        self.setTracks(self.audio_tracks, self.video_tracks, self.gif_tracks)

    def setPlayheadPosition(self, position_ms: int):
        self.playhead_position_ms = position_ms
        self.canvas.setPlayheadPosition(position_ms)

        if self.total_duration_ms > 0 and self.is_playing:
            playhead_x = int((position_ms / 1000) * self.pixels_per_second)
            canvas_w = self.canvas.width()

            target_x = int(canvas_w * 0.4)
            desired_scroll = playhead_x - target_x

            if playhead_x < self.scroll_offset + int(canvas_w * 0.3) or playhead_x > self.scroll_offset + int(canvas_w * 0.7):
                max_scroll = max(1, self._get_total_width() - canvas_w)
                new_scroll = max(0, min(desired_scroll, max_scroll))

                self.scroll_offset = int(self.scroll_offset + (new_scroll - self.scroll_offset) * 0.3)
                self.canvas.setScrollOffset(self.scroll_offset)

                self.scrollbar.blockSignals(True)
                self.scrollbar.setValue(int((self.scroll_offset / max_scroll) * 1000))
                self.scrollbar.blockSignals(False)

    def setPlaying(self, playing: bool):
        self.is_playing = playing
        if playing:
            self.playhead_timer.start(33)
        else:
            self.playhead_timer.stop()

    def _update_playhead(self):
        self.canvas.update()
