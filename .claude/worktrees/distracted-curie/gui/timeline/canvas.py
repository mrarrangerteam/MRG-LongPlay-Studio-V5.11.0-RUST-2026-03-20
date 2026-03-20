"""
Timeline canvas and track control widgets.

Classes:
    TrackControlButton — Small icon button for track controls
    TrackControlsPanel — Left panel with CapCut-style track controls
    TimelineCanvas     — Main timeline canvas with tracks and playhead
"""

import random
from typing import List

from gui.utils.compat import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QPainter, QPen, QColor, QFont,
    QPolygon, QPoint, QMenu, Qt, pyqtSignal, QPainterPath,
)
from gui.styles import Colors
from gui.audio_player import MediaFile, TrackState
from gui.widgets.waveform import WaveformCache, ThumbnailCache


class TrackControlButton(QPushButton):
    """Small icon button for track controls"""

    def __init__(self, icon: str, tooltip: str, parent=None):
        super().__init__(icon, parent)
        self.setFixedSize(24, 24)
        self.setToolTip(tooltip)
        self.setCheckable(True)
        self.active_icon = icon
        self.inactive_icon = icon
        self._update_style()
        self.clicked.connect(self._update_style)

    def setIcons(self, active: str, inactive: str):
        self.active_icon = active
        self.inactive_icon = inactive
        self._update_style()

    def _update_style(self):
        if self.isChecked():
            self.setText(self.active_icon)
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {Colors.ACCENT};
                    border: none;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background: {Colors.BG_SECONDARY};
                    border-radius: 4px;
                }}
            """)
        else:
            self.setText(self.inactive_icon)
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {Colors.TEXT_TERTIARY};
                    border: none;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background: {Colors.BG_SECONDARY};
                    border-radius: 4px;
                }}
            """)


class TrackControlsPanel(QFrame):
    """Left panel with track controls like CapCut"""

    stateChanged = pyqtSignal(int, str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(120)
        self.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_PRIMARY};
                border-right: 1px solid {Colors.BORDER};
            }}
        """)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.track_rows: List[QWidget] = []
        self.track_states: List[TrackState] = []

    def clear(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.track_rows.clear()
        self.track_states.clear()

    def addTrackRow(self, track_type: str, track_name: str, track_index: int):
        row = QFrame()
        row.setFixedHeight(45)
        row.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_PRIMARY};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)

        layout = QHBoxLayout(row)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(3)

        icon_map = {"gif": "\U0001f5bc", "video": "\U0001f3ac", "audio": "\U0001f3b5"}
        icon_label = QLabel(icon_map.get(track_type, "\U0001f4c1"))
        icon_label.setFixedWidth(20)
        layout.addWidget(icon_label)

        lock_btn = TrackControlButton("\U0001f513", "Lock track")
        lock_btn.setIcons("\U0001f512", "\U0001f513")
        lock_btn.clicked.connect(lambda: self._emit_state(track_index, "locked", lock_btn.isChecked()))
        layout.addWidget(lock_btn)

        eye_btn = TrackControlButton("\U0001f441", "Show/Hide")
        eye_btn.setIcons("\U0001f441", "\U0001f441\u200d\U0001f5e8")
        eye_btn.setChecked(True)
        eye_btn.clicked.connect(lambda: self._emit_state(track_index, "visible", eye_btn.isChecked()))
        layout.addWidget(eye_btn)

        if track_type == "audio":
            mute_btn = TrackControlButton("\U0001f50a", "Mute/Unmute")
            mute_btn.setIcons("\U0001f507", "\U0001f50a")
            mute_btn.clicked.connect(lambda: self._emit_state(track_index, "muted", mute_btn.isChecked()))
            layout.addWidget(mute_btn)
        else:
            spacer = QWidget()
            spacer.setFixedWidth(24)
            layout.addWidget(spacer)

        more_btn = QPushButton("\u22ef")
        more_btn.setFixedSize(24, 24)
        more_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {Colors.TEXT_TERTIARY};
                border: none;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.BG_SECONDARY};
                border-radius: 4px;
            }}
        """)
        more_btn.clicked.connect(lambda: self._show_menu(more_btn, track_index))
        layout.addWidget(more_btn)

        self.layout.addWidget(row)
        self.track_rows.append(row)
        self.track_states.append(TrackState())

    def _emit_state(self, index: int, prop: str, value: bool):
        self.stateChanged.emit(index, prop, value)

    def _show_menu(self, button, track_index):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                padding: 5px;
            }}
            QMenu::item {{
                padding: 8px 20px;
            }}
            QMenu::item:selected {{
                background: {Colors.ACCENT};
            }}
        """)

        menu.addAction("Delete track")
        menu.addAction("Duplicate track")
        menu.addSeparator()
        menu.addAction("Move up")
        menu.addAction("Move down")

        menu.exec(button.mapToGlobal(QPoint(0, button.height())))

    def addRulerSpacer(self, height: int):
        spacer = QWidget()
        spacer.setFixedHeight(height)
        self.layout.insertWidget(0, spacer)


class TimelineCanvas(QWidget):
    """Main timeline canvas with tracks and playhead"""

    seekRequested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.audio_tracks: List[MediaFile] = []
        self.video_tracks: List[MediaFile] = []
        self.gif_tracks: List[MediaFile] = []
        self.total_duration_ms = 0
        self.playhead_position_ms = 0
        self.crossfade_sec = 5
        self.pixels_per_second = 15
        self.scroll_offset = 0
        self.is_dragging = False

        self.track_height = 45
        self.track_spacing = 0
        self.ruler_height = 30

    def _calculate_needed_height(self):
        height = self.ruler_height
        height += self.track_height  # GIF
        height += self.track_height  # Video
        height += len(self.audio_tracks) * self.track_height
        return max(300, height + 10)

    def setData(self, audio, video, gif, total_ms, crossfade, pps):
        self.audio_tracks = audio or []
        self.video_tracks = video or []
        self.gif_tracks = gif or []
        self.total_duration_ms = total_ms
        self.crossfade_sec = crossfade
        self.pixels_per_second = pps
        needed = self._calculate_needed_height()
        self.setMinimumHeight(needed)
        self.setFixedHeight(needed)
        self.update()

    def setScrollOffset(self, offset):
        self.scroll_offset = offset
        self.update()

    def setPlayheadPosition(self, position_ms):
        self.playhead_position_ms = position_ms
        self.update()

    def mousePressEvent(self, event):
        if self.total_duration_ms > 0 and event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self._seek_to_mouse(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging and self.total_duration_ms > 0:
            self._seek_to_mouse(event)

    def mouseReleaseEvent(self, event):
        self.is_dragging = False

    def _seek_to_mouse(self, event):
        x = event.position().x() if hasattr(event, 'position') else event.x()
        pixel_pos = x + self.scroll_offset
        if self.pixels_per_second > 0:
            seek_sec = pixel_pos / self.pixels_per_second
            seek_ms = int(max(0, min(self.total_duration_ms, seek_sec * 1000)))
            self.playhead_position_ms = seek_ms
            self.seekRequested.emit(seek_ms)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(self.rect(), QColor(Colors.BG_PRIMARY))

        self._draw_ruler(painter, w)

        y_offset = self.ruler_height

        y_offset = self._draw_track_background(painter, y_offset, w, "GIF", Colors.GIF_COLOR)
        if self.gif_tracks:
            self._draw_media_bar(painter, y_offset - self.track_height + 5, w,
                                 self.gif_tracks, Colors.GIF_COLOR, loop=True)

        y_offset = self._draw_track_background(painter, y_offset, w, "Video", Colors.VIDEO_COLOR)
        if self.video_tracks:
            self._draw_media_bar(painter, y_offset - self.track_height + 5, w,
                                 self.video_tracks, Colors.VIDEO_COLOR, loop=True)

        for i, audio in enumerate(self.audio_tracks):
            y_offset = self._draw_track_background(painter, y_offset, w, f"Audio {i+1}", Colors.AUDIO_COLOR)
            self._draw_audio_staircase(painter, y_offset - self.track_height + 5, w, audio, i)

        self._draw_playhead(painter, h)

    def _draw_ruler(self, painter, w):
        painter.fillRect(0, 0, w, self.ruler_height, QColor(Colors.BG_SECONDARY))

        painter.setPen(QColor(Colors.TEXT_TERTIARY))
        painter.setFont(QFont("Inter", 9))

        if self.total_duration_ms > 0:
            if self.total_duration_ms < 60000:
                interval_sec = 10
            elif self.total_duration_ms < 300000:
                interval_sec = 30
            else:
                interval_sec = 60

            total_sec = self.total_duration_ms / 1000

            for sec in range(0, int(total_sec) + 1, interval_sec):
                x = int(sec * self.pixels_per_second) - self.scroll_offset
                if 0 <= x <= w:
                    hrs = sec // 3600
                    mins = (sec % 3600) // 60
                    secs = sec % 60
                    time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}:00"
                    painter.drawText(x + 2, 18, time_str)
                    painter.drawLine(x, 22, x, self.ruler_height)

    def _draw_track_background(self, painter, y_start, w, label, color):
        bg_color = Colors.BG_PRIMARY if y_start % 90 == self.ruler_height else Colors.TRACK_BG
        painter.fillRect(0, y_start, w, self.track_height, QColor(bg_color))

        painter.setPen(QColor(Colors.BORDER))
        painter.drawLine(0, y_start + self.track_height, w, y_start + self.track_height)

        return y_start + self.track_height

    def _draw_media_bar(self, painter, y, w, tracks, color, loop=False):
        if not tracks or self.total_duration_ms <= 0:
            return

        total_track_duration = sum(t.duration for t in tracks)
        if total_track_duration <= 0:
            return

        total_sec = self.total_duration_ms / 1000
        bar_height = self.track_height - 10

        current_x = -self.scroll_offset
        time_pos = 0

        while time_pos < total_sec:
            for track in tracks:
                if time_pos >= total_sec:
                    break

                seg_width = int(track.duration * self.pixels_per_second)

                if current_x + seg_width > 0 and current_x < w:
                    draw_x = max(0, current_x)
                    draw_width = min(seg_width, w - draw_x)
                    if current_x < 0:
                        draw_width = seg_width + current_x

                    painter.setBrush(QColor(color))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRoundedRect(int(draw_x), y, int(draw_width), bar_height, 4, 4)

                    is_gif_file = track.path.lower().endswith('.gif')
                    thumbnails = []
                    if not is_gif_file:
                        thumb_width = int(bar_height * 16 / 9)
                        num_thumbs_needed = max(5, seg_width // max(1, thumb_width))
                        thumbnails = ThumbnailCache.get_thumbnails(
                            track.path, num_frames=min(num_thumbs_needed, 40), thumb_height=bar_height
                        )

                    if thumbnails:
                        painter.save()
                        clip_path = QPainterPath()
                        clip_path.addRoundedRect(float(draw_x), float(y),
                                                 float(draw_width), float(bar_height), 4.0, 4.0)
                        painter.setClipPath(clip_path)

                        step_px = max(1, seg_width // len(thumbnails)) if len(thumbnails) > 0 else int(bar_height * 16 / 9)

                        for idx, pix in enumerate(thumbnails):
                            frame_x = int(current_x) + idx * step_px
                            if frame_x + step_px < 0:
                                continue
                            if frame_x > w:
                                break
                            scaled = pix.scaled(step_px, bar_height,
                                                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                Qt.TransformationMode.FastTransformation)
                            painter.drawPixmap(int(frame_x), y, step_px, bar_height, scaled)

                        overlay_color = QColor(color)
                        overlay_color.setAlpha(45)
                        painter.setBrush(overlay_color)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawRect(int(draw_x), y, int(draw_width), bar_height)

                        painter.restore()
                    else:
                        if is_gif_file:
                            painter.setPen(QPen(QColor("#ffffff70"), 1))
                            painter.setFont(QFont("Inter", 7))
                            icon_spacing = 80
                            for gx in range(int(draw_x) + 10, int(draw_x + draw_width) - 10, icon_spacing):
                                painter.drawText(gx, y + bar_height // 2 + 4, "GIF")
                        else:
                            import hashlib
                            seed = int(hashlib.md5(track.path.encode()).hexdigest()[:8], 16)
                            rng = random.Random(seed)
                            painter.setPen(QPen(QColor("#ffffff40"), 1))
                            for wx in range(int(draw_x) + 5, int(draw_x + draw_width) - 5, 3):
                                wave_h = rng.randint(5, bar_height - 10)
                                wy = y + (bar_height - wave_h) // 2
                                painter.drawLine(wx, wy, wx, wy + wave_h)

                    if draw_width > 50:
                        painter.setBrush(QColor(0, 0, 0, 140))
                        painter.setPen(Qt.PenStyle.NoPen)
                        label_text = track.name
                        dur_min = int(track.duration) // 60
                        dur_sec = int(track.duration) % 60
                        dur_str = f"{dur_min}:{dur_sec:02d}"
                        display = f"{label_text[:20]}  {dur_str}"
                        painter.setFont(QFont("Inter", 8, QFont.Weight.Bold))
                        text_w = painter.fontMetrics().horizontalAdvance(display) + 12
                        painter.drawRoundedRect(int(draw_x) + 4, y + 2, min(int(text_w), int(draw_width) - 8), 16, 3, 3)
                        painter.setPen(QColor("#ffffff"))
                        painter.drawText(int(draw_x) + 10, y + 14, display)

                current_x += seg_width
                time_pos += track.duration

            if not loop:
                break

    def _draw_audio_staircase(self, painter, y, w, audio, index):
        if not audio or self.total_duration_ms <= 0:
            return

        bar_height = self.track_height - 10

        start_sec = 0
        for i in range(index):
            if i < len(self.audio_tracks):
                prev_duration = self.audio_tracks[i].duration
                if i == 0:
                    start_sec += prev_duration
                else:
                    start_sec += max(0, prev_duration - self.crossfade_sec)

        x = int(start_sec * self.pixels_per_second) - self.scroll_offset
        seg_width = int(audio.duration * self.pixels_per_second)

        if x + seg_width > 0 and x < w:
            draw_x = max(0, x)
            draw_width = min(seg_width, w - draw_x)
            if x < 0:
                draw_width = seg_width + x

            painter.setBrush(QColor(Colors.AUDIO_COLOR))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(int(draw_x), y, int(draw_width), bar_height, 4, 4)

            waveform = WaveformCache.get_waveform(audio.path, max(100, seg_width // 2))
            painter.setPen(QPen(QColor("#ffffff50"), 1))

            for wx_pixel in range(int(draw_x) + 2, int(draw_x + draw_width) - 2, 2):
                pixel_offset = wx_pixel - int(x)
                if pixel_offset < 0:
                    continue
                sample_idx = int((pixel_offset / seg_width) * (len(waveform) - 1))
                sample_idx = max(0, min(sample_idx, len(waveform) - 1))

                val = waveform[sample_idx]
                wave_h = max(3, int(val * (bar_height - 6)))
                wy = y + (bar_height - wave_h) // 2
                painter.drawLine(wx_pixel, wy, wx_pixel, wy + wave_h)

            if self.playhead_position_ms > 0:
                playhead_sec = self.playhead_position_ms / 1000
                if start_sec <= playhead_sec <= start_sec + audio.duration:
                    played_width = int((playhead_sec - start_sec) * self.pixels_per_second)
                    played_draw_x = max(int(draw_x), int(x))
                    played_draw_w = min(played_width, int(draw_width))
                    if played_draw_w > 0:
                        painter.setBrush(QColor(255, 165, 0, 40))
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawRoundedRect(int(played_draw_x), y, played_draw_w, bar_height, 4, 4)

            if draw_width > 60:
                dur_min = int(audio.duration) // 60
                dur_sec = int(audio.duration) % 60
                dur_str = f"{dur_min}:{dur_sec:02d}"
                name = audio.name[:22]
                display = f"{name}  {dur_str}"
                painter.setFont(QFont("Inter", 8, QFont.Weight.Bold))
                text_w = painter.fontMetrics().horizontalAdvance(display) + 12
                painter.setBrush(QColor(0, 0, 0, 140))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(int(draw_x) + 4, y + 2, min(int(text_w), int(draw_width) - 8), 16, 3, 3)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(int(draw_x) + 10, y + 14, display)

    def _draw_playhead(self, painter, h):
        if self.total_duration_ms <= 0:
            return

        x = int((self.playhead_position_ms / 1000) * self.pixels_per_second) - self.scroll_offset

        if 0 <= x <= self.width():
            painter.setPen(QPen(QColor(Colors.ACCENT), 2))
            painter.drawLine(x, self.ruler_height, x, h)

            painter.setBrush(QColor(Colors.ACCENT))
            painter.setPen(Qt.PenStyle.NoPen)

            handle = QPolygon([
                QPoint(x - 8, self.ruler_height - 12),
                QPoint(x + 8, self.ruler_height - 12),
                QPoint(x, self.ruler_height)
            ])
            painter.drawPolygon(handle)
