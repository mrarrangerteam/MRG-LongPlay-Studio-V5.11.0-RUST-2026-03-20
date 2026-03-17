"""
Track list widgets for playlist display and reordering.

Classes:
    DraggableTrackListWidget — QListWidget with internal drag/drop reordering
    TrackListItem            — Single track in playlist (with crossfade preview)
"""

import os

from gui.utils.compat import (
    QListWidget, QListWidgetItem, QFrame, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QComboBox, Qt, pyqtSignal,
)
from gui.styles import Colors
from gui.audio_player import MediaFile


class DraggableTrackListWidget(QListWidget):
    """QListWidget with internal drag & drop reordering and track info display"""
    orderChanged = pyqtSignal(list)   # emits new order of file paths
    playRequested = pyqtSignal(int)   # emits track index to play

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks = []

        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        self.setStyleSheet(f"""
            QListWidget {{
                background: {Colors.BG_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
            }}
            QListWidget::item {{
                color: {Colors.TEXT_PRIMARY};
                padding: 12px 8px;
                border-bottom: 1px solid {Colors.BORDER};
                margin: 2px;
            }}
            QListWidget::item:hover {{
                background: {Colors.BG_TERTIARY};
            }}
            QListWidget::item:selected {{
                background: {Colors.ACCENT};
                color: white;
            }}
        """)

        self.model().rowsMoved.connect(self._on_rows_moved)

    def _on_rows_moved(self, parent, start, end, destination, row):
        new_tracks = []
        for i in range(self.count()):
            item = self.item(i)
            if item and hasattr(item, 'track_data'):
                new_tracks.append(item.track_data)
        self.tracks = new_tracks
        self.orderChanged.emit([t['file_path'] for t in self.tracks])

    def set_tracks(self, tracks: list):
        """Set tracks to display."""
        self.clear()
        self.tracks = tracks

        for i, track in enumerate(tracks):
            item = QListWidgetItem()
            item.track_data = track

            name = track.get('name', 'Unknown')
            duration_sec = track.get('duration_sec', 0)
            duration_str = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"
            bpm = track.get('bpm', 0)
            key = track.get('key', '')
            energy = track.get('energy', 0)
            energy_bars = "\u2588" * int(energy * 6) + "\u2591" * (6 - int(energy * 6))

            display_text = f"\u2261  {i+1}.  {name}    [{duration_str}]"
            if bpm > 0:
                display_text += f"    BPM: {int(bpm)}"
            if key:
                display_text += f"    Key: {key}"
            if energy > 0:
                display_text += f"    {energy_bars}"

            item.setText(display_text)
            item.setToolTip("Drag to reorder\nDouble-click to play")
            self.addItem(item)

    def get_ordered_paths(self) -> list:
        paths = []
        for i in range(self.count()):
            item = self.item(i)
            if item and hasattr(item, 'track_data'):
                paths.append(item.track_data['file_path'])
        return paths

    def get_ordered_tracks(self) -> list:
        tracks = []
        for i in range(self.count()):
            item = self.item(i)
            if item and hasattr(item, 'track_data'):
                tracks.append(item.track_data)
        return tracks

    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(event.pos())
        if item:
            row = self.row(item)
            self.playRequested.emit(row)
        super().mouseDoubleClickEvent(event)


class TrackListItem(QFrame):
    """Single track in playlist"""
    previewCrossfadeRequested = pyqtSignal(int)

    def __init__(self, index: int, track: MediaFile, parent=None):
        super().__init__(parent)
        self.track = track
        self.index = index

        self.setFixedHeight(45)
        self.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_TERTIARY};
                border-radius: 6px;
                margin: 2px;
            }}
            QFrame:hover {{
                background: {Colors.BG_SECONDARY};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)

        idx_label = QLabel(str(index + 1))
        idx_label.setFixedWidth(20)
        idx_label.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 12px; font-weight: bold;")
        layout.addWidget(idx_label)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(0)

        name = QLabel(track.name)
        name.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 11px;")
        info_layout.addWidget(name)

        details = QLabel(f"{track.duration_str} \u2022 {track.lufs:.1f} LUFS")
        details.setStyleSheet(f"color: {Colors.TEXT_TERTIARY}; font-size: 9px;")
        info_layout.addWidget(details)

        layout.addLayout(info_layout)
        layout.addStretch()

        if index > 0:
            self.preview_btn = QPushButton("\U0001f3a7")
            self.preview_btn.setFixedSize(28, 28)
            self.preview_btn.setToolTip(f"Preview crossfade: Track {index} \u2192 {index + 1}")
            self.preview_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.BG_PRIMARY};
                    border: 1px solid {Colors.BORDER};
                    color: {Colors.ACCENT};
                    font-size: 12px;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background: {Colors.ACCENT};
                    color: white;
                }}
            """)
            self.preview_btn.clicked.connect(lambda: self.previewCrossfadeRequested.emit(index))
            layout.addWidget(self.preview_btn)

        self.video_combo = QComboBox()
        self.video_combo.addItems(["V1"])
        self.video_combo.setFixedWidth(55)
        self.video_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_PRIMARY};
                border: 1px solid {Colors.BORDER};
                color: {Colors.TEXT_PRIMARY};
                padding: 3px;
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self.video_combo)

    def update_video_options(self, num_videos: int):
        current = self.video_combo.currentIndex()
        self.video_combo.clear()
        options = [f"V{i+1}" for i in range(max(1, num_videos))]
        self.video_combo.addItems(options)
        if current < len(options):
            self.video_combo.setCurrentIndex(current)
