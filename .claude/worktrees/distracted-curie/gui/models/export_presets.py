"""
Export format presets — platform-optimised encoding configurations.

Story 3.6 — Epic 3: CapCut Features.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from gui.utils.compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QGroupBox, QListWidget, QListWidgetItem,
    QDoubleSpinBox,
    Qt, QSize,
    QFont, QColor,
    pyqtSignal,
)


# ---------------------------------------------------------------------------
# ExportPreset model
# ---------------------------------------------------------------------------

@dataclass
class ExportPreset:
    """An export encoding configuration.

    Attributes:
        name:          Human-readable preset name.
        resolution:    (width, height) or (0, 0) for audio-only.
        fps:           Frames per second (0 for audio-only).
        video_codec:   FFmpeg video codec name (e.g. libx264).
        audio_codec:   FFmpeg audio codec name (e.g. aac).
        video_bitrate: Video bitrate string (e.g. '8M').
        audio_bitrate: Audio bitrate string (e.g. '320k').
        container:     Container format (e.g. mp4, mov, wav).
        platform_name: Target platform label.
        extra_flags:   Additional FFmpeg flags.
    """
    name: str = "Custom"
    resolution: Tuple[int, int] = (1920, 1080)
    fps: float = 30.0
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    video_bitrate: str = "8M"
    audio_bitrate: str = "192k"
    container: str = "mp4"
    platform_name: str = ""
    extra_flags: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def is_audio_only(self) -> bool:
        return self.resolution == (0, 0) or self.fps == 0

    def to_ffmpeg_args(self, input_path: str, output_path: str) -> List[str]:
        """Build FFmpeg command-line arguments for this preset."""
        args = ["ffmpeg", "-y", "-i", input_path]

        if not self.is_audio_only:
            args.extend(["-c:v", self.video_codec])
            args.extend(["-b:v", self.video_bitrate])
            w, h = self.resolution
            args.extend(["-vf", f"scale={w}:{h}"])
            args.extend(["-r", str(self.fps)])

        args.extend(["-c:a", self.audio_codec])
        args.extend(["-b:a", self.audio_bitrate])

        if self.extra_flags:
            args.extend(self.extra_flags.split())

        args.append(output_path)
        return args


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

def _builtin_presets() -> List[ExportPreset]:
    """Return the list of built-in export presets."""
    return [
        # YouTube
        ExportPreset(
            name="YouTube 1080p",
            resolution=(1920, 1080),
            fps=30.0,
            video_codec="libx264",
            audio_codec="aac",
            video_bitrate="8M",
            audio_bitrate="192k",
            container="mp4",
            platform_name="YouTube",
            extra_flags="-pix_fmt yuv420p -profile:v high -level 4.1",
        ),
        ExportPreset(
            name="YouTube 4K",
            resolution=(3840, 2160),
            fps=30.0,
            video_codec="libx264",
            audio_codec="aac",
            video_bitrate="35M",
            audio_bitrate="192k",
            container="mp4",
            platform_name="YouTube",
            extra_flags="-pix_fmt yuv420p -profile:v high -level 5.1",
        ),
        ExportPreset(
            name="YouTube Shorts",
            resolution=(1080, 1920),
            fps=30.0,
            video_codec="libx264",
            audio_codec="aac",
            video_bitrate="6M",
            audio_bitrate="192k",
            container="mp4",
            platform_name="YouTube Shorts",
            extra_flags="-pix_fmt yuv420p",
        ),
        # Instagram
        ExportPreset(
            name="Instagram Reels",
            resolution=(1080, 1920),
            fps=30.0,
            video_codec="libx264",
            audio_codec="aac",
            video_bitrate="6M",
            audio_bitrate="128k",
            container="mp4",
            platform_name="Instagram",
            extra_flags="-pix_fmt yuv420p -movflags +faststart",
        ),
        ExportPreset(
            name="Instagram Feed",
            resolution=(1080, 1080),
            fps=30.0,
            video_codec="libx264",
            audio_codec="aac",
            video_bitrate="5M",
            audio_bitrate="128k",
            container="mp4",
            platform_name="Instagram",
            extra_flags="-pix_fmt yuv420p -movflags +faststart",
        ),
        # TikTok
        ExportPreset(
            name="TikTok",
            resolution=(1080, 1920),
            fps=30.0,
            video_codec="libx264",
            audio_codec="aac",
            video_bitrate="6M",
            audio_bitrate="128k",
            container="mp4",
            platform_name="TikTok",
            extra_flags="-pix_fmt yuv420p -movflags +faststart",
        ),
        # Audio platforms
        ExportPreset(
            name="Spotify",
            resolution=(0, 0),
            fps=0,
            video_codec="",
            audio_codec="libmp3lame",
            video_bitrate="0",
            audio_bitrate="320k",
            container="mp3",
            platform_name="Spotify",
        ),
        ExportPreset(
            name="Apple Music",
            resolution=(0, 0),
            fps=0,
            video_codec="",
            audio_codec="alac",
            video_bitrate="0",
            audio_bitrate="0",
            container="m4a",
            platform_name="Apple Music",
            extra_flags="-acodec alac",
        ),
        # Master / archival
        ExportPreset(
            name="ProRes 422 (Master)",
            resolution=(1920, 1080),
            fps=30.0,
            video_codec="prores_ks",
            audio_codec="pcm_s24le",
            video_bitrate="0",
            audio_bitrate="0",
            container="mov",
            platform_name="Master",
            extra_flags="-profile:v 2 -pix_fmt yuv422p10le",
        ),
        ExportPreset(
            name="WAV (Audio Master)",
            resolution=(0, 0),
            fps=0,
            video_codec="",
            audio_codec="pcm_s24le",
            video_bitrate="0",
            audio_bitrate="0",
            container="wav",
            platform_name="Master",
        ),
    ]


BUILTIN_PRESETS: List[ExportPreset] = _builtin_presets()


# ---------------------------------------------------------------------------
# ExportPresetPanel widget
# ---------------------------------------------------------------------------

class ExportPresetPanel(QWidget):
    """Panel for selecting and customising export presets."""

    preset_selected = pyqtSignal(object)  # emits ExportPreset

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._presets: List[ExportPreset] = list(BUILTIN_PRESETS)
        self._current: Optional[ExportPreset] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Export Preset")
        header.setStyleSheet("color: #FFFFFF; font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        # Preset list
        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget { background: #2A2A2A; border: none; }"
            "QListWidget::item { padding: 6px; color: #DDDDDD; }"
            "QListWidget::item:selected { background: #3A3A3A; }"
        )
        for preset in self._presets:
            platform_tag = f" [{preset.platform_name}]" if preset.platform_name else ""
            item = QListWidgetItem(f"{preset.name}{platform_tag}")
            item.setData(Qt.ItemDataRole.UserRole, preset)
            self._list.addItem(item)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)

        # Details group
        details = QGroupBox("Settings")
        dl = QGridLayout(details)

        self._res_label = QLabel("—")
        dl.addWidget(QLabel("Resolution:"), 0, 0)
        dl.addWidget(self._res_label, 0, 1)

        self._fps_label = QLabel("—")
        dl.addWidget(QLabel("FPS:"), 1, 0)
        dl.addWidget(self._fps_label, 1, 1)

        self._vcodec_label = QLabel("—")
        dl.addWidget(QLabel("Video Codec:"), 2, 0)
        dl.addWidget(self._vcodec_label, 2, 1)

        self._acodec_label = QLabel("—")
        dl.addWidget(QLabel("Audio Codec:"), 3, 0)
        dl.addWidget(self._acodec_label, 3, 1)

        self._vbitrate_label = QLabel("—")
        dl.addWidget(QLabel("Video Bitrate:"), 4, 0)
        dl.addWidget(self._vbitrate_label, 4, 1)

        self._abitrate_label = QLabel("—")
        dl.addWidget(QLabel("Audio Bitrate:"), 5, 0)
        dl.addWidget(self._abitrate_label, 5, 1)

        self._container_label = QLabel("—")
        dl.addWidget(QLabel("Container:"), 6, 0)
        dl.addWidget(self._container_label, 6, 1)

        layout.addWidget(details)

        # Export button
        export_btn = QPushButton("Export")
        export_btn.setStyleSheet(
            "QPushButton { background: #FF6F00; color: white; padding: 10px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #FF8F00; }"
        )
        export_btn.clicked.connect(self._on_export)
        layout.addWidget(export_btn)

    # -- public API ---------------------------------------------------------

    def get_selected_preset(self) -> Optional[ExportPreset]:
        return self._current

    def add_custom_preset(self, preset: ExportPreset) -> None:
        self._presets.append(preset)
        platform_tag = f" [{preset.platform_name}]" if preset.platform_name else ""
        item = QListWidgetItem(f"{preset.name}{platform_tag}")
        item.setData(Qt.ItemDataRole.UserRole, preset)
        self._list.addItem(item)

    # -- slots --------------------------------------------------------------

    def _on_selection_changed(self, current: Optional[QListWidgetItem], _prev: Optional[QListWidgetItem] = None) -> None:
        if current is None:
            return
        preset = current.data(Qt.ItemDataRole.UserRole)
        if preset is None:
            return
        self._current = preset
        self._update_details(preset)

    def _update_details(self, p: ExportPreset) -> None:
        if p.is_audio_only:
            self._res_label.setText("Audio Only")
            self._fps_label.setText("—")
            self._vcodec_label.setText("—")
            self._vbitrate_label.setText("—")
        else:
            w, h = p.resolution
            self._res_label.setText(f"{w} x {h}")
            self._fps_label.setText(f"{p.fps}")
            self._vcodec_label.setText(p.video_codec)
            self._vbitrate_label.setText(p.video_bitrate)
        self._acodec_label.setText(p.audio_codec)
        self._abitrate_label.setText(p.audio_bitrate if p.audio_bitrate != "0" else "Lossless")
        self._container_label.setText(p.container)

    def _on_export(self) -> None:
        if self._current is not None:
            self.preset_selected.emit(self._current)
