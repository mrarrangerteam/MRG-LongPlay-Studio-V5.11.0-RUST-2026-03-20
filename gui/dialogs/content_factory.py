"""
Content Factory — Wizard Dialog

4-page wizard for batch content production:
  Page 1: Import — Add songs + background videos
  Page 2: Configure — Mastering, long/short settings, channel info
  Page 3: Preview — Review production plan, estimated sizes
  Page 4: Produce — Progress bars, cancel, open output folder

Uses ContentFactoryWorker for background processing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QStackedWidget,
        QPushButton, QLabel, QListWidget, QListWidgetItem,
        QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
        QCheckBox, QProgressBar, QGroupBox, QFormLayout,
        QFileDialog, QSizePolicy, QTextEdit, QFrame,
    )
    from PyQt6.QtCore import Qt, QSize
    from PyQt6.QtGui import QFont, QColor
except ImportError:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QStackedWidget,
        QPushButton, QLabel, QListWidget, QListWidgetItem,
        QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
        QCheckBox, QProgressBar, QGroupBox, QFormLayout,
        QFileDialog, QSizePolicy, QTextEdit, QFrame,
    )
    from PySide6.QtCore import Qt, QSize
    from PySide6.QtGui import QFont, QColor

from gui.content_factory.models import (
    ContentFactoryConfig, SongEntry, BackgroundVideo,
    BatchMasterConfig, VerticalStrategy, CrossfadeCurve,
)
from gui.content_factory.planner import ContentPlanner
from gui.content_factory.workers import ContentFactoryWorker


# ─── Styled helpers ──────────────────────────────────────────────────

DIALOG_STYLE = """
    QDialog { background: #1A1510; color: #F0E8D8; }
    QLabel { color: #C89B3C; }
    QGroupBox { color: #C89B3C; border: 1px solid #382E28;
                border-radius: 4px; margin-top: 10px; padding-top: 14px; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; }
    QListWidget { background: #0E0A06; color: #F0E8D8;
                  border: 1px solid #382E28; }
    QListWidget::item:selected { background: #4A3828; }
    QPushButton { background: #382E28; color: #F0E8D8;
                  border: 1px solid #4A3828; padding: 6px 16px;
                  border-radius: 3px; }
    QPushButton:hover { background: #4A3828; }
    QPushButton:pressed { background: #5A4838; }
    QPushButton:disabled { color: #6A6258; }
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background: #0E0A06; color: #F0E8D8;
        border: 1px solid #382E28; padding: 4px; }
    QProgressBar { background: #0E0A06; border: 1px solid #382E28;
                   text-align: center; color: #F0E8D8; }
    QProgressBar::chunk { background: #C89B3C; }
    QTextEdit { background: #0E0A06; color: #F0E8D8;
                border: 1px solid #382E28; }
    QCheckBox { color: #F0E8D8; }
"""


class ContentFactoryDialog(QDialog):
    """4-page Content Factory wizard dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Content Factory — Batch Production")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(DIALOG_STYLE)

        self._config = ContentFactoryConfig()
        self._worker: Optional[ContentFactoryWorker] = None

        self._build_ui()

    # ─── UI Construction ─────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Title
        title = QLabel("CONTENT FACTORY")
        title.setFont(QFont("Georgia", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Stacked pages
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        self._page_import = self._build_import_page()
        self._page_config = self._build_config_page()
        self._page_preview = self._build_preview_page()
        self._page_produce = self._build_produce_page()

        self._stack.addWidget(self._page_import)
        self._stack.addWidget(self._page_config)
        self._stack.addWidget(self._page_preview)
        self._stack.addWidget(self._page_produce)

        # Navigation
        nav_layout = QHBoxLayout()

        self._btn_back = QPushButton("← Back")
        self._btn_back.clicked.connect(self._go_back)
        nav_layout.addWidget(self._btn_back)

        nav_layout.addStretch()

        self._page_indicator = QLabel("Page 1 of 4")
        self._page_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._page_indicator)

        nav_layout.addStretch()

        self._btn_next = QPushButton("Next →")
        self._btn_next.clicked.connect(self._go_next)
        nav_layout.addWidget(self._btn_next)

        layout.addLayout(nav_layout)
        self._update_nav()

    # ─── Page 1: Import ──────────────────────────────────────────

    def _build_import_page(self) -> QFrame:
        page = QFrame()
        layout = QVBoxLayout(page)

        # Songs section
        songs_group = QGroupBox("Songs")
        songs_layout = QVBoxLayout(songs_group)

        self._songs_list = QListWidget()
        songs_layout.addWidget(self._songs_list)

        btn_row = QHBoxLayout()
        btn_add_songs = QPushButton("Add Songs...")
        btn_add_songs.clicked.connect(self._add_songs)
        btn_row.addWidget(btn_add_songs)

        btn_remove_song = QPushButton("Remove Selected")
        btn_remove_song.clicked.connect(
            lambda: self._songs_list.takeItem(self._songs_list.currentRow())
        )
        btn_row.addWidget(btn_remove_song)

        btn_clear_songs = QPushButton("Clear All")
        btn_clear_songs.clicked.connect(self._songs_list.clear)
        btn_row.addWidget(btn_clear_songs)
        songs_layout.addLayout(btn_row)

        layout.addWidget(songs_group)

        # Background videos section
        bg_group = QGroupBox("Background Videos")
        bg_layout = QVBoxLayout(bg_group)

        self._bg_list = QListWidget()
        bg_layout.addWidget(self._bg_list)

        bg_btn_row = QHBoxLayout()
        btn_add_bg = QPushButton("Add Videos...")
        btn_add_bg.clicked.connect(self._add_bg_videos)
        bg_btn_row.addWidget(btn_add_bg)

        btn_remove_bg = QPushButton("Remove Selected")
        btn_remove_bg.clicked.connect(
            lambda: self._bg_list.takeItem(self._bg_list.currentRow())
        )
        bg_btn_row.addWidget(btn_remove_bg)
        bg_layout.addLayout(bg_btn_row)

        layout.addWidget(bg_group)

        return page

    # ─── Page 2: Configure ───────────────────────────────────────

    def _build_config_page(self) -> QFrame:
        page = QFrame()
        layout = QVBoxLayout(page)

        # Channel info
        channel_group = QGroupBox("Channel Info")
        channel_form = QFormLayout(channel_group)

        self._channel_name = QLineEdit()
        self._channel_name.setPlaceholderText("e.g. Chillin' Vibes Official")
        channel_form.addRow("Channel Name:", self._channel_name)

        self._channel_genre = QComboBox()
        self._channel_genre.addItems([
            "Chill", "Lofi", "Jazz", "R&B", "Pop", "Rock",
            "EDM", "Hip-Hop", "Classical", "Ambient",
        ])
        channel_form.addRow("Genre:", self._channel_genre)

        layout.addWidget(channel_group)

        # Long video settings
        long_group = QGroupBox("Long Videos (16:9)")
        long_form = QFormLayout(long_group)

        self._long_count = QSpinBox()
        self._long_count.setRange(0, 10)
        self._long_count.setValue(1)
        long_form.addRow("Number of Long Videos:", self._long_count)

        self._songs_per_long = QSpinBox()
        self._songs_per_long.setRange(0, 100)
        self._songs_per_long.setValue(20)
        self._songs_per_long.setSpecialValueText("Auto")
        long_form.addRow("Songs per Video:", self._songs_per_long)

        self._crossfade_sec = QDoubleSpinBox()
        self._crossfade_sec.setRange(0, 10)
        self._crossfade_sec.setValue(3.0)
        self._crossfade_sec.setSuffix(" sec")
        long_form.addRow("Crossfade:", self._crossfade_sec)

        self._crossfade_curve = QComboBox()
        self._crossfade_curve.addItems([c.value for c in CrossfadeCurve])
        self._crossfade_curve.setCurrentText("exp")
        long_form.addRow("Crossfade Curve:", self._crossfade_curve)

        layout.addWidget(long_group)

        # Short video settings
        short_group = QGroupBox("Shorts (9:16, ≤29s)")
        short_form = QFormLayout(short_group)

        self._short_count = QSpinBox()
        self._short_count.setRange(0, 200)
        self._short_count.setValue(0)
        self._short_count.setSpecialValueText("One per song")
        short_form.addRow("Number of Shorts:", self._short_count)

        self._short_max_sec = QDoubleSpinBox()
        self._short_max_sec.setRange(5, 60)
        self._short_max_sec.setValue(29.0)
        self._short_max_sec.setSuffix(" sec")
        short_form.addRow("Max Duration:", self._short_max_sec)

        self._vertical_strategy = QComboBox()
        self._vertical_strategy.addItems([s.value for s in VerticalStrategy])
        short_form.addRow("Vertical Style:", self._vertical_strategy)

        layout.addWidget(short_group)

        # Mastering
        master_group = QGroupBox("Batch Mastering (Dynamics + Loudness + Imager)")
        master_form = QFormLayout(master_group)

        self._master_enabled = QCheckBox("Enable Batch Mastering")
        self._master_enabled.setChecked(True)
        master_form.addRow(self._master_enabled)

        self._target_lufs = QDoubleSpinBox()
        self._target_lufs.setRange(-24, 0)
        self._target_lufs.setValue(-14.0)
        self._target_lufs.setSuffix(" LUFS")
        master_form.addRow("Target Loudness:", self._target_lufs)

        layout.addWidget(master_group)

        # Upload
        upload_group = QGroupBox("YouTube Upload")
        upload_form = QFormLayout(upload_group)

        self._auto_upload = QCheckBox("Auto-upload after rendering")
        upload_form.addRow(self._auto_upload)

        self._upload_privacy = QComboBox()
        self._upload_privacy.addItems(["private", "unlisted", "public"])
        upload_form.addRow("Privacy:", self._upload_privacy)

        layout.addWidget(upload_group)

        return page

    # ─── Page 3: Preview ─────────────────────────────────────────

    def _build_preview_page(self) -> QFrame:
        page = QFrame()
        layout = QVBoxLayout(page)

        lbl = QLabel("Production Plan Preview")
        lbl.setFont(QFont("Georgia", 12, QFont.Weight.Bold))
        layout.addWidget(lbl)

        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setFont(QFont("Menlo", 10))
        layout.addWidget(self._preview_text)

        # Output dir
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output:"))
        self._output_dir = QLineEdit(str(Path.home() / "LongPlay_Output"))
        dir_row.addWidget(self._output_dir)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(btn_browse)
        layout.addLayout(dir_row)

        return page

    # ─── Page 4: Produce ─────────────────────────────────────────

    def _build_produce_page(self) -> QFrame:
        page = QFrame()
        layout = QVBoxLayout(page)

        self._phase_label = QLabel("Ready to produce")
        self._phase_label.setFont(QFont("Georgia", 14, QFont.Weight.Bold))
        self._phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._phase_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setMinimumHeight(30)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        # Log area
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Menlo", 9))
        self._log_text.setMaximumHeight(200)
        layout.addWidget(self._log_text)

        # Action buttons
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("START PRODUCTION")
        self._btn_start.setStyleSheet(
            "QPushButton { background: #5C9A3C; font-weight: bold; font-size: 14px; padding: 10px; }"
            "QPushButton:hover { background: #6CAA4C; }"
        )
        self._btn_start.clicked.connect(self._start_production)
        btn_row.addWidget(self._btn_start)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_production)
        btn_row.addWidget(self._btn_cancel)

        self._btn_open_folder = QPushButton("Open Output Folder")
        self._btn_open_folder.setEnabled(False)
        self._btn_open_folder.clicked.connect(self._open_output_folder)
        btn_row.addWidget(self._btn_open_folder)

        layout.addLayout(btn_row)
        layout.addStretch()

        return page

    # ─── Navigation ──────────────────────────────────────────────

    def _update_nav(self):
        idx = self._stack.currentIndex()
        self._btn_back.setEnabled(idx > 0)

        if idx == 3:
            self._btn_next.setText("Close")
        elif idx == 2:
            self._btn_next.setText("Produce →")
        else:
            self._btn_next.setText("Next →")

        self._page_indicator.setText(f"Page {idx + 1} of 4")

    def _go_back(self):
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
            self._update_nav()

    def _go_next(self):
        idx = self._stack.currentIndex()
        if idx == 3:
            self.accept()
            return

        if idx == 1:
            # Moving to Preview: generate plan
            self._generate_preview()
        elif idx == 2:
            # Moving to Produce
            pass

        if idx < 3:
            self._stack.setCurrentIndex(idx + 1)
            self._update_nav()

    # ─── Import Actions ──────────────────────────────────────────

    def _add_songs(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Songs", "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma);;All Files (*)",
        )
        for f in files:
            item = QListWidgetItem(Path(f).name)
            item.setData(Qt.ItemDataRole.UserRole, f)
            self._songs_list.addItem(item)

    def _add_bg_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Background Videos", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)",
        )
        for f in files:
            item = QListWidgetItem(Path(f).name)
            item.setData(Qt.ItemDataRole.UserRole, f)
            self._bg_list.addItem(item)

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._output_dir.setText(d)

    # ─── Build Config from UI ────────────────────────────────────

    def _build_config(self) -> ContentFactoryConfig:
        """Collect UI values into ContentFactoryConfig."""
        songs = []
        for i in range(self._songs_list.count()):
            item = self._songs_list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            songs.append(SongEntry(file_path=path))

        bg_videos = []
        for i in range(self._bg_list.count()):
            item = self._bg_list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            bg_videos.append(BackgroundVideo(file_path=path))

        curve = CrossfadeCurve(self._crossfade_curve.currentText())
        v_strategy = VerticalStrategy(self._vertical_strategy.currentText())

        config = ContentFactoryConfig(
            songs=songs,
            bg_videos=bg_videos,
            long_count=self._long_count.value(),
            songs_per_long=self._songs_per_long.value(),
            crossfade_sec=self._crossfade_sec.value(),
            crossfade_curve=curve,
            short_count=self._short_count.value(),
            short_max_sec=self._short_max_sec.value(),
            vertical_strategy=v_strategy,
            master_config=BatchMasterConfig(
                enabled=self._master_enabled.isChecked(),
                target_lufs=self._target_lufs.value(),
            ),
            channel_name=self._channel_name.text(),
            channel_genre=self._channel_genre.currentText(),
            output_dir=self._output_dir.text(),
            auto_upload=self._auto_upload.isChecked(),
            upload_privacy=self._upload_privacy.currentText(),
        )
        return config

    # ─── Preview ─────────────────────────────────────────────────

    def _generate_preview(self):
        """Generate and display production plan preview."""
        config = self._build_config()
        self._config = config

        planner = ContentPlanner()
        plan = planner.create_plan(config)

        lines = []
        lines.append(f"Channel: {config.channel_name or '(unnamed)'}")
        lines.append(f"Genre: {config.channel_genre}")
        lines.append(f"Songs: {len(config.songs)}")
        lines.append(f"Background Videos: {len(config.bg_videos)}")
        lines.append("")

        # Long videos
        lines.append(f"═══ Long Videos: {len(plan.long_videos)} ═══")
        for i, lv in enumerate(plan.long_videos):
            dur_min = lv.estimated_duration / 60
            lines.append(f"  #{i + 1}: {len(lv.songs)} songs, ~{dur_min:.0f} min")
            for j, song in enumerate(lv.songs):
                lines.append(f"    {j + 1}. {song.title} ({song.duration_sec:.0f}s)")

        lines.append("")

        # Shorts
        lines.append(f"═══ Shorts: {len(plan.shorts)} ═══")
        for i, sv in enumerate(plan.shorts):
            if sv.song:
                lines.append(f"  #{i + 1}: {sv.song.title} (≤{sv.max_duration_sec}s)")

        lines.append("")

        # Estimate
        lines.append("═══ Estimated Output ═══")
        total_long_min = sum(lv.estimated_duration for lv in plan.long_videos) / 60
        lines.append(f"  Long videos: ~{total_long_min:.0f} min total")
        lines.append(f"  Shorts: {len(plan.shorts)} clips")
        lines.append(f"  Output dir: {config.output_dir}")
        lines.append(f"  Mastering: {'ON' if config.master_config.enabled else 'OFF'}")
        lines.append(f"  Auto-upload: {'ON' if config.auto_upload else 'OFF'}")

        if planner.errors:
            lines.append("")
            lines.append("⚠ Warnings:")
            for err in planner.errors:
                lines.append(f"  - {err}")

        self._preview_text.setText("\n".join(lines))

    # ─── Production ──────────────────────────────────────────────

    def _start_production(self):
        """Start the Content Factory worker."""
        config = self._build_config()
        self._config = config

        self._worker = ContentFactoryWorker(config, self)
        self._worker.progress_changed.connect(self._on_progress)
        self._worker.phase_changed.connect(self._on_phase)
        self._worker.job_completed.connect(self._on_completed)
        self._worker.job_failed.connect(self._on_failed)

        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._btn_back.setEnabled(False)
        self._log_text.clear()
        self._log_text.append("Starting Content Factory...")

        self._worker.start()

    def _cancel_production(self):
        if self._worker:
            self._worker.cancel()
            self._log_text.append("Cancellation requested...")
            self._btn_cancel.setEnabled(False)

    def _on_progress(self, pct: float, msg: str):
        self._progress_bar.setValue(int(pct))
        self._status_label.setText(msg)
        self._log_text.append(f"[{pct:.0f}%] {msg}")

    def _on_phase(self, phase: str):
        self._phase_label.setText(phase)

    def _on_completed(self, job):
        self._phase_label.setText("COMPLETE")
        self._progress_bar.setValue(100)
        self._status_label.setText(
            f"Done! {len(job.output_files)} files produced"
        )
        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(False)
        self._btn_open_folder.setEnabled(True)

        if job.errors:
            self._log_text.append("\nNotes:")
            for err in job.errors:
                self._log_text.append(f"  - {err}")

    def _on_failed(self, error_msg: str):
        self._phase_label.setText("FAILED")
        self._status_label.setText(f"Error: {error_msg}")
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._log_text.append(f"\nFATAL ERROR: {error_msg}")

    def _open_output_folder(self):
        """Open output folder in file manager."""
        output_dir = self._config.output_dir
        if os.path.isdir(output_dir):
            import subprocess
            import sys
            if sys.platform == "darwin":
                subprocess.Popen(["open", output_dir])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", output_dir])
            else:
                subprocess.Popen(["xdg-open", output_dir])
