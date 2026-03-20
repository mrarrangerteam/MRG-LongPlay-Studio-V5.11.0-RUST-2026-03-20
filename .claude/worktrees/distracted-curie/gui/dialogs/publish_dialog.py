"""
One-click publish dialog — upload video to YouTube with auto-generated SEO metadata.

Combines video preview, SEO metadata editing, platform selection,
privacy settings, and upload progress into a single dialog.

Classes:
    PublishDialog — Main publish/upload dialog.
"""

import os
import logging
from typing import Optional, List

from gui.utils.compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QComboBox, QLineEdit, QTextEdit,
    QCheckBox, QProgressBar, QMessageBox, QApplication,
    Qt, pyqtSignal, QThread, QTimer, QPixmap, QFont, QImage,
)
from gui.styles import Colors

logger = logging.getLogger(__name__)


# ======================================================================
# Upload worker thread
# ======================================================================

class _UploadWorker(QThread):
    """Background thread that drives the YouTube upload."""

    progress = pyqtSignal(float)      # percentage 0-100
    finished = pyqtSignal(str)        # video URL on success (empty on failure)
    error = pyqtSignal(str)           # error message

    def __init__(
        self,
        uploader,
        video_path: str,
        title: str,
        description: str,
        tags: list,
        category_id: str,
        privacy: str,
        parent=None,
    ):
        super().__init__(parent)
        self._uploader = uploader
        self._video_path = video_path
        self._title = title
        self._description = description
        self._tags = tags
        self._category_id = category_id
        self._privacy = privacy

    def run(self):
        try:
            url = self._uploader.upload_video(
                video_path=self._video_path,
                title=self._title,
                description=self._description,
                tags=self._tags,
                category_id=self._category_id,
                privacy=self._privacy,
                progress_callback=self._on_progress,
            )
            if url:
                self.finished.emit(url)
            else:
                self.error.emit("Upload failed — check logs for details.")
        except Exception as exc:
            self.error.emit(str(exc))

    def _on_progress(self, pct: float):
        self.progress.emit(pct)


# ======================================================================
# Publish Dialog
# ======================================================================

class PublishDialog(QDialog):
    """One-click publish dialog for uploading video to YouTube.

    Usage from gui.py::

        from gui.dialogs.publish_dialog import PublishDialog
        dlg = PublishDialog(
            video_path="/path/to/video.mp4",
            audio_files=self.audio_files,
            parent=self,
        )
        dlg.exec()
    """

    def __init__(
        self,
        video_path: str = "",
        audio_files: Optional[list] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Publish to YouTube")
        self.setMinimumSize(720, 780)
        self.setStyleSheet(f"QDialog {{ background: {Colors.BG_PRIMARY}; }}")

        self._video_path = video_path
        self._audio_files = audio_files or []
        self._uploader = None
        self._worker: Optional[_UploadWorker] = None
        self._upload_url: str = ""

        self._setup_ui()
        self._populate_metadata()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        # -- Header ------------------------------------------------
        header = QLabel("Publish to YouTube")
        header.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 20px; font-weight: bold;"
        )
        root.addWidget(header)

        # -- Video preview -----------------------------------------
        preview_frame = QFrame()
        preview_frame.setStyleSheet(
            f"QFrame {{ background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER}; border-radius: 8px; }}"
        )
        preview_layout = QHBoxLayout(preview_frame)
        preview_layout.setContentsMargins(16, 12, 16, 12)

        # Thumbnail placeholder
        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(160, 90)
        self._thumb_label.setStyleSheet(
            f"background: {Colors.BG_TERTIARY}; border-radius: 4px;"
        )
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_label.setText("No Preview")
        self._thumb_label.setStyleSheet(
            f"background: {Colors.BG_TERTIARY}; color: {Colors.TEXT_TERTIARY}; "
            f"border-radius: 4px; font-size: 11px;"
        )
        preview_layout.addWidget(self._thumb_label)

        # File info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        fname = os.path.basename(self._video_path) if self._video_path else "No video selected"
        self._file_label = QLabel(fname)
        self._file_label.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 14px; font-weight: bold;"
        )
        info_layout.addWidget(self._file_label)

        fsize = ""
        if self._video_path and os.path.isfile(self._video_path):
            size_bytes = os.path.getsize(self._video_path)
            if size_bytes > 1_073_741_824:
                fsize = f"{size_bytes / 1_073_741_824:.2f} GB"
            else:
                fsize = f"{size_bytes / 1_048_576:.1f} MB"
        self._size_label = QLabel(fsize)
        self._size_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        info_layout.addWidget(self._size_label)

        info_layout.addStretch()
        preview_layout.addLayout(info_layout, 1)
        root.addWidget(preview_frame)

        # -- Metadata section --------------------------------------
        meta_grid = QGridLayout()
        meta_grid.setSpacing(10)

        # Title
        meta_grid.addWidget(self._make_label("Title:"), 0, 0, Qt.AlignmentFlag.AlignTop)
        self._title_edit = QLineEdit()
        self._title_edit.setMaxLength(100)
        self._title_edit.setPlaceholderText("Video title (max 100 characters)")
        self._title_edit.setStyleSheet(self._input_style())
        meta_grid.addWidget(self._title_edit, 0, 1)

        # Description
        meta_grid.addWidget(self._make_label("Description:"), 1, 0, Qt.AlignmentFlag.AlignTop)
        self._desc_edit = QTextEdit()
        self._desc_edit.setPlaceholderText("Description with timestamps...")
        self._desc_edit.setStyleSheet(self._textedit_style())
        self._desc_edit.setMinimumHeight(160)
        self._desc_edit.setMaximumHeight(220)
        meta_grid.addWidget(self._desc_edit, 1, 1)

        # Tags
        meta_grid.addWidget(self._make_label("Tags:"), 2, 0, Qt.AlignmentFlag.AlignTop)
        self._tags_edit = QLineEdit()
        self._tags_edit.setPlaceholderText("tag1, tag2, tag3 (comma-separated, max 500 chars)")
        self._tags_edit.setStyleSheet(self._input_style())
        meta_grid.addWidget(self._tags_edit, 2, 1)

        root.addLayout(meta_grid)

        # -- Auto-generate button ----------------------------------
        gen_btn = QPushButton("Auto-Generate Metadata")
        gen_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT_DIM};
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
                color: white;
            }}
        """)
        gen_btn.clicked.connect(self._auto_generate_metadata)
        root.addWidget(gen_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # -- Platform selection ------------------------------------
        platform_frame = QFrame()
        platform_frame.setStyleSheet(
            f"QFrame {{ background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER}; border-radius: 8px; }}"
        )
        platform_layout = QHBoxLayout(platform_frame)
        platform_layout.setContentsMargins(16, 10, 16, 10)

        plat_label = self._make_label("Platforms:")
        platform_layout.addWidget(plat_label)

        self._yt_check = QCheckBox("YouTube")
        self._yt_check.setChecked(True)
        self._yt_check.setStyleSheet(self._checkbox_style())
        platform_layout.addWidget(self._yt_check)

        self._fb_check = QCheckBox("Facebook (coming soon)")
        self._fb_check.setEnabled(False)
        self._fb_check.setStyleSheet(self._checkbox_style())
        platform_layout.addWidget(self._fb_check)

        platform_layout.addStretch()

        # Privacy
        priv_label = self._make_label("Privacy:")
        platform_layout.addWidget(priv_label)

        self._privacy_combo = QComboBox()
        self._privacy_combo.addItems(["Private", "Unlisted", "Public"])
        self._privacy_combo.setCurrentIndex(0)
        self._privacy_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 110px;
            }}
        """)
        platform_layout.addWidget(self._privacy_combo)

        root.addWidget(platform_frame)

        # -- Progress bar ------------------------------------------
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                height: 24px;
                text-align: center;
                color: {Colors.TEXT_PRIMARY};
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Colors.METER_GREEN}, stop:1 {Colors.TEAL});
                border-radius: 5px;
            }}
        """)
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        # -- Status label ------------------------------------------
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        # -- Bottom buttons ----------------------------------------
        btn_layout = QHBoxLayout()

        self._upload_btn = QPushButton("Upload to YouTube")
        self._upload_btn.setStyleSheet(f"""
            QPushButton {{
                background: #FF0000;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 14px 32px;
                font-size: 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #CC0000;
            }}
            QPushButton:disabled {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_TERTIARY};
            }}
        """)
        self._upload_btn.clicked.connect(self._on_upload_clicked)
        btn_layout.addWidget(self._upload_btn)

        self._cancel_btn = QPushButton("Cancel Upload")
        self._cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.METER_RED};
                border: 1px solid {Colors.METER_RED};
                border-radius: 8px;
                padding: 14px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.METER_RED};
                color: white;
            }}
        """)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel_upload)
        btn_layout.addWidget(self._cancel_btn)

        btn_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 14px 30px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        root.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _populate_metadata(self):
        """Pre-fill metadata from audio files if available."""
        if not self._audio_files:
            return

        # Build a basic tracklist with timestamps
        lines = []
        offset = 0
        track_names = []
        for af in self._audio_files:
            name = os.path.splitext(os.path.basename(af.path))[0] if hasattr(af, "path") else str(af)
            track_names.append(name)
            mins, secs = divmod(int(offset), 60)
            hrs, mins = divmod(mins, 60)
            if hrs > 0:
                ts = f"{hrs}:{mins:02d}:{secs:02d}"
            else:
                ts = f"{mins}:{secs:02d}"
            lines.append(f"{ts} {name}")
            duration = getattr(af, "duration", None)
            offset += duration if duration and duration > 0 else 180

        total_mins = offset // 60
        n_tracks = len(self._audio_files)

        # Title
        self._title_edit.setText(f"Chill Music Mix | {n_tracks} Tracks | {total_mins} Min")

        # Description
        desc_parts = [
            f"A curated mix of {n_tracks} tracks for your listening pleasure.",
            "",
            "Tracklist:",
        ]
        desc_parts.extend(lines)
        desc_parts.extend([
            "",
            "Produced with LongPlay Studio",
        ])
        self._desc_edit.setPlainText("\n".join(desc_parts))

        # Tags
        base_tags = ["music", "mix", "chill", "playlist", "longplay"]
        self._tags_edit.setText(", ".join(base_tags))

    def _auto_generate_metadata(self):
        """Try to use the YouTubeGeneratorDialog logic to produce richer metadata."""
        try:
            from gui.dialogs.youtube_gen import YouTubeGeneratorDialog
            gen_dlg = YouTubeGeneratorDialog(audio_files=self._audio_files, parent=self)
            if gen_dlg.yt_gen is not None:
                gen_dlg._do_generate_all()
                self._title_edit.setText(gen_dlg.title_edit.text())
                self._desc_edit.setPlainText(gen_dlg.desc_edit.toPlainText())
                self._tags_edit.setText(gen_dlg.tags_edit.toPlainText())
                self._set_status("Metadata auto-generated from YouTube Generator.")
            else:
                self._set_status("YouTube Generator module (ai_dj) not available; using defaults.")
        except Exception as exc:
            logger.warning("Auto-generate metadata failed: %s", exc)
            self._set_status(f"Auto-generate failed: {exc}")

    # ------------------------------------------------------------------
    # Upload logic
    # ------------------------------------------------------------------

    def _on_upload_clicked(self):
        """Validate inputs, authenticate if needed, and start upload."""
        if not self._video_path or not os.path.isfile(self._video_path):
            QMessageBox.warning(self, "No Video", "Please select a valid video file first.")
            return

        if not self._yt_check.isChecked():
            QMessageBox.information(self, "No Platform", "Please select at least one upload platform.")
            return

        title = self._title_edit.text().strip()
        if not title:
            QMessageBox.warning(self, "Missing Title", "Please enter a video title.")
            return

        # Lazy import uploader
        if self._uploader is None:
            try:
                from modules.upload.youtube_upload import YouTubeUploader
                self._uploader = YouTubeUploader()
            except ImportError as exc:
                QMessageBox.critical(
                    self, "Missing Dependencies",
                    f"YouTube upload dependencies not installed:\n{exc}\n\n"
                    "Install with: pip install google-auth google-auth-oauthlib google-api-python-client"
                )
                return

        # Authenticate
        self._set_status("Authenticating with YouTube...")
        QApplication.processEvents()

        if not self._uploader.authenticate():
            QMessageBox.warning(
                self, "Authentication Failed",
                "Could not authenticate with YouTube.\n\n"
                "Make sure client_secrets.json is placed at:\n"
                "~/.longplay_studio/client_secrets.json\n\n"
                "You can download it from the Google Cloud Console."
            )
            self._set_status("Authentication failed.")
            return

        # Prepare data
        description = self._desc_edit.toPlainText().strip()
        tags_text = self._tags_edit.text().strip()
        tags = [t.strip() for t in tags_text.split(",") if t.strip()] if tags_text else []
        privacy = self._privacy_combo.currentText().lower()
        category_id = "10"  # Music

        # Show progress UI
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._upload_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._set_status("Uploading...")

        # Start upload in background thread
        self._worker = _UploadWorker(
            uploader=self._uploader,
            video_path=self._video_path,
            title=title,
            description=description,
            tags=tags,
            category_id=category_id,
            privacy=privacy,
            parent=self,
        )
        self._worker.progress.connect(self._on_upload_progress)
        self._worker.finished.connect(self._on_upload_finished)
        self._worker.error.connect(self._on_upload_error)
        self._worker.start()

    def _on_cancel_upload(self):
        """Cancel the running upload."""
        if self._uploader:
            self._uploader.cancel_upload()
        self._set_status("Cancelling upload...")

    def _on_upload_progress(self, pct: float):
        """Update progress bar from worker signal."""
        self._progress_bar.setValue(int(pct))
        self._set_status(f"Uploading... {pct:.1f}%")

    def _on_upload_finished(self, url: str):
        """Handle successful upload."""
        self._upload_url = url
        self._progress_bar.setValue(100)
        self._upload_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._set_status(f"Upload complete!  {url}")

        QMessageBox.information(
            self, "Upload Successful",
            f"Your video has been uploaded to YouTube!\n\n{url}"
        )

    def _on_upload_error(self, msg: str):
        """Handle upload failure."""
        self._progress_bar.setVisible(False)
        self._upload_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._set_status(f"Upload failed: {msg}")

        QMessageBox.warning(self, "Upload Failed", f"Upload failed:\n\n{msg}")

    # ------------------------------------------------------------------
    # Style / widget helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        """Update the status label."""
        self._status_label.setText(text)

    def _make_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 13px; font-weight: bold;"
        )
        lbl.setFixedWidth(100)
        return lbl

    @staticmethod
    def _input_style() -> str:
        return f"""
            QLineEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {Colors.ACCENT};
            }}
        """

    @staticmethod
    def _textedit_style() -> str:
        return f"""
            QTextEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 10px;
                font-family: 'Menlo', 'Courier New', monospace;
                font-size: 12px;
            }}
            QTextEdit:focus {{
                border-color: {Colors.ACCENT};
            }}
        """

    @staticmethod
    def _checkbox_style() -> str:
        return f"""
            QCheckBox {{
                color: {Colors.TEXT_PRIMARY};
                font-size: 13px;
                spacing: 8px;
            }}
            QCheckBox:disabled {{
                color: {Colors.TEXT_TERTIARY};
            }}
        """
