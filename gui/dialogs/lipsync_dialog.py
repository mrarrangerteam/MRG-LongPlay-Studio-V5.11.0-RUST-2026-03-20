"""
Lip-sync dialog — Dreemina-powered avatar video generation.

Workflow:
    1. Select hooks (extracted from Hook Extractor)
    2. Master each hook (via MasterChain)
    3. Select avatar image
    4. Generate lip-sync video via Dreemina API
    5. Optionally batch-upload to YouTube as Shorts

Classes:
    LipSyncDialog — Main dialog for lip-sync workflow.
"""

import os
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List

from gui.utils.compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QComboBox, QLineEdit,
    QCheckBox, QProgressBar, QMessageBox, QApplication,
    QListWidget, QListWidgetItem, QFileDialog,
    Qt, pyqtSignal, QThread, QPixmap,
)
from gui.styles import Colors

logger = logging.getLogger(__name__)


# ======================================================================
# Worker: Master + Lip-sync + Upload pipeline
# ======================================================================

class _LipSyncPipelineWorker(QThread):
    """Background thread for: master hooks -> lip-sync -> upload Shorts."""

    progress = pyqtSignal(int, int, float, str)  # (current_idx, total, pct, message)
    item_done = pyqtSignal(int, str)              # (index, output_path or error)
    all_done = pyqtSignal(list)                    # list of (path, youtube_url, error) tuples
    error = pyqtSignal(str)

    def __init__(
        self,
        hook_paths: List[str],
        avatar_path: str,
        api_key: str,
        api_base: str,
        aspect_ratio: str,
        do_master: bool,
        do_upload: bool,
        upload_privacy: str,
        parent=None,
    ):
        super().__init__(parent)
        self._hook_paths = hook_paths
        self._avatar_path = avatar_path
        self._api_key = api_key
        self._api_base = api_base
        self._aspect_ratio = aspect_ratio
        self._do_master = do_master
        self._do_upload = do_upload
        self._upload_privacy = upload_privacy
        self._cancelled = False
        self._temp_dirs: List[str] = []  # Track temp dirs for cleanup

    def cancel(self):
        self._cancelled = True

    def run(self):
        results = []
        total = len(self._hook_paths)

        try:
            # Lazy imports
            from modules.lipsync.dreemina_client import DreeminaClient

            client = DreeminaClient(api_key=self._api_key, api_base=self._api_base)

            # Optional: mastering chain
            chain = None
            if self._do_master:
                try:
                    from modules.master.chain import MasterChain
                    chain = MasterChain()
                except ImportError:
                    logger.warning("MasterChain not available, skipping mastering")

            # Optional: YouTube uploader
            uploader = None
            if self._do_upload:
                try:
                    from modules.upload.youtube_upload import YouTubeUploader
                    uploader = YouTubeUploader()
                    self.progress.emit(0, total, 0.0, "Waiting for YouTube authentication (check browser)...")
                    if not uploader.authenticate():
                        self.error.emit("YouTube authentication failed")
                        uploader = None
                except ImportError:
                    logger.warning("YouTube uploader not available")

            # Create unique temp directories (M1: avoid predictable paths)
            master_tmp = tempfile.mkdtemp(prefix="lipsync_master_")
            output_tmp = tempfile.mkdtemp(prefix="lipsync_output_")
            self._temp_dirs.extend([master_tmp, output_tmp])

            for idx, hook_path in enumerate(self._hook_paths):
                if self._cancelled:
                    break

                hook_name = Path(hook_path).stem
                self.progress.emit(idx, total, 0.0, f"[{idx+1}/{total}] Processing: {hook_name}")

                try:
                    audio_for_lipsync = hook_path

                    # Step 1: Master the hook
                    if chain and self._do_master:
                        self.progress.emit(idx, total, 10.0, f"[{idx+1}/{total}] Mastering: {hook_name}")
                        # Reset chain state between hooks (HIGH-3)
                        if hasattr(chain, 'reset_all'):
                            chain.reset_all()
                        mastered_path = os.path.join(master_tmp, f"{hook_name}_mastered.wav")
                        if chain.load_audio(hook_path):
                            result_path = chain.render(output_path=mastered_path)
                            if result_path and os.path.isfile(result_path):
                                audio_for_lipsync = result_path
                                self.progress.emit(idx, total, 25.0, f"[{idx+1}/{total}] Mastered: {hook_name}")
                            else:
                                logger.warning("Mastering failed for %s, using original", hook_name)

                    if self._cancelled:
                        break

                    # Step 2: Generate lip-sync video
                    self.progress.emit(idx, total, 30.0, f"[{idx+1}/{total}] Generating lip-sync: {hook_name}")
                    output_path = os.path.join(output_tmp, f"{hook_name}_lipsync.mp4")

                    # Capture idx by value (C2: closure variable fix)
                    def _on_lipsync_progress(pct, msg, _idx=idx, _total=total):
                        mapped = 30.0 + (pct / 100.0) * 50.0
                        self.progress.emit(_idx, _total, mapped, f"[{_idx+1}/{_total}] {msg}")

                    video_path = client.generate_lipsync(
                        audio_path=audio_for_lipsync,
                        avatar_path=self._avatar_path,
                        output_path=output_path,
                        aspect_ratio=self._aspect_ratio,
                        progress_callback=_on_lipsync_progress,
                    )

                    if not video_path or self._cancelled:
                        results.append((hook_path, None, "Lip-sync failed or cancelled"))
                        continue

                    self.progress.emit(idx, total, 85.0, f"[{idx+1}/{total}] Lip-sync complete: {hook_name}")

                    # Step 3: Upload to YouTube as Short
                    youtube_url = None
                    if uploader and self._do_upload:
                        self.progress.emit(idx, total, 88.0, f"[{idx+1}/{total}] Uploading to YouTube: {hook_name}")
                        title = f"{hook_name} #Shorts"
                        description = (
                            f"{hook_name}\n\n"
                            "Generated with LongPlay Studio - Dreemina Lip-sync\n"
                            "#Shorts #Music #LipSync"
                        )
                        tags = ["Shorts", "Music", "LipSync", "AI", hook_name]

                        youtube_url = uploader.upload_video(
                            video_path=video_path,
                            title=title[:100],
                            description=description,
                            tags=tags,
                            category_id="10",
                            privacy=self._upload_privacy,
                        )

                        if youtube_url:
                            self.progress.emit(idx, total, 100.0,
                                               f"[{idx+1}/{total}] Uploaded: {youtube_url}")
                        else:
                            self.progress.emit(idx, total, 95.0,
                                               f"[{idx+1}/{total}] Upload failed for {hook_name}")

                    results.append((video_path, youtube_url, None))
                    self.item_done.emit(idx, video_path)

                except Exception as exc:
                    logger.error("Pipeline error for %s: %s", hook_path, exc)
                    results.append((hook_path, None, str(exc)))
                    self.item_done.emit(idx, f"ERROR: {exc}")

            self.all_done.emit(results)

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self._cleanup_temp()

    def _cleanup_temp(self):
        """Clean up temporary mastered files (H-CQ4)."""
        for d in self._temp_dirs:
            try:
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass


# ======================================================================
# LipSync Dialog
# ======================================================================

class LipSyncDialog(QDialog):
    """Dialog for Dreemina lip-sync avatar video generation.

    Workflow:
        Hook files -> Master -> Dreemina lip-sync -> YouTube Shorts upload

    Usage::
        dialog = LipSyncDialog(
            hook_paths=["/path/to/hook1.wav", "/path/to/hook2.wav"],
            parent=self,
        )
        dialog.exec()
    """

    def __init__(
        self,
        hook_paths: Optional[List[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Lip-Sync Avatar — Dreemina")
        self.setMinimumSize(800, 700)
        self.setStyleSheet(f"QDialog {{ background: {Colors.BG_PRIMARY}; }}")

        self._hook_paths = list(hook_paths) if hook_paths else []
        self._avatar_path = ""
        self._worker: Optional[_LipSyncPipelineWorker] = None
        self._results: list = []

        self._setup_ui()

    # ------------------------------------------------------------------
    # QThread lifecycle (C1: safe cleanup on close)
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Ensure the worker thread is stopped before dialog is destroyed."""
        self._stop_worker()
        super().closeEvent(event)

    def reject(self):
        """Handle Escape key / X button — stop worker first."""
        self._stop_worker()
        super().reject()

    def _stop_worker(self):
        """Cancel and wait for the worker thread to finish."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            # Disconnect signals to prevent callbacks to destroyed widgets
            try:
                self._worker.progress.disconnect()
                self._worker.item_done.disconnect()
                self._worker.all_done.disconnect()
                self._worker.error.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._worker.wait(5000)  # Wait up to 5 seconds
            if self._worker.isRunning():
                logger.warning("Lip-sync worker still running after 5s, terminating")
                self._worker.terminate()
                self._worker.wait(2000)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(14)

        # Header
        header = QLabel("Lip-Sync Avatar — Dreemina API")
        header.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-size: 20px; font-weight: bold;"
        )
        root.addWidget(header)

        desc = QLabel(
            "Hook Extract → Master → Dreemina Lip-sync → YouTube Shorts Upload"
        )
        desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        root.addWidget(desc)

        # ── Section 1: API Key ──
        api_frame = self._make_section("Dreemina API Settings")
        api_layout = QGridLayout()
        api_layout.setSpacing(8)

        api_layout.addWidget(self._make_label("API Key:"), 0, 0)
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setPlaceholderText("Enter your Dreemina API key")
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setStyleSheet(self._input_style())
        api_layout.addWidget(self._api_key_edit, 0, 1)

        save_key_btn = QPushButton("Save Key")
        save_key_btn.setStyleSheet(self._small_btn_style(Colors.ACCENT))
        save_key_btn.clicked.connect(self._save_api_key)
        api_layout.addWidget(save_key_btn, 0, 2)

        api_layout.addWidget(self._make_label("API Base:"), 1, 0)
        self._api_base_edit = QLineEdit()
        self._api_base_edit.setText("https://api.dreemina.com/v1")
        self._api_base_edit.setStyleSheet(self._input_style())
        api_layout.addWidget(self._api_base_edit, 1, 1, 1, 2)

        api_frame.layout().addLayout(api_layout)
        root.addWidget(api_frame)

        # Load saved config
        self._load_saved_config()

        # ── Section 2: Hook Files ──
        hook_frame = self._make_section("Hook Audio Files (to lip-sync)")
        hook_inner = QVBoxLayout()

        self._hook_list = QListWidget()
        self._hook_list.setStyleSheet(f"""
            QListWidget {{
                background: {Colors.BG_SECONDARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 6px;
                color: {Colors.TEXT_PRIMARY};
            }}
            QListWidget::item {{
                padding: 6px;
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        self._hook_list.setMinimumHeight(100)
        for hp in self._hook_paths:
            self._hook_list.addItem(os.path.basename(hp))
        hook_inner.addWidget(self._hook_list)

        hook_btn_row = QHBoxLayout()
        add_hooks_btn = QPushButton("+ Add Hook Files")
        add_hooks_btn.setStyleSheet(self._small_btn_style(Colors.METER_GREEN))
        add_hooks_btn.clicked.connect(self._add_hook_files)
        hook_btn_row.addWidget(add_hooks_btn)

        clear_hooks_btn = QPushButton("Clear All")
        clear_hooks_btn.setStyleSheet(self._small_btn_style(Colors.METER_RED))
        clear_hooks_btn.clicked.connect(self._clear_hooks)
        hook_btn_row.addWidget(clear_hooks_btn)
        hook_btn_row.addStretch()
        hook_inner.addLayout(hook_btn_row)

        hook_frame.layout().addLayout(hook_inner)
        root.addWidget(hook_frame)

        # ── Section 3: Avatar Image ──
        avatar_frame = self._make_section("Avatar Image / Video")
        avatar_layout = QHBoxLayout()

        self._avatar_preview = QLabel("No avatar selected")
        self._avatar_preview.setFixedSize(120, 120)
        self._avatar_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._avatar_preview.setStyleSheet(
            f"background: {Colors.BG_TERTIARY}; color: {Colors.TEXT_TERTIARY}; "
            f"border: 2px dashed {Colors.BORDER}; border-radius: 8px; font-size: 11px;"
        )
        avatar_layout.addWidget(self._avatar_preview)

        avatar_btn_col = QVBoxLayout()
        select_avatar_btn = QPushButton("Select Avatar Image")
        select_avatar_btn.setStyleSheet(self._small_btn_style(Colors.ACCENT))
        select_avatar_btn.clicked.connect(self._select_avatar)
        avatar_btn_col.addWidget(select_avatar_btn)

        self._avatar_path_label = QLabel("No file selected")
        self._avatar_path_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        self._avatar_path_label.setWordWrap(True)
        avatar_btn_col.addWidget(self._avatar_path_label)
        avatar_btn_col.addStretch()

        avatar_layout.addLayout(avatar_btn_col, 1)
        avatar_frame.layout().addLayout(avatar_layout)
        root.addWidget(avatar_frame)

        # ── Section 4: Options ──
        options_frame = self._make_section("Options")
        opt_layout = QGridLayout()
        opt_layout.setSpacing(10)

        # Aspect ratio
        opt_layout.addWidget(self._make_label("Aspect Ratio:"), 0, 0)
        self._aspect_combo = QComboBox()
        self._aspect_combo.addItems(["9:16 (Shorts/Reels)", "16:9 (Landscape)", "1:1 (Square)"])
        self._aspect_combo.setStyleSheet(self._combo_style())
        opt_layout.addWidget(self._aspect_combo, 0, 1)

        # Master before lip-sync
        self._master_check = QCheckBox("Master hooks before lip-sync (recommended)")
        self._master_check.setChecked(True)
        self._master_check.setStyleSheet(self._checkbox_style())
        opt_layout.addWidget(self._master_check, 1, 0, 1, 2)

        # Upload to YouTube
        self._upload_check = QCheckBox("Upload to YouTube as Shorts (API quota limits apply)")
        self._upload_check.setChecked(True)
        self._upload_check.setStyleSheet(self._checkbox_style())
        opt_layout.addWidget(self._upload_check, 2, 0, 1, 2)

        # Privacy
        opt_layout.addWidget(self._make_label("YouTube Privacy:"), 3, 0)
        self._privacy_combo = QComboBox()
        self._privacy_combo.addItems(["Public", "Unlisted", "Private"])
        self._privacy_combo.setCurrentIndex(0)
        self._privacy_combo.setStyleSheet(self._combo_style())
        opt_layout.addWidget(self._privacy_combo, 3, 1)

        options_frame.layout().addLayout(opt_layout)
        root.addWidget(options_frame)

        # ── Progress ──
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
                    stop:0 {Colors.GIF_COLOR}, stop:1 {Colors.ACCENT});
                border-radius: 5px;
            }}
        """)
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        # ── Bottom Buttons ──
        btn_layout = QHBoxLayout()

        self._start_btn = QPushButton("Start Lip-Sync Pipeline")
        self._start_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Colors.GIF_COLOR}, stop:1 {Colors.ACCENT});
                color: white;
                border: none;
                border-radius: 8px;
                padding: 14px 32px;
                font-size: 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT};
            }}
            QPushButton:disabled {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_TERTIARY};
            }}
        """)
        self._start_btn.clicked.connect(self._start_pipeline)
        btn_layout.addWidget(self._start_btn)

        self._cancel_btn = QPushButton("Cancel")
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
            QPushButton:hover {{ background: {Colors.METER_RED}; color: white; }}
        """)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_pipeline)
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
            QPushButton:hover {{ background: {Colors.BORDER}; }}
        """)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        root.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _load_saved_config(self):
        """Load saved Dreemina API config."""
        try:
            from modules.lipsync.dreemina_client import DreeminaClient
            client = DreeminaClient()
            if client.api_key:
                self._api_key_edit.setText(client.api_key)
            if client.api_base:
                self._api_base_edit.setText(client.api_base)
        except Exception:
            pass

    def _save_api_key(self):
        """Save API key to config."""
        try:
            from modules.lipsync.dreemina_client import DreeminaClient
            client = DreeminaClient()
            client.api_key = self._api_key_edit.text().strip()
            client.api_base = self._api_base_edit.text().strip()
            client.save_config()
            self._set_status("API key saved.")
        except Exception as exc:
            self._set_status(f"Failed to save: {exc}")

    def _add_hook_files(self):
        """Add hook audio files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Hook Audio Files",
            "", "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*)"
        )
        if files:
            for f in files:
                if f not in self._hook_paths:
                    self._hook_paths.append(f)
                    self._hook_list.addItem(os.path.basename(f))

    def _clear_hooks(self):
        """Clear all hook files."""
        self._hook_paths.clear()
        self._hook_list.clear()

    def _select_avatar(self):
        """Select avatar image or video."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Avatar Image",
            "", "Images (*.jpg *.jpeg *.png *.webp);;Videos (*.mp4 *.mov);;All Files (*)"
        )
        if path:
            self._avatar_path = path
            self._avatar_path_label.setText(os.path.basename(path))

            # Try to show preview
            try:
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        120, 120,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    self._avatar_preview.setPixmap(scaled)
                    self._avatar_preview.setText("")
            except Exception:
                self._avatar_preview.setText(os.path.basename(path))

    def _start_pipeline(self):
        """Validate inputs and start the lip-sync pipeline."""
        # Validation
        api_key = self._api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "No API Key", "Please enter your Dreemina API key.")
            return

        if not self._hook_paths:
            QMessageBox.warning(self, "No Hooks", "Please add hook audio files first.")
            return

        if not self._avatar_path or not os.path.isfile(self._avatar_path):
            QMessageBox.warning(self, "No Avatar", "Please select an avatar image.")
            return

        # Parse aspect ratio
        aspect_text = self._aspect_combo.currentText()
        if "9:16" in aspect_text:
            aspect = "9:16"
        elif "16:9" in aspect_text:
            aspect = "16:9"
        else:
            aspect = "1:1"

        # HIGH-5: Warn if uploading non-vertical video as Shorts
        if self._upload_check.isChecked() and aspect != "9:16":
            reply = QMessageBox.warning(
                self, "Non-Vertical Shorts",
                f"YouTube Shorts require 9:16 (vertical) aspect ratio.\n"
                f"You selected '{aspect_text}' — videos may not be classified as Shorts.\n\n"
                f"Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # HIGH-4: Validate hook durations for Shorts (<=60s)
        if self._upload_check.isChecked():
            try:
                import soundfile as sf
                long_files = []
                for hp in self._hook_paths:
                    try:
                        info = sf.info(hp)
                        if info.duration > 60.0:
                            long_files.append(f"  {os.path.basename(hp)}: {info.duration:.1f}s")
                    except Exception:
                        pass
                if long_files:
                    reply = QMessageBox.warning(
                        self, "Hooks Too Long for Shorts",
                        f"YouTube Shorts must be 60 seconds or less.\n"
                        f"These hooks exceed the limit:\n\n"
                        + "\n".join(long_files) + "\n\n"
                        f"Continue anyway? (Videos may not classify as Shorts)",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
            except ImportError:
                pass  # soundfile not available, skip validation

        # UI state
        self._start_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)

        # Start worker
        self._worker = _LipSyncPipelineWorker(
            hook_paths=self._hook_paths,
            avatar_path=self._avatar_path,
            api_key=api_key,
            api_base=self._api_base_edit.text().strip(),
            aspect_ratio=aspect,
            do_master=self._master_check.isChecked(),
            do_upload=self._upload_check.isChecked(),
            upload_privacy=self._privacy_combo.currentText().lower(),
            parent=self,
        )
        self._worker.progress.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        self._worker.item_done.connect(self._on_item_done, Qt.ConnectionType.QueuedConnection)
        self._worker.all_done.connect(self._on_all_done, Qt.ConnectionType.QueuedConnection)
        self._worker.error.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        self._worker.start()

    def _cancel_pipeline(self):
        """Cancel running pipeline."""
        if self._worker:
            self._worker.cancel()
        self._set_status("Cancelling...")

    def _on_progress(self, idx: int, total: int, pct: float, msg: str):
        """Update progress from worker."""
        overall = ((idx + pct / 100.0) / max(total, 1)) * 100.0
        self._progress_bar.setValue(int(overall))
        self._set_status(msg)

    def _on_item_done(self, idx: int, path: str):
        """One item completed."""
        item = self._hook_list.item(idx)
        if item:
            current = item.text()
            if path.startswith("ERROR:"):
                item.setText(f"[FAIL] {current}")
            else:
                item.setText(f"[DONE] {current}")

    def _on_all_done(self, results: list):
        """All items completed."""
        self._results = results
        self._progress_bar.setValue(100)
        self._start_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)

        # Count successes
        success = sum(1 for _, url, err in results if err is None)
        uploaded = sum(1 for _, url, err in results if url)
        total = len(results)

        summary = f"Pipeline complete: {success}/{total} lip-sync videos generated"
        if uploaded:
            summary += f", {uploaded} uploaded to YouTube"
        self._set_status(summary)

        # Show results dialog
        msg = "Lip-sync complete!\n\n"
        msg += f"Generated: {success}/{total}\n"
        if uploaded:
            msg += f"Uploaded to YouTube: {uploaded}\n\n"
            for video_path, url, err in results:
                if url:
                    name = os.path.basename(video_path) if video_path else "unknown"
                    msg += f"  {name}: {url}\n"
        QMessageBox.information(self, "Pipeline Complete", msg)

    def _on_error(self, msg: str):
        """Handle pipeline error."""
        self._progress_bar.setVisible(False)
        self._start_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._set_status(f"Error: {msg}")
        QMessageBox.critical(self, "Pipeline Error", f"Error:\n\n{msg}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        self._status_label.setText(text)

    def _make_section(self, title: str) -> QFrame:
        """Create a styled section frame with title."""
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {Colors.BG_SECONDARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: 8px; }}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        lbl = QLabel(title)
        lbl.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 13px; font-weight: bold; "
            f"border: none; background: transparent;"
        )
        layout.addWidget(lbl)
        return frame

    def _make_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: 12px; font-weight: bold; "
            f"border: none; background: transparent;"
        )
        lbl.setFixedWidth(110)
        return lbl

    @staticmethod
    def _input_style() -> str:
        return f"""
            QLineEdit {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }}
            QLineEdit:focus {{ border-color: {Colors.ACCENT}; }}
        """

    @staticmethod
    def _combo_style() -> str:
        return f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 160px;
            }}
        """

    @staticmethod
    def _checkbox_style() -> str:
        return f"""
            QCheckBox {{
                color: {Colors.TEXT_PRIMARY};
                font-size: 12px;
                spacing: 8px;
                border: none;
                background: transparent;
            }}
        """

    @staticmethod
    def _small_btn_style(color: str) -> str:
        return f"""
            QPushButton {{
                background: {color};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: {Colors.ACCENT_DIM}; }}
        """
