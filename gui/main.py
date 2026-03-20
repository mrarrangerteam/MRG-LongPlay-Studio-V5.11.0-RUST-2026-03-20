"""
Main application window and license dialog.

Classes:
    LongPlayStudioV4 — Main QMainWindow for LongPlay Studio
    LicenseDialog     — License activation dialog

Functions:
    check_and_show_license — Check license and show activation dialog if needed
    main                   — Application entry point
"""

import sys
import os
import re
import subprocess
import json
import shutil
import tempfile
import time
import random
import platform
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np


from gui.utils import ffmpeg_escape_path as _ffmpeg_escape_path

from gui.utils.compat import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QSlider, QComboBox,
    QTabWidget, QListWidget, QListWidgetItem, QProgressBar,
    QSplitter, QSizePolicy, QSpacerItem, QStackedWidget,
    QApplication, QStyle, QStyleOption, QFileDialog, QDialog,
    QLineEdit, QTextEdit, QCheckBox, QGroupBox, QToolButton,
    QMessageBox, QMenu, QSpinBox, QDoubleSpinBox, QDial,
    QGraphicsOpacityEffect,
    Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QUrl,
    QMimeData, QPoint, QThread, QRect, QRectF,
    pyqtSignal, pyqtSlot,
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QPalette,
    QIcon, QPixmap, QDragEnterEvent, QDropEvent, QPolygon, QAction,
    QShortcut, QKeySequence, QImage,
    QMediaPlayer, QAudioOutput, QVideoWidget,
    PYQT6,
)
from gui.styles import Colors
from gui.audio_player import (
    AudioPlayerWidget, MediaFile, TrackState, AudioAnalysisEngine,
)
from gui.widgets.meter import RealTimeMeter, LUFSDisplay
from gui.widgets.drop_zone import DropZoneListWidget
from gui.widgets.collapsible import CollapsibleSection
from gui.video.preview import VideoPreviewCard
from gui.timeline.capcut_timeline import CapCutTimeline
from gui.timeline.track_list import TrackListItem
from gui.dialogs.ai_dj import AIDJDialog
from gui.dialogs.ai_video import AIVideoDialog
from gui.dialogs.youtube_gen import YouTubeGeneratorDialog
from gui.dialogs.hook_extractor import HookExtractorDialog
from gui.dialogs.video_prompt import VideoPromptDialog
from gui.dialogs.timestamp import TimestampDialog
from gui.dialogs.lipsync_dialog import LipSyncDialog

# Import Ozone 12 / Waves WLM meter widgets from Master Module
try:
    from modules.master.ui_panel import WavesWLMMeter, GainReductionHistoryWidget
    _HAS_MASTER_WIDGETS = True
except ImportError:
    _HAS_MASTER_WIDGETS = False

# Import IRC modes & mastering presets
try:
    from modules.master.genre_profiles import (
        IRC_MODES, MASTERING_PRESET_NAMES, MASTERING_PRESET_CATEGORIES,
        get_mastering_preset, get_irc_sub_modes
    )
    _HAS_PRESETS = True
except ImportError:
    _HAS_PRESETS = False
    IRC_MODES = {}
    MASTERING_PRESET_NAMES = []
    MASTERING_PRESET_CATEGORIES = {}

# Import functions from the original gui module-level scope.
# These are used by LongPlayStudioV4 methods.
# We import them lazily from the top-level gui.py (which will still exist as
# a thin shim) via a small helper -- OR we can just define a stub that re-imports.
# For now we import the key helpers that live outside any class in the original.
try:
    from gui._original_helpers import (
        detect_hw_encoder, HW_ENCODER, get_smart_temp_dir,
        get_hwaccel_input_params, HW_INPUT_PARAMS,
        QUALITY_PRESETS, CURRENT_QUALITY_MODE, set_quality_mode, get_quality_mode,
        get_encoder_params, run_ffmpeg_with_progress,
        setup_ffmpeg_path, _natural_sort_key,
    )
except ImportError:
    # Fallback: these will be available from the shim gui.py when it re-exports
    pass

# ═══════════════════════════════════════════════════════════════════
#  PHASE 2 INTEGRATION IMPORTS — Wire all Phase 1 modules into GUI
# ═══════════════════════════════════════════════════════════════════

# I-1: Undo/Redo
try:
    from gui.models.commands import CommandHistory
    _HAS_UNDO = True
except ImportError:
    _HAS_UNDO = False

# I-2: AutoSave
try:
    from gui.models.autosave import AutoSaveManager, project_to_dict, dict_to_project
    _HAS_AUTOSAVE = True
except ImportError:
    _HAS_AUTOSAVE = False

# I-3: Vintage Theme
try:
    from gui.styles_vintage import get_theme, get_theme_names, apply_theme, THEMES
    _HAS_VINTAGE_THEME = True
except ImportError:
    _HAS_VINTAGE_THEME = False

# I-9: Multi-Track Timeline
try:
    from gui.timeline.multi_track_timeline import MultiTrackTimeline
    _HAS_MULTI_TRACK = True
except ImportError:
    _HAS_MULTI_TRACK = False

# I-10: Clip Drag/Trim/Split
try:
    from gui.timeline.clip_drag import ClipDragHandler
    from gui.timeline.clip_trim import ClipTrimSplitHandler
    _HAS_CLIP_TOOLS = True
except ImportError:
    _HAS_CLIP_TOOLS = False

# I-11: Keyframe Editor
try:
    from gui.timeline.keyframe_editor import KeyframeEditor
    _HAS_KEYFRAMES = True
except ImportError:
    _HAS_KEYFRAMES = False

# I-12: Text Overlay
try:
    from gui.timeline.text_layer import TextClip, TextAnimation
    _HAS_TEXT_OVERLAY = True
except ImportError:
    _HAS_TEXT_OVERLAY = False

# I-13: Transitions
try:
    from gui.models.transitions import TransitionLibraryPanel
    _HAS_TRANSITIONS = True
except ImportError:
    _HAS_TRANSITIONS = False

# I-14: Effects
try:
    from gui.models.effects import EffectsPanel
    _HAS_EFFECTS = True
except ImportError:
    _HAS_EFFECTS = False

# I-15: Speed Ramp
try:
    from gui.timeline.speed_ramp import SpeedCurveEditor
    _HAS_SPEED_RAMP = True
except ImportError:
    _HAS_SPEED_RAMP = False

# I-16: Export Presets
try:
    from gui.models.export_presets import ExportPresetPanel
    _HAS_EXPORT_PRESETS = True
except ImportError:
    _HAS_EXPORT_PRESETS = False



class LongPlayStudioV4(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LongPlay Studio V5.0 - AI DJ + YouTube Generator + Hook Extractor + Video Prompt + AI Master")
        self.setMinimumSize(1400, 900)
        
        # Logo - try multiple paths (V4.31.1: support PNG)
        logo_paths = [
            os.path.join(os.path.dirname(__file__), "mrarranger_logo.png"),
            os.path.join(os.path.dirname(__file__), "mrarranger_logo.jpg"),
            os.path.join(os.path.dirname(__file__), "logo.png"),
            os.path.join(os.path.dirname(__file__), "logo.jpg"),
            os.path.join(os.path.dirname(__file__), "..", "assets", "logo.png"),
            os.path.join(os.path.dirname(__file__), "..", "assets", "logo.jpg"),
        ]
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                self.setWindowIcon(QIcon(logo_path))
                break
        
        # Data
        self.audio_files: List[MediaFile] = []
        self.video_files: List[MediaFile] = []
        self.gif_files: List[MediaFile] = []
        self.logo_files: List[MediaFile] = []
        
        # Video transition settings
        self.video_transition_enabled = True
        self.video_transition_duration = 1.0  # seconds
        self.video_transition_style = "fade"  # fade, dissolve, wipe, etc.
        self.tracks_per_video = 4  # default
        
        self.current_audio_index = 0

        # ── Phase 2: I-1 Undo/Redo ──
        self.undo_manager = None
        if _HAS_UNDO:
            self.undo_manager = CommandHistory(on_change=self._on_undo_change)
            print("[PHASE2] Undo/Redo system initialized")

        # ── Phase 2: I-2 AutoSave ──
        self.auto_save = None
        if _HAS_AUTOSAVE:
            self.auto_save = AutoSaveManager()
            self.auto_save.set_save_callback(lambda p: print(f"[AUTOSAVE] Saved: {p}"))
            print("[PHASE2] AutoSave system initialized")

        self._setup_ui()
        self._setup_shortcuts()
        self._connect_signals()

        # ── Phase 2: I-3 Apply Vintage Theme ──
        if _HAS_VINTAGE_THEME:
            try:
                theme_cls = get_theme("classic_dark")
                app = QApplication.instance()
                if app:
                    app.setStyleSheet(theme_cls.get_global_stylesheet())
                    print(f"[PHASE2] Applied theme: {theme_cls.name}")
            except Exception as e:
                print(f"[PHASE2] Theme error: {e}")

        # ── Phase 2: I-2 Start AutoSave + Check Recovery ──
        if self.auto_save:
            if self.auto_save.has_recovery():
                info = self.auto_save.get_recovery_info()
                if info:
                    reply = QMessageBox.question(
                        self, "Recover Project",
                        f"Found auto-save from {info.get('saved_at', 'unknown')}.\n"
                        f"Tracks: {info.get('tracks', 0)}\n\nRestore?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        recovered = self.auto_save.recover()
                        if recovered:
                            print(f"[AUTOSAVE] Recovered project with {len(recovered.tracks)} tracks")
            self.auto_save.start()
        
    def _on_undo_change(self) -> None:
        """Callback when undo/redo stack changes — update status bar."""
        if self.undo_manager:
            stack = self.undo_manager._undo_stack
            label = stack[-1].description if stack else "—"
            self.statusBar().showMessage(f"Last action: {label}", 3000)

    def closeEvent(self, event) -> None:
        """Clean up on app close."""
        # I-2: Stop auto-save
        if self.auto_save:
            self.auto_save.stop()
        super().closeEvent(event)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Spacebar = Play/Pause
        space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space_shortcut.activated.connect(self._toggle_playback)

        # I-1: Undo (Ctrl+Z / Cmd+Z)
        if _HAS_UNDO and self.undo_manager:
            undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self)
            undo_sc.activated.connect(lambda: self.undo_manager.undo())
            redo_sc = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
            redo_sc.activated.connect(lambda: self.undo_manager.redo())
            print("[PHASE2] Undo/Redo shortcuts registered (Ctrl+Z / Ctrl+Shift+Z)")
        
        # Command+ = Zoom In
        zoom_in_shortcut = QShortcut(QKeySequence("Ctrl+="), self)
        zoom_in_shortcut.activated.connect(self._zoom_in)
        zoom_in_shortcut2 = QShortcut(QKeySequence("Ctrl++"), self)
        zoom_in_shortcut2.activated.connect(self._zoom_in)
        
        # Command- = Zoom Out
        zoom_out_shortcut = QShortcut(QKeySequence("Ctrl+-"), self)
        zoom_out_shortcut.activated.connect(self._zoom_out)
        
        # Command+0 = Reset Zoom
        zoom_reset_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        zoom_reset_shortcut.activated.connect(self._zoom_reset)
        
    def _zoom_in(self):
        """Zoom in timeline"""
        self.timeline.zoomIn()
        
    def _zoom_out(self):
        """Zoom out timeline"""
        self.timeline.zoomOut()
        
    def _zoom_reset(self):
        """Fit entire timeline to screen (like CapCut)"""
        self.timeline.zoomFit()
        
    def _toggle_playback(self):
        """Toggle play/pause with spacebar"""
        try:
            if self.audio_player.is_playing:
                self.audio_player.pause()
                if hasattr(self, 'video_preview') and self.video_preview:
                    self.video_preview.pause()
                self.timeline.setPlaying(False)
            else:
                if not self.audio_files:
                    print("[PLAY] ⚠️ No audio files loaded - add audio first")
                    return
                self.audio_player.play()
                if hasattr(self, 'video_preview') and self.video_preview:
                    self.video_preview.play()
                self.timeline.setPlaying(True)
        except Exception as e:
            print(f"[PLAY] Toggle playback error: {e}")
            import traceback
            traceback.print_exc()
        
    def _setup_ui(self):
        """Setup main UI with resizable splitters like CapCut"""
        # ══ Global Waves-Style Hardware Theme ══
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {Colors.BG_PRIMARY};
            }}
            QWidget {{
                color: {Colors.TEXT_PRIMARY};
                font-family: 'Menlo', 'Menlo', 'Courier New', 'Courier New', monospace;
            }}
            QLabel {{
                background: transparent;
            }}
            QToolTip {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.ACCENT};
                border: 1px solid {Colors.BORDER_LIGHT};
                padding: 4px 8px;
                font-size: 11px;
            }}
            QScrollBar:vertical {{
                background: {Colors.BG_PRIMARY};
                width: 8px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Colors.TEAL};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background: {Colors.BG_PRIMARY};
                height: 8px;
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: {Colors.BORDER};
                border-radius: 4px;
                min-width: 30px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {Colors.TEAL};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Main horizontal splitter (left | center | right)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(4)
        self.main_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {Colors.BORDER};
            }}
            QSplitter::handle:hover {{
                background: {Colors.ACCENT};
            }}
        """)
        
        # Left sidebar (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: {Colors.BG_SECONDARY};
            }}
            QScrollBar:vertical {{
                background: {Colors.BG_TERTIARY};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Colors.ACCENT};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        self._create_left_sidebar(left_scroll)
        self.main_splitter.addWidget(left_scroll)
        
        # Center content (scrollable)
        center_scroll = QScrollArea()
        center_scroll.setWidgetResizable(True)
        center_scroll.setStyleSheet(left_scroll.styleSheet().replace(Colors.BG_SECONDARY, Colors.BG_PRIMARY))
        self._create_center_content(center_scroll)
        self.main_splitter.addWidget(center_scroll)
        
        # Right panel (scrollable)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet(left_scroll.styleSheet())
        self._create_right_panel(right_scroll)
        self.main_splitter.addWidget(right_scroll)
        
        # Set initial sizes (left: 200, center: stretch, right: 220)
        self.main_splitter.setSizes([200, 800, 220])
        
        # Set minimum sizes
        self.main_splitter.widget(0).setMinimumWidth(150)  # Left min
        self.main_splitter.widget(1).setMinimumWidth(400)  # Center min
        self.main_splitter.widget(2).setMinimumWidth(180)  # Right min
        
        main_layout.addWidget(self.main_splitter)
        
    def _create_left_sidebar(self, parent_scroll: QScrollArea):
        """Create left sidebar with media lists"""
        sidebar = QFrame()
        sidebar.setMinimumWidth(180)
        sidebar.setStyleSheet(f"background: {Colors.BG_SECONDARY};")
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Logo Brand - MISTER ARRANGER
        logo_layout = QHBoxLayout()

        # V4.31.1: Try multiple logo paths (jpg and png)
        logo_paths = [
            os.path.join(os.path.dirname(__file__), "mrarranger_logo.png"),
            os.path.join(os.path.dirname(__file__), "mrarranger_logo.jpg"),
            os.path.join(os.path.dirname(__file__), "logo.png"),
            os.path.join(os.path.dirname(__file__), "logo.jpg"),
            os.path.join(os.path.dirname(__file__), "..", "assets", "logo.png"),
            os.path.join(os.path.dirname(__file__), "..", "assets", "logo.jpg"),
        ]

        logo_found = False
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                logo_label = QLabel()
                pixmap = QPixmap(logo_path).scaled(45, 45, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(pixmap)
                logo_layout.addWidget(logo_label)
                logo_found = True
                print(f"[LOGO] Loaded: {logo_path}")
                break

        if not logo_found:
            # Fallback: show emoji
            logo_emoji = QLabel("🎵")
            logo_emoji.setStyleSheet(f"font-size: 28px;")
            logo_layout.addWidget(logo_emoji)
            print("[LOGO] No logo found, using fallback emoji")

        title = QLabel("MISTER ARRANGER")
        title.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 14px; font-weight: bold; letter-spacing: 2px;")
        logo_layout.addWidget(title)
        logo_layout.addStretch()
        layout.addLayout(logo_layout)
        
        # Audio section
        audio_section = CollapsibleSection("AUDIO FILES", "🎵")
        self.audio_list = DropZoneListWidget(
            accepted_extensions=[".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg"],
            placeholder_text="Drop audio files here"
        )
        self.audio_list.filesDropped.connect(self._on_audio_dropped)
        self.audio_list.currentRowChanged.connect(self._on_audio_track_selected)
        self.audio_list.setMinimumHeight(100)
        audio_section.addWidget(self.audio_list)
        
        add_audio_btn = QPushButton("+ Add Audio (or drag & drop)")
        add_audio_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {Colors.BG_TERTIARY}, stop:0.05 {Colors.BG_SECONDARY},
                    stop:0.95 {Colors.BG_PRIMARY}, stop:1 #0A0A0C);
                color: {Colors.TEXT_SECONDARY};
                border: 1px dashed {Colors.BORDER};
                border-top: 1px dashed {Colors.BORDER_LIGHT};
                border-radius: 4px;
                padding: 10px;
                font-family: 'Menlo', monospace;
                font-size: 10px;
            }}
            QPushButton:hover {{
                border-color: {Colors.ACCENT};
                color: {Colors.ACCENT};
                border-style: solid;
            }}
        """)
        add_audio_btn.clicked.connect(self._add_audio)
        audio_section.addWidget(add_audio_btn)
        
        # Clear Audio button
        clear_audio_btn = QPushButton("🗑️ Clear All")
        clear_audio_btn.setStyleSheet(f"""
            QPushButton {{
                background: #FF4444;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: #FF6666;
            }}
        """)
        clear_audio_btn.clicked.connect(self._clear_audio_files)
        audio_section.addWidget(clear_audio_btn)
        
        # AI DJ button
        ai_dj_btn = QPushButton("🎧 AI DJ - Smart Order")
        ai_dj_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {Colors.ACCENT_BRIGHT}, stop:0.1 {Colors.ACCENT},
                    stop:0.9 {Colors.ACCENT_DIM}, stop:1 #6E5010);
                color: #1A1A1E;
                border: 1px solid {Colors.ACCENT_DIM};
                border-top: 1px solid {Colors.ACCENT_BRIGHT};
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #FFD85A, stop:0.5 {Colors.ACCENT_BRIGHT}, stop:1 {Colors.ACCENT});
            }}
        """)
        ai_dj_btn.clicked.connect(self._show_ai_dj_dialog)
        audio_section.addWidget(ai_dj_btn)
        
        # Hook Extractor button (NEW V4.26!)
        hook_btn = QPushButton("🎵 Hook Extractor")
        hook_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        hook_btn.setToolTip("Extract hook sections from audio using waveform analysis (max 20 files)")
        hook_btn.clicked.connect(self._show_hook_extractor_dialog)
        audio_section.addWidget(hook_btn)

        # Lip-Sync Avatar button (Dreemina API)
        lipsync_btn = QPushButton("Lip-Sync Avatar")
        lipsync_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.GIF_COLOR};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #B07CC8;
            }}
        """)
        lipsync_btn.setToolTip("Generate lip-sync avatar videos from hooks via Dreemina API → upload as YouTube Shorts")
        lipsync_btn.clicked.connect(self._show_lipsync_dialog)
        audio_section.addWidget(lipsync_btn)

        layout.addWidget(audio_section)
        
        # Video section
        video_section = CollapsibleSection("VIDEO FILES", "🎬")
        self.video_list = DropZoneListWidget(
            accepted_extensions=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
            placeholder_text="Drop video files here"
        )
        self.video_list.filesDropped.connect(self._on_video_dropped)
        self.video_list.setMinimumHeight(80)

        # Enable drag reorder for video list (using new method)
        self.video_list.enableInternalMove()
        self.video_list.model().rowsMoved.connect(self._on_video_reordered)
        
        video_section.addWidget(self.video_list)
        
        add_video_btn = QPushButton("+ Add Video (or drag & drop)")
        add_video_btn.setStyleSheet(add_audio_btn.styleSheet())
        add_video_btn.clicked.connect(self._add_video)
        video_section.addWidget(add_video_btn)
        
        # Video buttons row
        video_btn_layout = QHBoxLayout()
        
        # Auto Assign Video button
        auto_assign_btn = QPushButton("🎬 Auto Assign")
        auto_assign_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        auto_assign_btn.clicked.connect(self._auto_assign_videos)
        video_btn_layout.addWidget(auto_assign_btn)
        
        # Shuffle Video button
        shuffle_video_btn = QPushButton("🔀 Shuffle")
        shuffle_video_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        shuffle_video_btn.clicked.connect(self._shuffle_videos)
        video_btn_layout.addWidget(shuffle_video_btn)
        
        # Clear Video button
        clear_video_btn = QPushButton("🗑️ Clear All")
        clear_video_btn.setStyleSheet(f"""
            QPushButton {{
                background: #FF4444;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: #FF6666;
            }}
        """)
        clear_video_btn.clicked.connect(self._clear_video_files)
        video_btn_layout.addWidget(clear_video_btn)
        
        video_section.content_layout.addLayout(video_btn_layout)
        
        layout.addWidget(video_section)
        
        # GIF section
        gif_section = CollapsibleSection("GIF FILES", "🖼")
        self.gif_list = DropZoneListWidget(
            accepted_extensions=[".gif"],
            placeholder_text="Drop GIF files here"
        )
        self.gif_list.filesDropped.connect(self._on_gif_dropped)
        self.gif_list.setMinimumHeight(60)
        gif_section.addWidget(self.gif_list)
        
        add_gif_btn = QPushButton("+ Add GIF (or drag & drop)")
        add_gif_btn.setStyleSheet(add_audio_btn.styleSheet())
        add_gif_btn.clicked.connect(self._add_gif)
        gif_section.addWidget(add_gif_btn)
        
        # GIF Settings
        gif_settings_frame = QFrame()
        gif_settings_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_TERTIARY};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        gif_settings_layout = QVBoxLayout(gif_settings_frame)
        gif_settings_layout.setContentsMargins(8, 8, 8, 8)
        gif_settings_layout.setSpacing(8)
        
        # Size dropdown
        size_row = QHBoxLayout()
        size_label = QLabel("📏 Size:")
        size_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        size_row.addWidget(size_label)
        
        self.gif_size_combo = QComboBox()
        self.gif_size_combo.addItems([
            "Full Screen (100%)",
            "Large (80%)",
            "Medium (50%)",
            "Small (30%)"
        ])
        self.gif_size_combo.setCurrentIndex(0)  # Default: Full Screen
        self.gif_size_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_PRIMARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background: {Colors.BG_PRIMARY};
                color: {Colors.TEXT_PRIMARY};
                selection-background-color: {Colors.ACCENT};
            }}
        """)
        size_row.addWidget(self.gif_size_combo)
        gif_settings_layout.addLayout(size_row)
        
        # Position dropdown
        pos_row = QHBoxLayout()
        pos_label = QLabel("📍 Position:")
        pos_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        pos_row.addWidget(pos_label)
        
        self.gif_position_combo = QComboBox()
        self.gif_position_combo.addItems([
            "Center",
            "Top-Left",
            "Top-Right",
            "Bottom-Left",
            "Bottom-Right"
        ])
        self.gif_position_combo.setCurrentIndex(0)  # Default: Center
        self.gif_position_combo.setStyleSheet(self.gif_size_combo.styleSheet())
        pos_row.addWidget(self.gif_position_combo)
        gif_settings_layout.addLayout(pos_row)
        
        # Opacity dropdown
        opacity_row = QHBoxLayout()
        opacity_label = QLabel("🔲 Opacity:")
        opacity_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        opacity_row.addWidget(opacity_label)
        
        self.gif_opacity_combo = QComboBox()
        self.gif_opacity_combo.addItems([
            "100%",
            "80%",
            "60%",
            "40%"
        ])
        self.gif_opacity_combo.setCurrentIndex(0)  # Default: 100%
        self.gif_opacity_combo.setStyleSheet(self.gif_size_combo.styleSheet())
        opacity_row.addWidget(self.gif_opacity_combo)
        gif_settings_layout.addLayout(opacity_row)
        
        gif_section.addWidget(gif_settings_frame)
        layout.addWidget(gif_section)
        
        # Logo section
        logo_section = CollapsibleSection("LOGO OVERLAY", "🏷")
        self.logo_list = DropZoneListWidget(
            accepted_extensions=[".jpg", ".jpeg", ".png", ".webp"],
            placeholder_text="Drop logo/image here"
        )
        self.logo_list.filesDropped.connect(self._on_logo_dropped)
        self.logo_list.setMinimumHeight(60)
        logo_section.addWidget(self.logo_list)
        
        add_logo_btn = QPushButton("+ Add Logo (or drag & drop)")
        add_logo_btn.setStyleSheet(add_audio_btn.styleSheet())
        add_logo_btn.clicked.connect(self._add_logo)
        logo_section.addWidget(add_logo_btn)
        
        # Logo Settings
        logo_settings_frame = QFrame()
        logo_settings_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_TERTIARY};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        logo_settings_layout = QVBoxLayout(logo_settings_frame)
        logo_settings_layout.setContentsMargins(8, 8, 8, 8)
        logo_settings_layout.setSpacing(8)
        
        # Logo Size dropdown
        logo_size_row = QHBoxLayout()
        logo_size_label = QLabel("📏 Size:")
        logo_size_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        logo_size_row.addWidget(logo_size_label)
        
        self.logo_size_combo = QComboBox()
        self.logo_size_combo.addItems([
            "Large (20%)",
            "Medium (15%)",
            "Small (10%)",
            "Tiny (5%)"
        ])
        self.logo_size_combo.setCurrentIndex(1)  # Default: Medium
        self.logo_size_combo.setStyleSheet(self.gif_size_combo.styleSheet())
        logo_size_row.addWidget(self.logo_size_combo)
        logo_settings_layout.addLayout(logo_size_row)
        
        # Logo Position dropdown
        logo_pos_row = QHBoxLayout()
        logo_pos_label = QLabel("📍 Position:")
        logo_pos_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        logo_pos_row.addWidget(logo_pos_label)
        
        self.logo_position_combo = QComboBox()
        self.logo_position_combo.addItems([
            "Top-Left",
            "Top-Right",
            "Bottom-Left",
            "Bottom-Right",
            "Center"
        ])
        self.logo_position_combo.setCurrentIndex(1)  # Default: Top-Right
        self.logo_position_combo.setStyleSheet(self.gif_size_combo.styleSheet())
        logo_pos_row.addWidget(self.logo_position_combo)
        logo_settings_layout.addLayout(logo_pos_row)
        
        # Logo Opacity dropdown
        logo_opacity_row = QHBoxLayout()
        logo_opacity_label = QLabel("🔲 Opacity:")
        logo_opacity_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        logo_opacity_row.addWidget(logo_opacity_label)
        
        self.logo_opacity_combo = QComboBox()
        self.logo_opacity_combo.addItems([
            "100%",
            "80%",
            "60%",
            "40%"
        ])
        self.logo_opacity_combo.setCurrentIndex(1)  # Default: 80%
        self.logo_opacity_combo.setStyleSheet(self.gif_size_combo.styleSheet())
        logo_opacity_row.addWidget(self.logo_opacity_combo)
        logo_settings_layout.addLayout(logo_opacity_row)
        
        logo_section.addWidget(logo_settings_frame)
        layout.addWidget(logo_section)
        
        layout.addStretch()
        parent_scroll.setWidget(sidebar)
        
    def _create_center_content(self, parent_scroll: QScrollArea):
        """Create center content area with vertical splitter"""
        center = QWidget()
        center.setStyleSheet(f"background: {Colors.BG_PRIMARY};")
        
        layout = QVBoxLayout(center)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Header
        header = QHBoxLayout()
        
        logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "logo.jpg")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pixmap = QPixmap(logo_path).scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            header.addWidget(logo_label)
            
        title = QLabel("LongPlay Studio")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 22px; font-weight: bold;")
        header.addWidget(title)
        
        version = QLabel("V5.10")
        version.setStyleSheet(f"""
            background: {Colors.ACCENT};
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: bold;
        """)
        header.addWidget(version)
        header.addStretch()
        
        # Settings button
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        settings_btn.clicked.connect(self._show_settings)
        header.addWidget(settings_btn)
        
        # AI DJ button
        ai_dj_header_btn = QPushButton("🎧 AI DJ")
        ai_dj_header_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        ai_dj_header_btn.clicked.connect(self._show_ai_dj_dialog)
        header.addWidget(ai_dj_header_btn)
        
        # AI VDO button
        ai_vdo_header_btn = QPushButton("🎬 AI VDO")
        ai_vdo_header_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        ai_vdo_header_btn.clicked.connect(self._show_ai_vdo_dialog)
        header.addWidget(ai_vdo_header_btn)
        
        # Video Prompt button (NEW V4.26!)
        video_prompt_btn = QPushButton("🎬 VDO Prompt")
        video_prompt_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        video_prompt_btn.setToolTip("Generate Midjourney-style prompts for meta.ai video")
        video_prompt_btn.clicked.connect(self._show_video_prompt_dialog)
        header.addWidget(video_prompt_btn)
        
        # YouTube button
        yt_btn = QPushButton("📺 YouTube")
        yt_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        yt_btn.clicked.connect(self._show_youtube_generator)
        header.addWidget(yt_btn)
        
        # Export button
        export_btn = QPushButton("🚀 EXPORT")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {Colors.LED_GREEN}, stop:0.5 #2E7D32, stop:1 #1B5E20);
                color: white;
                border: 1px solid #2E7D32;
                border-top: 1px solid {Colors.LED_GREEN};
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #4CAF50, stop:0.5 {Colors.LED_GREEN}, stop:1 #2E7D32);
            }}
        """)
        export_btn.clicked.connect(self._export_video)
        header.addWidget(export_btn)
        
        layout.addLayout(header)
        
        # Create vertical splitter for 3 sections
        self.center_splitter = QSplitter(Qt.Orientation.Vertical)
        self.center_splitter.setHandleWidth(6)
        self.center_splitter.setStyleSheet(f"""
            QSplitter::handle:vertical {{
                background: {Colors.BORDER};
                height: 6px;
                margin: 2px 0;
            }}
            QSplitter::handle:vertical:hover {{
                background: {Colors.ACCENT};
            }}
        """)
        
        # Section 1: Video Preview
        video_container = QWidget()
        video_container.setStyleSheet(f"background: {Colors.BG_PRIMARY};")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 5)
        video_layout.setSpacing(0)
        
        self.video_preview = VideoPreviewCard()
        video_layout.addWidget(self.video_preview)
        self.center_splitter.addWidget(video_container)
        
        # Section 2: Track List
        track_container = QWidget()
        track_container.setStyleSheet(f"background: {Colors.BG_PRIMARY};")
        track_layout = QVBoxLayout(track_container)
        track_layout.setContentsMargins(0, 5, 0, 5)
        track_layout.setSpacing(5)
        
        track_section = CollapsibleSection("Track List", "📋")
        
        self.track_list_widget = QWidget()
        self.track_list_layout = QVBoxLayout(self.track_list_widget)
        self.track_list_layout.setContentsMargins(0, 0, 0, 0)
        self.track_list_layout.setSpacing(5)
        
        track_scroll = QScrollArea()
        track_scroll.setWidget(self.track_list_widget)
        track_scroll.setWidgetResizable(True)
        track_scroll.setStyleSheet(f"background: transparent; border: none;")
        track_section.addWidget(track_scroll)
        
        track_layout.addWidget(track_section)
        self.center_splitter.addWidget(track_container)
        
        # Section 3: Timeline Preview
        timeline_container = QWidget()
        timeline_container.setStyleSheet(f"background: {Colors.BG_PRIMARY};")
        timeline_layout = QVBoxLayout(timeline_container)
        timeline_layout.setContentsMargins(0, 5, 0, 0)
        timeline_layout.setSpacing(5)
        
        timeline_section = CollapsibleSection("Timeline Preview", "⏱")
        
        # Time display
        time_widget = QWidget()
        time_layout = QHBoxLayout(time_widget)
        time_layout.setContentsMargins(0, 0, 0, 0)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        time_layout.addWidget(self.time_label)
        time_layout.addStretch()
        timeline_section.addWidget(time_widget)
        
        # ── Phase 2: I-9 Multi-Track Timeline (replaces CapCutTimeline) ──
        if _HAS_MULTI_TRACK:
            self.timeline = MultiTrackTimeline()
            self.timeline.setMinimumHeight(180)
            print("[PHASE2] Multi-track timeline loaded (Video/Audio/Text/Effects)")
        else:
            self.timeline = CapCutTimeline()
            print("[PHASE2] Fallback: using CapCutTimeline")
        self.timeline.setMinimumHeight(150)
        if hasattr(self.timeline, 'seekRequested'):
            self.timeline.seekRequested.connect(self._on_seek)
        timeline_section.addWidget(self.timeline)

        # ── Phase 2: I-10 Wire clip drag/trim/split handlers ──
        if _HAS_CLIP_TOOLS and _HAS_MULTI_TRACK and hasattr(self.timeline, '_scene'):
            try:
                self._clip_drag_handler = ClipDragHandler(self.timeline._scene)
                self._clip_trim_handler = ClipTrimSplitHandler(self.timeline._scene)
                print("[PHASE2] Clip drag/trim/split handlers attached")
            except Exception as e:
                print(f"[PHASE2] Clip tools init warning: {e}")

        # V5.5: Timeline Transport Bar — Play/Stop controls under timeline
        transport_bar = QWidget()
        transport_bar.setStyleSheet(f"background: {Colors.BG_SECONDARY}; border-radius: 6px;")
        transport_layout = QHBoxLayout(transport_bar)
        transport_layout.setContentsMargins(10, 4, 10, 4)
        transport_layout.setSpacing(8)

        self.tl_play_btn = QPushButton("▶  PLAY")
        self.tl_play_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 20px;
                font-size: 12px;
                font-weight: bold;
                font-family: 'Menlo', monospace;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_BRIGHT};
            }}
        """)
        self.tl_play_btn.setFixedWidth(120)
        self.tl_play_btn.clicked.connect(self._on_tl_play_pause)
        transport_layout.addWidget(self.tl_play_btn)

        self.tl_stop_btn = QPushButton("⏹  STOP")
        self.tl_stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 6px 20px;
                font-size: 12px;
                font-weight: bold;
                font-family: 'Menlo', monospace;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        self.tl_stop_btn.setFixedWidth(120)
        self.tl_stop_btn.clicked.connect(lambda: self.audio_player.stop())
        transport_layout.addWidget(self.tl_stop_btn)

        # ── Phase 2: I-1 Undo/Redo buttons ──
        if _HAS_UNDO and self.undo_manager:
            _tb_style = (f"QPushButton {{ background: {Colors.BG_TERTIARY}; color: {Colors.TEXT_SECONDARY}; "
                         f"border: 1px solid {Colors.BORDER}; border-radius: 4px; padding: 4px 10px; "
                         f"font-size: 11px; font-weight: bold; }} "
                         f"QPushButton:hover {{ background: {Colors.BORDER}; color: {Colors.TEXT_PRIMARY}; }}")
            undo_btn = QPushButton("↩ Undo")
            undo_btn.setStyleSheet(_tb_style)
            undo_btn.setFixedWidth(80)
            undo_btn.clicked.connect(lambda: self.undo_manager.undo())
            transport_layout.addWidget(undo_btn)
            redo_btn = QPushButton("↪ Redo")
            redo_btn.setStyleSheet(_tb_style)
            redo_btn.setFixedWidth(80)
            redo_btn.clicked.connect(lambda: self.undo_manager.redo())
            transport_layout.addWidget(redo_btn)

        # ── Phase 2: I-10 Razor tool button ──
        if _HAS_CLIP_TOOLS:
            razor_btn = QPushButton("✂ Razor")
            razor_btn.setStyleSheet(_tb_style if _HAS_UNDO else (
                f"QPushButton {{ background: {Colors.BG_TERTIARY}; color: {Colors.TEXT_SECONDARY}; "
                f"border: 1px solid {Colors.BORDER}; border-radius: 4px; padding: 4px 10px; "
                f"font-size: 11px; font-weight: bold; }}"))
            razor_btn.setFixedWidth(80)
            razor_btn.setCheckable(True)
            razor_btn.setToolTip("Toggle razor tool — click on clip to split")
            transport_layout.addWidget(razor_btn)
            self._razor_btn = razor_btn

        # ── Phase 4 F-6: Pipeline button ──
        pipeline_btn = QPushButton("⚡ Pipeline")
        pipeline_btn.setStyleSheet(f"QPushButton {{ background: {Colors.BG_TERTIARY}; color: #00d4aa; "
                                   f"border: 1px solid #00d4aa; border-radius: 4px; padding: 4px 10px; "
                                   f"font-size: 11px; font-weight: bold; }} "
                                   f"QPushButton:hover {{ background: #00d4aa; color: #1a1a2e; }}")
        pipeline_btn.setFixedWidth(90)
        pipeline_btn.setToolTip("Open Production Pipeline wizard")
        pipeline_btn.clicked.connect(self._on_production_pipeline)
        transport_layout.addWidget(pipeline_btn)

        # ── Phase 2: I-12 Text tool button ──
        if _HAS_TEXT_OVERLAY:
            text_btn = QPushButton("T Text")
            text_btn.setStyleSheet(f"QPushButton {{ background: {Colors.BG_TERTIARY}; color: {Colors.TEXT_SECONDARY}; "
                                   f"border: 1px solid {Colors.BORDER}; border-radius: 4px; padding: 4px 10px; "
                                   f"font-size: 11px; font-weight: bold; }} "
                                   f"QPushButton:hover {{ background: {Colors.BORDER}; color: {Colors.TEXT_PRIMARY}; }}")
            text_btn.setFixedWidth(70)
            text_btn.setToolTip("Add text overlay to timeline")
            text_btn.clicked.connect(self._on_add_text_clip)
            transport_layout.addWidget(text_btn)

        transport_layout.addStretch()

        # ── Phase 2: I-3 Theme selector ──
        if _HAS_VINTAGE_THEME:
            theme_combo = QComboBox()
            theme_combo.addItems(list(THEMES.keys()))
            theme_combo.setCurrentText("classic_dark")
            theme_combo.setFixedWidth(120)
            theme_combo.setStyleSheet(f"QComboBox {{ background: {Colors.BG_TERTIARY}; color: {Colors.TEXT_SECONDARY}; "
                                      f"border: 1px solid {Colors.BORDER}; border-radius: 3px; padding: 3px; font-size: 10px; }}")
            theme_combo.currentTextChanged.connect(self._on_theme_changed)
            theme_combo.setToolTip("Switch UI theme")
            transport_layout.addWidget(theme_combo)

        self.tl_position_label = QLabel("00:00 / 00:00")
        self.tl_position_label.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 13px; font-weight: bold; "
            f"font-family: 'Menlo', monospace; letter-spacing: 1px;")
        transport_layout.addWidget(self.tl_position_label)

        timeline_section.addWidget(transport_bar)

        # ── Phase 2: I-11 Keyframe editor panel (below timeline) ──
        if _HAS_KEYFRAMES:
            self.keyframe_editor = KeyframeEditor()
            self.keyframe_editor.setMaximumHeight(150)
            self.keyframe_editor.setVisible(False)  # toggle on clip select
            timeline_section.addWidget(self.keyframe_editor)
            print("[PHASE2] Keyframe editor panel added below timeline")

        timeline_layout.addWidget(timeline_section)
        self.center_splitter.addWidget(timeline_container)
        
        # Set initial sizes (Video: 40%, Track: 25%, Timeline: 35%)
        self.center_splitter.setSizes([400, 250, 350])
        
        # Set minimum heights
        self.center_splitter.widget(0).setMinimumHeight(100)  # Video min
        self.center_splitter.widget(1).setMinimumHeight(80)   # Track min
        self.center_splitter.widget(2).setMinimumHeight(100)  # Timeline min
        
        layout.addWidget(self.center_splitter, 1)
        
        # Audio player (hidden)
        self.audio_player = AudioPlayerWidget()
        self.audio_player.hide()
        layout.addWidget(self.audio_player)
        
        parent_scroll.setWidget(center)
        
    def _create_right_panel(self, parent_scroll: QScrollArea):
        """Create right panel — Ozone 12 Maximizer + Waves WLM Plus Loudness Meter"""
        panel = QFrame()
        panel.setMinimumWidth(280)
        panel.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #0E0E12, stop:0.02 #141418, stop:0.98 #111115, stop:1 #0A0A0C);
                border-left: 1px solid #2A2A32;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ═══════════════════════════════════════════════
        #  MAXIMIZER — iZotope Ozone 12 Style
        # ═══════════════════════════════════════════════
        maximizer_header = QLabel("MAXIMIZER")
        maximizer_header.setStyleSheet("""
            color: #48CAE4;
            font-size: 11px;
            font-weight: bold;
            font-family: 'Menlo', monospace;
            letter-spacing: 3px;
            padding: 4px 0px 3px 0px;
            border-bottom: 1px solid #00B4D8;
        """)
        layout.addWidget(maximizer_header)

        # ── Mastering Preset dropdown ──
        if _HAS_PRESETS and MASTERING_PRESET_NAMES:
            preset_row = QHBoxLayout()
            preset_row.setSpacing(4)
            preset_lbl = QLabel("PRESET")
            preset_lbl.setStyleSheet("color: #8E8A82; font-size: 8px; font-weight: bold; font-family: 'Menlo', monospace; letter-spacing: 1px;")
            preset_row.addWidget(preset_lbl)
            self.right_mastering_preset = QComboBox()
            self.right_mastering_preset.addItem("— None —")
            for name in MASTERING_PRESET_NAMES:
                self.right_mastering_preset.addItem(name)
            self.right_mastering_preset.setStyleSheet("""
                QComboBox {
                    background: #0A0A0C; color: #48CAE4;
                    border: 1px solid #2A2A32; border-radius: 3px;
                    padding: 3px 6px; font-size: 10px; font-family: 'Menlo', monospace;
                }
                QComboBox::drop-down { border: none; }
                QComboBox QAbstractItemView {
                    background: #141418; color: #E0DCD4;
                    selection-background-color: #00B4D8;
                }
            """)
            self.right_mastering_preset.currentTextChanged.connect(self._on_right_mastering_preset_changed)
            preset_row.addWidget(self.right_mastering_preset, 1)
            layout.addLayout(preset_row)

        # ── IRC Mode dropdown (Ozone 12: IRC 1-5, IRC LL) ──
        irc_row = QHBoxLayout()
        irc_row.setSpacing(4)
        irc_lbl = QLabel("IRC MODE")
        irc_lbl.setStyleSheet("color: #8E8A82; font-size: 8px; font-weight: bold; font-family: 'Menlo', monospace; letter-spacing: 1px;")
        irc_row.addWidget(irc_lbl)
        self.right_irc_combo = QComboBox()
        irc_mode_names = [k for k in IRC_MODES.keys() if " - " not in k] if IRC_MODES else ["IRC 1", "IRC 2", "IRC 3", "IRC 4", "IRC 5", "IRC LL"]
        for m in irc_mode_names:
            self.right_irc_combo.addItem(m)
        self.right_irc_combo.setCurrentText("IRC 2")
        self.right_irc_combo.setStyleSheet("""
            QComboBox {
                background: #0A0A0C; color: #48CAE4;
                border: 1px solid #2A2A32; border-radius: 3px;
                padding: 3px 6px; font-size: 10px; font-weight: bold; font-family: 'Menlo', monospace;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #141418; color: #E0DCD4;
                selection-background-color: #00B4D8;
            }
        """)
        self.right_irc_combo.currentTextChanged.connect(self._on_right_irc_mode_changed)
        irc_row.addWidget(self.right_irc_combo, 1)
        layout.addLayout(irc_row)

        # ── IRC Sub-mode dropdown (only IRC 3 & 4: Pumping/Balanced/Crisp) ──
        self.right_irc_submode_row = QHBoxLayout()
        self.right_irc_submode_row.setSpacing(4)
        self.right_irc_submode_lbl = QLabel("SUB-MODE")
        self.right_irc_submode_lbl.setStyleSheet("color: #8E8A82; font-size: 8px; font-weight: bold; font-family: 'Menlo', monospace; letter-spacing: 1px;")
        self.right_irc_submode_row.addWidget(self.right_irc_submode_lbl)
        self.right_irc_submode = QComboBox()
        self.right_irc_submode.setStyleSheet("""
            QComboBox {
                background: #0A0A0C; color: #F5C04A;
                border: 1px solid #2A2A32; border-radius: 3px;
                padding: 3px 6px; font-size: 10px; font-family: 'Menlo', monospace;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #141418; color: #E0DCD4;
                selection-background-color: #F5C04A;
            }
        """)
        self.right_irc_submode_row.addWidget(self.right_irc_submode, 1)
        # Container widget for sub-mode row (hide/show)
        self.right_irc_submode_widget = QWidget()
        self.right_irc_submode_widget.setLayout(self.right_irc_submode_row)
        self.right_irc_submode_widget.setVisible(False)  # Hidden by default (IRC 2 has no sub-modes)
        layout.addWidget(self.right_irc_submode_widget)

        # ── Gain Knob + Display (OzoneRotaryKnob — custom QPainter) ──
        # NOTE: QDial + CSS border-radius breaks mouse events on macOS/PyQt6.
        from modules.widgets.rotary_knob import OzoneRotaryKnob

        gain_section = QHBoxLayout()
        gain_section.setSpacing(8)

        self.right_gain_dial = OzoneRotaryKnob(
            name="GAIN", min_val=0.0, max_val=20.0, default=0.0,
            unit="dB", decimals=1, large=True)
        self.right_gain_dial.valueChanged.connect(
            lambda v: self._on_right_gain_changed(int(v * 10)))
        gain_section.addWidget(self.right_gain_dial, alignment=Qt.AlignmentFlag.AlignCenter)

        # Gain display
        gain_info_col = QVBoxLayout()
        gain_info_col.setSpacing(2)
        gain_info_col.addStretch()
        self.right_gain_display = QLabel("+0.0")
        self.right_gain_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_gain_display.setFont(QFont("Menlo", 24, QFont.Weight.Bold))
        self.right_gain_display.setStyleSheet("color: #48CAE4;")
        gain_info_col.addWidget(self.right_gain_display)

        gain_unit = QLabel("dB")
        gain_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gain_unit.setStyleSheet("color: #8E8A82; font-size: 10px; font-family: 'Menlo', monospace;")
        gain_info_col.addWidget(gain_unit)
        gain_info_col.addStretch()
        gain_section.addLayout(gain_info_col)
        layout.addLayout(gain_section)

        # ── OUTPUT Ceiling ──
        output_row = QHBoxLayout()
        output_row.setSpacing(6)
        out_label = QLabel("OUTPUT")
        out_label.setStyleSheet("color: #8E8A82; font-size: 9px; font-weight: bold; font-family: 'Menlo', monospace; letter-spacing: 1px;")
        output_row.addWidget(out_label)

        self.right_ceiling_spin = QDoubleSpinBox()
        self.right_ceiling_spin.setRange(-3.0, -0.1)
        self.right_ceiling_spin.setValue(-1.0)
        self.right_ceiling_spin.setSingleStep(0.1)
        self.right_ceiling_spin.setSuffix(" dBTP")
        self.right_ceiling_spin.setDecimals(2)
        self.right_ceiling_spin.setStyleSheet("""
            QDoubleSpinBox {
                background: #0A0A0C; color: #48CAE4;
                border: 1px solid #2A2A32; border-radius: 3px;
                padding: 4px 8px; font-size: 11px; font-weight: bold; font-family: 'Menlo', monospace;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 14px; background: #141418; border: 1px solid #2A2A32;
            }
        """)
        self.right_ceiling_spin.valueChanged.connect(self._on_right_ceiling_changed)
        output_row.addWidget(self.right_ceiling_spin, 1)

        tp_indicator = QLabel("TRUE PEAK")
        tp_indicator.setStyleSheet("color: #43A047; font-size: 8px; font-weight: bold; font-family: 'Menlo', monospace;")
        output_row.addWidget(tp_indicator)
        layout.addLayout(output_row)

        # ── Gain Reduction History (Ozone 12 waveform) ──
        if _HAS_MASTER_WIDGETS:
            self.right_gr_history = GainReductionHistoryWidget()
            layout.addWidget(self.right_gr_history, alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            self.right_gr_history = None

        # ── Separator ──
        sep_gain = QFrame()
        sep_gain.setFixedHeight(2)
        sep_gain.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0A0A0C, stop:0.5 #00B4D8, stop:1 #0A0A0C);")
        layout.addWidget(sep_gain)

        # ═══════════════════════════════════════════════
        #  LOUDNESS METER — Waves WLM Plus Style
        # ═══════════════════════════════════════════════

        # Audio Analysis Engine for REAL meter levels
        self.audio_engine = AudioAnalysisEngine()

        # RealTimeMeter — still used internally for audio analysis data
        self.meter = RealTimeMeter()
        self.meter.set_audio_engine(self.audio_engine)
        self.meter.setVisible(False)  # Hidden — replaced by WavesWLMMeter visually

        if _HAS_MASTER_WIDGETS:
            self.right_wlm_meter = WavesWLMMeter(target_lufs=-14.0)
            layout.addWidget(self.right_wlm_meter, alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            # Fallback: show old meter if import failed
            self.right_wlm_meter = None
            self.meter.setVisible(True)
            self.meter.setMinimumHeight(120)
            self.meter.setMaximumHeight(140)
            layout.addWidget(self.meter)

        # Hidden backward-compat LUFS display references (data still fed by _on_position_changed)
        self.lufs_momentary = LUFSDisplay("MOMENTARY")
        self.lufs_momentary.setVisible(False)
        self.lufs_shortterm = LUFSDisplay("SHORT-TERM")
        self.lufs_shortterm.setVisible(False)
        self.lufs_integrated = LUFSDisplay("INTEGRATED")
        self.lufs_integrated.setVisible(False)
        self.lufs_lra = LUFSDisplay("LRA")
        self.lufs_lra.setVisible(False)
        self.true_peak_label = QLabel("")
        self.true_peak_label.setVisible(False)
        self.lufs_target_label = QLabel("")
        self.lufs_target_label.setVisible(False)

        # ── Separator ──
        separator = QFrame()
        separator.setFixedHeight(2)
        separator.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0A0A0C, stop:0.5 #3E3E46, stop:1 #0A0A0C);")
        layout.addWidget(separator)

        # DJ Crossfade section
        crossfade_header = QLabel("DJ CROSSFADE")
        crossfade_header.setStyleSheet("""
            color: #C89B3C;
            font-size: 11px;
            font-weight: bold;
            font-family: 'Menlo', monospace;
            letter-spacing: 2px;
            padding-bottom: 4px;
            border-bottom: 1px solid #2C2C34;
        """)
        layout.addWidget(crossfade_header)
        
        # Duration slider
        dur_layout = QHBoxLayout()
        dur_label = QLabel("Duration:")
        dur_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        dur_layout.addWidget(dur_label)
        
        self.crossfade_slider = QSlider(Qt.Orientation.Horizontal)
        self.crossfade_slider.setMinimum(1)
        self.crossfade_slider.setMaximum(10)
        self.crossfade_slider.setValue(5)
        self.crossfade_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #0A0A0C, stop:0.5 #0E0E10, stop:1 #3E3E46);
                height: 6px;
                border-radius: 3px;
                border: 1px solid #0A0A0C;
            }
            QSlider::handle:horizontal {
                width: 18px; height: 18px;
                margin: -7px 0;
                border-radius: 9px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #6E6E74, stop:0.3 #55555B, stop:0.7 #45454B, stop:1 #3A3A40);
                border: 1px solid #0A0A0C;
                border-top: 1px solid #78787E;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #0077B6, stop:1 #00B4D8);
                border-radius: 3px;
            }
        """)
        self.crossfade_slider.valueChanged.connect(self._update_crossfade)
        dur_layout.addWidget(self.crossfade_slider)
        
        self.crossfade_label = QLabel("5 sec")
        self.crossfade_label.setStyleSheet(f"color: {Colors.ACCENT}; font-weight: bold;")
        dur_layout.addWidget(self.crossfade_label)
        layout.addLayout(dur_layout)
        
        # Style combo
        style_layout = QHBoxLayout()
        style_label = QLabel("Style:")
        style_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        style_layout.addWidget(style_label)
        
        self.crossfade_style = QComboBox()
        self.crossfade_style.addItems(["Linear", "Exponential", "Equal Power", "S-Curve"])
        self.crossfade_style.setCurrentText("Equal Power")
        self.crossfade_style.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                selection-background-color: {Colors.ACCENT};
            }}
        """)
        style_layout.addWidget(self.crossfade_style, 1)
        layout.addLayout(style_layout)
        
        # Timestamp section
        ts_header = QLabel("📋 TIMESTAMP")
        ts_header.setStyleSheet(f"color: {Colors.VIDEO_COLOR}; font-size: 14px; font-weight: bold;")
        layout.addWidget(ts_header)
        
        ts_btn = QPushButton("📋 Generate Timestamps")
        ts_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #3A80C9;
            }}
        """)
        ts_btn.clicked.connect(self._generate_timestamps)
        layout.addWidget(ts_btn)

        # V5.5.1: Master buttons removed — Maximizer on right panel replaces Master Module

        # ── Phase 2: I-13 Transitions Panel ──
        if _HAS_TRANSITIONS:
            trans_section = CollapsibleSection("Transitions", "⬡")
            self.transitions_panel = TransitionLibraryPanel()
            trans_section.addWidget(self.transitions_panel)
            layout.addWidget(trans_section)
            print("[PHASE2] Transitions panel added (12+ types)")

        # ── Phase 2: I-14 Effects Panel ──
        if _HAS_EFFECTS:
            fx_section = CollapsibleSection("Effects", "✦")
            self.effects_panel = EffectsPanel()
            fx_section.addWidget(self.effects_panel)
            layout.addWidget(fx_section)
            print("[PHASE2] Effects panel added (10+ types)")

        # ── Phase 2: I-16 Export Presets Panel ──
        if _HAS_EXPORT_PRESETS:
            export_section = CollapsibleSection("Export Presets", "📦")
            self.export_preset_panel = ExportPresetPanel()
            export_section.addWidget(self.export_preset_panel)
            layout.addWidget(export_section)
            print("[PHASE2] Export presets panel added")

        layout.addStretch()
        parent_scroll.setWidget(panel)
        
    def _connect_signals(self):
        """Connect audio player signals"""
        self.audio_player.position_changed.connect(self._on_position_changed)
        self.audio_player.duration_changed.connect(self._on_duration_changed)
        self.audio_player.play_state_changed.connect(self._on_play_state_changed)
        self.video_preview.speed_changed.connect(self.audio_player.player.setPlaybackRate)
        # V5.5: Load audio into analysis engine when track changes
        self.audio_player.track_changed.connect(self._on_track_changed_for_meter)
        
    # ═══════════════════════════════════════════════════════════════
    #  PHASE 2 INTEGRATION METHODS
    # ═══════════════════════════════════════════════════════════════

    def _on_theme_changed(self, theme_key: str) -> None:
        """I-3: Switch UI theme."""
        if not _HAS_VINTAGE_THEME:
            return
        try:
            app = QApplication.instance()
            if app:
                apply_theme(app, theme_key)
                print(f"[PHASE2] Theme changed to: {theme_key}")
        except Exception as e:
            print(f"[PHASE2] Theme change error: {e}")

    def _on_add_text_clip(self) -> None:
        """I-12: Add a text clip to the text track on the timeline."""
        if not _HAS_TEXT_OVERLAY:
            return
        try:
            clip = TextClip(
                text="Title",
                font_family="Inter",
                font_size=48,
                color="#FFFFFF",
                animation=TextAnimation.FADE_IN,
                start_time=0.0,
                duration=5.0,
            )
            # If multi-track timeline, add to text track
            if _HAS_MULTI_TRACK and hasattr(self.timeline, '_project'):
                from gui.models.track import Track, TrackType, Clip
                project = self.timeline._project
                # find or create text track
                text_track = None
                for t in project.tracks:
                    if t.type == TrackType.TEXT:
                        text_track = t
                        break
                if text_track is None:
                    text_track = Track(name="Text", type=TrackType.TEXT)
                    project.add_track(text_track)
                tc = Clip(start_time=clip.start_time, duration=clip.duration,
                          name=clip.text, properties={"text_clip": True, "font": clip.font_family,
                                                       "font_size": clip.font_size, "color": clip.color})
                text_track.add_clip(tc)
                if hasattr(self.timeline, 'refresh'):
                    self.timeline.refresh()
                print(f"[PHASE2] Text clip added: '{clip.text}'")

            # Show text properties panel if available
            if hasattr(self, 'keyframe_editor') and _HAS_TEXT_OVERLAY:
                # toggle keyframe editor to show text properties
                pass
        except Exception as e:
            print(f"[PHASE2] Add text clip error: {e}")

    def _on_speed_ramp(self) -> None:
        """I-15: Open speed ramp editor for selected clip."""
        if not _HAS_SPEED_RAMP:
            return
        try:
            editor = SpeedCurveEditor()
            editor.setWindowTitle("Speed Ramp Editor")
            editor.show()
            print("[PHASE2] Speed ramp editor opened")
        except Exception as e:
            print(f"[PHASE2] Speed ramp error: {e}")

    def _on_production_pipeline(self) -> None:
        """F-6: Production Flow Pipeline wizard dialog."""
        try:
            dlg = _ProductionPipelineDialog(self)
            dlg.exec()
        except Exception as e:
            print(f"[PHASE4] Pipeline dialog error: {e}")
            # Fallback message
            QMessageBox.information(self, "Production Pipeline",
                "1. IMPORT → 2. AI DJ → 3. COMPILE → 4. MASTER → "
                "5. HOOK EXTRACT → 6. VIDEO → 7. EXPORT")

    def _on_position_changed(self, position_ms: int):
        """Handle playback position change"""
        try:
            self.timeline.setPlayheadPosition(position_ms)
            if hasattr(self, 'video_preview') and self.video_preview:
                self.video_preview.seek(position_ms)
        except Exception as e:
            print(f"[PLAY] Position update error: {e}")

        # Update meter with TRACK-LOCAL position (not timeline position)
        # AudioAnalysisEngine needs position within the current file, not global timeline
        track_local_pos = self.audio_player.player.position()
        self.meter.setPosition(track_local_pos)

        # V5.10.1: LUFS measurement using pyloudnorm (ITU-R BS.1770-4 K-weighted)
        if self.meter.is_playing:
            import math

            # Initialize LUFS state once
            if not hasattr(self, '_lufs_pyln'):
                try:
                    import pyloudnorm as pyln
                    self._lufs_pyln = pyln
                except ImportError:
                    self._lufs_pyln = None
                self._lufs_meter_obj = None
                self._lufs_short_history = []
                self._lufs_int_last_pos = 0
                self._lufs_integrated_val = -70.0

            levels_db = getattr(self.meter, '_last_levels_db', None)

            # Auto-load current track into audio_engine if not loaded yet
            if (hasattr(self, 'audio_engine') and self.audio_engine._current_data is None
                    and hasattr(self, 'audio_player') and self.audio_player.files):
                idx = self.audio_player.current_file_index
                if 0 <= idx < len(self.audio_player.files):
                    try:
                        self.audio_engine.load_file(self.audio_player.files[idx])
                        print(f"[LUFS] Auto-loaded track for metering")
                    except Exception:
                        pass

            has_audio = (hasattr(self, 'audio_engine') and
                        self.audio_engine._current_data is not None and
                        self._lufs_pyln is not None)

            if has_audio:
                import numpy as np
                sr = self.audio_engine._current_sr
                data = self.audio_engine._current_data

                # Create/update meter for this sample rate
                if self._lufs_meter_obj is None or getattr(self, '_lufs_sr', 0) != sr:
                    self._lufs_meter_obj = self._lufs_pyln.Meter(sr)
                    self._lufs_sr = sr

                pos_samples = int(track_local_pos / 1000.0 * sr)

                # === Momentary LUFS (400ms K-weighted) ===
                mom_n = int(sr * 0.4)
                mom_start = max(0, pos_samples - mom_n)
                mom_end = min(len(data), pos_samples)
                if mom_end - mom_start > 2048:
                    chunk = data[mom_start:mom_end]
                    if chunk.ndim == 1:
                        chunk = np.column_stack([chunk, chunk])
                    try:
                        momentary_lufs = self._lufs_meter_obj.integrated_loudness(chunk)
                        if np.isinf(momentary_lufs) or np.isnan(momentary_lufs):
                            momentary_lufs = -70.0
                    except Exception:
                        momentary_lufs = -70.0
                else:
                    momentary_lufs = -70.0

                # === Short-term LUFS (3s K-weighted) ===
                short_n = int(sr * 3.0)
                short_start = max(0, pos_samples - short_n)
                short_end = min(len(data), pos_samples)
                if short_end - short_start > sr:
                    chunk = data[short_start:short_end]
                    if chunk.ndim == 1:
                        chunk = np.column_stack([chunk, chunk])
                    try:
                        shortterm_lufs = self._lufs_meter_obj.integrated_loudness(chunk)
                        if np.isinf(shortterm_lufs) or np.isnan(shortterm_lufs):
                            shortterm_lufs = -70.0
                    except Exception:
                        shortterm_lufs = momentary_lufs
                else:
                    shortterm_lufs = momentary_lufs

                # === Integrated LUFS (gated, from start — update every 3s) ===
                if pos_samples - self._lufs_int_last_pos > sr * 3 or self._lufs_integrated_val <= -70.0:
                    int_end = min(len(data), pos_samples)
                    int_n = min(int_end, sr * 60)
                    int_start = max(0, int_end - int_n)
                    if int_end - int_start > sr:
                        chunk = data[int_start:int_end]
                        if chunk.ndim == 1:
                            chunk = np.column_stack([chunk, chunk])
                        try:
                            integrated_lufs = self._lufs_meter_obj.integrated_loudness(chunk)
                            if np.isinf(integrated_lufs) or np.isnan(integrated_lufs):
                                integrated_lufs = -70.0
                            self._lufs_integrated_val = integrated_lufs
                        except Exception:
                            integrated_lufs = self._lufs_integrated_val
                    else:
                        integrated_lufs = self._lufs_integrated_val
                    self._lufs_int_last_pos = pos_samples
                else:
                    integrated_lufs = self._lufs_integrated_val

                # === LRA (10th-95th percentile of short-term) ===
                if shortterm_lufs > -70.0:
                    self._lufs_short_history.append(shortterm_lufs)
                    if len(self._lufs_short_history) > 120:
                        self._lufs_short_history.pop(0)
                if len(self._lufs_short_history) >= 4:
                    s = sorted(self._lufs_short_history)
                    n = len(s)
                    lra = max(0.0, s[min(n-1, int(n*0.95))] - s[max(0, int(n*0.10))])
                else:
                    lra = 0.0

                # === True Peak ===
                if levels_db and levels_db.get("left_peak_db", -70) > -70:
                    tp_l = levels_db["left_peak_db"]
                    tp_r = levels_db["right_peak_db"]
                    peak_db = max(tp_l, tp_r)
                else:
                    peak_db = -70.0
                    tp_l = tp_r = -70.0

            else:
                # Fallback: RMS approximation (no pyloudnorm)
                if levels_db and levels_db.get("left_rms_db", -70) > -70:
                    avg_rms = (levels_db["left_rms_db"] + levels_db["right_rms_db"]) / 2.0
                    momentary_lufs = max(-70.0, min(0.0, avg_rms))
                    peak_db = max(levels_db["left_peak_db"], levels_db["right_peak_db"])
                else:
                    avg_level = (self.meter.left_level + self.meter.right_level) / 2.0
                    momentary_lufs = 20 * math.log10(max(avg_level, 1e-10)) - 6.0
                    momentary_lufs = max(-70.0, min(0.0, momentary_lufs))
                    peak_db = max(
                        20 * math.log10(max(self.meter.peak_left, 1e-10)),
                        20 * math.log10(max(self.meter.peak_right, 1e-10)),
                    )
                tp_l = tp_r = peak_db
                shortterm_lufs = momentary_lufs
                integrated_lufs = momentary_lufs
                lra = 0.0

            self.lufs_momentary.setValue(momentary_lufs)
            self.lufs_shortterm.setValue(shortterm_lufs)
            self.lufs_integrated.setValue(integrated_lufs)
            self.lufs_lra.setValue(lra)
            self.true_peak_label.setText(f"True Peak: {peak_db:.1f} dBTP")

            # V5.6: Feed Waves WLM Plus meter
            if hasattr(self, 'right_wlm_meter') and self.right_wlm_meter is not None:
                self.right_wlm_meter.set_levels(
                    momentary=momentary_lufs,
                    short_term=shortterm_lufs,
                    integrated=integrated_lufs,
                    lra=lra,
                    tp_left=tp_l,
                    tp_right=tp_r,
                )

            # V5.7: Feed GR to both WLM meter and GR history widget
            gain_db_val = getattr(self, '_right_gain_db', 0.0)
            ceiling_val = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0
            if gain_db_val > 0.01 and peak_db > ceiling_val:
                gr_sim = min(gain_db_val, max(0.0, peak_db - ceiling_val) + gain_db_val * 0.2)
            else:
                gr_sim = 0.0

            if hasattr(self, 'right_gr_history') and self.right_gr_history is not None:
                self.right_gr_history.set_gr(gr_sim)

            if hasattr(self, 'right_wlm_meter') and self.right_wlm_meter is not None:
                self.right_wlm_meter.set_gr(gr_sim)

            # V5.5: Bridge to Master Module meters if window is open
            self._feed_master_module_meters(position_ms, levels_db)

        # Update time label
        pos_min = position_ms // 60000
        pos_sec = (position_ms % 60000) // 1000
        total_ms = self.timeline.total_duration_ms
        total_min = total_ms // 60000
        total_sec = (total_ms % 60000) // 1000
        time_text = f"{pos_min:02d}:{pos_sec:02d} / {total_min:02d}:{total_sec:02d}"
        self.time_label.setText(time_text)
        # V5.5: Also update transport bar position label
        if hasattr(self, 'tl_position_label'):
            self.tl_position_label.setText(time_text)

    def _on_duration_changed(self, duration_ms: int):
        pass

    # V5.5.2: Maximizer Gain — apply gain to AUDIO DATA (not QAudioOutput which caps at 1.0)
    def _on_right_gain_changed(self, value: int):
        """Gain dial changed (0-200 → 0.0-20.0 dB).
        1) Instantly updates AudioAnalysisEngine gain → meter shows gained values
        2) Debounced: creates gained temp WAV → reloads player for REAL louder audio
        """
        gain_db = value / 10.0
        self._right_gain_db = gain_db

        # INSTANT: Update AudioAnalysisEngine gain → meter shows correct levels immediately
        ceiling = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0
        if hasattr(self, 'audio_engine'):
            self.audio_engine.set_gain(gain_db, ceiling)

        # DEBOUNCED: Apply gain to actual audio file for real louder playback
        if not hasattr(self, '_gain_apply_timer'):
            self._gain_apply_timer = QTimer()
            self._gain_apply_timer.setSingleShot(True)
            self._gain_apply_timer.timeout.connect(self._apply_gain_preview)
        self._gain_apply_timer.start(400)

        # Update display
        if hasattr(self, 'right_gain_display'):
            self.right_gain_display.setText(f"+{gain_db:.1f}")
            if gain_db < 6.0:
                color = "#00CED1"
            elif gain_db < 12.0:
                color = "#FFD700"
            elif gain_db < 16.0:
                color = "#FF8C00"
            else:
                color = "#FF4444"
            self.right_gain_display.setStyleSheet(f"color: {color};")

    def _on_right_ceiling_changed(self, value: float):
        """Output ceiling changed — update engine gain and re-render preview."""
        gain_db = getattr(self, '_right_gain_db', 0.0)
        if hasattr(self, 'audio_engine'):
            self.audio_engine.set_gain(gain_db, value)
        # Trigger preview re-render if gain > 0
        if gain_db > 0.01:
            if not hasattr(self, '_gain_apply_timer'):
                self._gain_apply_timer = QTimer()
                self._gain_apply_timer.setSingleShot(True)
                self._gain_apply_timer.timeout.connect(self._apply_gain_preview)
            self._gain_apply_timer.start(400)

    def _on_right_irc_mode_changed(self, mode_name: str):
        """V5.7: IRC mode changed — update sub-mode, trigger re-render with new limiter."""
        if _HAS_PRESETS:
            sub_modes = get_irc_sub_modes(mode_name)
            if sub_modes:
                self.right_irc_submode.clear()
                for sm in sub_modes:
                    self.right_irc_submode.addItem(sm)
                self.right_irc_submode.setCurrentText("Balanced")
                self.right_irc_submode_widget.setVisible(True)
            else:
                self.right_irc_submode_widget.setVisible(False)

        # Sync to Master Module if open
        if hasattr(self, '_master_window') and self._master_window is not None:
            mw = self._master_window
            if hasattr(mw, 'chain') and hasattr(mw.chain, 'maximizer'):
                try:
                    mw.chain.maximizer.set_irc_mode(mode_name)
                except Exception:
                    pass

        # Re-render with new IRC mode if gain > 0 or preset active
        self._trigger_master_rerender()

    def _on_right_mastering_preset_changed(self, preset_name: str):
        """V5.7: Mastering preset changed — apply REAL processing through MasterChain."""
        if not _HAS_PRESETS or preset_name == "— None —":
            # Reset if deselected
            if hasattr(self, '_gained_preview_active') and self._gained_preview_active:
                gain_db = getattr(self, '_right_gain_db', 0.0)
                if gain_db <= 0.01:
                    self._apply_gain_preview()  # Will reset to original
            return

        preset = get_mastering_preset(preset_name)
        if not preset:
            return

        # Update gain dial from preset if user hasn't set custom gain
        mx = preset.get("maximizer", {})
        if mx and getattr(self, '_right_gain_db', 0.0) < 0.01:
            preset_gain = mx.get("gain", 0.0)
            if preset_gain > 0 and hasattr(self, 'right_gain_dial'):
                self.right_gain_dial.setValue(int(preset_gain * 10))

        # Update IRC mode from preset
        if mx and "irc_mode" in mx and hasattr(self, 'right_irc_combo'):
            idx = self.right_irc_combo.findText(mx["irc_mode"])
            if idx >= 0:
                self.right_irc_combo.blockSignals(True)
                self.right_irc_combo.setCurrentIndex(idx)
                self.right_irc_combo.blockSignals(False)

        # Sync to Master Module if open
        if hasattr(self, '_master_window') and self._master_window is not None:
            mw = self._master_window
            if hasattr(mw, '_mastering_combo'):
                idx = mw._mastering_combo.findText(preset_name)
                if idx >= 0:
                    mw._mastering_combo.setCurrentIndex(idx)

        # Trigger REAL processing
        self._trigger_master_rerender()

    def _trigger_master_rerender(self):
        """V5.7: Debounced trigger for MasterChain re-render."""
        if not hasattr(self, '_master_rerender_timer'):
            self._master_rerender_timer = QTimer()
            self._master_rerender_timer.setSingleShot(True)
            self._master_rerender_timer.timeout.connect(self._apply_gain_preview)
        self._master_rerender_timer.start(500)

    def _get_right_panel_chain(self):
        """V5.7: Get or create MasterChain for right panel real-time processing."""
        if hasattr(self, '_right_chain') and self._right_chain is not None:
            return self._right_chain
        try:
            from modules.master import MasterChain
            self._right_chain = MasterChain()
            print("[MASTER] Right panel MasterChain created")
            return self._right_chain
        except Exception as e:
            print(f"[MASTER] Cannot create chain: {e}")
            self._right_chain = None
            return None

    def _sync_right_panel_to_chain(self):
        """V5.7: Sync all right-panel controls to the MasterChain.
        Works with both Rust proxy (rust_chain.py) and Python chain (chain.py).
        """
        chain = self._get_right_panel_chain()
        if chain is None:
            return

        gain_db = getattr(self, '_right_gain_db', 0.0)
        ceiling = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0

        # Maximizer: use proxy methods (works for both Rust and Python)
        chain.maximizer.set_gain(gain_db)
        chain.maximizer.set_ceiling(ceiling)

        # IRC Mode
        if hasattr(self, 'right_irc_combo'):
            irc_mode = self.right_irc_combo.currentText()
            sub_mode = "Balanced"
            if hasattr(self, 'right_irc_submode') and self.right_irc_submode_widget.isVisible():
                sub_mode = self.right_irc_submode.currentText() or "Balanced"
            chain.maximizer.set_irc_mode(irc_mode, sub_mode)

        # Mastering Preset (dynamics + imager)
        if hasattr(self, 'right_mastering_preset') and _HAS_PRESETS:
            preset_name = self.right_mastering_preset.currentText()
            if preset_name != "— None —":
                preset = get_mastering_preset(preset_name)
                if preset:
                    # Dynamics — use proxy methods if available, else direct access
                    comp = preset.get("compressor", {})
                    if comp:
                        dyn = chain.dynamics
                        if hasattr(dyn, 'set_threshold'):
                            dyn.set_threshold(comp.get("threshold", -16))
                            dyn.set_ratio(comp.get("ratio", 2.5))
                            dyn.set_attack(comp.get("attack", 10))
                            dyn.set_release(comp.get("release", 100))
                            dyn.set_makeup_gain(comp.get("makeup", 2.0))
                        elif hasattr(dyn, 'single_band'):
                            dyn.single_band.threshold = comp.get("threshold", -16)
                            dyn.single_band.ratio = comp.get("ratio", 2.5)
                            dyn.single_band.attack = comp.get("attack", 10)
                            dyn.single_band.release = comp.get("release", 100)
                            dyn.single_band.makeup = comp.get("makeup", 2.0)
                            dyn.enabled = True

                    # Imager
                    img = preset.get("imager", {})
                    if img:
                        chain.imager.set_width(int(img.get("width", 100)))

                    # Maximizer overrides from preset
                    mx = preset.get("maximizer", {})
                    if mx:
                        if "gain" in mx and gain_db < 0.01:
                            chain.maximizer.set_gain(mx["gain"])
                        if "ceiling" in mx:
                            chain.maximizer.set_ceiling(mx["ceiling"])
                        if "irc_mode" in mx:
                            chain.maximizer.set_irc_mode(mx["irc_mode"])

    def _apply_gain_preview(self):
        """V5.7: Process current track through REAL MasterChain (Dynamics + Imager + Maximizer).
        Uses the full mastering chain for actual audio processing, not just gain.
        Falls back to simple gain+limiter if chain unavailable.
        """
        try:
            gain_db = getattr(self, '_right_gain_db', 0.0)
            has_preset = (hasattr(self, 'right_mastering_preset') and
                         self.right_mastering_preset.currentText() != "— None —")

            # Need either gain or preset to process
            if gain_db <= 0.01 and not has_preset:
                # Reset to original file
                if hasattr(self, '_gained_preview_active') and self._gained_preview_active:
                    original = self.audio_engine._current_file if hasattr(self, 'audio_engine') else None
                    if original and os.path.exists(original):
                        self.audio_player.player.stop()
                        self.audio_player.player.setSource(QUrl.fromLocalFile(original))
                        self._gained_preview_active = False
                        was_playing = self.audio_player.is_playing
                        current_pos = self.audio_player.player.position()
                        QTimer.singleShot(150, lambda: self._restore_after_gain(current_pos, was_playing))
                        print(f"[MASTER] Reset to original file")
                return

            if not hasattr(self, 'audio_engine') or not self.audio_engine._has_soundfile:
                return

            # Auto-load current track if not yet in memory
            if self.audio_engine._current_data is None:
                current_file = None
                if hasattr(self, 'audio_player') and self.audio_player.files:
                    idx = self.audio_player.current_file_index
                    if 0 <= idx < len(self.audio_player.files):
                        current_file = self.audio_player.files[idx]
                elif hasattr(self, 'audio_files') and self.audio_files:
                    current_file = self.audio_files[0].path
                if current_file and os.path.exists(current_file):
                    self.audio_engine.load_file(current_file)
                    print(f"[MASTER] Auto-loaded: {os.path.basename(current_file)}")
                else:
                    return
            if self.audio_engine._current_data is None:
                return

            was_playing = self.audio_player.is_playing
            current_pos = self.audio_player.player.position()

            import tempfile
            temp_dir = os.path.join(tempfile.gettempdir(), "longplay_gain")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "master_preview.wav")

            # Try MasterChain for REAL processing
            chain = self._get_right_panel_chain()
            processed = False

            if chain is not None:
                try:
                    self._sync_right_panel_to_chain()
                    original_file = self.audio_engine._current_file
                    if original_file and os.path.exists(original_file):
                        chain.load_audio(original_file)
                        result = chain.render(output_path=temp_path)
                        if result and os.path.exists(result):
                            processed = True
                            print(f"[MASTER] Full chain render → {os.path.basename(result)}")
                except Exception as e:
                    print(f"[MASTER] Chain render failed, fallback to gain: {e}")

            # Fallback: simple gain + limiter
            if not processed:
                gained_data, sr = self.audio_engine.get_gained_audio()
                if gained_data is None:
                    return
                sf = self.audio_engine._sf
                sf.write(temp_path, gained_data, sr, subtype='PCM_16')
                print(f"[MASTER] Fallback gain-only → {os.path.basename(temp_path)}")

            # Hot-swap player source
            self.audio_player.player.stop()
            self.audio_player.player.setSource(QUrl.fromLocalFile(temp_path))
            self._gained_preview_active = True
            self._gained_preview_path = temp_path
            QTimer.singleShot(200, lambda: self._restore_after_gain(current_pos, was_playing))

        except Exception as e:
            print(f"[MASTER] Preview error: {e}")
            import traceback
            traceback.print_exc()

    def _restore_after_gain(self, position_ms: int, was_playing: bool):
        """Restore playback position and state after gain preview reload."""
        try:
            self.audio_player.player.setPosition(position_ms)
            if was_playing:
                self.audio_player.player.play()
        except Exception as e:
            print(f"[GAIN] Restore error: {e}")

    # V5.5: Timeline Transport Bar — Play/Pause toggle
    def _on_tl_play_pause(self):
        """Toggle play/pause from timeline transport bar."""
        if self.audio_player.is_playing:
            self.audio_player.pause()
        else:
            self.audio_player.play()

    def _on_play_state_changed(self, is_playing: bool):
        self.timeline.setPlaying(is_playing)
        # V5.5: Update transport bar button text
        if hasattr(self, 'tl_play_btn'):
            self.tl_play_btn.setText("⏸  PAUSE" if is_playing else "▶  PLAY")

        # V5.5: RT engine removed — QMediaPlayer always at full volume
        # No dual audio output issue anymore (Offline-Only architecture)

        if is_playing:
            self.meter.start()
            self.video_preview.play()
            # Reset LUFS tracking for new playback
            self._lufs_short_buf = []
            self._lufs_integrated_sum = 0.0
            self._lufs_integrated_count = 0
            self._lufs_max = -70.0
            self._lufs_min = 0.0
        else:
            self.meter.stop()
            self.video_preview.pause()

    def _on_track_changed_for_meter(self, index: int, filename: str):
        """V5.5: Load new track into AudioAnalysisEngine when player switches tracks."""
        if hasattr(self, 'audio_engine') and hasattr(self, 'audio_files'):
            if 0 <= index < len(self.audio_files):
                path = self.audio_files[index].path
                self.audio_engine.load_file(path)
                print(f"[METER] Track changed → loaded: {filename}")
                # V5.5.2: Re-apply gain preview for new track if gain > 0
                gain_db = getattr(self, '_right_gain_db', 0.0)
                if gain_db > 0.01:
                    QTimer.singleShot(300, self._apply_gain_preview)
                # V5.5: Also load into Master Module if window is open
                if hasattr(self, '_master_window') and self._master_window is not None:
                    try:
                        mw = self._master_window
                        if hasattr(mw, 'set_audio'):
                            mw.set_audio(path)
                    except Exception:
                        pass

    def _on_audio_track_selected(self, row: int):
        """V5.9: Auto-sync selected audio track to Master Module."""
        if not hasattr(self, 'audio_files') or not self.audio_files:
            return
        if 0 <= row < len(self.audio_files):
            path = self.audio_files[row].path
            if os.path.exists(path):
                if hasattr(self, '_master_window') and self._master_window is not None:
                    try:
                        self._master_window.set_audio(path)
                    except Exception:
                        pass
                if hasattr(self, 'audio_player'):
                    try:
                        self.audio_player.play_index(row)
                    except Exception:
                        pass

    def _feed_master_module_meters(self, position_ms: int, levels_db: dict):
        """V5.5: Send audio levels to Master Module's meters during playback.

        V5.5 Offline-Only: No RT engine anymore. This sends basic level data
        for input metering only. POST-processing meter values come from
        the chain._send_meter() callback during offline rendering.
        """
        if not hasattr(self, '_master_window') or self._master_window is None:
            return
        if levels_db is None:
            return

        try:
            mw = self._master_window
            if hasattr(mw, '_on_meter_data'):
                meter_levels = {
                    "stage": "playback",
                    "left_peak_db": levels_db.get("left_peak_db", -60.0),
                    "right_peak_db": levels_db.get("right_peak_db", -60.0),
                    "left_rms_db": levels_db.get("left_rms_db", -60.0),
                    "right_rms_db": levels_db.get("right_rms_db", -60.0),
                    "lufs_momentary": levels_db.get("left_rms_db", -70.0),
                    "lufs_short_term": levels_db.get("right_rms_db", -70.0),
                    "lufs_integrated": -70.0,
                    "gain_reduction_db": 0.0,
                }
                mw._on_meter_data(meter_levels)
        except Exception:
            pass  # Non-fatal — Master window might be closing
            
    def _on_seek(self, position_ms: int):
        """Handle timeline seek"""
        self.audio_player.seek(position_ms)
        self.video_preview.seek(position_ms)
        
    def _add_audio(self):
        """Add audio files"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Audio Files", "",
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a);;All Files (*)"
        )
        self._process_audio_files(files)
        
    def _on_audio_dropped(self, files: list):
        """Handle dropped audio files"""
        self._process_audio_files(files)
        
    def _process_audio_files(self, files: list):
        """Process audio files from dialog or drag & drop"""
        try:
            if not files:
                return
            # V4.31.2 FIX: Natural sort files by filename number prefix
            # Ensures 1, 2, 3... 10, 11... 20 (not 1, 10, 11, 2, 20, 3...)
            files = sorted(files, key=_natural_sort_key)
            print(f"[AUDIO] Loaded {len(files)} files (natural sorted):")
            for i, f in enumerate(files):
                print(f"  {i+1}. {os.path.basename(f)}")

            for filepath in files:
                if not os.path.exists(filepath):
                    print(f"[AUDIO] ⚠️ File not found, skipped: {filepath}")
                    continue
                duration = self._get_duration(filepath)
                if duration is None:
                    duration = 180.0
                media = MediaFile(
                    path=filepath,
                    name=os.path.basename(filepath),
                    duration=duration,
                    lufs=-14.0,
                    file_type="audio"
                )
                self.audio_files.append(media)

                item = QListWidgetItem(f"{media.name}\n{media.duration_str}")
                self.audio_list.addItem(item)

            self._update_ui()

            # V5.1 FIX: Auto-sync audio to AI Master if it's already open
            self._sync_audio_to_master()
        except Exception as e:
            print(f"[AUDIO] Error processing files: {e}")
            import traceback
            traceback.print_exc()

    def _sync_audio_to_master(self):
        """V5.1: Send current audio to AI Master Module if it's open and has no audio."""
        try:
            if (hasattr(self, '_master_window') and self._master_window is not None
                    and self._master_window.isVisible() and self.audio_files):
                # Check if Master already has audio loaded
                if not self._master_window.chain.input_path:
                    selected_row = self.audio_list.currentRow()
                    if 0 <= selected_row < len(self.audio_files):
                        audio_path = self.audio_files[selected_row].path
                    else:
                        audio_path = self.audio_files[0].path

                    if os.path.exists(audio_path):
                        result = self._master_window.set_audio(audio_path)
                        if result:
                            print(f"[MASTER] ✅ Auto-synced audio: {os.path.basename(audio_path)}")
                        else:
                            print(f"[MASTER] ⚠️ Auto-sync failed for: {audio_path}")
        except Exception as e:
            print(f"[MASTER] Auto-sync error: {e}")

    def _add_video(self):
        """Add video files"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Video Files", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        self._process_video_files(files)
        
    def _on_video_dropped(self, files: list):
        """Handle dropped video files"""
        self._process_video_files(files)
        
    def _process_video_files(self, files: list):
        """Process video files from dialog or drag & drop"""
        try:
            if not files:
                return
            # V4.31.2 FIX: Natural sort video files too
            files = sorted(files, key=_natural_sort_key)
            for filepath in files:
                if not os.path.exists(filepath):
                    print(f"[VIDEO] ⚠️ File not found, skipped: {filepath}")
                    continue
                duration = self._get_duration(filepath)
                if duration is None:
                    duration = 180.0
                media = MediaFile(
                    path=filepath,
                    name=os.path.basename(filepath),
                    duration=duration,
                    file_type="video"
                )
                self.video_files.append(media)

                item = QListWidgetItem(f"{media.name}\n{media.duration_str}")
                item.media_file = media  # Store reference for reordering
                self.video_list.addItem(item)

            # V4.31.3 FIX: Auto-distribute videos across audio tracks when loaded
            # So all 3 videos get used instead of only video[0]
            num_videos = len(self.video_files)
            num_tracks = len(self.audio_files)
            if num_videos > 1 and num_tracks > 0:
                tracks_per_video = max(1, num_tracks // num_videos)
                for i, af in enumerate(self.audio_files):
                    af.video_assignment = min(i // tracks_per_video, num_videos - 1)
                print(f"[VIDEO] Auto-distributed {num_videos} videos across {num_tracks} tracks ({tracks_per_video} tracks/video)")
                for i, af in enumerate(self.audio_files):
                    print(f"  Track {i+1} → V{af.video_assignment+1}")

            self._update_ui()
        except Exception as e:
            print(f"[VIDEO] Error processing files: {e}")
            import traceback
            traceback.print_exc()

    def _add_gif(self):
        """Add GIF files"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add GIF Files", "",
            "GIF Files (*.gif);;All Files (*)"
        )
        self._process_gif_files(files)
        
    def _on_gif_dropped(self, files: list):
        """Handle dropped GIF files — PHASE 1: just store path, do NOTHING else"""
        print(f"[GIF DROP] ===== RECEIVED {len(files)} files =====")
        for f in files:
            print(f"[GIF DROP] File: {f}")
        # Store paths for deferred processing — NO Qt widget updates during drop!
        if not hasattr(self, '_pending_gif_paths'):
            self._pending_gif_paths = []
        self._pending_gif_paths.extend(files)
        print(f"[GIF DROP] Stored. Scheduling deferred load in 500ms...")
        QTimer.singleShot(500, self._safe_gif_load)

    def _process_gif_files(self, files: list):
        """Process GIF files from file dialog (NOT from drag-drop)"""
        print(f"[GIF DIALOG] Processing {len(files)} files via dialog...")
        if not hasattr(self, '_pending_gif_paths'):
            self._pending_gif_paths = []
        self._pending_gif_paths.extend(files)
        QTimer.singleShot(200, self._safe_gif_load)

    def _safe_gif_load(self):
        """PHASE 2: Safely load GIF files AFTER drop event is completely done"""
        paths = getattr(self, '_pending_gif_paths', [])
        if not paths:
            return
        self._pending_gif_paths = []
        print(f"[GIF LOAD] Phase 2: processing {len(paths)} files safely...")

        try:
            for filepath in paths:
                print(f"[GIF LOAD] Creating MediaFile for: {os.path.basename(filepath)}")
                gif_duration = self.timeline.total_duration_ms / 1000.0 if self.timeline.total_duration_ms > 0 else 60.0
                media = MediaFile(
                    path=filepath,
                    name=os.path.basename(filepath),
                    duration=gif_duration,
                    file_type="gif"
                )
                self.gif_files.append(media)
                item = QListWidgetItem(media.name)
                self.gif_list.addItem(item)
                print(f"[GIF LOAD] Added to list: {media.name}")

            # Update timeline
            print(f"[GIF LOAD] Updating timeline...")
            self.timeline.setTracks(self.audio_files, self.video_files, self.gif_files)
            print(f"[GIF LOAD] Timeline updated OK")

            # Load GIF overlay
            if self.gif_files:
                print(f"[GIF LOAD] Loading GIF overlay...")
                self.video_preview.setGIF(self.gif_files[0].path)
                print(f"[GIF LOAD] GIF overlay loaded OK")

            print(f"[GIF LOAD] ===== ALL DONE =====")
        except Exception as e:
            print(f"[GIF LOAD] ERROR: {e}")
            import traceback
            traceback.print_exc()
        
    def _add_logo(self):
        """Add Logo/Image files for overlay"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Logo Image", "",
            "Image Files (*.png *.jpg *.jpeg *.webp);;All Files (*)"
        )
        self._process_logo_files(files)
        
    def _on_logo_dropped(self, files: list):
        """Handle dropped logo files"""
        self._process_logo_files(files)
        
    def _process_logo_files(self, files: list):
        """Process logo files from dialog or drag & drop"""
        if not files:
            return
        for filepath in files:
            media = MediaFile(
                path=filepath,
                name=os.path.basename(filepath),
                duration=0,  # Static image
                file_type="logo"
            )
            self.logo_files.append(media)
            
            item = QListWidgetItem(f"🏷 {media.name}")
            self.logo_list.addItem(item)
            
        if self.logo_files:
            QMessageBox.information(
                self, "✅ Logo Added",
                f"Logo จะแสดงบน Video ตอน Export\n\n"
                f"📍 Position: {self.logo_position_combo.currentText()}\n"
                f"📏 Size: {self.logo_size_combo.currentText()}\n"
                f"🔲 Opacity: {self.logo_opacity_combo.currentText()}"
            )
    
    def _shuffle_videos(self):
        """Shuffle video order randomly"""
        import random
        if not self.video_files:
            QMessageBox.warning(self, "No Videos", "Please add videos first.")
            return
            
        random.shuffle(self.video_files)
        self._refresh_video_list()
        self.timeline.setTracks(self.audio_files, self.video_files)
        
    def _on_video_reordered(self, parent, start, end, destination, row):
        """Handle video list reorder via drag & drop"""
        # Rebuild video_files list from current widget order
        new_video_files = []
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            if item and hasattr(item, 'media_file'):
                new_video_files.append(item.media_file)
        
        self.video_files = new_video_files
        self.timeline.setTracks(self.audio_files, self.video_files)
        
    def _refresh_video_list(self):
        """Refresh video list display"""
        self.video_list.clear()
        for vf in self.video_files:
            item = QListWidgetItem(f"🎬 {vf.name} ({vf.duration_str})")
            item.media_file = vf  # Store reference for reordering
            item.setForeground(QColor(Colors.VIDEO_COLOR))
            self.video_list.addItem(item)
    
    def _clear_audio_files(self):
        """Clear all audio files"""
        if not self.audio_files:
            return
            
        reply = QMessageBox.question(
            self,
            "Clear Audio Files?",
            f"Remove all {len(self.audio_files)} audio files from the project?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.audio_files.clear()
            self._refresh_audio_list()
            self._refresh_track_list()
            self.timeline.setTracks(self.audio_files, self.video_files)
            
    def _clear_video_files(self):
        """Clear all video files"""
        if not self.video_files:
            return
            
        reply = QMessageBox.question(
            self,
            "Clear Video Files?",
            f"Remove all {len(self.video_files)} video files from the project?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.video_files.clear()
            self._refresh_video_list()
            self._refresh_track_list()
            self.timeline.setTracks(self.audio_files, self.video_files)

    def _auto_assign_videos(self):
        """Auto assign videos to tracks - with video fade option"""
        num_tracks = len(self.audio_files)
        num_videos = len(self.video_files)
        
        if num_tracks == 0:
            QMessageBox.warning(self, "⚠️ No Tracks", "กรุณาเพิ่มเพลงก่อน")
            return
            
        if num_videos == 0:
            QMessageBox.warning(self, "⚠️ No Videos", "กรุณาเพิ่ม Video ก่อน")
            return
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("🎬 Auto Assign Video")
        dialog.setMinimumWidth(450)
        dialog.setStyleSheet(f"background: {Colors.BG_PRIMARY}; color: {Colors.TEXT_PRIMARY};")
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        
        # Info
        info = QLabel(f"📊 มี {num_tracks} เพลง และ {num_videos} Videos")
        info.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {Colors.ACCENT};")
        layout.addWidget(info)
        
        # Tracks per video
        tpv_layout = QHBoxLayout()
        tpv_label = QLabel("🎵 จำนวนเพลงต่อ Video:")
        tpv_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        tpv_layout.addWidget(tpv_label)
        
        suggested = max(1, num_tracks // num_videos) if num_videos > 0 else 4
        
        tpv_spin = QSpinBox()
        tpv_spin.setRange(1, num_tracks)
        tpv_spin.setValue(suggested)
        tpv_spin.setStyleSheet(f"""
            QSpinBox {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
            }}
        """)
        tpv_layout.addWidget(tpv_spin)
        layout.addLayout(tpv_layout)
        
        # Suggestion
        suggest_label = QLabel(f"💡 แนะนำ: {suggested} เพลงต่อ Video")
        suggest_label.setStyleSheet(f"color: {Colors.VIDEO_COLOR}; font-size: 11px;")
        layout.addWidget(suggest_label)
        
        # ===== VIDEO FADE SECTION =====
        fade_group = QGroupBox("🎞️ Video Transition (Fade)")
        fade_group.setStyleSheet(f"""
            QGroupBox {{
                color: {Colors.TEXT_PRIMARY};
                font-weight: bold;
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }}
        """)
        fade_layout = QVBoxLayout(fade_group)
        
        # Enable checkbox
        fade_check = QCheckBox("☑️ Enable Fade Transition ระหว่าง Video")
        fade_check.setChecked(self.video_transition_enabled)
        fade_check.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        fade_layout.addWidget(fade_check)
        
        # Duration
        dur_layout = QHBoxLayout()
        dur_label = QLabel("⏱️ Fade Duration:")
        dur_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        dur_layout.addWidget(dur_label)
        
        dur_combo = QComboBox()
        dur_combo.addItems(["0.5 sec", "1.0 sec ⭐", "1.5 sec", "2.0 sec"])
        dur_combo.setCurrentIndex(1)  # Default 1.0 sec
        dur_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 5px;
            }}
        """)
        dur_layout.addWidget(dur_combo)
        fade_layout.addLayout(dur_layout)
        
        # Style
        style_layout = QHBoxLayout()
        style_label = QLabel("🎨 Fade Style:")
        style_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        style_layout.addWidget(style_label)
        
        style_combo = QComboBox()
        style_combo.addItems(["Fade ⭐", "Dissolve", "Wipe Left", "Wipe Right", "Fade to Black"])
        style_combo.setStyleSheet(dur_combo.styleSheet())
        style_layout.addWidget(style_combo)
        fade_layout.addLayout(style_layout)
        
        layout.addWidget(fade_group)
        
        # Preview area
        preview_text = QTextEdit()
        preview_text.setReadOnly(True)
        preview_text.setMaximumHeight(180)
        preview_text.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-family: 'Menlo', 'Courier New';
                font-size: 11px;
            }}
        """)
        layout.addWidget(preview_text)
        
        def update_preview():
            tracks_per = tpv_spin.value()
            fade_enabled = fade_check.isChecked()
            lines = []
            current_video = -1
            
            for i in range(num_tracks):
                video_idx = min(i // tracks_per, num_videos - 1)
                video_name = os.path.basename(self.video_files[video_idx].name)[:20]
                
                # Show transition point
                if video_idx != current_video and current_video >= 0:
                    if fade_enabled:
                        lines.append(f"  🎞️ ─── FADE ───")
                    else:
                        lines.append(f"  ✂️ ─── CUT ───")
                current_video = video_idx
                
                lines.append(f"Track {i+1:2d} → V{video_idx+1} ({video_name})")
            preview_text.setPlainText("\n".join(lines))
        
        tpv_spin.valueChanged.connect(update_preview)
        fade_check.stateChanged.connect(update_preview)
        update_preview()  # Initial preview
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 10px 20px;
            }}
        """)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("✅ Apply")
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }}
        """)
        
        def apply_assignment():
            tracks_per = tpv_spin.value()
            self.tracks_per_video = tracks_per
            
            # Save video transition settings
            self.video_transition_enabled = fade_check.isChecked()
            dur_text = dur_combo.currentText()
            self.video_transition_duration = float(dur_text.split()[0])
            style_text = style_combo.currentText()
            self.video_transition_style = style_text.split()[0].lower()
            
            # V4.25.10: Apply to BOTH track items AND audio_files
            print(f"[AUTO-ASSIGN] Applying: {tracks_per} tracks per video, {num_videos} videos, {num_tracks} tracks")
            for i in range(num_tracks):
                video_idx = min(i // tracks_per, num_videos - 1)
                
                # Update audio_files directly
                self.audio_files[i].video_assignment = video_idx
                
                # Update UI if available
                if i < self.track_list_layout.count():
                    item = self.track_list_layout.itemAt(i)
                    if item and item.widget():
                        track_item = item.widget()
                        if hasattr(track_item, 'video_combo'):
                            track_item.update_video_options(num_videos)
                            track_item.video_combo.setCurrentIndex(video_idx)
                
                print(f"[AUTO-ASSIGN] Track {i+1} → Video {video_idx+1}")
            
            print(f"[AUTO-ASSIGN] Done! Verifying...")
            # Verify assignments from audio_files
            for i, track in enumerate(self.audio_files):
                print(f"[VERIFY] Track {i+1} = V{track.video_assignment+1}")
            
            dialog.accept()
            
            fade_status = "Enabled" if self.video_transition_enabled else "Disabled"
            QMessageBox.information(
                self, "✅ Done!",
                f"Video assignment เสร็จสิ้น!\n\n"
                f"🎵 {num_tracks} tracks assigned\n"
                f"🎬 {num_videos} videos\n"
                f"📊 {tracks_per} tracks per video\n"
                f"🎞️ Video Fade: {fade_status}"
            )
        
        apply_btn.clicked.connect(apply_assignment)
        btn_layout.addWidget(apply_btn)
        
        layout.addLayout(btn_layout)
        dialog.exec()
        
    def _get_duration(self, filepath: str) -> float:
        """Get media duration using ffprobe"""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", filepath
            ], capture_output=True, text=True, timeout=15)
            return float(result.stdout.strip())
        except subprocess.TimeoutExpired:
            print(f"[ERROR] ffprobe timeout for {filepath}")
            return 180.0  # Default 3 minutes
        except Exception:
            return 180.0  # Default 3 minutes
            
    def _update_ui(self):
        """Update all UI elements"""
        # Update timeline
        self.timeline.setTracks(self.audio_files, self.video_files, self.gif_files)
        
        # Update video preview - load video first, then GIF as overlay
        if self.video_files:
            self.video_preview.setVideos(self.video_files)

        # V4.31.2: Pass audio context to video preview for track-synced video switching
        crossfade_val = self.crossfade_slider.value() if hasattr(self, 'crossfade_slider') else 5
        self.video_preview.set_audio_context(self.audio_files, crossfade_val)
        
        # Load GIF as OVERLAY (on top of video)
        # Update GIF duration to span full timeline
        if self.gif_files:
            gif_duration = self.timeline.total_duration_ms / 1000.0 if self.timeline.total_duration_ms > 0 else 60.0
            for gf in self.gif_files:
                gf.duration = gif_duration
            self.video_preview.setGIF(self.gif_files[0].path)
            
        # Update track list
        while self.track_list_layout.count():
            item = self.track_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        for i, track in enumerate(self.audio_files):
            item = TrackListItem(i, track)
            # Update video options based on actual video count
            item.update_video_options(len(self.video_files))
            # V4.31.1 FIX: Restore video assignment from audio_files
            if hasattr(track, 'video_assignment') and track.video_assignment < len(self.video_files):
                item.video_combo.setCurrentIndex(track.video_assignment)
            # Connect preview crossfade signal
            item.previewCrossfadeRequested.connect(self._preview_crossfade)
            self.track_list_layout.addWidget(item)

        # Update audio player
        if self.audio_files:
            paths = [f.path for f in self.audio_files]
            crossfade_val = self.crossfade_slider.value() if hasattr(self, 'crossfade_slider') else 5
            self.audio_player.set_crossfade(crossfade_val)
            self.audio_player.load_files(paths)

            # V5.5: Load first track into AudioAnalysisEngine for real meter levels
            if hasattr(self, 'audio_engine') and paths:
                self.audio_engine.load_file(paths[0])

    def _update_crossfade(self, value: int):
        """Update crossfade duration"""
        self.crossfade_label.setText(f"{value} sec")
        self.timeline.setCrossfade(value)
        self.audio_player.set_crossfade(value)  # Sync crossfade to audio player for position calc
        
    def _preview_crossfade(self, track_index: int):
        """Preview crossfade zone between track_index-1 and track_index"""
        if track_index < 1 or track_index >= len(self.audio_files):
            return
            
        # Calculate position to seek to (end of previous track - crossfade duration)
        crossfade_sec = self.crossfade_slider.value()
        
        # Sum up duration of all tracks before this one
        position_sec = 0
        for i in range(track_index):
            if i == 0:
                position_sec += self.audio_files[i].duration
            else:
                position_sec += max(0, self.audio_files[i].duration - crossfade_sec)
        
        # Go back by crossfade duration to hear the transition start
        preview_start = max(0, position_sec - crossfade_sec - 2)  # 2 sec before crossfade
        
        # Seek audio player to this position
        position_ms = int(preview_start * 1000)
        self.audio_player.seek(position_ms)
        self.audio_player.play()
        
        # Update timeline playhead
        self.timeline.setPlayhead(position_ms)
        
        # Show notification
        track1 = self.audio_files[track_index - 1].name
        track2 = self.audio_files[track_index].name
        QMessageBox.information(
            self, "🎧 Preview Crossfade",
            f"Playing transition:\n\n{track1}\n→ {track2}\n\nCrossfade: {crossfade_sec} sec"
        )
        
    # ═══════════════════════════════════════════
    # V5.5: MASTER LONGPLAY + MASTER INDIVIDUAL
    # ═══════════════════════════════════════════

    def _master_longplay(self):
        """V5.5: Join all tracks into one WAV → open MasterPanel to master the compilation."""
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "กรุณาเพิ่มเพลงก่อน (Add audio files first)")
            return

        if len(self.audio_files) < 2:
            # Single track — just open master module directly
            self._open_master_module()
            return

        # Show progress dialog for mixing
        progress = QDialog(self)
        progress.setWindowTitle("🎛 MASTER LONGPLAY")
        progress.setFixedSize(400, 120)
        progress.setStyleSheet(f"background: {Colors.BG_PRIMARY}; color: {Colors.TEXT_PRIMARY};")
        p_layout = QVBoxLayout(progress)
        p_label = QLabel(f"🎵 Joining {len(self.audio_files)} tracks with crossfade...")
        p_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {Colors.ACCENT};")
        p_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        p_layout.addWidget(p_label)
        p_bar = QProgressBar()
        p_bar.setRange(0, 0)  # Indeterminate
        p_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                background: {Colors.BG_TERTIARY};
                height: 20px;
            }}
            QProgressBar::chunk {{
                background: {Colors.TEAL};
                border-radius: 4px;
            }}
        """)
        p_layout.addWidget(p_bar)
        progress.show()
        QApplication.processEvents()

        try:
            import tempfile, shutil, glob as glob_mod

            # V5.5 FIX: Clean up OLD temp dirs first to prevent disk fill-up
            old_temps = glob_mod.glob(os.path.join(tempfile.gettempdir(), "longplay_master_*"))
            for old_dir in old_temps:
                try:
                    shutil.rmtree(old_dir, ignore_errors=True)
                    print(f"[MASTER LONGPLAY] 🗑️ Cleaned old temp: {os.path.basename(old_dir)}")
                except Exception:
                    pass

            temp_dir = tempfile.mkdtemp(prefix="longplay_master_")
            self._longplay_temp_dir = temp_dir  # Store for cleanup later
            crossfade_sec = self.crossfade_slider.value() if hasattr(self, 'crossfade_slider') else 5
            curve = getattr(self, '_crossfade_curve', 'equal_power')

            mixed_path = self._mix_audio_with_crossfade_fast(temp_dir, crossfade_sec, curve)

            progress.close()

            if mixed_path and os.path.exists(mixed_path):
                print(f"[MASTER LONGPLAY] ✅ Mixed audio: {mixed_path}")
                self._open_master_module_with_path(mixed_path)
            else:
                # Clean up on failure
                shutil.rmtree(temp_dir, ignore_errors=True)
                QMessageBox.warning(self, "Mix Failed",
                                    "Could not join tracks. Please try again.")
        except Exception as e:
            progress.close()
            # Clean up on error
            if 'temp_dir' in locals():
                import shutil as _sh
                _sh.rmtree(temp_dir, ignore_errors=True)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error",
                                 f"Failed to join tracks for mastering:\n{e}")

    def _master_individual(self):
        """V5.5: Open MasterPanel for batch mastering individual tracks."""
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "กรุณาเพิ่มเพลงก่อน (Add audio files first)")
            return

        # Open Master Module first
        self._open_master_module()

        # Auto-trigger batch mode after window is shown
        if hasattr(self, '_master_window') and self._master_window is not None:
            QTimer.singleShot(600, self._master_window.btn_batch_render.click)

    def _open_master_module_with_path(self, audio_path: str):
        """V5.5: Open MasterPanel with a specific audio file path (e.g. joined compilation)."""
        try:
            from modules.master.ui_panel import MasterPanel

            # Create or reuse master window
            if not hasattr(self, '_master_window') or self._master_window is None:
                self._master_window = MasterPanel(ffmpeg_path="ffmpeg")
                self._master_window.setWindowTitle("🎛 LongPlay Studio — AI Master Module")
                self._master_window.setMinimumSize(1060, 700)
                self._master_window.resize(1120, 780)
                self._master_window.master_complete.connect(self._on_master_complete)
                self._master_window.master_closed.connect(self._on_master_closed)

            # Load the specified audio file
            result = self._master_window.set_audio(audio_path)
            if not result:
                QMessageBox.warning(self, "Load Error",
                                    f"Could not load audio:\n{audio_path}")

            self._master_window.show()
            self._master_window.raise_()
            self._master_window.activateWindow()
            self._set_mini_meter_opacity(0.35)

        except ImportError as e:
            QMessageBox.warning(self, "Module Not Found",
                                f"AI Master Module could not be loaded.\n\nError: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error",
                                 f"Failed to open AI Master Module:\n{e}")

    def _open_master_module(self):
        """V5.0: Open AI Master Module window"""
        try:
            from modules.master.ui_panel import MasterPanel

            # Create or show master window
            if not hasattr(self, '_master_window') or self._master_window is None:
                self._master_window = MasterPanel(ffmpeg_path="ffmpeg")
                self._master_window.setWindowTitle("🎛 LongPlay Studio — AI Master Module")
                self._master_window.setMinimumSize(1060, 700)
                self._master_window.resize(1120, 780)

                # Connect signals
                self._master_window.master_complete.connect(self._on_master_complete)
                self._master_window.master_closed.connect(self._on_master_closed)
                # V5.5 REMOVED: position_changed signal (no RT engine → no position sync)

            # V5.1 FIX: Robust audio passing — try ALL tracks until one works
            audio_loaded = False
            if self.audio_files:
                # Build list of paths to try: selected track first, then all others
                paths_to_try = []
                selected_row = self.audio_list.currentRow()
                if 0 <= selected_row < len(self.audio_files):
                    paths_to_try.append(self.audio_files[selected_row].path)

                # Add all other tracks as fallback
                for i, mf in enumerate(self.audio_files):
                    if mf.path not in paths_to_try:
                        paths_to_try.append(mf.path)

                print(f"[MASTER] Trying to load audio, {len(paths_to_try)} candidates:")
                for p in paths_to_try:
                    exists = os.path.exists(p)
                    print(f"  {'✅' if exists else '❌'} {p}")

                for audio_path in paths_to_try:
                    if os.path.exists(audio_path):
                        result = self._master_window.set_audio(audio_path)
                        if result:
                            audio_loaded = True
                            print(f"[MASTER] ✅ Audio loaded: {audio_path}")
                            break
                        else:
                            print(f"[MASTER] ⚠️ set_audio returned False for: {audio_path}")

                if not audio_loaded:
                    print("[MASTER] ❌ No audio could be loaded!")
                    QMessageBox.warning(
                        self, "Audio File Not Found",
                        f"Cannot load audio for mastering.\n\n"
                        f"Tried {len(paths_to_try)} file(s) but none could be loaded.\n"
                        f"Files may be on a disconnected drive.\n\n"
                        f"Use the BROWSE button in AI Master to select a file manually."
                    )
            else:
                print("[MASTER] ⚠️ No audio files in playlist — opening Master without audio")

            self._master_window.show()
            self._master_window.raise_()
            self._master_window.activateWindow()

            # V5.5 TASK 4: Dim Mini Meter when Master Module is open
            self._set_mini_meter_opacity(0.35)

        except ImportError as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(
                self, "Module Not Found",
                f"AI Master Module could not be loaded.\n\n"
                f"Error: {e}\n\n"
                f"Please ensure the 'modules/master/' folder is present."
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Error",
                f"Failed to open AI Master Module:\n{e}"
            )

    def _on_master_complete(self, output_path):
        """V5.0: Handle mastering complete — optionally replace audio in compilation"""
        QMessageBox.information(
            self, "Mastering Complete",
            f"Mastered audio saved to:\n{output_path}\n\n"
            f"You can now use this file in your compilation."
        )

    def _on_master_closed(self):
        """V5.5: Restore Mini Meter opacity when Master Module closes."""
        self._set_mini_meter_opacity(1.0)
        # V5.5: Clean up longplay temp directory to free disk space
        if hasattr(self, '_longplay_temp_dir') and self._longplay_temp_dir:
            try:
                import shutil
                if os.path.exists(self._longplay_temp_dir):
                    shutil.rmtree(self._longplay_temp_dir, ignore_errors=True)
                    print(f"[MASTER] 🗑️ Cleaned temp: {self._longplay_temp_dir}")
                self._longplay_temp_dir = None
            except Exception:
                pass

    def _set_mini_meter_opacity(self, opacity: float):
        """V5.5 TASK 4: Context-aware Mini Meter opacity.

        When Master Module is open, the Mini Meter dims to 0.35 opacity
        to visually indicate the full metering is available in the Master
        Module. When closed, it restores to full opacity.
        """
        try:
            if hasattr(self, 'meter'):
                effect = self.meter.graphicsEffect()
                if effect is None or not isinstance(effect, QGraphicsOpacityEffect):
                    effect = QGraphicsOpacityEffect(self.meter)
                    self.meter.setGraphicsEffect(effect)
                effect.setOpacity(opacity)

                # Also dim/restore the Waves WLM meter and GR history
                for widget in [
                    getattr(self, 'right_wlm_meter', None),
                    getattr(self, 'right_gr_history', None),
                    getattr(self, 'true_peak_label', None),
                    getattr(self, 'lufs_integrated', None),
                    getattr(self, 'lufs_lra', None),
                    getattr(self, 'lufs_target_label', None),
                ]:
                    if widget is not None:
                        w_effect = widget.graphicsEffect()
                        if w_effect is None or not isinstance(w_effect, QGraphicsOpacityEffect):
                            w_effect = QGraphicsOpacityEffect(widget)
                            widget.setGraphicsEffect(w_effect)
                        w_effect.setOpacity(opacity)
        except Exception:
            pass

    def _generate_timestamps(self):
        """Generate YouTube timestamps"""
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "Please add audio files first.")
            return
            
        crossfade = self.crossfade_slider.value()
        timestamps = []
        current_time = 0
        
        for i, track in enumerate(self.audio_files):
            # Format time
            mins = int(current_time // 60)
            secs = int(current_time % 60)
            if mins >= 60:
                hours = mins // 60
                mins = mins % 60
                time_str = f"{hours}:{mins:02d}:{secs:02d}"
            else:
                time_str = f"{mins}:{secs:02d}"
                
            # Clean track name
            name = track.name
            name = re.sub(r'^[\d]+[\.\-\s]+', '', name)  # Remove numbering
            name = os.path.splitext(name)[0]  # Remove extension
            
            timestamps.append(f"{time_str} {name}")
            
            # Calculate next start time with crossfade overlap
            if i == 0:
                current_time += track.duration
            else:
                current_time += max(0, track.duration - crossfade)
                
        # Total duration
        total_mins = int(current_time // 60)
        total_secs = int(current_time % 60)
        total_str = f"{total_mins}:{total_secs:02d}"
        
        # Show dialog
        dialog = TimestampDialog(timestamps, total_str, self)
        dialog.exec()
    
    def _show_ai_dj_dialog(self):
        """Show AI DJ dialog for smart playlist ordering"""
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "Please add audio files first.")
            return
            
        dialog = AIDJDialog(self.audio_files, self)
        dialog.orderApplied.connect(self._apply_ai_dj_order)
        dialog.exec()
        
    def _apply_ai_dj_order(self, new_order: list):
        """Apply new track order from AI DJ"""
        try:
            # Handle clear all case
            if not new_order:
                self.audio_files.clear()
                self._refresh_audio_list()
                self._refresh_track_list()
                self.timeline.setTracks(self.audio_files, self.video_files)
                QMessageBox.information(self, "Cleared", "🗑️ ลบเพลงทั้งหมดแล้ว!")
                return
                
            # Rebuild audio_files in new order
            path_to_file = {af.path: af for af in self.audio_files}
            new_audio_files = [path_to_file[p] for p in new_order if p in path_to_file]
            
            # Only update if we have files
            if new_audio_files:
                self.audio_files = new_audio_files
            
            # Update UI - refresh both list and track list
            self._refresh_audio_list()
            self._refresh_track_list()
            
            # Update timeline
            self.timeline.setTracks(self.audio_files, self.video_files)
            
            QMessageBox.information(self, "Applied", "✅ New track order applied!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply order: {str(e)}")
        
    def _show_ai_vdo_dialog(self):
        """Show AI VDO dialog for smart video ordering"""
        if not self.video_files:
            QMessageBox.warning(self, "No Video", "Please add video files first.")
            return
            
        dialog = AIVideoDialog(self.video_files, self)
        dialog.orderApplied.connect(self._apply_ai_vdo_order)
        dialog.exec()
        
    def _apply_ai_vdo_order(self, new_order: list):
        """Apply new video order from AI VDO"""
        try:
            # Handle clear all case
            if not new_order:
                self.video_files.clear()
                self._refresh_video_list()
                self._refresh_track_list()
                self.timeline.setTracks(self.audio_files, self.video_files)
                QMessageBox.information(self, "Cleared", "🗑️ ลบวิดีโอทั้งหมดแล้ว!")
                return
                
            # Rebuild video_files in new order
            path_to_file = {vf.path: vf for vf in self.video_files}
            new_video_files = [path_to_file[p] for p in new_order if p in path_to_file]
            
            # Only update if we have files
            if new_video_files:
                self.video_files = new_video_files
            
            # Update UI
            self._refresh_video_list()
            
            # Update timeline
            self.timeline.setTracks(self.audio_files, self.video_files)
            
            # Also update track list video combos
            self._refresh_track_list()
            
            QMessageBox.information(self, "Applied", "✅ New video order applied!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply order: {str(e)}")
        
    def _show_youtube_generator(self):
        """Show YouTube Generator dialog"""
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "Please add audio files first.")
            return
            
        dialog = YouTubeGeneratorDialog(self.audio_files, self)
        dialog.exec()
    
    def _show_video_prompt_dialog(self):
        """Show Video Prompt Generator dialog - Midjourney Style for meta.ai (NEW V4.26!)"""
        if not self.video_files:
            QMessageBox.warning(self, "No Video", "Please add video files first.")
            return
            
        dialog = VideoPromptDialog(self.video_files, self)
        dialog.exec()
    
    def _show_hook_extractor_dialog(self):
        """Show Hook Extractor dialog - Audio Waveform Analysis (NEW V4.26!)"""
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "Please add audio files first.")
            return

        dialog = HookExtractorDialog(self.audio_files, self)
        dialog.exec()

    def _show_lipsync_dialog(self):
        """Show Lip-Sync Avatar dialog — Dreemina API (Hook → Master → Lip-sync → YouTube Shorts)."""
        # Pre-populate with current audio files (user can add/remove in dialog)
        hook_paths = []
        for af in self.audio_files:
            path = af.path if hasattr(af, "path") else str(af)
            if os.path.isfile(path):
                hook_paths.append(path)

        dialog = LipSyncDialog(hook_paths=hook_paths, parent=self)
        dialog.exec()
        
    def _refresh_audio_list(self):
        """Refresh audio list display"""
        self.audio_list.clear()
        for af in self.audio_files:
            self.audio_list.addItem(af.name)
        
        # Update timeline
        self.timeline.setTracks(self.audio_files, self.video_files)
    
    def _refresh_track_list(self):
        """Refresh track list widget (TrackListItem widgets)"""
        # Clear existing track items
        while self.track_list_layout.count():
            item = self.track_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Rebuild track items
        for i, track in enumerate(self.audio_files):
            item = TrackListItem(i, track)
            # Update video options based on actual video count
            item.update_video_options(len(self.video_files))
            # V4.31.1 FIX: Restore video assignment from audio_files
            if hasattr(track, 'video_assignment') and track.video_assignment < len(self.video_files):
                item.video_combo.setCurrentIndex(track.video_assignment)
            # Connect preview crossfade signal
            item.previewCrossfadeRequested.connect(self._preview_crossfade)
            self.track_list_layout.addWidget(item)

        # Update audio player
        if self.audio_files:
            paths = [f.path for f in self.audio_files]
            self.audio_player.load_files(paths)

    def _apply_auto_numbers(self):
        """Add track numbers to audio file names (rename files)"""
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "Please add audio files first.")
            return
            
        # Confirm with user
        reply = QMessageBox.question(
            self, "Apply Numbers",
            f"This will rename {len(self.audio_files)} files with track numbers.\n\n"
            "Example: 'Song.wav' → '01.Song.wav'\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        renamed_count = 0
        errors = []
        
        for i, af in enumerate(self.audio_files, 1):
            old_path = af.path
            directory = os.path.dirname(old_path)
            filename = os.path.basename(old_path)
            
            # Remove existing number prefix if present
            import re
            clean_name = re.sub(r'^[\d]+[\.\-\s]+', '', filename)
            
            # Add new number prefix
            new_name = f"{i:02d}.{clean_name}"
            new_path = os.path.join(directory, new_name)
            
            try:
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    af.path = new_path
                    af.name = new_name
                    renamed_count += 1
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")
                
        # Refresh UI
        self._refresh_audio_list()
        
        # Show result
        if errors:
            QMessageBox.warning(
                self, "Partial Success",
                f"Renamed {renamed_count} files.\n\nErrors:\n" + "\n".join(errors[:5])
            )
        else:
            QMessageBox.information(
                self, "Success",
                f"✅ Renamed {renamed_count} files with track numbers!"
            )
    
    def _show_settings(self):
        """Show settings dialog with Auto Video Mode"""
        dialog = QDialog(self)
        dialog.setWindowTitle("⚙️ Settings")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet(f"background: {Colors.BG_PRIMARY}; color: {Colors.TEXT_PRIMARY};")
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Settings")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {Colors.ACCENT};")
        layout.addWidget(title)
        
        # === Auto Video Mode ===
        video_group = QGroupBox("🎬 Auto Video Mode")
        video_group.setStyleSheet(f"""
            QGroupBox {{
                color: {Colors.TEXT_PRIMARY};
                font-weight: bold;
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        video_layout = QVBoxLayout(video_group)
        
        # Auto select description
        desc = QLabel("เลือก Video ให้อัตโนมัติตามจำนวนเพลง:")
        desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        video_layout.addWidget(desc)
        
        # Mode selection
        self.auto_video_mode = QComboBox()
        self.auto_video_mode.addItems([
            "Manual (เลือกเอง)",
            "Sequential (เรียงลำดับ V1→V2→V3)",
            "Random (สุ่ม)",
            "Even Split (แบ่งเท่าๆ กัน)"
        ])
        self.auto_video_mode.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                color: {Colors.TEXT_PRIMARY};
                padding: 8px;
                border-radius: 6px;
            }}
        """)
        video_layout.addWidget(self.auto_video_mode)
        
        # Apply auto video button
        apply_btn = QPushButton("🎯 Apply Auto Video")
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        apply_btn.clicked.connect(lambda: self._apply_auto_video(dialog))
        video_layout.addWidget(apply_btn)
        
        layout.addWidget(video_group)
        
        # === Video Transition ===
        transition_group = QGroupBox("🎞️ Video Transition")
        transition_group.setStyleSheet(video_group.styleSheet())
        transition_layout = QVBoxLayout(transition_group)
        
        # Enable transition checkbox
        self.transition_enabled_check = QCheckBox("☑️ Enable Fade Transition between videos")
        self.transition_enabled_check.setChecked(self.video_transition_enabled)
        self.transition_enabled_check.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        transition_layout.addWidget(self.transition_enabled_check)
        
        # Tracks per video
        tracks_row = QHBoxLayout()
        tracks_label = QLabel("🎵 Tracks per Video:")
        tracks_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        tracks_row.addWidget(tracks_label)
        
        self.tracks_per_video_spin = QSpinBox()
        self.tracks_per_video_spin.setRange(1, 20)
        self.tracks_per_video_spin.setValue(self.tracks_per_video)
        self.tracks_per_video_spin.setStyleSheet(f"""
            QSpinBox {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                color: {Colors.TEXT_PRIMARY};
                padding: 5px;
                border-radius: 4px;
            }}
        """)
        tracks_row.addWidget(self.tracks_per_video_spin)
        
        # Suggestion label
        if len(self.audio_files) > 0 and len(self.video_files) > 0:
            suggested = max(1, len(self.audio_files) // len(self.video_files))
            suggest_label = QLabel(f"💡 แนะนำ: {suggested} tracks")
            suggest_label.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 11px;")
            tracks_row.addWidget(suggest_label)
        
        transition_layout.addLayout(tracks_row)
        
        # Transition duration - CapCut style slider
        duration_row = QHBoxLayout()
        duration_label = QLabel("⏱️ Fade Duration:")
        duration_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        duration_row.addWidget(duration_label)
        
        self.transition_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.transition_duration_slider.setRange(1, 50)  # 0.1 to 5.0 seconds (x10)
        self.transition_duration_slider.setValue(10)  # Default: 1.0 sec
        self.transition_duration_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.transition_duration_slider.setTickInterval(5)
        self.transition_duration_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {Colors.BORDER};
                height: 8px;
                background: {Colors.BG_TERTIARY};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {Colors.VIDEO_COLOR};
                border: 2px solid {Colors.VIDEO_COLOR};
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #FF6B9D;
            }}
            QSlider::sub-page:horizontal {{
                background: {Colors.VIDEO_COLOR};
                border-radius: 4px;
            }}
        """)
        self.transition_duration_slider.valueChanged.connect(self._on_fade_duration_changed)
        duration_row.addWidget(self.transition_duration_slider, 1)
        
        self.fade_duration_label = QLabel("1.0s")
        self.fade_duration_label.setStyleSheet(f"color: {Colors.VIDEO_COLOR}; font-weight: bold; min-width: 45px;")
        duration_row.addWidget(self.fade_duration_label)
        
        transition_layout.addLayout(duration_row)
        
        # Transition style
        style_row = QHBoxLayout()
        style_label = QLabel("🎨 Style:")
        style_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        style_row.addWidget(style_label)
        
        self.transition_style_combo = QComboBox()
        self.transition_style_combo.addItems([
            "Fade ⭐",
            "Dissolve",
            "Wipe Left",
            "Wipe Right",
            "Fade to Black"
        ])
        self.transition_style_combo.setCurrentIndex(0)  # Default: Fade
        self.transition_style_combo.setStyleSheet(self.auto_video_mode.styleSheet())
        style_row.addWidget(self.transition_style_combo)
        transition_layout.addLayout(style_row)
        
        # Preview assignment button
        preview_assign_btn = QPushButton("📋 Preview Assignment")
        preview_assign_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background: #3A80C9;
            }}
        """)
        preview_assign_btn.clicked.connect(self._preview_video_assignment)
        transition_layout.addWidget(preview_assign_btn)
        
        layout.addWidget(transition_group)
        
        # === GIF Overlay ===
        gif_group = QGroupBox("🖼️ GIF Overlay")
        gif_group.setStyleSheet(video_group.styleSheet())
        gif_layout = QVBoxLayout(gif_group)
        
        self.gif_overlay_check = QCheckBox("Show GIF overlay on video preview")
        self.gif_overlay_check.setChecked(True)
        self.gif_overlay_check.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        gif_layout.addWidget(self.gif_overlay_check)
        
        layout.addWidget(gif_group)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background: {Colors.BG_SECONDARY};
            }}
        """)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def _apply_auto_video(self, dialog):
        """Apply auto video selection to all tracks"""
        if not self.video_files:
            QMessageBox.warning(self, "No Videos", "Please add video files first.")
            return
            
        mode = self.auto_video_mode.currentIndex()
        num_videos = len(self.video_files)
        num_tracks = len(self.audio_files)
        
        if num_tracks == 0:
            QMessageBox.warning(self, "No Tracks", "Please add audio tracks first.")
            return
        
        # Generate video assignments
        assignments = []
        
        if mode == 0:  # Manual
            QMessageBox.information(self, "Manual Mode", "Select videos manually using V1/V2/V3 dropdowns.")
            return
        elif mode == 1:  # Sequential
            for i in range(num_tracks):
                assignments.append(i % num_videos)
        elif mode == 2:  # Random
            for i in range(num_tracks):
                assignments.append(random.randint(0, num_videos - 1))
        elif mode == 3:  # Even Split
            tracks_per_video = max(1, num_tracks // num_videos)
            for i in range(num_tracks):
                assignments.append(min(i // tracks_per_video, num_videos - 1))
        
        # V4.31.2 FIX: Apply to BOTH track items AND audio_files (was UI-only before!)
        print(f"[AUTO-VIDEO] Applying {len(assignments)} assignments to {num_tracks} tracks")
        for i in range(num_tracks):
            if i < len(assignments):
                # Save to data model (CRITICAL - was missing before!)
                self.audio_files[i].video_assignment = assignments[i]
                print(f"[AUTO-VIDEO] Track {i+1} → V{assignments[i]+1}")

            # Update UI if available
            if i < self.track_list_layout.count():
                item = self.track_list_layout.itemAt(i)
                if item and item.widget():
                    track_item = item.widget()
                    if hasattr(track_item, 'video_combo'):
                        track_item.update_video_options(num_videos)
                        if i < len(assignments):
                            track_item.video_combo.setCurrentIndex(assignments[i])

        # Verify assignments
        for i, track in enumerate(self.audio_files):
            print(f"[AUTO-VIDEO VERIFY] Track {i+1} = V{track.video_assignment+1}")

        dialog.accept()
        QMessageBox.information(self, "✅ Auto Video Applied",
            f"Video assigned to {num_tracks} tracks automatically!\n"
            f"🎬 {num_videos} videos used")
    
    def _on_fade_duration_changed(self, value: int):
        """Update fade duration label when slider changes"""
        duration = value / 10.0  # Convert to seconds
        self.fade_duration_label.setText(f"{duration:.1f}s")
    
    def _preview_video_assignment(self):
        """Preview video assignment with transition points"""
        num_tracks = len(self.audio_files)
        num_videos = len(self.video_files)
        
        if num_tracks == 0:
            QMessageBox.warning(self, "No Tracks", "Please add audio tracks first.")
            return
        if num_videos == 0:
            QMessageBox.warning(self, "No Videos", "Please add video files first.")
            return
        
        # Get settings
        tracks_per_video = self.tracks_per_video_spin.value()
        transition_enabled = self.transition_enabled_check.isChecked()
        # Get duration from slider (value is x10)
        duration = self.transition_duration_slider.value() / 10.0
        style_text = self.transition_style_combo.currentText()
        
        # Generate preview
        preview_lines = []
        preview_lines.append("📋 VIDEO ASSIGNMENT PREVIEW\n")
        preview_lines.append(f"🎵 Total Tracks: {num_tracks}")
        preview_lines.append(f"🎬 Total Videos: {num_videos}")
        preview_lines.append(f"📊 Tracks per Video: {tracks_per_video}\n")
        preview_lines.append("─" * 35 + "\n")
        
        transition_points = []
        current_video = 0
        
        for i in range(num_tracks):
            video_idx = min(i // tracks_per_video, num_videos - 1)
            track_name = self.audio_files[i].name[:30]
            
            # Check for transition point
            if video_idx != current_video:
                transition_points.append(i)
                current_video = video_idx
                preview_lines.append(f"\n🎞️ ── TRANSITION ({style_text.split()[0]} {duration}s) ──\n")
            
            preview_lines.append(f"Track {i+1:2d} → V{video_idx+1}  │ {track_name}")
        
        preview_lines.append("\n" + "─" * 35)
        if transition_enabled:
            preview_lines.append(f"\n✨ Transitions: {len(transition_points)} points")
            preview_lines.append(f"🎨 Style: {style_text}")
            preview_lines.append(f"⏱️ Duration: {duration}s each")
        else:
            preview_lines.append("\n⚠️ Transitions: DISABLED (hard cuts)")
        
        # Show preview dialog
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("📋 Video Assignment Preview")
        preview_dialog.setMinimumSize(450, 500)
        preview_dialog.setStyleSheet(f"background: {Colors.BG_PRIMARY}; color: {Colors.TEXT_PRIMARY};")
        
        layout = QVBoxLayout(preview_dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setText("\n".join(preview_lines))
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 15px;
                font-family: 'Menlo', 'Courier New';
                font-size: 12px;
            }}
        """)
        layout.addWidget(text_edit)
        
        # Apply button
        apply_btn = QPushButton("✅ Apply This Assignment")
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        apply_btn.clicked.connect(lambda: self._apply_video_assignment_from_preview(
            tracks_per_video, transition_enabled, duration, style_text, preview_dialog
        ))
        layout.addWidget(apply_btn)
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        close_btn.clicked.connect(preview_dialog.accept)
        layout.addWidget(close_btn)
        
        preview_dialog.exec()
    
    def _apply_video_assignment_from_preview(self, tracks_per_video, transition_enabled, duration, style, dialog):
        """Apply video assignment from preview"""
        num_videos = len(self.video_files)
        num_tracks = len(self.audio_files)
        
        # Save settings
        self.tracks_per_video = tracks_per_video
        self.video_transition_enabled = transition_enabled
        self.video_transition_duration = duration
        self.video_transition_style = style.split()[0].lower()  # "Fade ⭐" -> "fade"
        
        # V4.25.10: Apply to BOTH track items AND audio_files
        for i in range(num_tracks):
            video_idx = min(i // tracks_per_video, num_videos - 1)
            
            # Update audio_files directly
            self.audio_files[i].video_assignment = video_idx
            
            # Update UI if available
            if i < self.track_list_layout.count():
                item = self.track_list_layout.itemAt(i)
                if item and item.widget():
                    track_item = item.widget()
                    if hasattr(track_item, 'video_combo'):
                        track_item.video_combo.setCurrentIndex(video_idx)
        
        print(f"[APPLY-ASSIGN] Applied to {num_tracks} tracks")
        dialog.accept()
        QMessageBox.information(self, "✅ Applied", 
            f"Video assignment applied!\n\n"
            f"🎵 {num_tracks} tracks assigned\n"
            f"🎞️ Transition: {style} ({duration}s)")
    
    def _update_quality_description(self, index):
        """Update quality description based on selection"""
        descriptions = [
            "🚀 Quick preview • ~5-10 min export • ~500 MB",
            "⚖️ Good for YouTube • ~15-25 min export • ~1.5 GB",
            "💎 Best quality • ~45-60 min export • ~3-4 GB",
        ]
        if hasattr(self, 'quality_desc'):
            self.quality_desc.setText(descriptions[index])
        
    def _export_video(self):
        """Export separate video and audio files"""
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "Please add audio files first.")
            return
            
        # Allow audio-only export (no video required)
        has_any_video = bool(self.video_files) or bool(self.gif_files)

        # Create export dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("🚀 Export Settings")
        dialog.setMinimumWidth(450)
        dialog.setStyleSheet(f"background: {Colors.BG_PRIMARY}; color: {Colors.TEXT_PRIMARY};")
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Export Settings")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {Colors.ACCENT};")
        layout.addWidget(title)
        
        # Filename input
        name_group = QGroupBox("📝 Filename")
        name_group.setStyleSheet(f"""
            QGroupBox {{
                color: {Colors.TEXT_PRIMARY};
                font-weight: bold;
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        name_layout = QVBoxLayout(name_group)
        
        name_label = QLabel("ตั้งชื่อไฟล์ (ไม่ต้องใส่นามสกุล):")
        name_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        name_layout.addWidget(name_label)
        
        self.export_name_input = QLineEdit()
        self.export_name_input.setPlaceholderText("e.g. My_LongPlay_Mix")
        self.export_name_input.setText("LongPlay_Mix")
        self.export_name_input.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                color: {Colors.TEXT_PRIMARY};
                padding: 10px;
                border-radius: 6px;
                font-size: 14px;
            }}
        """)
        name_layout.addWidget(self.export_name_input)
        
        layout.addWidget(name_group)
        
        # Quality Mode - NEW!
        quality_group = QGroupBox("⚡ Video Quality")
        quality_group.setStyleSheet(name_group.styleSheet())
        quality_layout = QVBoxLayout(quality_group)
        
        # Quality dropdown
        quality_row = QHBoxLayout()
        quality_label = QLabel("Quality Mode:")
        quality_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 13px;")
        quality_row.addWidget(quality_label)
        
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["🚀 Fast", "⚖️ Balanced", "💎 Quality"])
        self.quality_combo.setCurrentIndex(1)  # Default: Balanced
        self.quality_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 150px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                selection-background-color: {Colors.ACCENT};
            }}
        """)
        quality_row.addWidget(self.quality_combo, 1)
        quality_layout.addLayout(quality_row)
        
        # Quality description
        self.quality_desc = QLabel("⚖️ Good for YouTube • ~15-25 min export • ~1.5 GB")
        self.quality_desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; padding-left: 5px;")
        quality_layout.addWidget(self.quality_desc)
        
        # Connect signal to update description
        self.quality_combo.currentIndexChanged.connect(self._update_quality_description)
        
        layout.addWidget(quality_group)
        
        # Export Options - NEW!
        options_group = QGroupBox("🎛️ Export Options")
        options_group.setStyleSheet(name_group.styleSheet())
        options_layout = QVBoxLayout(options_group)
        
        self.export_audio_check = QCheckBox("🎵 Export Audio (WAV 24-bit)")
        self.export_audio_check.setChecked(True)
        self.export_audio_check.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 13px; padding: 5px;")
        options_layout.addWidget(self.export_audio_check)
        
        self.export_video_check = QCheckBox("📹 Export Video (MP4)")
        if has_any_video:
            self.export_video_check.setChecked(True)
        else:
            self.export_video_check.setChecked(False)
            self.export_video_check.setEnabled(False)
        self.export_video_check.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 13px; padding: 5px;")
        options_layout.addWidget(self.export_video_check)

        # Combined export option (video + audio in one file)
        self.export_combined_check = QCheckBox("🎬 Combined MP4 (Video + Audio รวมไฟล์เดียว)")
        if has_any_video:
            self.export_combined_check.setChecked(True)
        else:
            self.export_combined_check.setChecked(False)
            self.export_combined_check.setEnabled(False)
        self.export_combined_check.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 13px; padding: 5px; font-weight: bold;")
        self.export_combined_check.setToolTip("Export ไฟล์ MP4 ที่มีทั้ง Video + Audio รวมกัน พร้อมอัพ YouTube/Social ได้เลย")
        options_layout.addWidget(self.export_combined_check)

        combined_hint = QLabel("💡 ได้ไฟล์ MP4 พร้อมอัพ YouTube/TikTok/IG ได้เลย!")
        combined_hint.setStyleSheet(f"color: {Colors.METER_GREEN}; font-size: 11px; padding-left: 20px;")
        options_layout.addWidget(combined_hint)

        # Speed hint
        speed_hint = QLabel("💡 Export แค่ Audio จะเร็วกว่ามาก (~10 วินาที)")
        speed_hint.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; padding-left: 20px;")
        options_layout.addWidget(speed_hint)
        
        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet(f"background: {Colors.BORDER};")
        options_layout.addWidget(sep1)
        
        # Batch Export Mode (NEW V4.26!)
        self.batch_export_check = QCheckBox("📦 Batch Export - Export แยกแต่ละไฟล์")
        self.batch_export_check.setChecked(False)
        self.batch_export_check.setStyleSheet(f"color: {Colors.VIDEO_COLOR}; font-size: 13px; padding: 5px;")
        self.batch_export_check.setToolTip("Export แต่ละ Video/Audio แยกกัน แทนที่จะรวมเป็นไฟล์เดียว")
        options_layout.addWidget(self.batch_export_check)
        
        batch_hint = QLabel("💡 เลือก 5 ไฟล์ = ได้ 5 ไฟล์ output (ไม่รวมกัน)")
        batch_hint.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; padding-left: 20px;")
        options_layout.addWidget(batch_hint)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"background: {Colors.BORDER};")
        options_layout.addWidget(sep2)
        
        # Seamless Loop Mode (NEW V4.26!)
        self.seamless_loop_check = QCheckBox("🔄 Seamless Loop - Video ต่อเนื่องไม่มีรอยต่อ")
        self.seamless_loop_check.setChecked(False)
        self.seamless_loop_check.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 13px; padding: 5px;")
        self.seamless_loop_check.setToolTip("ใช้ Crossfade ทำให้ Video Loop ต่อเนื่องเนียนๆ")
        options_layout.addWidget(self.seamless_loop_check)
        
        seamless_hint = QLabel("💡 ใช้ Crossfade 1 วินาที ทำให้ Loop ไม่มีรอยต่อ")
        seamless_hint.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; padding-left: 20px;")
        options_layout.addWidget(seamless_hint)
        
        layout.addWidget(options_group)
        
        # Output format info
        format_group = QGroupBox("📦 Output Files")
        format_group.setStyleSheet(name_group.styleSheet())
        format_layout = QVBoxLayout(format_group)
        
        format_info = QLabel(
            "🎬 [ชื่อ].mp4 - Combined (Video + Audio) พร้อมอัพได้เลย\n"
            "📹 [ชื่อ]_video.mp4 - Video only (ไม่มีเสียง)\n"
            "🎵 [ชื่อ]_audio.wav - Audio only (24-bit, 48kHz)\n\n"
            "💡 เลือก Combined เพื่อได้ไฟล์เดียวพร้อมใช้งาน"
        )
        format_info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; line-height: 1.5;")
        format_info.setWordWrap(True)
        format_layout.addWidget(format_info)
        
        layout.addWidget(format_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 12px 24px;
            }}
            QPushButton:hover {{
                background: {Colors.BG_SECONDARY};
            }}
        """)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        export_btn = QPushButton("🚀 Export")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        export_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(export_btn)
        
        layout.addLayout(btn_layout)
        
        # Show dialog
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # Get export options
        export_audio = self.export_audio_check.isChecked()
        export_video = self.export_video_check.isChecked()
        export_combined = self.export_combined_check.isChecked()  # NEW V5.6!
        batch_export = self.batch_export_check.isChecked()  # NEW V4.26!
        seamless_loop = self.seamless_loop_check.isChecked()  # NEW V4.26!
        
        # Set quality mode based on selection
        quality_index = self.quality_combo.currentIndex()
        quality_modes = ["fast", "balanced", "quality"]
        set_quality_mode(quality_modes[quality_index])
        
        if not export_audio and not export_video and not export_combined:
            QMessageBox.warning(self, "⚠️ Warning", "กรุณาเลือกอย่างน้อย 1 อย่าง (Audio, Video หรือ Combined)")
            return

        # If combined is checked, we need both audio and video processing
        if export_combined:
            export_video = True  # Force video processing for combined
        
        # V4.27: Check if all files still exist before export
        missing_files = []
        for af in self.audio_files:
            if not os.path.exists(af.path):
                missing_files.append(f"🎵 {os.path.basename(af.path)}")
        for vf in self.video_files:
            if not os.path.exists(vf.path):
                missing_files.append(f"📹 {os.path.basename(vf.path)}")
        
        if missing_files:
            missing_list = "\n".join(missing_files[:10])  # Show first 10
            if len(missing_files) > 10:
                missing_list += f"\n... และอีก {len(missing_files) - 10} ไฟล์"
            
            reply = QMessageBox.question(
                self, "⚠️ ไฟล์หายไป!",
                f"พบไฟล์ที่หายไป {len(missing_files)} ไฟล์:\n\n{missing_list}\n\n"
                "อาจเกิดจาก:\n"
                "• External drive ถูกถอดออก\n"
                "• ไฟล์ถูกย้ายหรือลบ\n"
                "• ชื่อโฟลเดอร์มีอักขระพิเศษ\n\n"
                "ต้องการลบไฟล์ที่หายไปออกจาก playlist แล้ว export ต่อหรือไม่?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Remove missing files from lists
                self.audio_files = [af for af in self.audio_files if os.path.exists(af.path)]
                self.video_files = [vf for vf in self.video_files if os.path.exists(vf.path)]
                self._refresh_track_list()
                
                if not self.audio_files:
                    QMessageBox.warning(self, "❌ Error", "ไม่มีไฟล์ Audio เหลือให้ export")
                    return
            else:
                return
        
        # NEW V4.26: Batch Export Mode
        if batch_export:
            self._do_batch_export(export_audio, export_video, seamless_loop)
            return
            
        # Get filename
        base_name = self.export_name_input.text().strip()
        if not base_name:
            base_name = "LongPlay_Mix"
        
        # Sanitize filename
        base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
            
        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not output_dir:
            return
        
        # Store seamless loop setting for use in export
        self._seamless_loop_enabled = seamless_loop
            
        # Generate filenames
        video_path = os.path.join(output_dir, f"{base_name}_video.mp4") if export_video else None
        audio_path = os.path.join(output_dir, f"{base_name}_audio.wav") if export_audio else None
        combined_path = os.path.join(output_dir, f"{base_name}.mp4") if export_combined else None
        
        # Show progress dialog with detailed status
        self.export_dialog = QDialog(self)
        self.export_dialog.setWindowTitle("🚀 Exporting...")
        self.export_dialog.setMinimumWidth(500)
        self.export_dialog.setWindowFlags(self.export_dialog.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.export_dialog.setStyleSheet(f"background: {Colors.BG_PRIMARY}; color: {Colors.TEXT_PRIMARY};")
        progress_layout = QVBoxLayout(self.export_dialog)
        progress_layout.setSpacing(15)
        
        # Title
        if export_combined:
            export_type = "Combined MP4" + (" + Separate Files" if (export_audio or (export_video and not export_combined)) else "")
        else:
            export_type = "Audio + Video" if (export_audio and export_video) else ("Audio Only" if export_audio else "Video Only")
        title_label = QLabel(f"🚀 Exporting ({export_type})")
        title_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {Colors.ACCENT};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(title_label)
        
        # Step indicator
        self.step_label = QLabel("Step 1/4: Preparing...")
        self.step_label.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_PRIMARY}; font-weight: bold;")
        progress_layout.addWidget(self.step_label)
        
        # Status text
        self.status_label = QLabel("Initializing export process...")
        self.status_label.setStyleSheet(f"font-size: 12px; color: {Colors.TEXT_SECONDARY};")
        self.status_label.setWordWrap(True)
        progress_layout.addWidget(self.status_label)
        
        # Progress bar
        self.export_progress = QProgressBar()
        self.export_progress.setMinimum(0)
        self.export_progress.setMaximum(100)
        self.export_progress.setTextVisible(True)
        self.export_progress.setFormat("%p%")
        self.export_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {Colors.BORDER};
                border-radius: 8px;
                background: {Colors.BG_TERTIARY};
                text-align: center;
                color: {Colors.TEXT_PRIMARY};
                font-weight: bold;
                height: 25px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Colors.ACCENT}, stop:1 {Colors.ACCENT_DIM});
                border-radius: 6px;
            }}
        """)
        progress_layout.addWidget(self.export_progress)
        
        # Time info frame
        time_frame = QFrame()
        time_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_TERTIARY};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        time_layout = QHBoxLayout(time_frame)
        
        # Elapsed time
        elapsed_container = QVBoxLayout()
        elapsed_title = QLabel("⏱️ Elapsed")
        elapsed_title.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        elapsed_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        elapsed_container.addWidget(elapsed_title)
        
        self.elapsed_label = QLabel("0:00")
        self.elapsed_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 16px; font-weight: bold;")
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        elapsed_container.addWidget(self.elapsed_label)
        time_layout.addLayout(elapsed_container)
        
        # Separator
        sep = QLabel("|")
        sep.setStyleSheet(f"color: {Colors.BORDER}; font-size: 20px;")
        time_layout.addWidget(sep)
        
        # Remaining time
        remaining_container = QVBoxLayout()
        remaining_title = QLabel("⏳ Est. Remaining")
        remaining_title.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        remaining_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        remaining_container.addWidget(remaining_title)
        
        self.remaining_label = QLabel("Calculating...")
        self.remaining_label.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 16px; font-weight: bold;")
        self.remaining_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        remaining_container.addWidget(self.remaining_label)
        time_layout.addLayout(remaining_container)
        
        progress_layout.addWidget(time_frame)
        
        # Speed info - show encoder being used
        encoder_info = f"🚀 GPU: {HW_ENCODER}" if HW_ENCODER != "libx264" else "⚡ CPU: libx264"
        speed_label = QLabel(f"{encoder_info} + Multi-threading")
        speed_label.setStyleSheet(f"color: {Colors.METER_GREEN}; font-size: 11px;")
        speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(speed_label)
        
        self.export_dialog.show()
        QApplication.processEvents()
        
        # Start timer for elapsed time
        self.export_start_time = time.time()
        self._progress_history = []  # Reset progress history for accurate time estimation
        self.export_timer = QTimer()
        self.export_timer.timeout.connect(self._update_export_time)
        self.export_timer.start(1000)  # Update every second
        
        try:
            self._do_export_separate(video_path, audio_path, output_dir, export_video, export_audio, combined_path)
            self.export_timer.stop()
            self.export_dialog.close()
            
            # Success dialog with open folder option
            elapsed = time.time() - self.export_start_time
            elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
            
            # Build success message
            files_msg = []
            if video_path:
                files_msg.append(f"📹 Video: {os.path.basename(video_path)}")
            if audio_path:
                files_msg.append(f"🎵 Audio: {os.path.basename(audio_path)}")
            
            result = QMessageBox(self)
            result.setWindowTitle("✅ Export Complete!")
            result.setIcon(QMessageBox.Icon.Information)
            result.setText(f"Export สำเร็จ! ใช้เวลา {elapsed_str}\n\n" + "\n".join(files_msg))
            result.setInformativeText(f"📂 Saved to: {output_dir}")
            
            result.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
            open_btn = result.addButton("📂 Open Folder", QMessageBox.ButtonRole.ActionRole)
            
            result.exec()
            
            if result.clickedButton() == open_btn:
                # Open folder in file manager
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", output_dir])
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["explorer", output_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", output_dir])
                    
        except Exception as e:
            self.export_timer.stop()
            self.export_dialog.close()
            error_msg = str(e)
            # V4.31.2: Better error message for disk space issues
            if "No space left on device" in error_msg or "Errno 28" in error_msg:
                import shutil as _sh
                try:
                    usage = _sh.disk_usage(output_dir)
                    free_gb = usage.free / (1024**3)
                    total_gb = usage.total / (1024**3)
                except Exception:
                    free_gb = 0
                    total_gb = 0
                QMessageBox.critical(self, "❌ Export Failed - Disk Full",
                    f"ไม่มีพื้นที่ดิสก์เพียงพอ (No space left on device)\n\n"
                    f"พื้นที่ว่างเหลือ: {free_gb:.1f} GB / {total_gb:.1f} GB\n\n"
                    f"วิธีแก้:\n"
                    f"1. ลบไฟล์ที่ไม่ใช้ เพื่อเพิ่มพื้นที่ว่าง\n"
                    f"2. ใช้ External Drive ที่มีพื้นที่มากกว่า\n"
                    f"3. ล้าง Temp files: ไป Finder → Go → ~/Library/Caches/\n\n"
                    f"แนะนำ: ต้องการพื้นที่ว่างอย่างน้อย 5-10 GB สำหรับ Export")
            else:
                QMessageBox.critical(self, "❌ Export Failed", f"Error:\n{error_msg}")

    def _update_export_time(self):
        """Update elapsed and estimated remaining time
        
        Fixed: Use progress history for more accurate time estimation
        """
        elapsed = time.time() - self.export_start_time
        elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
        self.elapsed_label.setText(elapsed_str)
        
        # Get current progress
        progress = self.export_progress.value()
        
        # Initialize progress history if not exists
        if not hasattr(self, '_progress_history'):
            self._progress_history = []
        
        # Record progress point
        current_time = time.time()
        if not self._progress_history or self._progress_history[-1][1] != progress:
            self._progress_history.append((current_time, progress))
        
        # Keep only last 10 data points to avoid memory issues
        if len(self._progress_history) > 10:
            self._progress_history = self._progress_history[-10:]
        
        # Calculate remaining time
        if progress >= 95:
            self.remaining_label.setText("Almost done...")
        elif progress > 5 and len(self._progress_history) >= 2:
            # Use recent progress rate for better estimation
            # Get oldest and newest points
            t1, p1 = self._progress_history[0]
            t2, p2 = self._progress_history[-1]
            
            time_diff = t2 - t1
            progress_diff = p2 - p1
            
            if progress_diff > 0 and time_diff > 0:
                # Calculate rate: seconds per percent
                rate = time_diff / progress_diff
                remaining_progress = 100 - progress
                remaining = remaining_progress * rate

                # Cap at reasonable maximum (24 hours)
                remaining = min(remaining, 86400)
                
                if remaining < 60:
                    remaining_str = f"~{int(remaining)}s"
                else:
                    remaining_str = f"~{int(remaining // 60)}:{int(remaining % 60):02d}"
                self.remaining_label.setText(remaining_str)
            else:
                # Fallback to simple calculation
                if progress > 0:
                    total_estimated = elapsed / (progress / 100)
                    remaining = max(0, total_estimated - elapsed)
                    if remaining < 60:
                        remaining_str = f"~{int(remaining)}s"
                    else:
                        remaining_str = f"~{int(remaining // 60)}:{int(remaining % 60):02d}"
                    self.remaining_label.setText(remaining_str)
        elif progress > 0:
            self.remaining_label.setText("Calculating...")
            
    def _update_export_status(self, step: int, total: int, status: str, progress_val: int):
        """Update export status display"""
        self.step_label.setText(f"Step {step}/{total}: {status}")
        self.status_label.setText(status)
        self.export_progress.setValue(progress_val)
        QApplication.processEvents()
    
    def _do_batch_export(self, export_audio: bool, export_video: bool, seamless_loop: bool):
        """NEW V4.26: Batch Export - Export each file separately instead of merging
        
        This exports each video/audio file individually without combining them.
        If user selects 5 files, they get 5 output files.
        """
        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Batch Export", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not output_dir:
            return
        
        # V4.27: Check if all files still exist before batch export
        missing_audio = [af for af in self.audio_files if not os.path.exists(af.path)]
        missing_video = [vf for vf in self.video_files if not os.path.exists(vf.path)]
        
        if missing_audio or missing_video:
            missing_count = len(missing_audio) + len(missing_video)
            missing_names = [f"🎵 {af.name}" for af in missing_audio[:5]] + [f"📹 {vf.name}" for vf in missing_video[:5]]
            missing_list = "\n".join(missing_names)
            if missing_count > 10:
                missing_list += f"\n... และอีก {missing_count - 10} ไฟล์"
            
            reply = QMessageBox.question(
                self, "⚠️ ไฟล์หายไป!",
                f"พบไฟล์ที่หายไป {missing_count} ไฟล์:\n\n{missing_list}\n\n"
                "ต้องการข้ามไฟล์ที่หายไปแล้ว export เฉพาะไฟล์ที่ยังอยู่หรือไม่?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Remove missing files from lists
                self.audio_files = [af for af in self.audio_files if os.path.exists(af.path)]
                self.video_files = [vf for vf in self.video_files if os.path.exists(vf.path)]
                self._refresh_track_list()
            else:
                return
        
        # Count total files to export
        video_count = len(self.video_files) if export_video else 0
        audio_count = len(self.audio_files) if export_audio else 0
        total_files = video_count + audio_count
        
        if total_files == 0:
            QMessageBox.warning(self, "No Files", "ไม่มีไฟล์ให้ Export")
            return
        
        # Show progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("📦 Batch Exporting...")
        progress_dialog.setMinimumWidth(500)
        progress_dialog.setStyleSheet(f"background: {Colors.BG_PRIMARY}; color: {Colors.TEXT_PRIMARY};")
        layout = QVBoxLayout(progress_dialog)
        
        title = QLabel(f"📦 Batch Export: {total_files} ไฟล์")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Colors.ACCENT};")
        layout.addWidget(title)
        
        status_label = QLabel("กำลังเตรียม...")
        status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        layout.addWidget(status_label)
        
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(total_files)
        progress_bar.setValue(0)
        progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background: {Colors.ACCENT};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(progress_bar)
        
        result_text = QTextEdit()
        result_text.setReadOnly(True)
        result_text.setMaximumHeight(200)
        result_text.setStyleSheet(f"background: {Colors.BG_TERTIARY}; color: {Colors.TEXT_PRIMARY}; border: 1px solid {Colors.BORDER};")
        layout.addWidget(result_text)
        
        progress_dialog.show()
        QApplication.processEvents()
        
        exported_files = []
        failed_files = []
        current = 0
        
        # Export Videos
        if export_video and self.video_files:
            for i, video_file in enumerate(self.video_files):
                current += 1
                status_label.setText(f"🎬 Exporting Video {i+1}/{video_count}: {video_file.name}")
                progress_bar.setValue(current)
                QApplication.processEvents()
                
                try:
                    # Generate output filename
                    base_name = os.path.splitext(video_file.name)[0]
                    base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
                    output_path = os.path.join(output_dir, f"{base_name}_exported.mp4")
                    
                    # Check if seamless loop is enabled
                    if seamless_loop:
                        self._export_video_seamless(video_file.path, output_path)
                    else:
                        # Simple copy/re-encode
                        cmd = [
                            "ffmpeg", "-y", "-i", video_file.path,
                            "-c:v", "copy", "-an",  # Copy video, no audio
                            output_path
                        ]
                        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
                    
                    exported_files.append(output_path)
                    result_text.append(f"✅ {video_file.name} -> {os.path.basename(output_path)}")
                except Exception as e:
                    failed_files.append((video_file.name, str(e)))
                    result_text.append(f"❌ {video_file.name}: {str(e)[:50]}")
                
                QApplication.processEvents()
        
        # Export Audios
        if export_audio and self.audio_files:
            for i, audio_file in enumerate(self.audio_files):
                current += 1
                status_label.setText(f"🎵 Exporting Audio {i+1}/{audio_count}: {audio_file.name}")
                progress_bar.setValue(current)
                QApplication.processEvents()
                
                try:
                    # Generate output filename
                    base_name = os.path.splitext(audio_file.name)[0]
                    base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
                    output_path = os.path.join(output_dir, f"{base_name}_exported.wav")
                    
                    # Export as high-quality WAV
                    cmd = [
                        "ffmpeg", "-y", "-i", audio_file.path,
                        "-c:a", "pcm_s24le", "-ar", "48000",
                        output_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
                    
                    exported_files.append(output_path)
                    result_text.append(f"✅ {audio_file.name} -> {os.path.basename(output_path)}")
                except Exception as e:
                    failed_files.append((audio_file.name, str(e)))
                    result_text.append(f"❌ {audio_file.name}: {str(e)[:50]}")
                
                QApplication.processEvents()
        
        # Complete
        status_label.setText(f"✅ เสร็จสิ้น! Export {len(exported_files)}/{total_files} ไฟล์")
        
        # Add close button
        close_btn = QPushButton("✅ ปิด")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
            }}
        """)
        close_btn.clicked.connect(progress_dialog.accept)
        layout.addWidget(close_btn)
        
        # Add open folder button
        open_btn = QPushButton("📂 เปิดโฟลเดอร์")
        open_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 10px 20px;
            }}
        """)
        open_btn.clicked.connect(lambda: subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", output_dir]))
        layout.addWidget(open_btn)
        
        progress_dialog.exec()
    
    def _export_video_seamless(self, input_path: str, output_path: str):
        """NEW V4.26: Export video with seamless loop using crossfade

        Creates a seamless loop by crossfading the end back to the beginning.
        """
        # V4.31.1: Use smart temp directory selection
        temp_dir = get_smart_temp_dir()
        
        try:
            # Get video duration
            duration = self._get_video_duration(input_path)
            
            # Crossfade duration (1 second for smooth transition)
            xfade_duration = 1.0
            
            if duration < xfade_duration * 3:
                # Video too short for seamless loop, just copy
                shutil.copy2(input_path, output_path)
                return
            
            # Create seamless loop using xfade filter
            # This crossfades the end of the video with the beginning
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-filter_complex", f"""
                    [0:v]split=2[v1][v2];
                    [v1]trim=0:{duration - xfade_duration},setpts=PTS-STARTPTS[head];
                    [v2]trim={duration - xfade_duration}:{duration},setpts=PTS-STARTPTS[tail];
                    [0:v]trim=0:{xfade_duration},setpts=PTS-STARTPTS[start];
                    [tail][start]xfade=transition=fade:duration={xfade_duration}:offset=0[xfaded];
                    [head][xfaded]concat=n=2:v=1:a=0[out]
                """.replace("\n", "").replace("  ", ""),
                "-map", "[out]",
                "-an",  # No audio
            ] + get_encoder_params() + [output_path]
            
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            
        except Exception as e:
            print(f"[SEAMLESS] Error: {e}, falling back to simple copy")
            # Fallback to simple copy
            shutil.copy2(input_path, output_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def _do_export_separate(self, video_output: str, audio_output: str, output_dir: str,
                           export_video: bool = True, export_audio: bool = True,
                           combined_output: str = None):
        """Export separate video (no audio), audio (crossfaded), and/or combined MP4 files"""
        # Check disk space before starting export
        try:
            usage = shutil.disk_usage(output_dir)
            free_bytes = usage.free
            required_bytes = 2 * 1024 * 1024 * 1024  # 2GB
            if free_bytes < required_bytes:
                free_gb = free_bytes / (1024 * 1024 * 1024)
                print(f"[WARNING] Insufficient disk space: {free_gb:.1f}GB available, 2GB required")
                self.export_finished.emit(False, "Insufficient disk space (minimum 2GB required)")
                return
        except Exception as e:
            print(f"[WARNING] Could not check disk space: {e}")

        crossfade_sec = self.crossfade_slider.value()
        crossfade_style = self.crossfade_style.currentText().lower().replace(" ", "_")

        # Determine total steps based on what we're exporting
        total_steps = 1  # Audio mixing always needed
        if export_audio:
            total_steps += 1
        if export_video:
            total_steps += 2  # Process + Encode video
        if combined_output:
            total_steps += 1  # Mux video + audio into combined MP4

        current_step = 0

        # Map style to FFmpeg curve
        curve_map = {
            "linear": "tri",
            "exponential": "exp",
            "equal_power": "esin",
            "s-curve": "hsin"
        }
        curve = curve_map.get(crossfade_style, "esin")

        # V4.31.1: Use smart temp directory selection
        temp_dir = get_smart_temp_dir()

        try:
            current_step += 1
            self._update_export_status(current_step, total_steps, "🎵 Mixing audio with crossfade...", 5)
            
            # ========== STEP 1: Mix audio with crossfade ==========
            if len(self.audio_files) > 1:
                mixed_audio = self._mix_audio_with_crossfade_fast(temp_dir, crossfade_sec, curve)
            else:
                mixed_audio = self.audio_files[0].path

            # V5.5.2: Apply Maximizer gain + brickwall ceiling to mixed audio
            gain_db = getattr(self, '_right_gain_db', 0.0)
            if gain_db > 0.01 and hasattr(self, 'audio_engine') and self.audio_engine._has_soundfile:
                ceiling_db = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0
                gain_linear = 10 ** (gain_db / 20.0)
                ceiling_linear = 10 ** (ceiling_db / 20.0)
                try:
                    sf = self.audio_engine._sf
                    audio_data, sr = sf.read(mixed_audio, dtype='float32')
                    audio_data = audio_data * gain_linear
                    audio_data = np.clip(audio_data, -ceiling_linear, ceiling_linear)
                    gained_path = os.path.join(temp_dir, "mixed_audio_gained.wav")
                    sf.write(gained_path, audio_data, sr, subtype='PCM_24')
                    mixed_audio = gained_path
                    print(f"[EXPORT] ✅ Maximizer applied: +{gain_db:.1f} dB, ceiling {ceiling_db:.1f} dBTP")
                except Exception as e:
                    print(f"[EXPORT] ⚠️ Maximizer skip: {e}")

            self._update_export_status(current_step, total_steps, "🎵 Audio mixing complete!", 25)
                
            # Get audio duration
            audio_duration = self._get_audio_duration(mixed_audio)
            
            # ========== STEP 2: Export Audio File (WAV) - if requested ==========
            if export_audio:
                current_step += 1
                self._update_export_status(current_step, total_steps, "💾 Exporting audio file (WAV 24-bit)...", 30)
                
                subprocess.run([
                    "ffmpeg", "-y", "-threads", "0",
                    "-i", mixed_audio,
                    "-c:a", "pcm_s24le",  # High quality WAV
                    "-ar", "48000",
                    audio_output
                ], check=True, capture_output=True)
                
                self._update_export_status(current_step, total_steps, "💾 Audio export complete!", 45)
            
            # ========== STEP 3: Process Video - if requested ==========
            if not export_video:
                # Skip video processing
                self._update_export_status(total_steps, total_steps, "✅ Audio-only export complete!", 100)
                return
            
            current_step += 1
            self._update_export_status(current_step, total_steps, "🎬 Processing video...", 50)
            
            has_video = len(self.video_files) > 0
            has_gif = len(self.gif_files) > 0
            has_logo = len(self.logo_files) > 0
            
            # Get crossfade for segment calculation
            crossfade_sec = self.crossfade_slider.value()
            
            # Debug video transition settings
            print(f"[EXPORT] Video transition enabled: {self.video_transition_enabled}")
            print(f"[EXPORT] Video transition duration: {self.video_transition_duration}")
            print(f"[EXPORT] Video transition style: {self.video_transition_style}")
            
            if has_video:
                # Check if multiple videos with different assignments
                assignments = self._get_video_assignments()
                unique_videos = len(set(assignments))
                use_multi_video = unique_videos > 1 and len(self.video_files) > 1
                
                print(f"[EXPORT] Assignments: {assignments}")
                print(f"[EXPORT] Unique videos used: {unique_videos}")
                print(f"[EXPORT] Use multi-video: {use_multi_video}")
                
                if use_multi_video:
                    fade_status = "with FADE" if self.video_transition_enabled else "NO fade"
                    self._update_export_status(current_step, total_steps, f"🎬 Creating multi-video ({unique_videos} videos, {fade_status})...", 55)
                    base_video = self._create_multi_video_from_assignments(temp_dir, audio_duration, crossfade_sec)
                else:
                    base_video = self.video_files[0].path
                
                if has_gif or has_logo:
                    overlay_info = []
                    if has_gif:
                        overlay_info.append("GIF")
                    if has_logo:
                        overlay_info.append("Logo")
                    self._update_export_status(current_step, total_steps, f"🎬 Adding {' + '.join(overlay_info)} overlay...", 65)
                    final_video = self._create_video_with_overlay(temp_dir, base_video, audio_duration)
                else:
                    if use_multi_video:
                        # Extend multi-video to match audio duration and add fade out
                        self._update_export_status(current_step, total_steps, "🎬 Extending video to match audio + fade out...", 60)
                        final_video = self._extend_video_with_fade(temp_dir, base_video, audio_duration)
                    else:
                        self._update_export_status(current_step, total_steps, "🎬 Looping video to match audio duration...", 55)
                        final_video = self._create_looped_video_fast(temp_dir, base_video, audio_duration)
            else:
                self._update_export_status(current_step, total_steps, "🎬 Converting GIF to video...", 55)
                final_video = self._create_looped_video_fast(temp_dir, self.gif_files[0].path, audio_duration)
            
            self._update_export_status(current_step, total_steps, "🎬 Video processing complete!", 75)
            
            # ========== STEP 4: Export Video File (NO AUDIO) — skip if only combined ==========
            if video_output:
                current_step += 1

                # V4.25.10: Check if we need to re-encode or can use stream copy
                needs_reencode = has_gif or has_logo

                if needs_reencode:
                    hw_status = "GPU" if HW_ENCODER != "libx264" else "CPU"
                    self._update_export_status(current_step, total_steps, f"📹 Encoding final video ({hw_status})...", 80)

                    cmd = [
                        "ffmpeg", "-y", "-threads", "0",
                        "-i", final_video,
                        "-an",  # NO AUDIO - separate export
                    ] + get_encoder_params() + [video_output]

                    subprocess.run(cmd, check=True, capture_output=True)
                else:
                    # Stream copy - INSTANT!
                    self._update_export_status(current_step, total_steps, "🚀 Copying video (stream copy - instant!)...", 80)

                    cmd = [
                        "ffmpeg", "-y",
                        "-i", final_video,
                        "-an",  # NO AUDIO
                        "-c", "copy",
                        video_output
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"[EXPORT] Stream copy failed, falling back to encode: {result.stderr}")
                        cmd_encode = [
                            "ffmpeg", "-y", "-threads", "0",
                            "-i", final_video,
                            "-an",
                        ] + get_encoder_params() + [video_output]
                        subprocess.run(cmd_encode, check=True, capture_output=True)
            
            # ========== STEP 5: Combined MP4 (Video + Audio) - if requested ==========
            if combined_output and mixed_audio:
                current_step += 1
                self._update_export_status(current_step, total_steps, "🎬 Muxing Video + Audio into combined MP4...", 90)

                # Use the final_video (with overlays) and mixed_audio (with gain)
                # Mux with AAC audio for maximum compatibility
                mux_cmd = [
                    "ffmpeg", "-y", "-threads", "0",
                    "-i", final_video,
                    "-i", mixed_audio,
                    "-c:v", "copy",        # Copy video stream (no re-encode!)
                    "-c:a", "aac",         # AAC audio for MP4 compatibility
                    "-b:a", "320k",        # High quality audio
                    "-ar", "48000",        # 48kHz sample rate
                    "-shortest",           # Match shortest stream
                    "-movflags", "+faststart",  # Web-optimized MP4
                    combined_output
                ]

                result = subprocess.run(mux_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"[EXPORT] Stream copy mux failed, falling back to encode: {result.stderr}")
                    # Fallback: re-encode video if stream copy fails
                    mux_cmd_encode = [
                        "ffmpeg", "-y", "-threads", "0",
                        "-i", final_video,
                        "-i", mixed_audio,
                    ] + get_encoder_params() + [
                        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
                        "-shortest", "-movflags", "+faststart",
                        combined_output
                    ]
                    subprocess.run(mux_cmd_encode, check=True, capture_output=True)

                combined_size = os.path.getsize(combined_output) / (1024 * 1024)
                print(f"[EXPORT] ✅ Combined MP4: {combined_output} ({combined_size:.1f} MB)")

            self._update_export_status(current_step, total_steps, "✅ Export complete!", 100)

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _mix_audio_with_crossfade_fast(self, temp_dir: str, crossfade_sec: int, curve: str) -> str:
        """Mix multiple audio files with crossfade - Batch approach for speed
        
        Process tracks in batches of 5, NO overlap between batches.
        Then combine batches with crossfade.
        
        Result: 5-6 writes instead of 19 = 2x faster, 75% less disk
        """
        import gc
        
        if len(self.audio_files) < 2:
            return self.audio_files[0].path
        
        all_paths = [af.path for af in self.audio_files]
        total_tracks = len(all_paths)

        # V4.31.3: Log exact track order for debugging
        print(f"[AUDIO MIX] Track order ({total_tracks} tracks):")
        for i, p in enumerate(all_paths):
            print(f"  {i+1}. {os.path.basename(p)}")

        # For small number of tracks, use simple approach
        BATCH_SIZE = 5
        if total_tracks <= BATCH_SIZE:
            print(f"[AUDIO] Simple mixing {total_tracks} tracks (≤{BATCH_SIZE})")
            return self._mix_batch_crossfade(temp_dir, all_paths, crossfade_sec, curve, "mixed_audio.wav")
        
        print(f"[AUDIO] Batch mixing {total_tracks} tracks ({BATCH_SIZE} per batch)")
        
        # Split into batches - NO OVERLAP
        batches = []
        for i in range(0, total_tracks, BATCH_SIZE):
            batch = all_paths[i:i + BATCH_SIZE]
            if len(batch) >= 2:
                batches.append(batch)
            elif len(batch) == 1 and batches:
                # Single leftover - add to previous batch
                batches[-1].append(batch[0])
        
        print(f"[AUDIO] Created {len(batches)} batches")
        
        # Mix each batch
        batch_outputs = []
        for idx, batch in enumerate(batches):
            gc.collect()
            output_name = f"batch_{idx}.wav"
            print(f"[AUDIO] Batch {idx+1}/{len(batches)}: {len(batch)} tracks")
            
            try:
                batch_output = self._mix_batch_crossfade(temp_dir, batch, crossfade_sec, curve, output_name)
                batch_outputs.append(batch_output)
                print(f"[AUDIO] ✅ Batch {idx+1} complete")
            except Exception as e:
                print(f"[AUDIO] ⚠️ Batch {idx+1} failed: {e}")
                # Fallback: concat without crossfade
                batch_output = self._concat_batch(temp_dir, batch, output_name)
                batch_outputs.append(batch_output)
                print(f"[AUDIO] ✅ Batch {idx+1} complete (concat fallback)")
        
        # V4.31.2 FIX: Combine batches WITH crossfade (not hard concat!)
        # This fixes the abrupt cut at batch boundaries (track 5→6, 10→11, etc.)
        if len(batch_outputs) == 1:
            final_output = os.path.join(temp_dir, "mixed_audio.wav")
            os.rename(batch_outputs[0], final_output)
            return final_output

        # Verify all batch files exist
        valid_batches = []
        for batch_path in batch_outputs:
            if os.path.exists(batch_path):
                valid_batches.append(batch_path)
            else:
                print(f"[AUDIO] Warning: Batch file not found: {batch_path}")

        if not valid_batches:
            raise Exception("No valid batch files to combine")

        if len(valid_batches) == 1:
            os.rename(valid_batches[0], os.path.join(temp_dir, "mixed_audio.wav"))
            return os.path.join(temp_dir, "mixed_audio.wav")

        print(f"[AUDIO] Combining {len(valid_batches)} batches WITH crossfade ({crossfade_sec}s)...")

        # Use crossfade between batches for smooth transitions
        final_output = os.path.join(temp_dir, "mixed_audio.wav")

        try:
            # Build crossfade chain for batches (same approach as _mix_batch_crossfade)
            inputs = []
            for bp in valid_batches:
                inputs.extend(["-i", bp])

            n = len(valid_batches)
            if n == 2:
                filter_complex = f"[0:a][1:a]acrossfade=d={crossfade_sec}:c1={curve}:c2={curve}[out]"
            else:
                parts = []
                parts.append(f"[0:a][1:a]acrossfade=d={crossfade_sec}:c1={curve}:c2={curve}[a0]")
                for i in range(2, n):
                    prev = f"[a{i-2}]"
                    curr = f"[{i}:a]"
                    out = "[out]" if i == n-1 else f"[a{i-1}]"
                    parts.append(f"{prev}{curr}acrossfade=d={crossfade_sec}:c1={curve}:c2={curve}{out}")
                filter_complex = ";".join(parts)

            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-c:a", "pcm_s24le",
                "-ar", "48000",
                final_output
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir, timeout=300)

            if result.returncode != 0:
                raise Exception(f"Batch crossfade failed: {result.stderr[:300]}")

            print(f"[AUDIO] ✅ Batch crossfade complete")

        except Exception as e:
            print(f"[AUDIO] ⚠️ Batch crossfade failed, falling back to concat: {e}")
            # Fallback: plain concat (better than crashing)
            concat_list = os.path.join(temp_dir, "concat_batches.txt")
            with open(concat_list, "w") as f:
                for batch_path in valid_batches:
                    f.write(f"file '{_ffmpeg_escape_path(os.path.basename(batch_path))}'\n")

            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c:a", "pcm_s24le",
                "-ar", "48000",
                final_output
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
            if result.returncode != 0:
                raise Exception(f"FFmpeg batch concat failed: {result.stderr}")
        
        # Cleanup batch files AFTER successful concat
        for batch_path in valid_batches:
            try: os.remove(batch_path)
            except Exception: pass
        
        print(f"[AUDIO] ✅ Batch mixing complete: {total_tracks} tracks merged")
        return final_output
    
    def _mix_batch_crossfade(self, temp_dir: str, audio_paths: list, crossfade_sec: int, curve: str, output_name: str) -> str:
        """Mix a batch of audio files (up to 5) with crossfade chain
        
        V4.25.11: Use FFmpeg stream copy instead of symlinks for better compatibility
        with external volumes and Unicode paths on macOS
        """
        if len(audio_paths) < 2:
            return audio_paths[0]
        
        # V4.31.3 FIX: Use hash-based unique name per file to avoid batch collision!
        # OLD BUG: audio_000.wav reused across batches = songs repeat/mix wrong
        safe_paths = []
        for i, path in enumerate(audio_paths):
            import hashlib
            path_hash = hashlib.md5(path.encode()).hexdigest()[:8]
            safe_name = f"audio_{path_hash}_{i:03d}.wav"
            safe_path = os.path.join(temp_dir, safe_name)

            if not os.path.exists(safe_path):
                print(f"[AUDIO] Normalizing file {i+1}: {os.path.basename(path)}")
                success = False

                # V5.5.1 FIX: Re-encode to uniform format (48kHz stereo WAV)
                # acrossfade REQUIRES identical sample rate + channel layout.
                # Using -c copy preserves original format → mismatch → exit 234!
                try:
                    cmd = [
                        "ffmpeg", "-y", "-i", path,
                        "-ar", "48000", "-ac", "2",
                        "-c:a", "pcm_s24le",
                        "-rf64", "auto",
                        safe_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    if result.returncode == 0 and os.path.exists(safe_path):
                        success = True
                        print(f"[AUDIO] Normalized OK: {os.path.basename(path)}")
                    else:
                        print(f"[AUDIO] Normalize failed: {result.stderr[:200] if result.stderr else 'unknown'}")
                except Exception as e:
                    print(f"[AUDIO] Normalize error: {e}")

                # Fallback: shutil.copy2 (format may differ → crossfade might fail)
                if not success:
                    try:
                        import shutil
                        shutil.copy2(path, safe_path)
                        if os.path.exists(safe_path):
                            success = True
                            print(f"[AUDIO] shutil.copy2 fallback OK: {os.path.basename(path)}")
                    except Exception as e2:
                        print(f"[AUDIO] shutil.copy2 failed: {e2}")

                if not success:
                    print(f"[AUDIO] All methods failed for {path}")
                    continue

            if os.path.exists(safe_path):
                safe_paths.append(safe_path)
            else:
                print(f"[AUDIO] Warning: File not created: {safe_path}")

        if len(safe_paths) < 2:
            if safe_paths:
                return safe_paths[0]
            raise Exception(f"Not enough valid audio files (got {len(safe_paths)}, need at least 2)")

        print(f"[AUDIO] Successfully prepared {len(safe_paths)} files for mixing (all 48kHz stereo)")
        
        inputs = []
        for path in safe_paths:
            inputs.extend(["-i", path])
        
        # Build filter chain for batch
        # V4.31.3 FIX: use safe_paths count (not audio_paths) in case some copies failed
        n = len(safe_paths)
        if n == 2:
            filter_complex = f"[0:a][1:a]acrossfade=d={crossfade_sec}:c1={curve}:c2={curve}[out]"
        else:
            parts = []
            parts.append(f"[0:a][1:a]acrossfade=d={crossfade_sec}:c1={curve}:c2={curve}[a0]")
            for i in range(2, n):
                prev = f"[a{i-2}]"
                curr = f"[{i}:a]"
                out = "[out]" if i == n-1 else f"[a{i-1}]"
                parts.append(f"{prev}{curr}acrossfade=d={crossfade_sec}:c1={curve}:c2={curve}{out}")
            filter_complex = ";".join(parts)
        
        output_path = os.path.join(temp_dir, output_name)
        
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:a", "pcm_s24le",
            "-ar", "48000",
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    
    def _concat_batch(self, temp_dir: str, audio_paths: list, output_name: str) -> str:
        """Concat audio files without crossfade (fallback)
        
        V4.25.11: Use FFmpeg stream copy instead of symlinks
        """
        # V4.31.3 FIX: Use hash-based unique name per file to avoid batch collision
        safe_paths = []
        for i, path in enumerate(audio_paths):
            import hashlib
            path_hash = hashlib.md5(path.encode()).hexdigest()[:8]
            safe_name = f"concat_{path_hash}_{i:03d}.wav"
            safe_path = os.path.join(temp_dir, safe_name)

            if not os.path.exists(safe_path):
                print(f"[AUDIO] Normalizing file {i+1}: {os.path.basename(path)}")
                success = False

                # V5.5.1: Normalize to uniform format for reliable concat
                try:
                    cmd = [
                        "ffmpeg", "-y", "-i", path,
                        "-ar", "48000", "-ac", "2",
                        "-c:a", "pcm_s24le",
                        "-rf64", "auto",
                        safe_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    if result.returncode == 0 and os.path.exists(safe_path):
                        success = True
                        print(f"[AUDIO] Normalized OK: {os.path.basename(path)}")
                except Exception as e:
                    print(f"[AUDIO] Normalize error: {e}")

                # Fallback: shutil.copy2
                if not success:
                    try:
                        import shutil
                        shutil.copy2(path, safe_path)
                        if os.path.exists(safe_path):
                            success = True
                            print(f"[AUDIO] shutil.copy2 fallback OK: {os.path.basename(path)}")
                    except Exception as e2:
                        print(f"[AUDIO] shutil.copy2 failed: {e2}")

                if not success:
                    print(f"[AUDIO] All methods failed for {path}")
                    continue
            
            if os.path.exists(safe_path):
                safe_paths.append(safe_path)
            else:
                print(f"[AUDIO] Warning: File not created: {safe_path}")
        
        if not safe_paths:
            raise Exception(f"No valid audio files to concat (tried {len(audio_paths)} files)")
        
        if len(safe_paths) == 1:
            # Only one file, just copy it
            output_path = os.path.join(temp_dir, output_name)
            shutil.copy2(safe_paths[0], output_path)
            return output_path
        
        concat_list = os.path.join(temp_dir, f"concat_{output_name}.txt")
        with open(concat_list, "w") as f:
            for path in safe_paths:
                # Use relative path for concat file
                f.write(f"file '{_ffmpeg_escape_path(os.path.basename(path))}'\n")
        
        output_path = os.path.join(temp_dir, output_name)
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list, "-c:a", "pcm_s24le", "-ar", "48000",
            output_path
        ], capture_output=True, text=True, cwd=temp_dir)  # Run in temp_dir for relative paths
        
        if result.returncode != 0:
            print(f"[AUDIO] FFmpeg concat error: {result.stderr}")
            raise Exception(f"FFmpeg concat failed: {result.stderr}")
        
        return output_path
    
    def _create_looped_video_fast(self, temp_dir: str, video_path: str, duration: float, seamless: bool = False) -> str:
        """Create looped video to match duration - Ultra-fast using stream_loop
        
        V4.25.11: Simplified approach using FFmpeg's -stream_loop option
        - Single command: stream_loop + trim + stream copy
        - Tested: 70 minutes in ~1 second!
        - No intermediate files needed
        
        V4.26: Added seamless loop option using crossfade
        
        Command: ffmpeg -stream_loop N -i input.mp4 -t duration -c copy output.mp4
        """
        # Check if seamless loop is enabled
        if seamless or getattr(self, '_seamless_loop_enabled', False):
            return self._create_seamless_looped_video(temp_dir, video_path, duration)
        video_duration = self._get_video_duration(video_path)
        if video_duration <= 0:
            print(f"[VIDEO LOOP] ERROR: Could not get video duration")
            raise Exception("Could not get video duration")
        
        # Calculate loop count: need enough loops to cover target duration
        # stream_loop N means play N+1 times total
        loop_count = int(duration / video_duration) + 1
        
        print(f"[VIDEO LOOP] Source: {video_duration:.1f}s, Target: {duration:.1f}s")
        print(f"[VIDEO LOOP] Using stream_loop={loop_count} (ultra-fast, ~1 second)")
        
        self._update_export_status_safe("🚀 Fast video looping (stream copy)...", 55)
        
        # V4.26: Handle Unicode paths and external volumes using FFmpeg copy
        safe_source = os.path.join(temp_dir, "source_video.mp4")
        try:
            # Use FFmpeg stream copy (works with external volumes and Unicode paths)
            if os.path.exists(safe_source):
                os.remove(safe_source)
            copy_result = subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-c", "copy", safe_source
            ], capture_output=True, text=True, timeout=60)
            if copy_result.returncode == 0:
                print(f"[VIDEO LOOP] Using FFmpeg copy for safe path")
            else:
                # Fallback: use original path directly
                safe_source = video_path
                print(f"[VIDEO LOOP] Using original path directly")
        except Exception as e:
            print(f"[VIDEO LOOP] FFmpeg copy failed: {e}, using original path")
            safe_source = video_path
        
        looped_video = os.path.join(temp_dir, "looped.mp4")
        
        # Single command: stream_loop + trim + stream copy
        # This is the fastest method - tested at ~1 second for 70 minutes!
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", str(loop_count),
            "-i", safe_source,
            "-t", str(duration),
            "-c", "copy",  # Stream copy = instant!
            looped_video
        ]
        
        print(f"[VIDEO LOOP] Command: {' '.join(cmd)}")
        self._update_export_status_safe("🔗 Creating looped video...", 65)
        
        # Timeout: generous but reasonable (1 min per 10 min of output + 60 base)
        timeout = max(120, int(duration / 600) * 60 + 60)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                print(f"[VIDEO LOOP] FFmpeg error: {result.stderr}")
                raise Exception(result.stderr)
            print(f"[VIDEO LOOP] ✅ Stream copy complete!")
        except subprocess.TimeoutExpired:
            print(f"[VIDEO LOOP] Timeout after {timeout}s, trying fallback...")
            return self._create_looped_video_fallback(temp_dir, video_path, duration)
        except Exception as e:
            print(f"[VIDEO LOOP] Stream copy failed: {e}")
            print(f"[VIDEO LOOP] Trying fallback method...")
            return self._create_looped_video_fallback(temp_dir, video_path, duration)
        
        # Verify output exists and has correct duration
        if os.path.exists(looped_video):
            output_duration = self._get_video_duration(looped_video)
            print(f"[VIDEO LOOP] Output duration: {output_duration:.1f}s (target: {duration:.1f}s)")
            if output_duration < duration * 0.9:  # Allow 10% tolerance
                print(f"[VIDEO LOOP] WARNING: Output shorter than expected!")
        
        # Cleanup
        try:
            if os.path.islink(safe_source):
                os.unlink(safe_source)
            elif os.path.exists(safe_source):
                os.remove(safe_source)
        except Exception:
            pass
        
        print(f"[VIDEO LOOP] ✅ Video looping complete!")
        return looped_video
    
    def _create_seamless_looped_video(self, temp_dir: str, video_path: str, duration: float) -> str:
        """NEW V4.26: Create seamless looped video using crossfade transitions
        
        This creates a video that loops smoothly without visible cuts.
        Uses xfade filter to blend the end of each loop with the beginning of the next.
        """
        video_duration = self._get_video_duration(video_path)
        if video_duration <= 0:
            print(f"[SEAMLESS LOOP] ERROR: Could not get video duration")
            raise Exception("Could not get video duration")
        
        # Crossfade duration (1 second for smooth transition)
        xfade_duration = 1.0
        
        # Calculate how many complete loops we need
        effective_loop_duration = video_duration - xfade_duration  # Account for crossfade overlap
        loop_count = int(duration / effective_loop_duration) + 2
        
        print(f"[SEAMLESS LOOP] Source: {video_duration:.1f}s, Target: {duration:.1f}s")
        print(f"[SEAMLESS LOOP] Creating {loop_count} seamless loops with {xfade_duration}s crossfade")
        
        self._update_export_status_safe("🔄 Creating seamless loop (crossfade)...", 55)
        
        # V4.26: Handle Unicode paths and external volumes using FFmpeg copy
        safe_source = os.path.join(temp_dir, "seamless_source.mp4")
        try:
            if os.path.exists(safe_source):
                os.remove(safe_source)
            copy_result = subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-c", "copy", safe_source
            ], capture_output=True, text=True, timeout=60)
            if copy_result.returncode == 0:
                print(f"[SEAMLESS LOOP] Using FFmpeg copy for safe path")
            else:
                safe_source = video_path
                print(f"[SEAMLESS LOOP] Using original path directly")
        except Exception as e:
            print(f"[SEAMLESS LOOP] FFmpeg copy failed: {e}, using original path")
            safe_source = video_path
        
        # Step 1: Create a single seamless loop unit (end crossfades to beginning)
        seamless_unit = os.path.join(temp_dir, "seamless_unit.mp4")
        
        try:
            # Create seamless unit: crossfade end to beginning
            cmd = [
                "ffmpeg", "-y",
                "-i", safe_source,
                "-filter_complex", f"""
                    [0:v]split=2[v1][v2];
                    [v1]trim=0:{video_duration - xfade_duration},setpts=PTS-STARTPTS[head];
                    [v2]trim={video_duration - xfade_duration}:{video_duration},setpts=PTS-STARTPTS[tail];
                    [0:v]trim=0:{xfade_duration},setpts=PTS-STARTPTS[start];
                    [tail][start]xfade=transition=fade:duration={xfade_duration}:offset=0[xfaded];
                    [head][xfaded]concat=n=2:v=1:a=0[out]
                """.replace("\n", "").replace("  ", ""),
                "-map", "[out]",
                "-an",
            ] + get_encoder_params() + [seamless_unit]
            
            print(f"[SEAMLESS LOOP] Creating seamless unit...")
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            
            # Step 2: Loop the seamless unit to target duration
            looped_video = os.path.join(temp_dir, "seamless_looped.mp4")
            unit_duration = self._get_video_duration(seamless_unit)
            final_loop_count = int(duration / unit_duration) + 1
            
            self._update_export_status_safe("🔄 Looping seamless video...", 70)
            
            cmd2 = [
                "ffmpeg", "-y",
                "-stream_loop", str(final_loop_count),
                "-i", seamless_unit,
                "-t", str(duration),
                "-c", "copy",
                looped_video
            ]
            
            print(f"[SEAMLESS LOOP] Looping seamless unit {final_loop_count} times...")
            subprocess.run(cmd2, check=True, capture_output=True, timeout=300)
            
            print(f"[SEAMLESS LOOP] ✅ Seamless loop complete!")
            return looped_video
            
        except Exception as e:
            print(f"[SEAMLESS LOOP] Error: {e}, falling back to regular loop")
            # Fallback to regular loop
            return self._create_looped_video_fallback(temp_dir, video_path, duration)
    
    def _create_looped_video_fallback(self, temp_dir: str, video_path: str, duration: float) -> str:
        """Fallback method using chunked stream_loop (V4.25.11)
        
        For very long videos or when main method fails.
        Uses two-stage approach: create intermediate, then loop intermediate.
        """
        video_duration = self._get_video_duration(video_path)
        if video_duration <= 0:
            raise Exception("Could not get video duration")
        
        total_loop_count = int(duration / video_duration) + 2
        
        print(f"[VIDEO LOOP FALLBACK] Loops needed: {total_loop_count}")
        
        # For reasonable loop counts, just use simple method
        MAX_SAFE_LOOPS = 1000  # FFmpeg can handle high loop counts with stream_loop
        
        if total_loop_count <= MAX_SAFE_LOOPS:
            return self._simple_loop_video_streamcopy(temp_dir, video_path, duration, total_loop_count)
        
        # For extreme cases (>1000 loops), use two-stage approach
        print(f"[VIDEO LOOP FALLBACK] Using two-stage approach for {total_loop_count} loops")
        
        # Stage 1: Create intermediate with 100 loops
        STAGE1_LOOPS = 100
        self._update_export_status_safe("🚀 Creating intermediate video...", 60)
        
        intermediate = os.path.join(temp_dir, "intermediate.mp4")
        cmd_int = [
            "ffmpeg", "-y",
            "-stream_loop", str(STAGE1_LOOPS - 1),
            "-i", video_path,
            "-c", "copy",
            intermediate
        ]
        
        try:
            subprocess.run(cmd_int, check=True, capture_output=True, timeout=120)
        except Exception as e:
            print(f"[VIDEO LOOP FALLBACK] Stage 1 failed: {e}")
            raise
        
        # Stage 2: Loop intermediate to target duration
        int_duration = video_duration * STAGE1_LOOPS
        final_loops = int(duration / int_duration) + 2
        
        looped_video = os.path.join(temp_dir, "looped.mp4")
        self._update_export_status_safe("🚀 Finalizing video...", 75)
        
        cmd_final = [
            "ffmpeg", "-y",
            "-stream_loop", str(final_loops - 1),
            "-i", intermediate,
            "-t", str(duration),
            "-c", "copy",
            looped_video
        ]
        
        timeout = max(120, int(duration / 600) * 60 + 60)
        
        try:
            subprocess.run(cmd_final, check=True, capture_output=True, timeout=timeout)
            print(f"[VIDEO LOOP FALLBACK] ✅ Two-stage complete!")
        except Exception as e:
            print(f"[VIDEO LOOP FALLBACK] Stage 2 failed: {e}")
            raise
        
        # Cleanup intermediate
        try:
            os.remove(intermediate)
        except Exception:
            pass
        
        return looped_video
    
    def _simple_loop_video_streamcopy(self, temp_dir: str, video_path: str, duration: float, loop_count: int) -> str:
        """Simple video loop with stream copy (V4.25.11 - tested & working)"""
        looped_video = os.path.join(temp_dir, "looped.mp4")
        
        # stream_loop N means play N+1 times total
        # So for loop_count loops, use loop_count-1
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", str(loop_count - 1),
            "-i", video_path, 
            "-t", str(duration),
            "-c", "copy",
            looped_video
        ]
        
        # Timeout: generous for large files
        timeout = max(120, int(duration / 600) * 60 + 60)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                raise Exception(result.stderr)
            print(f"[VIDEO LOOP] ✅ Simple stream copy complete!")
        except Exception as e:
            print(f"[VIDEO LOOP] Stream copy failed: {e}, falling back to encode")
            return self._simple_loop_video(temp_dir, video_path, duration, loop_count)
        
        return looped_video
    
    def _simple_loop_video(self, temp_dir: str, video_path: str, duration: float, loop_count: int) -> str:
        """Simple video loop with encoding (fallback for incompatible formats)"""
        looped_video = os.path.join(temp_dir, "looped.mp4")
        
        cmd = [
            "ffmpeg", "-y", "-threads", "0",
        ] + HW_INPUT_PARAMS + [
            "-stream_loop", str(loop_count - 1),
            "-i", video_path, 
            "-t", str(duration),
        ] + get_encoder_params() + [looped_video]
        
        def update_progress(pct):
            mapped = 55 + int(pct * 0.3)
            self.export_progress.setValue(mapped)
            QApplication.processEvents()
        
        try:
            run_ffmpeg_with_progress(cmd, duration, update_progress)
        except Exception:
            subprocess.run(cmd, check=True, capture_output=True)
        
        return looped_video
    
    def _update_export_status_safe(self, status: str, progress_val: int):
        """Safely update export status if dialog exists"""
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.setText(status)
        if hasattr(self, 'export_progress') and self.export_progress:
            self.export_progress.setValue(progress_val)
        QApplication.processEvents()
    
    def _get_video_assignments(self) -> list:
        """Get video assignment index for each track from audio_files
        
        V4.25.10: Read from audio_files.video_assignment instead of UI layout
        This fixes the issue where layout has fewer items than audio_files
        """
        # First, sync UI to audio_files if layout exists
        self._sync_video_assignments_from_ui()
        
        # Then read from audio_files
        assignments = []
        print(f"[GET-ASSIGN] Reading assignments from {len(self.audio_files)} audio files")
        for i, track in enumerate(self.audio_files):
            idx = track.video_assignment
            assignments.append(idx)
            print(f"[GET-ASSIGN] Track {i+1} → V{idx+1}")
        print(f"[GET-ASSIGN] Final: {assignments}")
        return assignments
    
    def _sync_video_assignments_from_ui(self):
        """Sync video assignments from UI combo boxes to audio_files

        V4.31.2 FIX: Only sync if combo box has enough options (matching video count).
        This prevents overwriting correct assignments with V1 defaults when
        combo boxes haven't been properly initialized yet.
        """
        layout_count = self.track_list_layout.count()
        num_videos = len(self.video_files)
        print(f"[SYNC-ASSIGN] Syncing from {layout_count} UI items to {len(self.audio_files)} audio files (videos: {num_videos})")

        for i in range(min(layout_count, len(self.audio_files))):
            item = self.track_list_layout.itemAt(i)
            if item and item.widget():
                track_item = item.widget()
                if hasattr(track_item, 'video_combo'):
                    combo_count = track_item.video_combo.count()
                    idx = track_item.video_combo.currentIndex()

                    # V4.31.2 FIX: Only sync if combo has correct number of options
                    # If combo has fewer options than videos, it was not properly updated
                    # and would overwrite good assignments with V1 defaults
                    if combo_count >= num_videos and num_videos > 0:
                        self.audio_files[i].video_assignment = idx
                        print(f"[SYNC-ASSIGN] Track {i+1}: UI V{idx+1} → audio_files (combo has {combo_count} options ✅)")
                    else:
                        print(f"[SYNC-ASSIGN] Track {i+1}: SKIPPED - combo has {combo_count} options but need {num_videos} ⚠️")
                        print(f"[SYNC-ASSIGN]   Keeping existing: V{self.audio_files[i].video_assignment+1}")

        print(f"[SYNC-ASSIGN] Done!")
    
    def _calculate_segment_durations(self, crossfade_sec: int) -> list:
        """Calculate actual duration for each track with crossfade"""
        durations = []
        for i, track in enumerate(self.audio_files):
            if i == 0:
                # First track: full duration minus half crossfade at end
                dur = track.duration - (crossfade_sec / 2)
            elif i == len(self.audio_files) - 1:
                # Last track: full duration minus half crossfade at start
                dur = track.duration - (crossfade_sec / 2)
            else:
                # Middle tracks: minus full crossfade
                dur = track.duration - crossfade_sec
            durations.append(max(1, dur))  # At least 1 second
        return durations
    
    def _create_multi_video_from_assignments(self, temp_dir: str, total_duration: float, crossfade_sec: int) -> str:
        """Create video with multiple source videos based on track assignments
        
        V4.31 FIX: Video sync with song start
        - Each song starts video from frame 0 (not continuous loop)
        - Video transition with fade effect between different videos
        """
        assignments = self._get_video_assignments()
        track_durations = self._calculate_segment_durations(crossfade_sec)
        
        print(f"[VIDEO SYNC] ========================================")
        print(f"[VIDEO SYNC] Creating video for {len(self.audio_files)} tracks")
        print(f"[VIDEO SYNC] Assignments: {assignments}")
        print(f"[VIDEO SYNC] Track durations: {[f'{d:.1f}s' for d in track_durations]}")
        
        # If all same video or only 1 video, use simple method
        if len(set(assignments)) <= 1 or len(self.video_files) <= 1:
            print(f"[VIDEO SYNC] Single video mode - using simple loop")
            return self._create_looped_video_fast(temp_dir, self.video_files[0].path, total_duration)
        
        # V4.31 NEW: Create segment for EACH TRACK (not grouped)
        # This ensures video restarts from frame 0 at each song start
        segments = []  # [(video_idx, track_index, duration), ...]
        
        for i, (video_idx, track_dur) in enumerate(zip(assignments, track_durations)):
            segments.append((video_idx, i, track_dur))
            print(f"[VIDEO SYNC] Track {i+1}: V{video_idx+1}, duration={track_dur:.1f}s")
        
        print(f"[VIDEO SYNC] Total segments: {len(segments)}")
        print(f"[VIDEO SYNC] ========================================")
        
        # Get transition settings
        transition_enabled = self.video_transition_enabled
        transition_duration = self.video_transition_duration
        transition_style = self.video_transition_style
        
        # Map transition style to FFmpeg xfade transition
        xfade_map = {
            "fade": "fade",
            "dissolve": "dissolve",
            "wipe": "wipeleft",
            "wipe left": "wipeleft",
            "wipe right": "wiperight",
            "fadeblack": "fadeblack",
            "fade to black": "fadeblack"
        }
        xfade_transition = xfade_map.get(transition_style.lower(), "fade")
        
        # Create individual segment videos - V4.31: Each track starts video from frame 0
        segment_files = []
        MAX_SAFE_LOOPS = 20  # Safe limit for VideoToolbox
        # V4.31.2: Cache video copies - reuse same copy for same source video
        _video_copy_cache = {}  # video_idx -> safe_video_path

        for idx, (video_idx, track_idx, seg_duration) in enumerate(segments):
            original_video_path = self.video_files[min(video_idx, len(self.video_files)-1)].path
            segment_file = os.path.join(temp_dir, f"segment_{idx}.mp4")

            print(f"[SEGMENT {idx}] Track {track_idx+1}: Creating video segment from V{video_idx+1}")
            print(f"[SEGMENT {idx}] Source: {os.path.basename(original_video_path)}")
            print(f"[SEGMENT {idx}] Target duration: {seg_duration:.1f}s")

            # V4.31.2: Reuse video copy if same source video was already copied
            if video_idx in _video_copy_cache and os.path.exists(_video_copy_cache[video_idx]):
                video_path = _video_copy_cache[video_idx]
                print(f"[SEGMENT {idx}] Reusing cached copy for V{video_idx+1}")
            else:
                # V4.31: Copy FULL video file for proper looping
                safe_video_path = os.path.join(temp_dir, f"video_v{video_idx}.mp4")
                try:
                    if os.path.exists(safe_video_path):
                        os.remove(safe_video_path)
                    copy_cmd = [
                        "ffmpeg", "-y", "-i", original_video_path,
                        "-c", "copy",
                        safe_video_path
                    ]
                    result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode == 0 and os.path.exists(safe_video_path):
                        video_path = safe_video_path
                        _video_copy_cache[video_idx] = safe_video_path
                        print(f"[SEGMENT {idx}] Using FFmpeg copy for safe path")
                    else:
                        video_path = original_video_path
                        print(f"[SEGMENT {idx}] Using original path directly")
                except Exception as e:
                    print(f"[SEGMENT {idx}] FFmpeg copy failed: {e}, using original path")
                    video_path = original_video_path
            
            video_duration = self._get_video_duration(video_path)
            if video_duration <= 0:
                print(f"[SEGMENT {idx}] ERROR: Could not get video duration, skipping")
                continue
                
            loop_count = int(seg_duration / video_duration) + 2
            
            print(f"[SEGMENT {idx}] Video: {video_duration:.1f}s, Target: {seg_duration:.1f}s, Loops: {loop_count}")
            
            if loop_count <= MAX_SAFE_LOOPS:
                # V4.25.10: Use stream copy for speed
                cmd = [
                    "ffmpeg", "-y",
                    "-stream_loop", str(loop_count - 1),
                    "-i", video_path,
                    "-t", str(seg_duration),
                    "-c", "copy",  # Stream copy!
                    segment_file
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode != 0:
                        raise Exception(result.stderr[:200])
                    print(f"[SEGMENT {idx}] ✅ Stream copy complete!")
                except Exception as e:
                    print(f"[SEGMENT {idx}] Stream copy failed: {str(e)[:100]}")
                    # Fallback 1: concat demuxer (more reliable than -stream_loop)
                    print(f"[SEGMENT {idx}] Trying concat demuxer fallback...")
                    concat_list = os.path.join(temp_dir, f"seg_{idx}_concat.txt")
                    with open(concat_list, "w") as cf:
                        for _ in range(loop_count + 1):
                            cf.write(f"file '{_ffmpeg_escape_path(video_path)}'\n")
                    cmd_concat = [
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_list,
                        "-t", str(seg_duration),
                        "-c", "copy",
                        segment_file
                    ]
                    try:
                        result2 = subprocess.run(cmd_concat, capture_output=True, text=True, timeout=120)
                        if result2.returncode != 0:
                            raise Exception(result2.stderr[:200])
                        print(f"[SEGMENT {idx}] ✅ Concat demuxer complete!")
                    except Exception as e2:
                        print(f"[SEGMENT {idx}] Concat failed: {str(e2)[:100]}, trying encode...")
                        cmd_encode = [
                            "ffmpeg", "-y", "-threads", "0",
                            "-f", "concat", "-safe", "0",
                            "-i", concat_list,
                            "-t", str(seg_duration),
                        ] + get_encoder_params() + ["-s", "1920x1080", segment_file]
                        subprocess.run(cmd_encode, capture_output=True, text=True, timeout=600)
                        print(f"[SEGMENT {idx}] ✅ Concat + encode complete!")
            else:
                # Smart Segment Approach for high loop counts
                print(f"[SEGMENT {idx}] Using Smart Segment Approach (loops={loop_count} > {MAX_SAFE_LOOPS})")
                
                # V4.25.10: Stage 1 with stream copy
                stage1_loops = MAX_SAFE_LOOPS
                stage1_duration = video_duration * stage1_loops
                intermediate = os.path.join(temp_dir, f"seg_{idx}_intermediate.mp4")
                
                cmd_s1 = [
                    "ffmpeg", "-y",
                    "-stream_loop", str(stage1_loops - 1),
                    "-i", video_path,
                    "-c", "copy",  # Stream copy!
                    intermediate
                ]
                
                try:
                    result = subprocess.run(cmd_s1, capture_output=True, text=True, timeout=60)
                    if result.returncode != 0:
                        raise Exception(result.stderr[:200])
                    print(f"[SEGMENT {idx}] Stage 1: ✅ Stream copy complete!")
                except Exception as e:
                    print(f"[SEGMENT {idx}] Stage 1 stream_loop failed: {str(e)[:100]}")
                    # Fallback: concat demuxer
                    print(f"[SEGMENT {idx}] Stage 1: Using concat demuxer fallback...")
                    concat_s1 = os.path.join(temp_dir, f"seg_{idx}_s1_concat.txt")
                    with open(concat_s1, "w") as cf:
                        for _ in range(stage1_loops):
                            cf.write(f"file '{_ffmpeg_escape_path(video_path)}'\n")
                    cmd_s1_concat = [
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_s1, "-c", "copy", intermediate
                    ]
                    try:
                        result2 = subprocess.run(cmd_s1_concat, capture_output=True, text=True, timeout=120)
                        if result2.returncode != 0:
                            raise Exception(result2.stderr[:200])
                        print(f"[SEGMENT {idx}] Stage 1: ✅ Concat demuxer complete!")
                    except Exception as e2:
                        print(f"[SEGMENT {idx}] Stage 1 concat failed, encoding...")
                        cmd_s1_encode = [
                            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", concat_s1,
                        ] + get_encoder_params() + ["-s", "1920x1080", intermediate]
                        subprocess.run(cmd_s1_encode, capture_output=True, text=True, timeout=600)
                        print(f"[SEGMENT {idx}] Stage 1: ✅ Concat + encode complete!")
                
                # Stage 2: Loop intermediate to target duration
                # Use chunking for very long segments (>400s) to avoid FFmpeg crash
                MAX_CHUNK_DURATION = 200  # 3.3 minutes - conservative safe limit per encode
                
                # V4.25.14: Stage 2 with STREAM COPY (no encoding!)
                # Calculate how many times to loop the intermediate
                stage2_loops = int(seg_duration / stage1_duration) + 2
                
                print(f"[SEGMENT {idx}] Stage 2: stream_loop={stage2_loops} → {seg_duration:.0f}s")
                
                cmd_s2 = [
                    "ffmpeg", "-y",
                    "-stream_loop", str(stage2_loops - 1),
                    "-i", intermediate,
                    "-t", str(seg_duration),
                    "-c", "copy",  # STREAM COPY - no encoding!
                    segment_file
                ]
                
                try:
                    result = subprocess.run(cmd_s2, capture_output=True, text=True, timeout=120)
                    if result.returncode != 0:
                        raise Exception(result.stderr[:200])
                    print(f"[SEGMENT {idx}] Stage 2: ✅ Stream copy complete!")
                except Exception as e:
                    print(f"[SEGMENT {idx}] Stage 2 stream_loop failed: {str(e)[:100]}")
                    print(f"[SEGMENT {idx}] Stage 2: Using concat demuxer fallback...")
                    # FALLBACK: concat demuxer (more reliable than -stream_loop)
                    # Create concat file listing intermediate N times
                    concat_list = os.path.join(temp_dir, f"seg_{idx}_concat.txt")
                    concat_repeats = int(seg_duration / max(1, stage1_duration)) + 2
                    with open(concat_list, "w") as cf:
                        for _ in range(concat_repeats):
                            cf.write(f"file '{_ffmpeg_escape_path(intermediate)}'\n")
                    cmd_concat = [
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_list,
                        "-t", str(seg_duration),
                        "-c", "copy",
                        segment_file
                    ]
                    try:
                        result2 = subprocess.run(cmd_concat, capture_output=True, text=True, timeout=180)
                        if result2.returncode != 0:
                            raise Exception(result2.stderr[:200])
                        print(f"[SEGMENT {idx}] Stage 2: ✅ Concat demuxer complete!")
                    except Exception as e2:
                        print(f"[SEGMENT {idx}] Stage 2 concat also failed: {str(e2)[:100]}")
                        # Last resort: re-encode
                        cmd_encode = [
                            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", concat_list,
                            "-t", str(seg_duration),
                            "-s", "1920x1080",
                        ] + get_encoder_params() + [segment_file]
                        subprocess.run(cmd_encode, capture_output=True, text=True, timeout=600)
                        print(f"[SEGMENT {idx}] Stage 2: ✅ Concat + encode complete!")
            
            segment_files.append((segment_file, seg_duration))
            print(f"[SEGMENT {idx}] ✅ Created successfully")

            # V4.31.2 FIX: Clean up intermediate files IMMEDIATELY after each segment
            # This prevents disk space from filling up with 20+ tracks of temp files
            # NOTE: video_v{video_idx}.mp4 is NOT cleaned here — it's cached for reuse
            cleanup_files = [
                os.path.join(temp_dir, f"seg_{idx}_intermediate.mp4"), # Stage 1 intermediate
                os.path.join(temp_dir, f"seg_{idx}_concat.txt"),       # Concat list
                os.path.join(temp_dir, f"seg_{idx}_s1_concat.txt"),    # Stage 1 concat list
            ]
            freed_bytes = 0
            for cleanup_path in cleanup_files:
                try:
                    if os.path.exists(cleanup_path):
                        fsize = os.path.getsize(cleanup_path)
                        os.remove(cleanup_path)
                        freed_bytes += fsize
                except Exception:
                    pass
            if freed_bytes > 0:
                print(f"[SEGMENT {idx}] 🧹 Cleaned up {freed_bytes / (1024*1024):.1f}MB of intermediate files")

        # V4.31.2: Clean up cached video copies now that all segments are created
        for cached_path in _video_copy_cache.values():
            try:
                if os.path.exists(cached_path):
                    fsize = os.path.getsize(cached_path)
                    os.remove(cached_path)
                    print(f"[VIDEO CLEANUP] 🧹 Removed cached copy: {os.path.basename(cached_path)} ({fsize/(1024*1024):.1f}MB)")
            except Exception:
                pass
        _video_copy_cache.clear()

        # If only one segment, return it
        if len(segment_files) == 1:
            return segment_files[0][0]

        # Use xfade if transition enabled, otherwise simple concat
        if transition_enabled and len(segment_files) > 1:
            return self._concat_with_xfade(temp_dir, segment_files, xfade_transition, transition_duration)
        else:
            # Simple concat without transition - use -c copy for speed!
            concat_file = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_file, "w") as f:
                for seg_file, _ in segment_files:
                    f.write(f"file '{_ffmpeg_escape_path(seg_file)}'\n")
            
            output_path = os.path.join(temp_dir, "multi_video.mp4")
            
            # Use -c copy for fast concat (no re-encode)
            cmd = [
                "ffmpeg", "-y", "-threads", "0",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy",  # Fast copy!
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # V4.31.2 FIX: Clean up individual segment files after concat
            freed = 0
            for seg_file, _ in segment_files:
                try:
                    if os.path.exists(seg_file) and seg_file != output_path:
                        freed += os.path.getsize(seg_file)
                        os.remove(seg_file)
                except Exception:
                    pass
            if freed > 0:
                print(f"[VIDEO CONCAT] 🧹 Cleaned up {freed / (1024*1024):.1f}MB of segment files")

            return output_path

    def _concat_with_xfade(self, temp_dir: str, segment_files: list, transition: str, duration: float) -> str:
        """Concat video segments with fade transitions
        
        V4.31: Use fade in/out instead of xfade for better compatibility
        - Each segment gets fade out at end
        - Next segment gets fade in at start
        - Simple concat after applying fades
        """
        if len(segment_files) < 2:
            return segment_files[0][0]
        
        output_path = os.path.join(temp_dir, "multi_video_fade.mp4")
        
        print(f"[VIDEO COMBINE] Fade transition combining {len(segment_files)} segments")
        print(f"[VIDEO COMBINE] Fade duration: {duration}s")
        
        # Apply fade in/out to each segment
        faded_segments = []
        
        for i, (seg_file, seg_duration) in enumerate(segment_files):
            faded_file = os.path.join(temp_dir, f"faded_{i}.mp4")
            
            # Calculate fade positions
            fade_out_start = max(0.1, seg_duration - duration)
            
            # Build fade filter
            if i == 0:
                # First segment: fade out only
                filter_str = f"fade=t=out:st={fade_out_start}:d={duration}"
            elif i == len(segment_files) - 1:
                # Last segment: fade in only
                filter_str = f"fade=t=in:st=0:d={duration}"
            else:
                # Middle segments: fade in and out
                filter_str = f"fade=t=in:st=0:d={duration},fade=t=out:st={fade_out_start}:d={duration}"
            
            print(f"[VIDEO COMBINE] Segment {i+1}/{len(segment_files)}: applying fade ({seg_duration:.0f}s)")
            
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", seg_file,
                    "-vf", filter_str,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    faded_file
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(faded_file):
                    faded_segments.append(faded_file)
                    print(f"[VIDEO COMBINE] Segment {i+1} ✅ fade applied")
                else:
                    # Fallback: use original without fade
                    faded_segments.append(seg_file)
                    print(f"[VIDEO COMBINE] Segment {i+1} ⚠️ using original (fade failed)")
                    
            except Exception as e:
                print(f"[VIDEO COMBINE] Segment {i+1} fade error: {str(e)[:50]}")
                faded_segments.append(seg_file)
        
        # Concat all faded segments
        print(f"[VIDEO COMBINE] Concatenating {len(faded_segments)} faded segments...")
        
        concat_list = os.path.join(temp_dir, "concat_faded.txt")
        with open(concat_list, "w") as f:
            for seg in faded_segments:
                f.write(f"file '{_ffmpeg_escape_path(seg)}'\n")
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list, "-c", "copy", output_path
            ], check=True, capture_output=True, timeout=120)
            print(f"[VIDEO COMBINE] ✅ All {len(segment_files)} segments combined with fade!")
        except Exception as e:
            print(f"[VIDEO COMBINE] Concat failed: {str(e)[:50]}")
            # Last resort: just use first segment
            output_path = segment_files[0][0]

        # V4.31.2 FIX: Clean up segment and faded files after final concat
        freed = 0
        for seg_file, _ in segment_files:
            try:
                if os.path.exists(seg_file) and seg_file != output_path:
                    freed += os.path.getsize(seg_file)
                    os.remove(seg_file)
            except Exception:
                pass
        for faded_f in faded_segments:
            try:
                if os.path.exists(faded_f) and faded_f != output_path:
                    freed += os.path.getsize(faded_f)
                    os.remove(faded_f)
            except Exception:
                pass
        if freed > 0:
            print(f"[VIDEO COMBINE] 🧹 Cleaned up {freed / (1024*1024):.1f}MB of segment files")

        return output_path
    
    def _extend_video_with_fade(self, temp_dir: str, base_video_path: str, duration: float) -> str:
        """Extend video to match audio duration with loop and add fade out at end
        
        V4.25.14: Uses 3-PART approach for ULTRA-FAST export:
        - HEAD: Stream copy (instant!) - 99% of video
        - TAIL: Encode with fade out - only last 2 seconds
        - Concat: Stream copy to combine
        
        Result: 70 min video in ~5-10 seconds instead of 15-30 minutes!
        """
        output_path = os.path.join(temp_dir, "extended_with_fade.mp4")
        
        # Get video duration
        video_duration = self._get_video_duration(base_video_path)
        
        # Fade out duration (2 seconds at the end)
        fade_duration = 2.0
        fade_start = max(0, duration - fade_duration)
        head_duration = fade_start  # Everything before fade
        
        print(f"[VIDEO EXTEND] Base: {video_duration}s, Target: {duration}s")
        print(f"[VIDEO EXTEND] V4.25.14: Using 3-PART approach (HEAD={head_duration:.0f}s + TAIL={fade_duration}s)")
        
        # ========== STEP 1: Create looped video with stream copy (INSTANT!) ==========
        looped_video = os.path.join(temp_dir, "extend_looped.mp4")
        loop_count = int(duration / video_duration) + 2
        
        # V4.26: Handle Unicode paths and external volumes using FFmpeg copy
        safe_source = os.path.join(temp_dir, "extend_source.mp4")
        try:
            if os.path.exists(safe_source):
                os.remove(safe_source)
            copy_result = subprocess.run([
                "ffmpeg", "-y", "-i", base_video_path,
                "-c", "copy", safe_source
            ], capture_output=True, text=True, timeout=60)
            if copy_result.returncode == 0:
                print(f"[VIDEO EXTEND] Using FFmpeg copy for safe path")
            else:
                safe_source = base_video_path
                print(f"[VIDEO EXTEND] Using original path directly")
        except Exception as e:
            print(f"[VIDEO EXTEND] FFmpeg copy failed: {e}, using original path")
            safe_source = base_video_path
        
        # Stream copy loop - INSTANT!
        cmd_loop = [
            "ffmpeg", "-y",
            "-stream_loop", str(loop_count - 1),
            "-i", safe_source,
            "-t", str(duration),
            "-c", "copy",
            looped_video
        ]
        
        print(f"[VIDEO EXTEND] Step 1: Stream copy loop (instant!)")
        try:
            result = subprocess.run(cmd_loop, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise Exception(result.stderr)
            print(f"[VIDEO EXTEND] Step 1: ✅ Looped video created!")
        except Exception as e:
            print(f"[VIDEO EXTEND] Stream copy failed: {e}, falling back to old method")
            return self._extend_video_with_fade_legacy(temp_dir, base_video_path, duration)
        
        # ========== STEP 2: Split into HEAD (stream copy) + TAIL (encode with fade) ==========
        head_file = os.path.join(temp_dir, "extend_head.mp4")
        tail_file = os.path.join(temp_dir, "extend_tail.mp4")
        
        # HEAD: Stream copy (instant!) - everything before fade
        cmd_head = [
            "ffmpeg", "-y",
            "-i", looped_video,
            "-t", str(head_duration),
            "-c", "copy",
            head_file
        ]
        
        print(f"[VIDEO EXTEND] Step 2a: HEAD stream copy ({head_duration:.0f}s)")
        try:
            subprocess.run(cmd_head, check=True, capture_output=True, timeout=30)
            print(f"[VIDEO EXTEND] Step 2a: ✅ HEAD created!")
        except Exception as e:
            print(f"[VIDEO EXTEND] HEAD failed: {e}")
            return self._extend_video_with_fade_legacy(temp_dir, base_video_path, duration)
        
        # TAIL: Encode with fade out (only 2 seconds!)
        cmd_tail = [
            "ffmpeg", "-y", "-threads", "0",
            "-ss", str(head_duration),
            "-i", looped_video,
            "-t", str(fade_duration),
            "-vf", f"fade=t=out:st=0:d={fade_duration}",
        ] + get_encoder_params() + [tail_file]
        
        print(f"[VIDEO EXTEND] Step 2b: TAIL encode with fade ({fade_duration}s)")
        try:
            subprocess.run(cmd_tail, check=True, capture_output=True, timeout=30)
            print(f"[VIDEO EXTEND] Step 2b: ✅ TAIL created!")
        except Exception as e:
            print(f"[VIDEO EXTEND] TAIL encode failed: {e}, trying software encoder")
            cmd_tail_sw = [
                "ffmpeg", "-y", "-threads", "0",
                "-ss", str(head_duration),
                "-i", looped_video,
                "-t", str(fade_duration),
                "-vf", f"fade=t=out:st=0:d={fade_duration}",
            ] + get_encoder_params(force_software=True) + [tail_file]
            subprocess.run(cmd_tail_sw, check=True, capture_output=True, timeout=60)
        
        # ========== STEP 3: Concat HEAD + TAIL ==========
        concat_file = os.path.join(temp_dir, "extend_concat.txt")
        with open(concat_file, "w") as f:
            f.write(f"file '{_ffmpeg_escape_path(head_file)}'\n")
            f.write(f"file '{_ffmpeg_escape_path(tail_file)}'\n")
        
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path
        ]
        
        print(f"[VIDEO EXTEND] Step 3: Concat HEAD + TAIL")
        try:
            subprocess.run(cmd_concat, check=True, capture_output=True, timeout=30)
            print(f"[VIDEO EXTEND] ✅ 3-PART complete! Video extended with fade out.")
        except Exception as e:
            print(f"[VIDEO EXTEND] Concat failed: {e}, falling back to old method")
            return self._extend_video_with_fade_legacy(temp_dir, base_video_path, duration)
        
        # Cleanup temp files
        for f in [looped_video, head_file, tail_file, concat_file, safe_source]:
            try:
                if os.path.islink(f):
                    os.unlink(f)
                elif os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        
        return output_path
    
    def _extend_video_with_fade_legacy(self, temp_dir: str, base_video_path: str, duration: float) -> str:
        """Legacy method for extending video with fade (fallback)
        
        Uses full re-encode - SLOW but reliable
        """
        output_path = os.path.join(temp_dir, "extended_with_fade.mp4")
        
        video_duration = self._get_video_duration(base_video_path)
        fade_duration = 2.0
        fade_start = max(0, duration - fade_duration)
        
        loop_count = int(duration / video_duration) + 2
        MAX_SAFE_LOOPS = 20
        
        print(f"[VIDEO EXTEND LEGACY] Using full re-encode method")
        
        if loop_count <= MAX_SAFE_LOOPS:
            cmd = [
                "ffmpeg", "-y", "-threads", "0",
            ] + HW_INPUT_PARAMS + [
                "-stream_loop", str(loop_count - 1),
                "-i", base_video_path,
                "-vf", f"trim=duration={duration},setpts=PTS-STARTPTS,fade=t=out:st={fade_start}:d={fade_duration}",
            ] + get_encoder_params() + ["-t", str(duration), output_path]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                return output_path
            except Exception:
                pass
        
        # Smart Segment Approach for high loop counts
        stage1_loops = MAX_SAFE_LOOPS
        stage1_duration = video_duration * stage1_loops
        intermediate = os.path.join(temp_dir, "extend_intermediate.mp4")
        
        cmd_s1 = [
            "ffmpeg", "-y", "-threads", "0",
        ] + HW_INPUT_PARAMS + [
            "-stream_loop", str(stage1_loops - 1),
            "-i", base_video_path,
        ] + get_encoder_params() + ["-s", "1920x1080", intermediate]
        
        try:
            subprocess.run(cmd_s1, check=True, capture_output=True)
        except Exception:
            cmd_s1_sw = [
                "ffmpeg", "-y", "-threads", "0",
                "-stream_loop", str(stage1_loops - 1),
                "-i", base_video_path,
            ] + get_encoder_params(force_software=True) + ["-s", "1920x1080", intermediate]
            subprocess.run(cmd_s1_sw, check=True, capture_output=True)
        
        stage2_loops = min(int(duration / stage1_duration) + 2, MAX_SAFE_LOOPS)
        
        cmd_s2 = [
            "ffmpeg", "-y", "-threads", "0",
        ] + HW_INPUT_PARAMS + [
            "-stream_loop", str(stage2_loops - 1),
            "-i", intermediate,
            "-vf", f"trim=duration={duration},setpts=PTS-STARTPTS,fade=t=out:st={fade_start}:d={fade_duration}",
        ] + get_encoder_params() + ["-t", str(duration), output_path]
        
        try:
            subprocess.run(cmd_s2, check=True, capture_output=True)
        except Exception:
            cmd_s2_sw = [
                "ffmpeg", "-y", "-threads", "0",
                "-stream_loop", str(stage2_loops - 1),
                "-i", intermediate,
                "-vf", f"trim=duration={duration},setpts=PTS-STARTPTS,fade=t=out:st={fade_start}:d={fade_duration}",
            ] + get_encoder_params(force_software=True) + ["-t", str(duration), output_path]
            subprocess.run(cmd_s2_sw, check=True, capture_output=True)
        
        return output_path
    
    def _create_video_with_overlay(self, temp_dir: str, base_video_path: str, duration: float) -> str:
        """Create video with GIF and Logo overlay on top of base video
        
        Uses Smart Segment Approach for high loop counts
        """
        gif_path = self.gif_files[0].path if self.gif_files else None
        logo_path = self.logo_files[0].path if self.logo_files else None
        
        # Get base video duration
        base_duration = self._get_video_duration(base_video_path)
        
        # Check if base video needs looping/extending
        MAX_SAFE_LOOPS = 20  # Safe limit for VideoToolbox
        force_software = False
        actual_input_path = base_video_path
        
        if base_duration >= duration - 0.5:
            # Video is long enough
            video_loops = 0
            print(f"[OVERLAY] Base video long enough ({base_duration}s >= {duration}s)")
        else:
            # Need to loop
            video_loops = int(duration / base_duration) + 2
            
            if video_loops > MAX_SAFE_LOOPS:
                # Use Smart Segment Approach
                print(f"[OVERLAY] Loop count {video_loops} > {MAX_SAFE_LOOPS}, using Smart Segment Approach")
                
                # Stage 1: Create intermediate segment
                stage1_loops = MAX_SAFE_LOOPS
                stage1_duration = base_duration * stage1_loops
                intermediate = os.path.join(temp_dir, "overlay_intermediate.mp4")
                
                cmd_s1 = [
                    "ffmpeg", "-y", "-threads", "0",
                ] + HW_INPUT_PARAMS + [
                    "-stream_loop", str(stage1_loops - 1),
                    "-i", base_video_path,
                ] + get_encoder_params() + ["-s", "1920x1080", intermediate]
                
                try:
                    subprocess.run(cmd_s1, check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    cmd_s1_sw = [
                        "ffmpeg", "-y", "-threads", "0",
                        "-stream_loop", str(stage1_loops - 1),
                        "-i", base_video_path,
                    ] + get_encoder_params(force_software=True) + ["-s", "1920x1080", intermediate]
                    subprocess.run(cmd_s1_sw, check=True, capture_output=True)
                
                # Use intermediate as input
                actual_input_path = intermediate
                base_duration = stage1_duration
                video_loops = min(int(duration / stage1_duration) + 2, MAX_SAFE_LOOPS)
                print(f"[OVERLAY] Using intermediate ({stage1_duration:.1f}s), new loops: {video_loops}")
            else:
                print(f"[OVERLAY] Base video too short ({base_duration}s < {duration}s), loop={video_loops}")
        
        output_path = os.path.join(temp_dir, "with_overlay.mp4")
        
        # Use -stream_loop for base video (memory efficient)
        inputs = []
        if video_loops > 0:
            inputs.extend(["-stream_loop", str(video_loops - 1)])
        inputs.extend(["-i", actual_input_path])
        input_idx = 1
        
        filter_parts = []
        current_output = "[base]"
        
        # Base video - scale and trim (loop handled by -stream_loop)
        filter_parts.append(
            f"[0:v]trim=duration={duration},setpts=PTS-STARTPTS,"
            f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[base]"
        )
        
        # Add GIF overlay if exists
        if gif_path:
            gif_duration = self._get_video_duration(gif_path)
            
            # For GIF: use -stream_loop -1 for infinite loop
            # This is the most reliable method for animated GIFs
            inputs.extend(["-stream_loop", "-1", "-i", gif_path])
            
            # Get GIF settings from UI
            size_text = self.gif_size_combo.currentText()
            position_text = self.gif_position_combo.currentText()
            opacity_text = self.gif_opacity_combo.currentText()
            
            size_map = {"Full Screen (100%)": 1.0, "Large (80%)": 0.8, "Medium (50%)": 0.5, "Small (30%)": 0.3}
            scale = size_map.get(size_text, 1.0)
            
            opacity_map = {"100%": 1.0, "80%": 0.8, "60%": 0.6, "40%": 0.4}
            opacity = opacity_map.get(opacity_text, 1.0)
            
            position_map = {
                "Center": ("(W-w)/2", "(H-h)/2"),
                "Top-Left": ("0", "0"),
                "Top-Right": ("W-w", "0"),
                "Bottom-Left": ("0", "H-h"),
                "Bottom-Right": ("W-w", "H-h")
            }
            overlay_x, overlay_y = position_map.get(position_text, ("(W-w)/2", "(H-h)/2"))
            
            gif_w = int(1920 * scale)
            gif_h = int(1080 * scale)
            
            # Filter with trim and scale (stream_loop handles looping)
            gif_filter = f"trim=duration={duration},setpts=PTS-STARTPTS,scale={gif_w}:{gif_h}"
            if opacity < 1.0:
                gif_filter += f",format=rgba,colorchannelmixer=aa={opacity}"
            
            filter_parts.append(f"[{input_idx}:v]{gif_filter}[gif]")
            filter_parts.append(f"[base][gif]overlay={overlay_x}:{overlay_y}:shortest=1[with_gif]")
            current_output = "[with_gif]"
            input_idx += 1
        
        # Add Logo overlay if exists
        if logo_path:
            inputs.extend(["-i", logo_path])
            
            logo_size_text = self.logo_size_combo.currentText()
            logo_position_text = self.logo_position_combo.currentText()
            logo_opacity_text = self.logo_opacity_combo.currentText()
            
            logo_size_map = {"Large (20%)": 0.20, "Medium (15%)": 0.15, "Small (10%)": 0.10, "Tiny (5%)": 0.05}
            logo_scale = logo_size_map.get(logo_size_text, 0.15)
            
            logo_opacity_map = {"100%": 1.0, "80%": 0.8, "60%": 0.6, "40%": 0.4}
            logo_opacity = logo_opacity_map.get(logo_opacity_text, 0.8)
            
            logo_position_map = {
                "Top-Left": ("20", "20"),
                "Top-Right": ("W-w-20", "20"),
                "Bottom-Left": ("20", "H-h-20"),
                "Bottom-Right": ("W-w-20", "H-h-20"),
                "Center": ("(W-w)/2", "(H-h)/2")
            }
            logo_x, logo_y = logo_position_map.get(logo_position_text, ("W-w-20", "20"))
            
            logo_width = int(1920 * logo_scale)
            logo_filter = f"scale={logo_width}:-1"
            if logo_opacity < 1.0:
                logo_filter += f",format=rgba,colorchannelmixer=aa={logo_opacity}"
            
            filter_parts.append(f"[{input_idx}:v]{logo_filter}[logo]")
            
            logo_input = current_output if gif_path else "[base]"
            filter_parts.append(f"{logo_input}[logo]overlay={logo_x}:{logo_y}[prefade]")
        else:
            if gif_path:
                filter_parts[-1] = filter_parts[-1].replace("[with_gif]", "[prefade]")
            else:
                filter_parts[0] = filter_parts[0].replace("[base]", "[prefade]")
        
        # Add fade out at end (2 seconds)
        fade_duration = 2.0
        fade_start = max(0, duration - fade_duration)
        filter_parts.append(f"[prefade]fade=t=out:st={fade_start}:d={fade_duration}[out]")
        
        filter_complex = ";".join(filter_parts)
        
        # Use hardware input only if not forcing software encoder
        hw_input = [] if force_software else HW_INPUT_PARAMS
        
        cmd = ["ffmpeg", "-y", "-threads", "0"] + hw_input + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
        ] + get_encoder_params(force_software=force_software) + ["-t", str(duration), output_path]
        
        print(f"[OVERLAY] FFmpeg command: {' '.join(cmd)}")
        print(f"[OVERLAY] Filter complex: {filter_complex}")
        print(f"[OVERLAY] Force software encoder: {force_software}")
        
        # Use progress tracking
        def update_progress(pct):
            # Map 0-100 to 65-90 range for overlay step
            mapped = 65 + int(pct * 0.25)
            self.export_progress.setValue(mapped)
            QApplication.processEvents()
        
        try:
            run_ffmpeg_with_progress(cmd, duration, update_progress)
        except subprocess.CalledProcessError as e:
            print(f"[OVERLAY] FFmpeg failed: {e}, retrying with software encoder")
            # Retry with software encoder
            cmd_fallback = ["ffmpeg", "-y", "-threads", "0"] + inputs + [
                "-filter_complex", filter_complex,
                "-map", "[out]",
            ] + get_encoder_params(force_software=True) + ["-t", str(duration), output_path]
            subprocess.run(cmd_fallback, check=True, capture_output=True)
        except Exception as e:
            # Fallback to standard run if progress tracking fails
            print(f"[OVERLAY] Progress tracking failed, using standard: {e}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[OVERLAY] FFmpeg stderr: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)
        
        return output_path
    
    def _create_video_with_gif_overlay_fast(self, temp_dir: str, duration: float) -> str:
        """Create video with GIF and Logo overlay composite - FAST version with settings"""
        video_path = self.video_files[0].path
        gif_path = self.gif_files[0].path if self.gif_files else None
        logo_path = self.logo_files[0].path if self.logo_files else None
        
        video_duration = self._get_video_duration(video_path)
        video_loops = int(duration / video_duration) + 2
        
        output_path = os.path.join(temp_dir, "composite.mp4")
        
        # Use -stream_loop for base video (memory efficient)
        inputs = ["-stream_loop", str(video_loops), "-i", video_path]
        input_idx = 1
        
        # Start building filter chain
        filter_parts = []
        current_output = "[base]"
        
        # Base video - trim and scale (loop handled by -stream_loop)
        filter_parts.append(
            f"[0:v]trim=duration={duration},setpts=PTS-STARTPTS,"
            f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[base]"
        )
        
        # Add GIF overlay if exists
        if gif_path:
            # Use -stream_loop -1 for GIF (infinite loop)
            inputs.extend(["-stream_loop", "-1", "-i", gif_path])
            
            # Get GIF settings from UI
            size_text = self.gif_size_combo.currentText()
            position_text = self.gif_position_combo.currentText()
            opacity_text = self.gif_opacity_combo.currentText()
            
            # Parse size percentage
            size_map = {
                "Full Screen (100%)": 1.0,
                "Large (80%)": 0.8,
                "Medium (50%)": 0.5,
                "Small (30%)": 0.3
            }
            scale = size_map.get(size_text, 1.0)
            
            # Parse opacity
            opacity_map = {"100%": 1.0, "80%": 0.8, "60%": 0.6, "40%": 0.4}
            opacity = opacity_map.get(opacity_text, 1.0)
            
            # Calculate GIF position and scale
            if scale >= 1.0:
                overlay_x, overlay_y = "0", "0"
                gif_scale = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
            else:
                position_map = {
                    "Center": ("(W-w)/2", "(H-h)/2"),
                    "Top-Left": ("20", "20"),
                    "Top-Right": ("W-w-20", "20"),
                    "Bottom-Left": ("20", "H-h-20"),
                    "Bottom-Right": ("W-w-20", "H-h-20")
                }
                overlay_x, overlay_y = position_map.get(position_text, ("(W-w)/2", "(H-h)/2"))
                gif_scale = f"scale=iw*{scale}:ih*{scale}"
            
            # Build GIF filter - trim and scale (loop handled by -stream_loop)
            gif_filter = f"trim=duration={duration},setpts=PTS-STARTPTS,{gif_scale}"
            if opacity < 1.0:
                gif_filter += f",format=rgba,colorchannelmixer=aa={opacity}"
            
            filter_parts.append(f"[{input_idx}:v]{gif_filter}[gif]")
            filter_parts.append(f"[base][gif]overlay={overlay_x}:{overlay_y}:shortest=1[with_gif]")
            current_output = "[with_gif]"
            input_idx += 1
        
        # Add Logo overlay if exists
        if logo_path:
            inputs.extend(["-i", logo_path])
            
            # Get Logo settings from UI
            logo_size_text = self.logo_size_combo.currentText()
            logo_position_text = self.logo_position_combo.currentText()
            logo_opacity_text = self.logo_opacity_combo.currentText()
            
            # Parse logo size
            logo_size_map = {
                "Large (20%)": 0.20,
                "Medium (15%)": 0.15,
                "Small (10%)": 0.10,
                "Tiny (5%)": 0.05
            }
            logo_scale = logo_size_map.get(logo_size_text, 0.15)
            
            # Parse logo opacity
            logo_opacity_map = {"100%": 1.0, "80%": 0.8, "60%": 0.6, "40%": 0.4}
            logo_opacity = logo_opacity_map.get(logo_opacity_text, 0.8)
            
            # Logo position
            logo_position_map = {
                "Top-Left": ("20", "20"),
                "Top-Right": ("W-w-20", "20"),
                "Bottom-Left": ("20", "H-h-20"),
                "Bottom-Right": ("W-w-20", "H-h-20"),
                "Center": ("(W-w)/2", "(H-h)/2")
            }
            logo_x, logo_y = logo_position_map.get(logo_position_text, ("W-w-20", "20"))
            
            # Build logo filter - scale to percentage of 1080p width
            logo_width = int(1920 * logo_scale)
            logo_filter = f"scale={logo_width}:-1"
            if logo_opacity < 1.0:
                logo_filter += f",format=rgba,colorchannelmixer=aa={logo_opacity}"
            
            filter_parts.append(f"[{input_idx}:v]{logo_filter}[logo]")
            
            # Determine input for logo overlay
            logo_input = current_output if gif_path else "[base]"
            filter_parts.append(f"{logo_input}[logo]overlay={logo_x}:{logo_y}[out]")
        else:
            # If no logo, final output is current state
            if gif_path:
                filter_parts[-1] = filter_parts[-1].replace("[with_gif]", "[out]")
            else:
                filter_parts[0] = filter_parts[0].replace("[base]", "[out]")
        
        filter_complex = ";".join(filter_parts)
        
        cmd = ["ffmpeg", "-y", "-threads", "0"] + HW_INPUT_PARAMS + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
        ] + get_encoder_params() + ["-t", str(duration), output_path]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
        
    def _create_video_with_logo_only(self, temp_dir: str, duration: float) -> str:
        """Create video with Logo overlay only (no GIF) - FAST version"""
        video_path = self.video_files[0].path
        logo_path = self.logo_files[0].path
        
        video_duration = self._get_video_duration(video_path)
        video_loops = int(duration / video_duration) + 2
        
        output_path = os.path.join(temp_dir, "with_logo.mp4")
        
        # Get Logo settings from UI
        logo_size_text = self.logo_size_combo.currentText()
        logo_position_text = self.logo_position_combo.currentText()
        logo_opacity_text = self.logo_opacity_combo.currentText()
        
        # Parse logo size
        logo_size_map = {
            "Large (20%)": 0.20,
            "Medium (15%)": 0.15,
            "Small (10%)": 0.10,
            "Tiny (5%)": 0.05
        }
        logo_scale = logo_size_map.get(logo_size_text, 0.15)
        
        # Parse logo opacity
        logo_opacity_map = {"100%": 1.0, "80%": 0.8, "60%": 0.6, "40%": 0.4}
        logo_opacity = logo_opacity_map.get(logo_opacity_text, 0.8)
        
        # Logo position
        logo_position_map = {
            "Top-Left": ("20", "20"),
            "Top-Right": ("W-w-20", "20"),
            "Bottom-Left": ("20", "H-h-20"),
            "Bottom-Right": ("W-w-20", "H-h-20"),
            "Center": ("(W-w)/2", "(H-h)/2")
        }
        logo_x, logo_y = logo_position_map.get(logo_position_text, ("W-w-20", "20"))
        
        # Calculate logo width (percentage of 1920)
        logo_width = int(1920 * logo_scale)
        logo_filter = f"scale={logo_width}:-1"
        if logo_opacity < 1.0:
            logo_filter += f",format=rgba,colorchannelmixer=aa={logo_opacity}"
        
        # Use trim instead of loop filter (loop handled by -stream_loop)
        filter_complex = (
            f"[0:v]trim=duration={duration},setpts=PTS-STARTPTS,"
            f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[base];"
            f"[1:v]{logo_filter}[logo];"
            f"[base][logo]overlay={logo_x}:{logo_y}[out]"
        )
        
        cmd = [
            "ffmpeg", "-y", "-threads", "0",
        ] + HW_INPUT_PARAMS + [
            "-stream_loop", str(video_loops), "-i", video_path,
            "-i", logo_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
        ] + get_encoder_params() + ["-t", str(duration), output_path]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
        
    def _create_looped_video(self, temp_dir: str, video_path: str, duration: float, progress: QProgressBar) -> str:
        """Create looped video to match duration (legacy)"""
        video_duration = self._get_video_duration(video_path)
        loop_count = int(duration / video_duration) + 2
        
        looped_video = os.path.join(temp_dir, "looped.mp4")
        cmd = [
            "ffmpeg", "-y",
        ] + HW_INPUT_PARAMS + [
            "-stream_loop", str(loop_count),
            "-i", video_path, "-t", str(duration),
        ] + get_encoder_params() + [looped_video]
        subprocess.run(cmd, check=True, capture_output=True)
        
        progress.setValue(70)
        return looped_video
    
    def _create_video_with_gif_overlay(self, temp_dir: str, duration: float, progress: QProgressBar) -> str:
        """Create video with GIF overlay composite (legacy)"""
        video_path = self.video_files[0].path
        gif_path = self.gif_files[0].path
        
        video_duration = self._get_video_duration(video_path)
        
        # Calculate loops for video only
        video_loops = int(duration / video_duration) + 2
        
        output_path = os.path.join(temp_dir, "composite.mp4")
        
        progress.setValue(60)
        
        # FFmpeg filter: GIF FULLSCREEN overlay on video
        # Use -stream_loop for both video and GIF
        filter_complex = (
            f"[0:v]trim=duration={duration},setpts=PTS-STARTPTS,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[base];"
            f"[1:v]trim=duration={duration},setpts=PTS-STARTPTS,"
            f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[gif];"
            f"[base][gif]overlay=0:0:shortest=1[out]"
        )
        
        cmd = [
            "ffmpeg", "-y",
        ] + HW_INPUT_PARAMS + [
            "-stream_loop", str(video_loops), "-i", video_path,
            "-stream_loop", "-1", "-i", gif_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
        ] + get_encoder_params() + ["-t", str(duration), output_path]
        subprocess.run(cmd, check=True, capture_output=True)
        
        progress.setValue(80)
        
        return output_path
            
    def _mix_audio_with_crossfade(self, temp_dir: str, crossfade_sec: int, curve: str, progress: QProgressBar) -> str:
        """Mix multiple audio files with crossfade"""
        if len(self.audio_files) < 2:
            return self.audio_files[0].path
            
        # Build complex filter
        inputs = []
        for i, audio in enumerate(self.audio_files):
            inputs.extend(["-i", audio.path])
            
        # Chain crossfades
        filter_parts = []
        current_input = "[0:a]"
        
        for i in range(1, len(self.audio_files)):
            next_input = f"[{i}:a]"
            output = f"[a{i}]"
            
            filter_parts.append(
                f"{current_input}{next_input}acrossfade=d={crossfade_sec}:c1={curve}:c2={curve}{output}"
            )
            current_input = output
            
        filter_complex = ";".join(filter_parts)
        
        output_path = os.path.join(temp_dir, "mixed_audio.wav")
        
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", current_input,
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        
        return output_path
        
    def _get_audio_duration(self, path: str) -> float:
        """Get audio duration"""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", path
            ], capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except subprocess.TimeoutExpired:
            print(f"[ERROR] ffprobe timeout for {path}, unable to determine duration")
            return None
        except Exception as e:
            print(f"[ERROR] ffprobe failed for {path}: {e}, unable to determine duration")
            return None
            
    def _get_video_duration(self, path: str) -> float:
        """Get video duration"""
        return self._get_audio_duration(path)


# ==================== Phase 4 F-6: Production Pipeline Dialog ====================


class _ProductionPipelineDialog(QDialog):
    """Step-by-step production pipeline wizard."""

    _STEPS = [
        ("1. IMPORT", "Load audio tracks into playlist"),
        ("2. AI DJ", "Auto-order & arrange playlist"),
        ("3. COMPILE", "Crossfade & concatenate tracks"),
        ("4. MASTER", "Ozone 12 mastering chain"),
        ("5. HOOK EXTRACT", "Extract hooks (before/after master)"),
        ("6. VIDEO", "Assemble video + text + transitions"),
        ("7. EXPORT", "Platform presets + loudness report"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Production Pipeline")
        self.setMinimumSize(500, 400)
        self.setStyleSheet("background: #1a1a2e; color: #ffffff;")

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("PRODUCTION PIPELINE")
        title.setStyleSheet("color: #00d4aa; font-size: 16px; font-weight: bold; letter-spacing: 2px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._status_labels = []
        self._action_btns = []

        for i, (name, desc) in enumerate(self._STEPS):
            row = QHBoxLayout()

            status = QLabel("⬜")
            status.setFixedWidth(24)
            status.setStyleSheet("font-size: 14px;")
            self._status_labels.append(status)
            row.addWidget(status)

            info = QVBoxLayout()
            n = QLabel(name)
            n.setStyleSheet("color: #00d4aa; font-size: 12px; font-weight: bold;")
            info.addWidget(n)
            d = QLabel(desc)
            d.setStyleSheet("color: #888899; font-size: 10px;")
            info.addWidget(d)

            # F-8: Hook extract radio option
            if "HOOK" in name:
                from gui.utils.compat import QHBoxLayout as QHL
                radio_row = QHL()
                radio_row.setSpacing(10)
                lbl = QLabel("Extract from:")
                lbl.setStyleSheet("color: #888899; font-size: 9px;")
                radio_row.addWidget(lbl)
                self._hook_original = QCheckBox("Original")
                self._hook_original.setChecked(True)
                self._hook_original.setStyleSheet("color: #cccccc; font-size: 9px;")
                radio_row.addWidget(self._hook_original)
                self._hook_mastered = QCheckBox("Mastered")
                self._hook_mastered.setStyleSheet("color: #00d4aa; font-size: 9px;")
                radio_row.addWidget(self._hook_mastered)
                radio_row.addStretch()
                info.addLayout(radio_row)

            row.addLayout(info, 1)

            btn = QPushButton("RUN")
            btn.setFixedSize(60, 28)
            btn.setStyleSheet(
                "QPushButton { background: #2a2a44; color: #00d4aa; border: 1px solid #00d4aa; "
                "border-radius: 4px; font-weight: bold; font-size: 10px; } "
                "QPushButton:hover { background: #00d4aa; color: #1a1a2e; }")
            btn.clicked.connect(lambda checked, idx=i: self._run_step(idx))
            self._action_btns.append(btn)
            row.addWidget(btn)

            layout.addLayout(row)

        layout.addStretch()

        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet(
            "QPushButton { background: #2a2a44; color: #ffffff; border: 1px solid #444; "
            "border-radius: 4px; padding: 8px; font-weight: bold; } "
            "QPushButton:hover { background: #444; }")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _run_step(self, idx: int) -> None:
        self._status_labels[idx].setText("⏳")
        self._status_labels[idx].repaint()
        QApplication.processEvents()

        parent = self.parent()
        try:
            if idx == 0:
                if parent and hasattr(parent, '_add_audio_files'):
                    parent._add_audio_files()
            elif idx == 1:
                if parent and hasattr(parent, '_open_ai_dj'):
                    parent._open_ai_dj()
            elif idx == 3:
                if parent and hasattr(parent, '_open_master'):
                    parent._open_master()
            elif idx == 4:
                if parent and hasattr(parent, '_open_hook_extractor'):
                    parent._open_hook_extractor()
            elif idx == 5:
                if parent and hasattr(parent, '_add_video_files'):
                    parent._add_video_files()
            elif idx == 6:
                if parent and hasattr(parent, '_export_video'):
                    parent._export_video()
        except Exception as e:
            print(f"[PIPELINE] Step {idx} error: {e}")

        self._status_labels[idx].setText("✅")


# ==================== License Dialog ====================


class LicenseDialog(QDialog):
    """License activation dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔐 LongPlay Studio - Activation")
        self.setFixedSize(500, 400)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Logo/Title
        title = QLabel("🎵 LongPlay Studio")
        title.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            color: {Colors.ACCENT};
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Professional Audio/Video Creator")
        subtitle.setStyleSheet(f"font-size: 14px; color: {Colors.TEXT_SECONDARY};")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # Serial Key Input
        key_group = QGroupBox("🔑 Enter Serial Key")
        key_group.setStyleSheet(f"""
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }}
        """)
        key_layout = QVBoxLayout(key_group)
        
        self.serial_input = QLineEdit()
        self.serial_input.setPlaceholderText("XXXX-XXXX-XXXX-XXXX")
        self.serial_input.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 2px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 15px;
                font-size: 18px;
                font-family: 'Menlo', 'Courier New';
                letter-spacing: 2px;
            }}
            QLineEdit:focus {{
                border-color: {Colors.ACCENT};
            }}
        """)
        self.serial_input.textChanged.connect(self._format_serial)
        key_layout.addWidget(self.serial_input)
        
        # Name input (optional)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Your Name (optional)")
        self.name_input.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
            }}
        """)
        key_layout.addWidget(self.name_input)
        
        layout.addWidget(key_group)
        
        # Status message
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"font-size: 12px; color: {Colors.TEXT_SECONDARY};")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.activate_btn = QPushButton("✅ Activate")
        self.activate_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
            QPushButton:disabled {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
            }}
        """)
        self.activate_btn.clicked.connect(self._activate)
        btn_layout.addWidget(self.activate_btn)
        
        quit_btn = QPushButton("❌ Quit")
        quit_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.BG_SECONDARY};
            }}
        """)
        quit_btn.clicked.connect(self.reject)
        btn_layout.addWidget(quit_btn)
        
        layout.addLayout(btn_layout)
        
        # Purchase link
        purchase_label = QLabel("💳 <a href='#' style='color: #FF6B35;'>Purchase License</a>")
        purchase_label.setStyleSheet(f"font-size: 12px; color: {Colors.TEXT_SECONDARY};")
        purchase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(purchase_label)
        
    def _format_serial(self, text):
        """Auto-format serial key input"""
        # Remove non-alphanumeric except dash
        clean = ''.join(c for c in text.upper() if c.isalnum() or c == '-')
        
        # Auto-add dashes
        parts = clean.replace('-', '')
        if len(parts) > 16:
            parts = parts[:16]
        
        formatted = '-'.join([parts[i:i+4] for i in range(0, len(parts), 4)])
        
        if formatted != text:
            self.serial_input.blockSignals(True)
            self.serial_input.setText(formatted)
            self.serial_input.blockSignals(False)
    
    def _activate(self):
        """Validate and activate license"""
        from license_manager import validate_serial_key, save_license, get_license_type
        
        serial = self.serial_input.text().strip().upper()
        name = self.name_input.text().strip()
        
        is_valid, message = validate_serial_key(serial)
        
        if is_valid:
            # Save license
            if save_license(serial, name):
                prefix = serial.split('-')[0]
                license_type = get_license_type(prefix)
                
                self.status_label.setStyleSheet(f"font-size: 12px; color: {Colors.METER_GREEN};")
                self.status_label.setText(f"✅ {license_type} License Activated Successfully!")
                
                # Close dialog with success
                QMessageBox.information(
                    self, 
                    "✅ Activation Successful",
                    f"Welcome to LongPlay Studio!\n\n"
                    f"License Type: {license_type}\n"
                    f"Registered to: {name or 'User'}\n\n"
                    f"Enjoy creating amazing content! 🎵"
                )
                self.accept()
            else:
                self.status_label.setStyleSheet(f"font-size: 12px; color: {Colors.METER_RED};")
                self.status_label.setText("❌ Error saving license. Please try again.")
        else:
            self.status_label.setStyleSheet(f"font-size: 12px; color: {Colors.METER_RED};")
            self.status_label.setText(f"❌ {message}")



def check_and_show_license(app) -> bool:
    """Check license and show activation dialog if needed"""
    from license_manager import check_license
    
    is_licensed, message, license_data = check_license()
    
    if is_licensed:
        print(f"✅ License valid: {message}")
        return True
    
    # Show license dialog
    dialog = LicenseDialog()
    result = dialog.exec()
    
    return result == QDialog.DialogCode.Accepted


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Dark theme is set per-widget using Colors class
    
    # Check license first
    if not check_and_show_license(app):
        print("❌ License required. Exiting.")
        sys.exit(1)
    
    window = LongPlayStudioV4()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
