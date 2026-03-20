#!/usr/bin/env python3
"""
LongPlay Studio V5.0 - AI DJ + YouTube Generator + Hook Extractor + Video Prompt + AI Master
Features:
1. CapCut-style timeline with track controls
2. GIF OVERLAY on Video (size/position/opacity controls)
3. LOGO OVERLAY - ใส่ Logo บน Video (size/position/opacity)
4. VIDEO TRANSITIONS - Fade ระหว่าง Video เปลี่ยน
5. Track controls (Lock, Eye, Speaker, More)
6. Spacebar = Play/Pause
7. DJ Crossfade with export
8. Real-time meter (monitoring only)
9. Timestamp generator
10. Separate Video + Audio export (FAST mode)
11. AUTO VIDEO MODE - เลือก Video อัตโนมัติ
12. PREVIEW CROSSFADE - ฟังรอยต่อแต่ละเพลง
13. Export Progress with Time Estimate

V4.25 NEW:
- 🎧 AI DJ Mode - Analyze BPM, Energy, Key for smart playlist ordering
- 🔄 Shuffle Again - Generate new orders until satisfied
- 🏆 Best #1 Suggestion - AI picks best opener track
- ≡ Drag & Drop Reorder - Manual track arrangement
- 📝 Auto Numbering - Rename files with track numbers
- 📺 YouTube Generator - Auto Title/Description/Tags/Timestamps
- 🎨 Theme Templates - Cafe, Driving, Sleep, Workout, Focus, Chill

V4.25.5 FIXES:
- ✅ Fixed YouTube Generator crash (missing Colors.SUCCESS)
- ✅ Fixed AI VDO video preview not showing (added QVideoWidget)
- ✅ Improved Audio Player - use ffprobe instead of pydub for faster loading
- ✅ Added error handling for audio player errors
- ✅ Added track_changed signal for better track info display

V4.25.6 FIXES:
- ✅ Fixed Clear VDO crash (missing _refresh_track_list method)
- ✅ Fixed Clear Audio crash (same issue)
- ✅ Fixed Apply Order crash in AI DJ and AI VDO dialogs
- ✅ Added Clear All button in AI DJ Dialog
- ✅ Added Clear All button in AI VDO Dialog
- ✅ Fixed Export crash (ffmpeg concat error 254) - improved path handling

V4.25.7 PERFORMANCE:
- 🚀 Ultra-fast video looping (CapCut-style Stream Copy + Concat)
- 🚀 10-50x faster for short videos (5 seconds)
- 🚀 Only encode once at the end instead of every loop
- 🚀 Reduced disk I/O and memory usage

V4.25.8 UI IMPROVEMENTS:
- 🎯 CapCut-style resizable panels (drag to resize left/center/right)
- 📜 Scrollable panels for all sections
- 🔽 Collapsible sections (click header to collapse/expand)
- 🖥️ Better layout for small screens
- 📐 Vertical splitter for center content (Video/Track/Timeline)

V4.25.9 FIXES:
- ✅ Fixed Unicode path issue on macOS (Thai/special characters)
- ✅ Use symlinks instead of file copy for better Unicode support
- ✅ Added multiple fallback methods for file operations

V4.25.10 ULTRA-FAST EXPORT:
- 🚀 Stream copy throughout entire pipeline (NO re-encode!)

V4.25.11 TESTED & FIXED:
- ✅ Fixed video looping using FFmpeg stream_loop (tested: 70 min in 1 sec!)
- ✅ Fixed Unicode path handling - use FFmpeg copy instead of symlinks
- ✅ Fixed Audio batch mixing on macOS with external volumes
- ✅ Fixed timeout issues for large files
- ✅ Simplified looping logic (single command instead of multiple steps)
- 🚀 Video looping: 5s → 74min in ~10 seconds (was 30+ minutes)
- 🚀 Export without overlay: instant stream copy
- 🚀 Only encode when overlay (GIF/Logo) is applied
- 🚀 CapCut-level export speed achieved!

V4.25.12 PRODUCTION READY:
- ✅ TESTED 50 ROUNDS - 100% success rate!
- ✅ Cross-platform: Mac (VideoToolbox) + PC (NVENC, AMD, Intel, Software)
- ✅ Fixed GIF loop: use -stream_loop -1 (more reliable than -ignore_loop)
- ✅ Automatic fallback to software encoder if hardware fails
- ✅ Proper timeout handling for large files
- 🚀 Tested scenarios: 2-5 tracks, with/without GIF, multi-video, crossfade 2-10s

V4.25.13 OPTIMIZED VIDEO TRANSITIONS:
- ✅ 3-Part xfade method: HEAD (copy) + XFADE (encode) + TAIL (copy)
- ✅ Only re-encode transition region (~4 seconds) instead of entire video

V4.25.14 ULTRA-FAST EXTEND WITH FADE:
- 🚀 3-Part approach for _extend_video_with_fade: HEAD (stream copy) + TAIL (encode fade)
- 🚀 70 min video in ~5-10 seconds instead of 15-30 minutes!
- 🚀 Only encode last 2 seconds for fade out effect
- ✅ Fallback to legacy method if stream copy fails
- ✅ Fixes memory issues with long videos (70+ minutes)
- 🚀 Video transition now 10x faster for long videos
- 🚀 No more "xfade failed" fallback to concat

V4.26 NEW FEATURES:
- 🎵 Hook Extractor - ตัดเพลงเฉพาะท่อน Hook รองรับ 20 เพลง ใช้ Audio Waveform Analysis
- 🎬 Video Prompt Generator - สร้าง Prompt คล้าย Midjourney รองรับ Video จาก meta.ai
- ⏱️ Realtime Display - แสดงเวลาแบบ Realtime ขณะเล่น Video (mm:ss.ms)
- ✅ Fixed Speed Controls (1x, 1.5x, 2x) - ปุ่มทำงานได้จริงแล้ว!
- 🎨 Speed indicator แสดงความเร็วปัจจุบันใน Realtime Display
- 📦 Batch Export Mode - Export แยกแต่ละไฟล์ (เลือก 5 ไฟล์ = ได้ 5 ไฟล์ output)
- 🔄 Seamless Video Loop - Video Loop ต่อเนื่องไม่มีรอยต่อ (ใช้ Crossfade 1 วินาที)

V4.27 FIXES:
- ✅ Fixed "No valid audio files to concat" error - เพิ่มการตรวจสอบไฟล์ก่อน Export
- ✅ แสดง Error ชัดเจนว่าไฟล์ไหนหายไป (External drive ถูกถอด)
- ✅ ถามผู้ใช้ว่าต้องการข้ามไฟล์ที่หายไปหรือไม่
- ✅ ลบไฟล์ที่หายไปออกจาก playlist อัตโนมัติ

V4.28 VIDEO PREVIEW FIXES:
- ✅ Fixed Video Preview ไม่แสดงผล - เพิ่ม auto-play เมื่อ load video
- ✅ เพิ่ม error handling สำหรับ QMediaPlayer
- ✅ เพิ่ม debug logging เพื่อตรวจสอบปัญหา
- ✅ Realtime display อัพเดทตลอดเวลา (50ms interval)
- ✅ ปุ่ม Play/Pause ทำงานได้จริง

V4.29 VIDEO PREVIEW DEBUG:
- ✅ เพิ่ม debug log ละเอียดมาก - แสดงทุกขั้นตอนการ load video
- ✅ แสดง media status, duration, hasVideo, hasAudio
- ✅ แสดง URL validation และ file path
- ✅ รอ media load นานขึ้น (500ms) ก่อน auto-play
- ✅ แสดง error ชัดเจนขึ้น (ResourceError, FormatError, etc.)

V4.30 VIDEO PREVIEW - OPENCV SOLUTION:
- ✅ แก้ปัญหาจอดำบน macOS - ใช้ OpenCV + QLabel แทน QVideoWidget
- ✅ แสดงภาพวิดีโอจริงๆ บนหน้าจอ (ไม่ใช่จอดำอีกต่อไป)
- ✅ แสดง frame แรกทันทีเมื่อโหลดไฟล์
- ✅ ทำงานได้ทุก platform (Windows, macOS, Linux)
- ✅ ปุ่ม Play/Pause ทำงานได้จริง
- ✅ Speed control (1x, 1.5x, 2x) ทำงานได้
- ✅ Seek ทำงานได้
- ✅ Loop video อัตโนมัติ
- ✅ Realtime display อัพเดทตลอดเวลา
- ⚠️ ต้องติดตั้ง: pip3 install opencv-python numpy

V4.31 VIDEO SYNC WITH SONG START:
- ✅ แก้ปัญหา Video ไม่ตรงกับหัวเพลง
- ✅ แต่ละเพลงเริ่ม Video จาก frame 0 (ไม่ใช่ loop ต่อเนื่อง)
- ✅ Video transition ด้วย fade effect เมื่อเปลี่ยน Video
- ✅ คัดลอก full video file แทน 10 วินาทีแรก
- ✅ แสดง log ชัดเจนขึ้นว่า Video ไหนใช้กับเพลงไหน

V4.31.1 SMART TEMP DIRECTORY:
- ✅ แก้ปัญหา "No space left on device" สำหรับ video export ยาว (68+ นาที)
- ✅ เพิ่ม get_smart_temp_dir() - เลือก temp directory อัตโนมัติ
- ✅ ตรวจสอบ External Drive (macOS: /Volumes/*) และเลือก drive ที่มีพื้นที่มากสุด
- ✅ รองรับ TMPDIR environment variable
- ✅ Fallback หลายระดับ: TMPDIR > External Drive > Home Temp > System Default
- ✅ รองรับ macOS, Windows, และ Linux
"""

import sys
import os
import platform
import faulthandler

# V5.0: Add modules path for AI Master Module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable faulthandler to dump traceback on SEGFAULT (macOS "Python quit unexpectedly")
# Crash log will be saved to ~/Desktop/longplay_crash.log
_crash_log_path = os.path.expanduser("~/Desktop/longplay_crash.log")
_crash_log_file = None
try:
    _crash_log_file = open(_crash_log_path, "w")
    faulthandler.enable(file=_crash_log_file, all_threads=True)
    print(f"[CRASH LOG] Enabled → {_crash_log_path}")
    # V5.5 FIX: Register cleanup to close file on exit
    import atexit
    atexit.register(lambda: _crash_log_file.close() if _crash_log_file and not _crash_log_file.closed else None)
except Exception:
    faulthandler.enable()  # Fallback to stderr

# ==================== FFmpeg Path Setup for macOS ====================
# MUST run before any subprocess calls to ffmpeg
def setup_ffmpeg_path():
    """Add Homebrew paths to PATH for macOS .app bundles"""
    if platform.system() == "Darwin":
        # Common Homebrew paths
        homebrew_paths = [
            "/opt/homebrew/bin",  # Apple Silicon
            "/usr/local/bin",     # Intel Mac
        ]

        current_path = os.environ.get("PATH", "")
        paths_to_add = []

        for path in homebrew_paths:
            if os.path.exists(path) and path not in current_path:
                paths_to_add.append(path)

        if paths_to_add:
            os.environ["PATH"] = ":".join(paths_to_add) + ":" + current_path
            print(f"✅ Added to PATH: {', '.join(paths_to_add)}")

# Run IMMEDIATELY before other imports
setup_ffmpeg_path()

import subprocess
import json
import shutil
import re
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import random
import numpy as np

def _natural_sort_key(filepath):
    """Natural sort key: sort '2.Song' before '10.Song' (numeric-aware)"""
    basename = os.path.basename(filepath)
    # Split filename into numeric and non-numeric parts
    parts = re.split(r'(\d+)', basename)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

# OpenCV for video preview (fallback for QVideoWidget issues on macOS)
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] OpenCV not installed. Video preview may not work on macOS.")
    print("[WARNING] Install with: pip3 install opencv-python numpy")

# Import Video Prompt Generator
try:
    from video_prompt_generator import VideoPromptGenerator, MIDJOURNEY_STYLES
except ImportError:
    VideoPromptGenerator = None
    MIDJOURNEY_STYLES = {}

# Import Hook Extractor
try:
    from hook_extractor import HookExtractor, HookResult
except ImportError:
    HookExtractor = None
    HookResult = None

# Import Ozone 12 / Waves WLM meter widgets from Master Module
try:
    from modules.master.ui_panel import WavesWLMMeter, GainReductionHistoryWidget, LogicChannelMeter
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


# Try to import shared FFmpeg escape utility; define inline fallback if gui package not available
try:
    from gui.utils import ffmpeg_escape_path as _ffmpeg_escape_path
except ImportError:
    def _ffmpeg_escape_path(p: str) -> str:
        return p.replace("'", "'\\''")


# Try PyQt6 first, then PySide6
try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QFrame, QScrollArea, QSlider, QComboBox,
        QTabWidget, QListWidget, QListWidgetItem, QProgressBar,
        QSplitter, QSizePolicy, QSpacerItem, QStackedWidget,
        QApplication, QStyle, QStyleOption, QFileDialog, QDialog,
        QLineEdit, QTextEdit, QCheckBox, QGroupBox, QToolButton,
        QMessageBox, QMenu, QSpinBox, QDoubleSpinBox, QDial, QGraphicsOpacityEffect
    )
    from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QUrl, QMimeData, QPoint, QThread, pyqtSlot, QRect, QRectF
    from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QPalette, QIcon, QPixmap, QDragEnterEvent, QDropEvent, QPolygon, QAction, QShortcut, QKeySequence, QImage
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    PYQT6 = True
except ImportError:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QFrame, QScrollArea, QSlider, QComboBox,
        QTabWidget, QListWidget, QListWidgetItem, QProgressBar,
        QSplitter, QSizePolicy, QSpacerItem, QStackedWidget,
        QApplication, QStyle, QStyleOption, QFileDialog, QDialog,
        QLineEdit, QTextEdit, QCheckBox, QGroupBox, QToolButton,
        QMessageBox, QMenu, QSpinBox, QDoubleSpinBox, QDial, QGraphicsOpacityEffect
    )
    from PySide6.QtCore import Qt, QSize, Signal as pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QUrl, QMimeData, QPoint, QThread, Slot as pyqtSlot
    from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QPalette, QIcon, QPixmap, QDragEnterEvent, QDropEvent, QPolygon, QAction, QShortcut, QKeySequence, QImage
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtMultimediaWidgets import QVideoWidget
    PYQT6 = False

# Styles and AudioPlayer are defined below


# ==================== Hardware Acceleration Detection ====================
def detect_hw_encoder():
    """Detect best available hardware encoder for the system"""
    import platform
    
    system = platform.system()
    
    # Mac - use VideoToolbox (works on all Apple Silicon and Intel Macs)
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            if "h264_videotoolbox" in result.stdout:
                print("✅ Hardware Acceleration: VideoToolbox (Apple Silicon/Intel Mac)")
                return "h264_videotoolbox"
        except Exception:
            pass
    
    # Windows - try NVIDIA NVENC, then AMD AMF, then Intel QSV
    elif system == "Windows":
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            if "h264_nvenc" in result.stdout:
                print("✅ Hardware Acceleration: NVIDIA NVENC")
                return "h264_nvenc"
            if "h264_amf" in result.stdout:
                print("✅ Hardware Acceleration: AMD AMF")
                return "h264_amf"
            if "h264_qsv" in result.stdout:
                print("✅ Hardware Acceleration: Intel QuickSync")
                return "h264_qsv"
        except Exception:
            pass
    
    # Linux - try VAAPI, then NVENC
    elif system == "Linux":
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            if "h264_vaapi" in result.stdout:
                print("✅ Hardware Acceleration: VAAPI (Linux)")
                return "h264_vaapi"
            if "h264_nvenc" in result.stdout:
                print("✅ Hardware Acceleration: NVIDIA NVENC")
                return "h264_nvenc"
        except Exception:
            pass
    
    # Fallback to software encoder
    print("⚠️ No Hardware Acceleration - using CPU (libx264)")
    return "libx264"


# Global encoder setting
HW_ENCODER = detect_hw_encoder()


# ==================== Smart Temp Directory Selection ====================
# V4.31.1 FIX: Auto-select temp directory with most free space
def get_smart_temp_dir(min_free_gb: float = 5.0) -> str:
    """
    Smart temp directory selection - automatically chooses the best location

    Priority:
    1. TMPDIR environment variable (if set and has enough space)
    2. External drives with most free space (macOS: /Volumes/*)
    3. User home temp folder
    4. System default temp

    V4.31.2: Reduced min from 50GB to 5GB (intermediate cleanup now happens per-segment)

    Args:
        min_free_gb: Minimum free space required in GB (default: 5GB for video export)

    Returns:
        Path to temp directory with most free space
    """
    import platform
    import shutil as sh

    min_free_bytes = min_free_gb * 1024 * 1024 * 1024
    candidates = []

    # Check TMPDIR environment variable first
    env_tmpdir = os.environ.get('TMPDIR') or os.environ.get('TEMP') or os.environ.get('TMP')
    if env_tmpdir and os.path.exists(env_tmpdir):
        try:
            usage = sh.disk_usage(env_tmpdir)
            if usage.free >= min_free_bytes:
                print(f"[SMART TEMP] Using TMPDIR: {env_tmpdir} ({usage.free / (1024**3):.1f}GB free)")
                return tempfile.mkdtemp(dir=env_tmpdir)
            candidates.append((env_tmpdir, usage.free))
        except Exception as e:
            print(f"[SMART TEMP] Warning: Cannot check TMPDIR: {e}")

    system = platform.system()

    if system == "Darwin":  # macOS
        # Check /Volumes for external drives
        volumes_path = "/Volumes"
        if os.path.exists(volumes_path):
            for volume in os.listdir(volumes_path):
                vol_path = os.path.join(volumes_path, volume)
                if os.path.isdir(vol_path) and volume != "Macintosh HD":
                    try:
                        usage = sh.disk_usage(vol_path)
                        # Only consider drives with enough space
                        if usage.free >= min_free_bytes:
                            temp_base = os.path.join(vol_path, "longplay_temp")
                            os.makedirs(temp_base, exist_ok=True)
                            candidates.append((temp_base, usage.free))
                    except Exception:
                        pass

        # Check user home temp
        home_temp = os.path.expanduser("~/Library/Caches/LongPlayStudio")
        try:
            os.makedirs(home_temp, exist_ok=True)
            usage = sh.disk_usage(home_temp)
            candidates.append((home_temp, usage.free))
        except Exception:
            pass

    elif system == "Windows":
        # Check all drive letters
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                try:
                    usage = sh.disk_usage(drive)
                    if usage.free >= min_free_bytes:
                        temp_base = os.path.join(drive, "LongPlayTemp")
                        os.makedirs(temp_base, exist_ok=True)
                        candidates.append((temp_base, usage.free))
                except Exception:
                    pass

    else:  # Linux
        # Check common mount points
        for mount in ["/mnt", "/media", os.path.expanduser("~")]:
            if os.path.exists(mount):
                try:
                    usage = sh.disk_usage(mount)
                    if usage.free >= min_free_bytes:
                        temp_base = os.path.join(mount, "longplay_temp")
                        os.makedirs(temp_base, exist_ok=True)
                        candidates.append((temp_base, usage.free))
                except Exception:
                    pass

    # Sort by free space (most space first)
    candidates.sort(key=lambda x: x[1], reverse=True)

    if candidates:
        best_path, best_free = candidates[0]
        print(f"[SMART TEMP] Selected: {best_path} ({best_free / (1024**3):.1f}GB free)")
        return tempfile.mkdtemp(dir=best_path)

    # Fallback to system default
    default_temp = tempfile.mkdtemp()
    print(f"[SMART TEMP] Fallback to system default: {default_temp}")
    return default_temp


def get_hwaccel_input_params():
    """Get FFmpeg hardware acceleration input parameters for faster decoding"""
    import platform
    system = platform.system()
    
    if system == "Darwin" and HW_ENCODER == "h264_videotoolbox":
        # Mac - VideoToolbox hardware decoding
        return ["-hwaccel", "videotoolbox"]
    elif system == "Windows":
        if HW_ENCODER == "h264_nvenc":
            return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif HW_ENCODER == "h264_qsv":
            return ["-hwaccel", "qsv"]
    elif system == "Linux":
        if HW_ENCODER == "h264_vaapi":
            return ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"]
        elif HW_ENCODER == "h264_nvenc":
            return ["-hwaccel", "cuda"]
    
    return []  # No hardware decoding available

# ==================== Quality Mode Settings ====================
# Quality modes: "fast", "balanced", "quality"
QUALITY_PRESETS = {
    "fast": {
        "description": "🚀 Fast - Quick preview, smaller file",
        "x264_preset": "ultrafast",
        "x264_crf": "28",
        "videotoolbox_q": "45",  # Lower = faster, less quality
        "nvenc_preset": "p1",    # Fastest
        "nvenc_cq": "28",
    },
    "balanced": {
        "description": "⚖️ Balanced - Good for YouTube",
        "x264_preset": "fast",
        "x264_crf": "23",
        "videotoolbox_q": "60",
        "nvenc_preset": "p4",
        "nvenc_cq": "23",
    },
    "quality": {
        "description": "💎 Quality - Best quality, slower",
        "x264_preset": "slow",
        "x264_crf": "18",
        "videotoolbox_q": "75",  # Higher = better quality
        "nvenc_preset": "p7",    # Best quality
        "nvenc_cq": "18",
    },
}

# Global quality mode setting (default: balanced)
CURRENT_QUALITY_MODE = "balanced"

def set_quality_mode(mode: str):
    """Set the global quality mode"""
    global CURRENT_QUALITY_MODE
    if mode in QUALITY_PRESETS:
        CURRENT_QUALITY_MODE = mode
        print(f"✅ Quality mode set to: {QUALITY_PRESETS[mode]['description']}")

def get_quality_mode() -> str:
    """Get the current quality mode"""
    return CURRENT_QUALITY_MODE

def get_encoder_params(force_software=False, quality_mode=None):
    """Get FFmpeg encoder parameters based on detected hardware and quality mode
    
    Args:
        force_software: If True, always use software encoder (for high loop counts)
        quality_mode: Override quality mode ("fast", "balanced", "quality")
    """
    mode = quality_mode or CURRENT_QUALITY_MODE
    preset = QUALITY_PRESETS.get(mode, QUALITY_PRESETS["balanced"])
    
    if force_software:
        # Force software encoder for reliability with high loop counts
        return [
            "-c:v", "libx264",
            "-crf", preset["x264_crf"],
            "-preset", preset["x264_preset"],
            "-pix_fmt", "yuv420p",
        ]
    if HW_ENCODER == "h264_videotoolbox":
        # Mac VideoToolbox
        return [
            "-c:v", "h264_videotoolbox",
            "-q:v", preset["videotoolbox_q"],
            "-pix_fmt", "yuv420p",
        ]
    elif HW_ENCODER == "h264_nvenc":
        # NVIDIA NVENC
        return [
            "-c:v", "h264_nvenc",
            "-preset", preset["nvenc_preset"],
            "-cq", preset["nvenc_cq"],
            "-pix_fmt", "yuv420p",
        ]
    elif HW_ENCODER == "h264_amf":
        # AMD AMF
        quality_setting = "speed" if mode == "fast" else ("balanced" if mode == "balanced" else "quality")
        return [
            "-c:v", "h264_amf",
            "-quality", quality_setting,
            "-rc", "cqp",
            "-qp", preset["x264_crf"],  # Use same CRF value
            "-pix_fmt", "yuv420p",
        ]
    elif HW_ENCODER == "h264_qsv":
        # Intel QuickSync
        qsv_preset = "veryfast" if mode == "fast" else ("fast" if mode == "balanced" else "slow")
        return [
            "-c:v", "h264_qsv",
            "-preset", qsv_preset,
            "-global_quality", preset["x264_crf"],
            "-pix_fmt", "nv12",
        ]
    elif HW_ENCODER == "h264_vaapi":
        # Linux VAAPI
        return [
            "-c:v", "h264_vaapi",
            "-qp", preset["x264_crf"],
        ]
    else:
        # Software fallback (libx264)
        return [
            "-c:v", "libx264",
            "-crf", preset["x264_crf"],
            "-preset", preset["x264_preset"],
            "-pix_fmt", "yuv420p",
        ]

# Get hardware input params once at startup
HW_INPUT_PARAMS = get_hwaccel_input_params()
if HW_INPUT_PARAMS:
    print(f"✅ Hardware Decoding: {HW_INPUT_PARAMS[1]}")


def run_ffmpeg_with_progress(cmd: list, duration: float, progress_callback=None):
    """Run FFmpeg command with real-time progress tracking"""
    import re
    
    # Add progress output flags
    cmd_with_progress = cmd.copy()
    # Insert after 'ffmpeg'
    idx = 1
    cmd_with_progress.insert(idx, "-progress")
    cmd_with_progress.insert(idx + 1, "pipe:1")
    cmd_with_progress.insert(idx + 2, "-stats_period")
    cmd_with_progress.insert(idx + 3, "0.5")
    
    process = subprocess.Popen(
        cmd_with_progress,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    time_pattern = re.compile(r'out_time_ms=(\d+)')
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        
        match = time_pattern.search(line)
        if match and duration > 0:
            current_ms = int(match.group(1))
            current_sec = current_ms / 1000000  # microseconds to seconds
            progress = min(100, int((current_sec / duration) * 100))
            if progress_callback:
                progress_callback(progress)
    
    process.wait()
    if process.returncode != 0:
        stderr = process.stderr.read()
        raise subprocess.CalledProcessError(process.returncode, cmd, stderr=stderr)


# ==================== Color Constants ====================
class Colors:
    # ══ Original LongPlay Studio Theme ══
    BG_PRIMARY = "#1a1a1a"       # Original dark background
    BG_SECONDARY = "#242424"     # Secondary background
    BG_TERTIARY = "#2d2d2d"      # Tertiary/raised surface
    BG_CARD = "#2a2a2a"          # Card background
    # ══ Accent Colors (Original Orange) ══
    ACCENT = "#FF9500"           # Original orange accent
    ACCENT_DIM = "#CC7700"       # Dimmed orange
    ACCENT_BRIGHT = "#FFB340"    # Bright orange glow
    # ══ Text ══
    TEXT_PRIMARY = "#ffffff"     # White text (original)
    TEXT_SECONDARY = "#aaaaaa"   # Secondary grey
    TEXT_TERTIARY = "#666666"    # Subtle/disabled
    # ══ Borders & Chrome ══
    BORDER = "#333333"           # Panel border
    BORDER_LIGHT = "#444444"     # Ridge highlight
    # ══ Track Colors (vibrant, saturated) ══
    VIDEO_COLOR = "#00CED1"      # Cyan for video
    AUDIO_COLOR = "#4FC3F7"      # Light blue for audio
    GIF_COLOR = "#CE93D8"        # Purple for GIF
    MASTER_COLOR = "#00B4D8"     # Teal for master
    # ══ Meter Colors (vivid LED) ══
    METER_GREEN = "#43A047"      # Green LED
    METER_YELLOW = "#FDD835"     # Yellow LED
    METER_RED = "#E53935"        # Red LED
    # ══ Hardware Accents ══
    CHROME_LIGHT = "#5A5A62"
    CHROME_DARK = "#2A2A30"
    CHROME_HIGHLIGHT = "#78787E"
    LED_AMBER = "#FF9500"
    LED_AMBER_GLOW = "#FFB340"
    PANEL_DEEP = "#141414"
    TEAL = "#00B4D8"
    TEAL_GLOW = "#48CAE4"
    TEAL_DIM = "#0077B6"
    # ══ Module Identity Colors ══
    MOD_EQ = "#4FC3F7"           # Light blue (SSL)
    MOD_DYN = "#FF8A65"          # Warm orange (CLA)
    MOD_IMG = "#CE93D8"          # Purple (Abbey Road)
    MOD_MAX = "#FFD54F"          # Gold (L2)
    # ══ LED Status ══
    LED_GREEN = "#43A047"
    LED_RED = "#E53935"
    LED_BLUE = "#42A5F5"
    # ══ Misc ══
    TRACK_BG = "#1a1a1a"
    SUCCESS = "#43A047"          # Green LED style
    GOLD = "#C89B3C"             # Muted gold for labels


# ==================== Drag & Drop Zone ====================
class DropZoneListWidget(QListWidget):
    """QListWidget with drag and drop support for files AND internal reordering"""
    filesDropped = pyqtSignal(list)  # emits list of file paths

    def __init__(self, accepted_extensions: list, placeholder_text: str = "Drop files here", parent=None):
        super().__init__(parent)
        self.accepted_extensions = [ext.lower() for ext in accepted_extensions]
        self.placeholder_text = placeholder_text
        self._is_dragging = False
        self._allow_internal_move = False  # Flag for internal reordering

        # Set default style
        self._update_style_dragging(False)

        # Enable drag and drop - MUST be after setStyleSheet
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DropOnly)

    def enableInternalMove(self):
        """Enable internal drag & drop reordering"""
        self._allow_internal_move = True
        self.setDragEnabled(True)
        self.setDragDropMode(QListWidget.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event: QDragEnterEvent):
        try:
            # Check for external file drop
            mime = event.mimeData()
            if mime and mime.hasUrls():
                for url in mime.urls():
                    file_path = url.toLocalFile()
                    if file_path:  # External file
                        ext = os.path.splitext(file_path)[1].lower()
                        if ext in self.accepted_extensions:
                            event.acceptProposedAction()
                            self._is_dragging = True
                            self._update_style_dragging(True)
                            return

            # Check for internal move
            if self._allow_internal_move and event.source() == self:
                event.acceptProposedAction()
                return

            event.ignore()
        except Exception as e:
            print(f"[DROP] dragEnterEvent error: {e}")
            event.ignore()

    def dragMoveEvent(self, event):
        """Required for proper drag and drop"""
        try:
            mime = event.mimeData()
            if mime and mime.hasUrls():
                # Check if external file
                for url in mime.urls():
                    if url.toLocalFile():
                        event.acceptProposedAction()
                        return

            # Internal move
            if self._allow_internal_move and event.source() == self:
                event.acceptProposedAction()
                return

            event.ignore()
        except Exception as e:
            print(f"[DROP] dragMoveEvent error: {e}")
            event.ignore()

    def dragLeaveEvent(self, event):
        self._is_dragging = False
        self._update_style_dragging(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        try:
            self._is_dragging = False
            self._update_style_dragging(False)

            # Check for external files
            mime = event.mimeData()
            if mime and mime.hasUrls():
                valid_files = []
                for url in mime.urls():
                    file_path = url.toLocalFile()
                    if file_path:  # External file
                        ext = os.path.splitext(file_path)[1].lower()
                        if ext in self.accepted_extensions:
                            valid_files.append(file_path)

                if valid_files:
                    self.filesDropped.emit(valid_files)
                    event.acceptProposedAction()
                    return

            # Handle internal move
            if self._allow_internal_move and event.source() == self:
                super().dropEvent(event)
                return

            event.ignore()
        except Exception as e:
            print(f"[DROP] dropEvent error: {e}")
            event.ignore()
            
    def _update_style_dragging(self, is_dragging: bool):
        if is_dragging:
            self.setStyleSheet(f"""
                QListWidget {{
                    background: {Colors.BG_TERTIARY};
                    border: 2px dashed {Colors.ACCENT};
                    border-radius: 6px;
                }}
                QListWidget::item {{
                    color: {Colors.TEXT_PRIMARY};
                    padding: 8px;
                    border-radius: 4px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QListWidget {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 #0A0A0C, stop:0.02 {Colors.BG_PRIMARY},
                        stop:0.98 {Colors.BG_PRIMARY}, stop:1 #0A0A0C);
                    border: 1px solid {Colors.BORDER};
                    border-top: 1px solid #0A0A0C;
                    border-radius: 4px;
                }}
                QListWidget::item {{
                    color: {Colors.TEXT_PRIMARY};
                    padding: 8px;
                    border-bottom: 1px solid {Colors.BORDER};
                    font-family: 'Menlo', monospace;
                    font-size: 11px;
                }}
                QListWidget::item:hover {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 {Colors.BG_TERTIARY}, stop:1 {Colors.BG_SECONDARY});
                    color: {Colors.ACCENT};
                }}
                QListWidget::item:selected {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 {Colors.ACCENT_DIM}, stop:0.5 {Colors.ACCENT},
                        stop:1 {Colors.ACCENT_DIM});
                    color: #1A1A1E;
                }}
            """)


# ==================== Draggable Track List Widget ====================
class DraggableTrackListWidget(QListWidget):
    """QListWidget with internal drag & drop reordering and track info display"""
    orderChanged = pyqtSignal(list)  # emits new order of file paths
    playRequested = pyqtSignal(int)  # emits track index to play
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks = []  # List of dicts with file_path, name, duration, bpm, key, energy, etc.
        
        # Enable internal drag & drop
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        
        # Style
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
        
        # Connect model signals
        self.model().rowsMoved.connect(self._on_rows_moved)
        
    def _on_rows_moved(self, parent, start, end, destination, row):
        """Handle row reorder"""
        # Rebuild tracks list from current widget order
        new_tracks = []
        for i in range(self.count()):
            item = self.item(i)
            if item and hasattr(item, 'track_data'):
                new_tracks.append(item.track_data)
        
        self.tracks = new_tracks
        self.orderChanged.emit([t['file_path'] for t in self.tracks])
        
    def set_tracks(self, tracks: list):
        """Set tracks to display. Each track is a dict with file_path, name, duration_sec, bpm, key, energy"""
        self.clear()
        self.tracks = tracks
        
        for i, track in enumerate(tracks):
            item = QListWidgetItem()
            item.track_data = track
            
            # Format display text
            name = track.get('name', 'Unknown')
            duration_sec = track.get('duration_sec', 0)
            duration_str = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"
            bpm = track.get('bpm', 0)
            key = track.get('key', '')
            energy = track.get('energy', 0)
            energy_bars = "█" * int(energy * 6) + "░" * (6 - int(energy * 6))
            
            # Build display text
            display_text = f"≡  {i+1}.  {name}    [{duration_str}]"
            if bpm > 0:
                display_text += f"    BPM: {int(bpm)}"
            if key:
                display_text += f"    Key: {key}"
            if energy > 0:
                display_text += f"    {energy_bars}"
                
            item.setText(display_text)
            item.setToolTip(f"Drag to reorder\nDouble-click to play")
            
            self.addItem(item)
            
    def get_ordered_paths(self) -> list:
        """Get current order of file paths"""
        paths = []
        for i in range(self.count()):
            item = self.item(i)
            if item and hasattr(item, 'track_data'):
                paths.append(item.track_data['file_path'])
        return paths
    
    def get_ordered_tracks(self) -> list:
        """Get current order of track data"""
        tracks = []
        for i in range(self.count()):
            item = self.item(i)
            if item and hasattr(item, 'track_data'):
                tracks.append(item.track_data)
        return tracks
        
    def mouseDoubleClickEvent(self, event):
        """Handle double-click to play"""
        item = self.itemAt(event.pos())
        if item:
            row = self.row(item)
            self.playRequested.emit(row)
        super().mouseDoubleClickEvent(event)


# ==================== Audio Player Widget ====================
class AudioPlayerWidget(QWidget):
    """Simple audio player using QMediaPlayer"""
    position_changed = pyqtSignal(int)  # position in ms
    duration_changed = pyqtSignal(int)  # duration in ms
    play_state_changed = pyqtSignal(bool)  # is_playing
    track_changed = pyqtSignal(int, str)  # index, filename
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_playing = False
        self.current_file_index = 0
        self.files = []
        self.durations = []  # duration in ms for each file
        self.crossfade_ms = 5000
        
        # Media player A (main)
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)

        # Media player B (crossfade next track)
        self.player_b = QMediaPlayer()
        self.audio_output_b = QAudioOutput()
        self.player_b.setAudioOutput(self.audio_output_b)
        self.audio_output_b.setVolume(0.0)
        self._crossfading = False
        self._crossfade_started_for = -1  # track index we already started crossfade into

        # Connect signals
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.player.errorOccurred.connect(self._on_error)
        self.player_b.mediaStatusChanged.connect(self._on_player_b_status_changed)

        # Timer for checking position / crossfade
        self.timer = QTimer()
        self.timer.timeout.connect(self._check_position)
        self.timer.setInterval(50)
        
    def _on_error(self, error, error_string=""):
        """Handle media player errors"""
        print(f"Audio Player Error: {error} - {error_string}")
        # Try to continue with next file on error
        if self.current_file_index < len(self.files) - 1:
            print(f"Skipping to next track...")
            self._load_file(self.current_file_index + 1)
            if self.is_playing:
                self.player.play()
        
    def load_files(self, file_paths: list):
        """Load multiple audio files"""
        self.files = file_paths
        self.durations = []
        self.current_file_index = 0
        
        # Get durations using ffprobe (faster and more reliable than pydub for many files)
        for path in file_paths:
            try:
                duration_ms = self._get_duration_ffprobe(path)
                self.durations.append(duration_ms)
            except Exception as e:
                print(f"Warning: Could not get duration for {path}: {e}")
                self.durations.append(180000)  # Default 3 minutes
        
        # Load first file
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
        """Handle position change - emit TIMELINE-ADJUSTED position (with crossfade subtracted)"""
        if not self.files or not self.durations:
            return
        # Audio global position = sum of full durations of previous tracks + current position
        audio_global_pos = sum(self.durations[:self.current_file_index]) + position
        # Timeline position = audio position adjusted for crossfade overlaps
        # Each track after the first starts crossfade_ms earlier in timeline view
        timeline_pos = audio_global_pos - (self.current_file_index * self.crossfade_ms)
        self.position_changed.emit(max(0, timeline_pos))

    def _on_duration_changed(self, duration: int):
        """Handle duration change - emit TIMELINE-ADJUSTED total duration (with crossfade subtracted)"""
        if len(self.durations) > 1:
            total_duration = sum(self.durations) - (len(self.durations) - 1) * self.crossfade_ms
        else:
            total_duration = sum(self.durations)
        self.duration_changed.emit(max(0, total_duration))
        
    def _on_media_status_changed(self, status):
        """Handle media status change"""
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if self._crossfading:
                # Crossfade in progress — swap players
                next_idx = self.current_file_index + 1
                if next_idx < len(self.files):
                    self._finish_crossfade(next_idx)
                    return
            # No crossfade — play next file directly
            if self.current_file_index < len(self.files) - 1:
                self._load_file(self.current_file_index + 1)
                self.player.play()
            else:
                self.is_playing = False
                self.play_state_changed.emit(False)
                self.timer.stop()
                
    def _check_position(self):
        """DJ crossfade: fade out current track, fade in next track during overlap."""
        if not self.is_playing or not self.durations:
            return
        position = self.player.position()
        duration = self.durations[self.current_file_index] if self.current_file_index < len(self.durations) else 0
        if duration <= 0:
            return
        remaining = duration - position
        next_idx = self.current_file_index + 1

        if remaining <= self.crossfade_ms and next_idx < len(self.files):
            # Start crossfade into next track
            if not self._crossfading or self._crossfade_started_for != next_idx:
                self._crossfading = True
                self._crossfade_started_for = next_idx
                self.player_b.setSource(QUrl.fromLocalFile(self.files[next_idx]))
                self.player_b.play()
                self.audio_output_b.setVolume(0.0)

            # Calculate crossfade progress (0.0 → 1.0)
            import math
            t = max(0.0, min(1.0, 1.0 - remaining / self.crossfade_ms))
            # Equal-power crossfade
            vol_out = math.cos(t * math.pi / 2)
            vol_in = math.sin(t * math.pi / 2)
            self.audio_output.setVolume(max(0.0, vol_out))
            self.audio_output_b.setVolume(max(0.0, vol_in))

        elif self._crossfading and remaining <= 0:
            # Crossfade complete — swap players
            self._finish_crossfade(next_idx)

    def _finish_crossfade(self, next_idx):
        """Complete crossfade: swap player B → A, advance track index."""
        self.player.stop()
        # Swap: B becomes the active player, A becomes standby
        self.player, self.player_b = self.player_b, self.player
        self.audio_output, self.audio_output_b = self.audio_output_b, self.audio_output
        self.audio_output.setVolume(1.0)
        self.audio_output_b.setVolume(0.0)
        # Re-connect signals to the new active player
        try:
            self.player_b.positionChanged.disconnect(self._on_position_changed)
        except (TypeError, RuntimeError):
            pass
        self.player.positionChanged.connect(self._on_position_changed)
        self.current_file_index = next_idx
        self.track_changed.emit(next_idx, os.path.basename(self.files[next_idx]))
        self._crossfading = False
        self._crossfade_started_for = -1

    def _on_player_b_status_changed(self, status):
        """Handle player B errors gracefully."""
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._crossfading = False
            self._crossfade_started_for = -1
        
    def play(self):
        """Start playback"""
        if not self.files:
            print("[PLAYER] ⚠️ No audio files loaded - cannot play")
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
        # Convert timeline position to audio file position
        # Timeline uses effective durations: track[0]=full, track[i>0]=duration-crossfade
        cumulative_timeline = 0
        for i, duration in enumerate(self.durations):
            effective_dur = duration if i == 0 else max(0, duration - self.crossfade_ms)
            if cumulative_timeline + effective_dur > position_ms:
                # Found the track — local position within the audio file
                local_pos = position_ms - cumulative_timeline
                if i != self.current_file_index:
                    self._load_file(i)
                self.player.setPosition(max(0, local_pos))
                return
            cumulative_timeline += effective_dur
            
    def set_crossfade(self, crossfade_sec: int):
        """Set crossfade duration"""
        self.crossfade_ms = crossfade_sec * 1000


# ==================== Data Classes ====================
@dataclass
class MediaFile:
    path: str
    name: str
    duration: float = 0.0
    lufs: float = -14.0
    file_type: str = "audio"  # audio, video, gif
    video_assignment: int = 0  # V4.25.10: Track which video this audio uses (0 = V1, 1 = V2, etc.)
    
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


# ==================== Audio Analysis Engine ====================
class AudioAnalysisEngine:
    """V5.5: Loads audio data into memory and provides real-time level analysis
    at any playback position. Used by both Mini Meter (main GUI) and
    Master Module meters to show REAL audio levels during playback.
    """

    def __init__(self):
        self._audio_data = {}   # {file_path: np.ndarray} — cached audio data
        self._sample_rates = {} # {file_path: int}
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
            print("[AUDIO ENGINE] ✅ soundfile available — real meter analysis enabled")
        except ImportError:
            print("[AUDIO ENGINE] ⚠️ soundfile not available — using fallback meter")

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
        """Set gain and ceiling for accurate metering + preview rendering.
        gain_db: 0.0 to 20.0 dB boost
        ceiling_dbtp: -3.0 to -0.1 dBTP brickwall ceiling
        """
        self._gain_linear = 10 ** (gain_db / 20.0) if gain_db > 0.01 else 1.0
        self._ceiling_linear = 10 ** (ceiling_dbtp / 20.0)

    def get_gained_audio(self):
        """Return current audio data with gain+ceiling applied (for writing temp WAV).
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
        # Convert position_ms to sample index
        center_sample = int((position_ms / 1000.0) * sr)
        window_samples = int((window_ms / 1000.0) * sr)
        start = max(0, center_sample - window_samples // 2)
        end = min(len(data), start + window_samples)

        if end <= start or start >= len(data):
            return self._empty_result()

        chunk = data[start:end]
        # V5.5.2: Apply gain + brickwall ceiling for accurate metering
        if self._gain_linear > 1.001:
            chunk = chunk * self._gain_linear
            chunk = np.clip(chunk, -self._ceiling_linear, self._ceiling_linear)
        left = chunk[:, 0]
        right = chunk[:, 1] if chunk.shape[1] > 1 else left

        left_peak = float(np.max(np.abs(left)))
        right_peak = float(np.max(np.abs(right)))
        left_rms = float(np.sqrt(np.mean(left ** 2)))
        right_rms = float(np.sqrt(np.mean(right ** 2)))

        import math
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


# ==================== Real-time Level Meter ====================
class RealTimeMeter(QWidget):
    """Professional real-time audio level meter - Waves-style with segmented LED bars"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.left_level = 0.0
        self.right_level = 0.0
        self.peak_left = 0.0
        self.peak_right = 0.0
        self.current_position_ms = 0
        self.is_playing = False
        self.num_segments = 35  # 35 LED segments per channel

        self.peak_hold_timer = QTimer()
        self.peak_hold_timer.timeout.connect(self._decay_peaks)
        self.peak_hold_timer.start(50)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)

    def start(self):
        self.is_playing = True
        self.update_timer.start(33)  # ~30fps for smoother animation

    def stop(self):
        self.is_playing = False
        self.update_timer.stop()
        self.left_level = 0.0
        self.right_level = 0.0
        self.peak_left = 0.0
        self.peak_right = 0.0
        self.update()

    def setPosition(self, position_ms: int):
        """Update meter based on playback position"""
        self.current_position_ms = position_ms
        if self.is_playing:
            self._generate_levels_for_position(position_ms)

    def set_audio_engine(self, engine: 'AudioAnalysisEngine'):
        """V5.5: Connect real audio analysis engine for actual level metering."""
        self._audio_engine = engine

    def _generate_levels_for_position(self, position_ms: int):
        """V5.5.2: Read REAL audio levels from AudioAnalysisEngine (with gain applied)."""
        # Use real audio analysis if available
        if hasattr(self, '_audio_engine') and self._audio_engine is not None:
            levels = self._audio_engine.analyze_at_position(position_ms, window_ms=50)
            # V5.5.2: Use dB-to-meter mapping for accurate display (0 dBFS = 1.0, -48 dBFS = 0.0)
            l_db = max(-48.0, levels["left_rms_db"])
            r_db = max(-48.0, levels["right_rms_db"])
            self.left_level = max(0.01, min(0.99, (l_db + 48.0) / 48.0))
            self.right_level = max(0.01, min(0.99, (r_db + 48.0) / 48.0))
            # Store dB values for LUFS/TruePeak in main GUI
            self._last_levels_db = levels
        else:
            # Fallback: simple sine wave simulation
            import math
            t = position_ms / 1000.0
            base_l = 0.5 + 0.3 * math.sin(t * 2.5) + 0.1 * math.sin(t * 7.3)
            base_r = 0.5 + 0.3 * math.sin(t * 2.7 + 0.5) + 0.1 * math.sin(t * 6.9)
            variation = (position_ms % 100) / 500.0
            self.left_level = max(0.1, min(0.95, base_l + variation))
            self.right_level = max(0.1, min(0.95, base_r - variation * 0.5))

        # Update peaks
        if self.left_level > self.peak_left:
            self.peak_left = self.left_level
        if self.right_level > self.peak_right:
            self.peak_right = self.right_level

    def _update_display(self):
        """Periodic display update"""
        self.update()

    def _decay_peaks(self):
        self.peak_left = max(0, self.peak_left - 0.02)
        self.peak_right = max(0, self.peak_right - 0.02)

    def _get_segment_color(self, segment_index: int) -> QColor:
        """Get color for segment based on its position (green/yellow/red zones)"""
        ratio = segment_index / self.num_segments
        if ratio < 0.6:
            return QColor(Colors.METER_GREEN)
        elif ratio < 0.8:
            return QColor(Colors.METER_YELLOW)
        else:
            return QColor(Colors.METER_RED)

    def _draw_meter_bar(self, painter: QPainter, x: int, y: int, width: int, height: int,
                        level: float, peak: float, is_left: bool):
        """Draw a single Waves-style segmented meter bar"""
        segment_height = height // self.num_segments
        gap = 2
        segment_h = segment_height - gap

        # Draw chrome bezel (dark gradient border)
        bezel_pen = QPen(QColor(Colors.CHROME_DARK), 1)
        painter.setPen(bezel_pen)
        painter.drawRect(x, y, width, height)

        # Draw inner highlight for chrome effect
        highlight_pen = QPen(QColor(Colors.CHROME_HIGHLIGHT), 1)
        painter.setPen(highlight_pen)
        painter.drawLine(x + 1, y + 1, x + width - 2, y + 1)
        painter.drawLine(x + 1, y + 1, x + 1, y + height - 2)

        # Fill recessed background
        painter.fillRect(x + 2, y + 2, width - 4, height - 4, QColor(Colors.PANEL_DEEP))

        # Draw LED segments from bottom to top
        filled_segments = int(level * self.num_segments)

        for i in range(self.num_segments):
            seg_y = y + height - (i + 1) * segment_height

            if i < filled_segments:
                # Segment is lit
                color = self._get_segment_color(i)
                painter.fillRect(x + 4, seg_y + gap // 2, width - 8, segment_h, color)

                # Add glow effect for lit segments
                glow_color = color
                glow_color.setAlpha(80)
                painter.fillRect(x + 3, seg_y + gap // 2 - 1, width - 6, segment_h + 2, glow_color)
            else:
                # Unlit segment - very dim
                painter.fillRect(x + 4, seg_y + gap // 2, width - 8, segment_h,
                                QColor(Colors.CHROME_DARK))

        # Draw peak hold line (bright horizontal line that slowly decays)
        if peak > 0:
            peak_segment = int(peak * self.num_segments)
            peak_y = y + height - peak_segment * segment_height

            # Bright red/white peak indicator
            peak_color = QColor(Colors.METER_RED) if peak > 0.8 else QColor("#ffffff")
            painter.setPen(QPen(peak_color, 3))
            painter.drawLine(x + 2, peak_y, x + width - 2, peak_y)

            # Glow around peak
            peak_color.setAlpha(100)
            glow_pen = QPen(peak_color, 5)
            painter.setPen(glow_pen)
            painter.drawLine(x + 2, peak_y, x + width - 2, peak_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w = self.width()
        h = self.height()

        # Main background
        painter.fillRect(self.rect(), QColor(Colors.BG_PRIMARY))

        # Layout: dB scale on left, then two meter bars (L and R)
        db_width = 28
        bar_width = (w - db_width - 60) // 2
        meter_height = h - 50
        meter_top = 15

        left_bar_x = db_width + 15
        right_bar_x = left_bar_x + bar_width + 15

        # Draw dB scale on left
        painter.setPen(QColor(Colors.TEXT_TERTIARY))
        painter.setFont(QFont("Inter", 7))
        db_values = [(0, "0"), (-3, "-3"), (-6, "-6"), (-12, "-12"),
                     (-18, "-18"), (-24, "-24"), (-36, "-36"), (-48, "-48")]

        for db_val, db_label in db_values:
            # Map dB to screen position (0 at top, -48 at bottom)
            if db_val == 0:
                ratio = 0.0
            else:
                ratio = min(1.0, abs(db_val) / 48.0)

            y = meter_top + int(ratio * meter_height)
            painter.drawText(5, y + 3, db_label)
            painter.drawLine(db_width - 3, y, db_width, y)

        # Draw left and right meter bars
        self._draw_meter_bar(painter, left_bar_x, meter_top, bar_width, meter_height,
                            self.left_level, self.peak_left, True)
        self._draw_meter_bar(painter, right_bar_x, meter_top, bar_width, meter_height,
                            self.right_level, self.peak_right, False)

        # Draw channel labels in amber/gold color at bottom
        painter.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        painter.setPen(QColor(Colors.LED_AMBER))

        left_label_x = left_bar_x + bar_width // 2 - 5
        right_label_x = right_bar_x + bar_width // 2 - 5

        painter.drawText(left_label_x, h - 10, "L")
        painter.drawText(right_label_x, h - 10, "R")


# ==================== LUFS Display ====================
class LUFSDisplay(QWidget):
    """Waves-style LUFS value display with recessed LCD appearance"""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label
        self.value = -14.0
        self.unit_text = "LUFS"
        self.setMinimumWidth(100)
        self.setFixedHeight(80)
        self.setStyleSheet(f"background: {Colors.BG_PRIMARY};")

    def setValue(self, value: float):
        self.value = value
        self.update()

    def _get_accent_color(self) -> str:
        """Get color based on label type"""
        label_upper = self.label.upper()
        if "INTEGRATED" in label_upper:
            return Colors.TEAL_GLOW
        elif "SHORT" in label_upper:
            return Colors.LED_AMBER_GLOW
        elif "MOMENTARY" in label_upper:
            return Colors.METER_RED
        else:
            return Colors.LED_AMBER_GLOW

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Main background
        painter.fillRect(self.rect(), QColor(Colors.BG_PRIMARY))

        # Frame/border area (chrome bezel)
        frame_margin = 4
        frame_rect = QRect(frame_margin, frame_margin, w - 2 * frame_margin, h - 2 * frame_margin)

        # Draw outer chrome bezel - dark border
        bezel_pen = QPen(QColor(Colors.CHROME_DARK), 1)
        painter.setPen(bezel_pen)
        painter.drawRect(frame_rect)

        # Draw inner highlight line for chrome effect
        highlight_pen = QPen(QColor(Colors.CHROME_HIGHLIGHT), 1)
        painter.setPen(highlight_pen)
        painter.drawLine(frame_margin + 1, frame_margin + 1,
                        w - frame_margin - 2, frame_margin + 1)
        painter.drawLine(frame_margin + 1, frame_margin + 1,
                        frame_margin + 1, h - frame_margin - 2)

        # Recessed LCD background with gradient
        lcd_rect = QRect(frame_margin + 2, frame_margin + 2,
                        w - 2 * frame_margin - 4, h - 2 * frame_margin - 4)

        # Create subtle gradient for LCD appearance
        gradient = QLinearGradient(0, lcd_rect.top(), 0, lcd_rect.bottom())
        gradient.setColorAt(0.0, QColor(Colors.PANEL_DEEP))
        gradient.setColorAt(1.0, QColor("#050507"))  # Slightly darker at bottom

        painter.fillRect(lcd_rect, gradient)

        # Draw inner shadow/inset effect
        shadow_pen = QPen(QColor(Colors.CHROME_DARK), 1)
        painter.setPen(shadow_pen)
        painter.drawLine(frame_margin + 2, h - frame_margin - 3,
                        w - frame_margin - 3, h - frame_margin - 3)
        painter.drawLine(w - frame_margin - 3, frame_margin + 2,
                        w - frame_margin - 3, h - frame_margin - 3)

        # Layout regions
        top_height = 18
        middle_height = h - 2 * frame_margin - 4 - top_height - 14
        bottom_height = 14

        # Top section: Label in gold/amber
        label_rect = QRect(frame_margin + 3, frame_margin + 3,
                          w - 2 * frame_margin - 6, top_height)
        painter.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        painter.setPen(QColor(Colors.LED_AMBER))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, self.label.upper())

        # Middle section: Large value display
        value_rect = QRect(frame_margin + 3, frame_margin + 3 + top_height,
                          w - 2 * frame_margin - 6, middle_height)
        painter.setFont(QFont("Courier New", 28, QFont.Weight.Bold))
        accent_color = self._get_accent_color()
        painter.setPen(QColor(accent_color))
        painter.drawText(value_rect, Qt.AlignmentFlag.AlignCenter, f"{self.value:.1f}")

        # Bottom section: Unit label in dimmed text
        unit_rect = QRect(frame_margin + 3,
                         frame_margin + 3 + top_height + middle_height,
                         w - 2 * frame_margin - 6, bottom_height)
        painter.setFont(QFont("Inter", 7))
        painter.setPen(QColor(Colors.TEXT_TERTIARY))
        painter.drawText(unit_rect, Qt.AlignmentFlag.AlignCenter, self.unit_text)


# ==================== Collapsible Section ====================
class CollapsibleSection(QWidget):
    """Collapsible section with header"""
    
    def __init__(self, title: str, icon: str = "", parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self.header = QPushButton(f"▼ {icon} {title}")
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {Colors.BG_TERTIARY}, stop:0.5 {Colors.BG_SECONDARY},
                    stop:1 {Colors.BG_PRIMARY});
                color: {Colors.GOLD};
                border: none;
                border-bottom: 1px solid {Colors.BORDER};
                border-top: 1px solid {Colors.BORDER_LIGHT};
                padding: 10px;
                text-align: left;
                font-size: 11px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                font-family: 'Menlo', 'Menlo', 'Courier New', monospace;
            }}
            QPushButton:hover {{
                color: {Colors.ACCENT};
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #2E2E36, stop:0.5 {Colors.BG_TERTIARY},
                    stop:1 {Colors.BG_SECONDARY});
            }}
        """)
        self.header.clicked.connect(self._toggle)
        layout.addWidget(self.header)
        
        # Content
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.content)
        
        self.title_text = title
        self.icon = icon
        
    def _toggle(self):
        visible = self.header.isChecked()
        self.content.setVisible(visible)
        arrow = "▼" if visible else "▶"
        self.header.setText(f"{arrow} {self.icon} {self.title_text}")
        
    def addWidget(self, widget):
        self.content_layout.addWidget(widget)


# ==================== Video Thread for OpenCV ====================
class VideoThread(QThread):
    """Thread stub for backward compatibility - frame rendering now handled by QTimer in VideoPreviewCard"""
    change_pixmap_signal = pyqtSignal(object)  # np.ndarray
    position_changed = pyqtSignal(int)  # position in ms
    duration_changed = pyqtSignal(int)  # duration in ms

    def __init__(self, video_path: str = None):
        super().__init__()
        self.video_path = video_path
        self._run_flag = False
        self._pause_flag = False
        self._speed = 1.0
        self._seek_position = -1
        self.duration_ms = 0
        self.current_position_ms = 0

    def set_video(self, path: str):
        """Set video path and prepare for playback"""
        self.video_path = path
        self._seek_position = 0

    def set_speed(self, speed: float):
        """Set playback speed"""
        self._speed = speed

    def seek(self, position_ms: int):
        """Seek to position"""
        self._seek_position = position_ms

    def run(self):
        """No-op stub - rendering handled by VideoPreviewCard's QTimer"""
        return

    def pause(self):
        self._pause_flag = True

    def resume(self):
        self._pause_flag = False

    def stop(self):
        self._run_flag = False
        self._pause_flag = False
        self.wait()


# ==================== Detached Video Window ====================
class DetachedVideoWindow(QWidget):
    """Floating video preview window - can be moved to any monitor"""
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Preview")
        self.setStyleSheet("background: #000000;")
        self.setGeometry(100, 100, 640, 360)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: #000000;")
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setScaledContents(False)
        layout.addWidget(self.video_label, 1)

        # Bottom info bar
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(8)

        # Track name + time label
        self.track_label = QLabel("Ready")
        self.track_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_PRIMARY};
                font-size: 11px;
                font-family: 'Menlo', 'Courier New', monospace;
            }}
        """)
        info_layout.addWidget(self.track_label)
        info_layout.addStretch()

        # Always on top button (pin)
        self.pin_btn = QPushButton("📌")
        self.pin_btn.setFixedSize(28, 28)
        self.pin_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                border-radius: 4px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
            QPushButton:checked {{
                background: {Colors.ACCENT};
                color: white;
            }}
        """)
        self.pin_btn.setCheckable(True)
        self.pin_btn.setChecked(True)
        self.pin_btn.clicked.connect(self._toggle_always_on_top)
        info_layout.addWidget(self.pin_btn)

        # Dock button
        self.dock_btn = QPushButton("⬅ Dock")
        self.dock_btn.setFixedSize(60, 28)
        self.dock_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        info_layout.addWidget(self.dock_btn)

        layout.addLayout(info_layout)

    def _toggle_always_on_top(self):
        """Toggle always on top flag"""
        if self.pin_btn.isChecked():
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    def set_frame(self, cv_frame):
        """Display a cv2 frame"""
        if not CV2_AVAILABLE or cv_frame is None:
            return

        try:
            rgb_image = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

            label_size = self.video_label.size()
            scaled = qt_image.scaled(
                label_size.width(),
                label_size.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.video_label.setPixmap(QPixmap.fromImage(scaled))
        except Exception as e:
            print(f"[DETACHED VIDEO] Frame display error: {e}")

    def set_track_info(self, track_name: str, time_str: str):
        """Update track info display"""
        self.track_label.setText(f"{track_name} · {time_str}")

    def closeEvent(self, event):
        """Handle window close"""
        self.closed.emit()
        super().closeEvent(event)


# ==================== Video Preview Card ====================
class VideoPreviewCard(QFrame):
    """Video/GIF preview with QTimer-based frame rendering - no threading overhead"""

    speed_changed = pyqtSignal(float)  # Emit speed changes to connect to audio player

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_CARD};
                border-radius: 12px;
                border: 1px solid {Colors.BORDER};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Header with play button
        header = QHBoxLayout()
        title = QLabel("🎬 Video Preview")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 13px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(32, 32)
        self.play_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        self.play_btn.clicked.connect(self._toggle_play)
        header.addWidget(self.play_btn)

        # Speed controls
        self.speed_buttons = {}
        self.current_speed = 1.0
        for speed_text, speed_value in [("1x", 1.0), ("1.5x", 1.5), ("2x", 2.0)]:
            btn = QPushButton(speed_text)
            btn.setFixedSize(32, 24)
            btn.setProperty("speed_value", speed_value)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.BG_TERTIARY if speed_value != 1.0 else Colors.ACCENT};
                    color: {Colors.TEXT_SECONDARY if speed_value != 1.0 else 'white'};
                    border: none;
                    border-radius: 4px;
                    font-size: 10px;
                }}
                QPushButton:hover {{
                    background: {Colors.BORDER};
                }}
            """)
            btn.clicked.connect(lambda checked, sv=speed_value: self._set_playback_speed(sv))
            header.addWidget(btn)
            self.speed_buttons[speed_value] = btn

        # Pop Out button
        self.popout_btn = QPushButton("Pop Out ⬜")
        self.popout_btn.setFixedSize(80, 32)
        self.popout_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        self.popout_btn.clicked.connect(self._toggle_detached_window)
        header.addWidget(self.popout_btn)

        layout.addLayout(header)

        # Realtime display label
        self.realtime_label = QLabel("00:00 / 00:00")
        self.realtime_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.realtime_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.ACCENT};
                font-size: 14px;
                font-weight: bold;
                font-family: 'Menlo', 'Courier New', monospace;
                background: {Colors.BG_TERTIARY};
                border-radius: 6px;
                padding: 4px 8px;
            }}
        """)
        layout.addWidget(self.realtime_label)

        # Container for video preview
        self.preview_container = QWidget()
        self.preview_container.setMinimumHeight(180)
        self.preview_container.setStyleSheet("background: #000000; border-radius: 8px;")
        self.preview_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        container_layout = QGridLayout(self.preview_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: #000000;")
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setScaledContents(False)
        container_layout.addWidget(self.video_label, 0, 0)

        # GIF label OVERLAY (top layer) - HIDDEN by default to avoid blocking video
        self.gif_label = QLabel()
        self.gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gif_label.setStyleSheet("background: transparent;")
        self.gif_label.setScaledContents(False)
        self.gif_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        container_layout.addWidget(self.gif_label, 0, 0)
        self.gif_label.raise_()
        self.gif_label.hide()  # CRITICAL: Hide until GIF is actually loaded

        layout.addWidget(self.preview_container, 1)

        # Caption
        self.caption = QLabel("No video loaded")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setStyleSheet(f"color: {Colors.TEXT_TERTIARY}; font-size: 10px;")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)

        # Video thread stub (for backward compatibility, no-op)
        self.video_thread = None

        # QTimer for frame rendering (12fps for smooth playback without blocking)
        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self._render_frame)
        self._render_timer.setInterval(83)  # ~12fps (enough for 5-sec looping videos)

        # PRE-CACHED FRAMES: All frames pre-loaded as QPixmap for instant rendering
        self._frame_cache: Dict[int, list] = {}     # {video_idx: [QPixmap, ...]}
        self._frame_fps: Dict[int, float] = {}      # {video_idx: fps}
        self._video_durations: Dict[int, int] = {}   # {video_idx: duration_ms}
        self._current_audio_pos_ms = 0
        self._force_render = False
        self._last_rendered_frame = (-1, -1)  # (video_idx, frame_idx) to avoid re-rendering same frame

        # State
        self.video_files: List[MediaFile] = []
        self.current_type = "video"
        self.is_playing = False
        self.gif_movie = None
        self._gif_frames: list = []       # QImage frames for GIF (replaces QMovie)
        self._gif_delays: list = []       # Frame delays in ms
        self._gif_frame_idx: int = 0
        self._gif_timer: Optional[QTimer] = None
        self.use_opencv = CV2_AVAILABLE

        # Detached window
        self.detached_window: Optional[DetachedVideoWindow] = None
        
    def _preload_video_frames(self, video_idx: int):
        """Pre-load ALL frames from a short video into memory as QImage.
        Uses cv2 first, falls back to ffmpeg+QImage if cv2 fails."""
        if video_idx in self._frame_cache:
            return  # Already cached

        if not (0 <= video_idx < len(self.video_files)):
            print(f"[VIDEO PREVIEW] ❌ video_idx {video_idx} out of range (have {len(self.video_files)} videos)")
            return

        video = self.video_files[video_idx]
        print(f"[VIDEO PREVIEW] 🔄 Preloading video {video_idx}: {video.path}")

        # Try cv2 first
        if CV2_AVAILABLE:
            frames = self._preload_via_cv2(video_idx, video.path)
            if frames:
                self._frame_cache[video_idx] = frames
                print(f"[VIDEO PREVIEW] ✅ cv2 cached video {video_idx}: {len(frames)} frames")
                return

        # Fallback: ffmpeg frame extraction
        print(f"[VIDEO PREVIEW] ⚠️ cv2 failed, trying ffmpeg fallback...")
        frames = self._preload_via_ffmpeg(video_idx, video.path)
        if frames:
            self._frame_cache[video_idx] = frames
            print(f"[VIDEO PREVIEW] ✅ ffmpeg cached video {video_idx}: {len(frames)} frames")
        else:
            # Last resort: create a colored test frame so we know display works
            print(f"[VIDEO PREVIEW] ❌ Both cv2 and ffmpeg failed! Creating test pattern...")
            self._create_test_frames(video_idx)

    def _preload_via_cv2(self, video_idx, path):
        """Load frames using cv2"""
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"[VIDEO PREVIEW] ❌ cv2 cannot open: {path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = int((total_frames / fps) * 1000) if fps > 0 else 5000

            self._video_durations[video_idx] = duration_ms
            self._frame_fps[video_idx] = fps
            print(f"[VIDEO PREVIEW] cv2 opened: fps={fps}, total={total_frames}, duration={duration_ms}ms")

            target_w, target_h = 640, 360
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                if not qimg.isNull():
                    frames.append(qimg)

            cap.release()
            if frames:
                print(f"[VIDEO PREVIEW] ✅ cv2: {len(frames)} frames, first={frames[0].width()}x{frames[0].height()}")
            else:
                print(f"[VIDEO PREVIEW] ❌ cv2: 0 frames read (codec issue?)")
            return frames if frames else None

        except Exception as e:
            print(f"[VIDEO PREVIEW] ❌ cv2 error: {e}")
            return None

    def _preload_via_ffmpeg(self, video_idx, path):
        """Load frames using ffmpeg subprocess (more codec support)"""
        import subprocess, tempfile, glob as glob_mod
        try:
            # Get video info
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "stream=r_frame_rate,duration,nb_frames",
                 "-select_streams", "v:0", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10
            )
            info = probe.stdout.strip().split(',') if probe.stdout.strip() else []

            if len(info) >= 1 and '/' in info[0]:
                num, den = info[0].split('/')
                fps = float(num) / float(den) if float(den) > 0 else 24
            else:
                fps = 24

            # Extract frames to temp directory
            tmpdir = tempfile.mkdtemp(prefix="longplay_frames_")
            result = subprocess.run(
                ["ffmpeg", "-i", path, "-vf", "scale=640:360", "-q:v", "3",
                 "-f", "image2", f"{tmpdir}/frame_%04d.jpg"],
                capture_output=True, text=True, timeout=30
            )

            # Load extracted frames as QImage
            frame_files = sorted(glob_mod.glob(f"{tmpdir}/frame_*.jpg"))
            frames = []
            for fpath in frame_files:
                qimg = QImage(fpath)
                if not qimg.isNull():
                    frames.append(qimg.copy())

            # Cleanup temp files
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

            if frames:
                duration_ms = int((len(frames) / fps) * 1000)
                self._video_durations[video_idx] = duration_ms
                self._frame_fps[video_idx] = fps
                print(f"[VIDEO PREVIEW] ✅ ffmpeg: {len(frames)} frames, fps={fps}")
            return frames if frames else None

        except Exception as e:
            print(f"[VIDEO PREVIEW] ❌ ffmpeg error: {e}")
            return None

    def _create_test_frames(self, video_idx):
        """Create colored test pattern to verify display pipeline works"""
        from PyQt6.QtGui import QPainter, QFont, QColor as QC
        target_w, target_h = 640, 360
        frames = []
        colors = [QC(255, 100, 50), QC(50, 200, 100), QC(50, 100, 255)]
        for i in range(24):  # 1 second at 24fps
            img = QImage(target_w, target_h, QImage.Format.Format_RGB888)
            img.fill(colors[i % len(colors)])
            p = QPainter(img)
            p.setPen(QC(255, 255, 255))
            f = QFont("Arial", 24)
            p.setFont(f)
            p.drawText(img.rect(), Qt.AlignmentFlag.AlignCenter,
                       f"TEST FRAME {i}\nVideo {video_idx}\nDisplay OK!")
            p.end()
            frames.append(img)
        self._frame_cache[video_idx] = frames
        self._video_durations[video_idx] = 1000
        self._frame_fps[video_idx] = 24
        print(f"[VIDEO PREVIEW] 🎨 Created {len(frames)} test frames for video {video_idx}")

    _render_debug_count = 0  # Class-level debug counter

    def _render_frame(self):
        """Called by QTimer - render current frame from pre-cached QImages.
        ZERO cv2 calls during playback - all frames are pre-loaded!"""
        if not self.is_playing and not self._force_render:
            return

        if not self.video_files:
            return

        pos = self._current_audio_pos_ms
        video_idx = self._get_video_for_position(pos)

        # Get cached frames for this video (lazy-load if needed)
        if video_idx not in self._frame_cache:
            self._preload_video_frames(video_idx)

        frames = self._frame_cache.get(video_idx)
        if not frames:
            if VideoPreviewCard._render_debug_count < 5:
                print(f"[VIDEO PREVIEW] ⚠️ No frames in cache for video {video_idx}")
                VideoPreviewCard._render_debug_count += 1
            self._force_render = False
            return

        # Calculate which frame to show
        duration_ms = self._video_durations.get(video_idx, 5000)
        video_pos_ms = pos % duration_ms if duration_ms > 0 else 0
        fps = self._frame_fps.get(video_idx, 24)
        frame_idx = int((video_pos_ms / 1000.0) * fps)
        frame_idx = max(0, min(frame_idx, len(frames) - 1))

        # Skip if same frame as last render (save CPU)
        cache_key = (video_idx, frame_idx)
        if cache_key == self._last_rendered_frame and not self._force_render:
            return
        self._last_rendered_frame = cache_key

        # Debug: log first few renders
        if VideoPreviewCard._render_debug_count < 10:
            print(f"[VIDEO PREVIEW] 🎬 Render: video={video_idx}, frame={frame_idx}/{len(frames)}, pos={pos}ms")
            VideoPreviewCard._render_debug_count += 1

        # Display the pre-cached QImage - convert to QPixmap at render time
        qimg = frames[frame_idx]
        self._display_qimage(qimg)
        self._force_render = False

    def _display_qimage(self, qimage):
        """Display a QImage on both embedded label and detached window.
        Converts QImage→QPixmap at render time (macOS safe)."""
        if qimage is None or qimage.isNull():
            print(f"[VIDEO PREVIEW] ⚠️ Null QImage, cannot display")
            return

        try:
            # Convert QImage → QPixmap (safe in main thread with active GUI)
            pixmap = QPixmap.fromImage(qimage)
            if pixmap.isNull():
                print(f"[VIDEO PREVIEW] ⚠️ QPixmap.fromImage returned null!")
                return

            # Scale to fit embedded label
            label_size = self.video_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled = pixmap.scaled(
                    label_size.width(),
                    label_size.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.video_label.setPixmap(scaled)
            else:
                # Fallback: set directly without scaling
                self.video_label.setPixmap(pixmap)

            # Send to detached window if open
            if self.detached_window and self.detached_window.isVisible():
                det_size = self.detached_window.video_label.size()
                if det_size.width() > 0 and det_size.height() > 0:
                    det_scaled = pixmap.scaled(
                        det_size.width(),
                        det_size.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.detached_window.video_label.setPixmap(det_scaled)
                else:
                    self.detached_window.video_label.setPixmap(pixmap)
        except Exception as e:
            import traceback
            print(f"[VIDEO PREVIEW] ❌ Display error: {e}")
            traceback.print_exc()

    def setVideos(self, videos: List[MediaFile]):
        """Load videos - prepare cv2.VideoCapture objects"""
        self.video_files = videos
        if videos:
            video = videos[0]
            print(f"[VIDEO PREVIEW] ========================================")
            print(f"[VIDEO PREVIEW] Loading video: {video.path}")
            print(f"[VIDEO PREVIEW] Total videos loaded: {len(videos)}")
            print(f"[VIDEO PREVIEW] File exists: {os.path.exists(video.path)}")
            print(f"[VIDEO PREVIEW] Using OpenCV: {self.use_opencv}")

            if not os.path.exists(video.path):
                print(f"[VIDEO PREVIEW] ERROR: File not found: {video.path}")
                self.caption.setText(f"❌ File not found: {video.name}")
                return

            if len(videos) > 1:
                self.caption.setText(f"🎬 {os.path.basename(video.path)} (+{len(videos)-1} more)")
            else:
                self.caption.setText(os.path.basename(video.path))
            self.current_type = "video"

            # CRITICAL: Hide GIF overlay so it doesn't block video
            self.gif_label.hide()

            # Reset render debug counter
            VideoPreviewCard._render_debug_count = 0

            # Pre-cache all video frames (cv2 → ffmpeg → test pattern fallback)
            for i, v in enumerate(videos):
                self.caption.setText(f"⏳ Loading video {i+1}/{len(videos)}...")
                QApplication.processEvents()
                self._preload_video_frames(i)

            total_cached = sum(len(f) for f in self._frame_cache.values())
            if total_cached > 0:
                self.caption.setText(f"🎬 {len(videos)} videos ({total_cached} frames)")
            else:
                self.caption.setText(f"❌ Cannot load video frames")
            print(f"[VIDEO PREVIEW] ✅ All videos cached: {total_cached} total frames")

            # Show first frame immediately
            self._force_render = True
            self._current_audio_pos_ms = 0
            self._render_frame()

            # Auto-play with small delay to ensure widget is laid out
            self.is_playing = True
            self.play_btn.setText("⏸")
            QTimer.singleShot(100, self._render_timer.start)
            print(f"[VIDEO PREVIEW] ✅ Frame rendering scheduled!")
            
    def setGIF(self, gif_path: str):
        """Load GIF as OVERLAY on video.
        ULTRA-SAFE: Uses ffmpeg subprocess to extract frames (crash-proof on macOS).
        NO QMovie, NO QImageReader (both can segfault on macOS with certain GIFs).
        ffmpeg crashes stay in subprocess and don't kill main app.
        """
        import glob as _glob
        print(f"[GIF setGIF] START: {gif_path}")
        try:
            # Stop existing GIF animation
            print(f"[GIF setGIF] Step 1: Stopping existing animation...")
            if self._gif_timer:
                self._gif_timer.stop()
                self._gif_timer = None
            self._gif_frames = []
            self._gif_delays = []
            self._gif_frame_idx = 0
            self.gif_movie = None
            print(f"[GIF setGIF] Step 1: OK")

            if not os.path.exists(gif_path):
                print(f"[GIF setGIF] File not found: {gif_path}")
                return

            # Extract GIF frames using ffmpeg SUBPROCESS (crash-safe)
            print(f"[GIF setGIF] Step 2: Creating temp dir...")
            tmpdir = tempfile.mkdtemp(prefix="longplay_gif_")
            print(f"[GIF setGIF] Step 2: tmpdir={tmpdir}")
            try:
                label_size = self.video_label.size()
                scale_w = max(320, label_size.width())
                scale_h = max(180, label_size.height())
                print(f"[GIF setGIF] Step 3: Extracting frames via ffmpeg ({scale_w}x{scale_h})...")

                out_path = os.path.join(tmpdir, "frame_%04d.png")
                cmd = [
                    "ffmpeg", "-i", gif_path,
                    "-vf", f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=decrease",
                    "-q:v", "2", "-v", "quiet",
                    "-f", "image2", out_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                print(f"[GIF setGIF] Step 3: ffmpeg returned {result.returncode}")

                if result.returncode != 0:
                    print(f"[GIF setGIF] Step 3b: ffmpeg stderr: {result.stderr[:200]}")
                    # Simpler fallback
                    cmd2 = ["ffmpeg", "-i", gif_path, "-v", "quiet",
                            "-f", "image2", os.path.join(tmpdir, "f_%04d.jpg")]
                    subprocess.run(cmd2, capture_output=True, text=True, timeout=30)

                # Load extracted frames as QImage
                print(f"[GIF setGIF] Step 4: Loading frames from disk...")
                frame_files = sorted(_glob.glob(os.path.join(tmpdir, "frame_*.png")))
                if not frame_files:
                    frame_files = sorted(_glob.glob(os.path.join(tmpdir, "f_*.jpg")))
                print(f"[GIF setGIF] Step 4: Found {len(frame_files)} frame files")

                for fpath in frame_files[:200]:  # Limit to 200 frames
                    qimg = QImage(fpath)
                    if not qimg.isNull():
                        self._gif_frames.append(qimg.copy())

                print(f"[GIF setGIF] Step 4: Loaded {len(self._gif_frames)} QImage frames")

            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            if not self._gif_frames:
                print(f"[GIF setGIF] No frames extracted, skipping display")
                return

            # Display first frame
            print(f"[GIF setGIF] Step 5: Displaying first frame...")
            pixmap = QPixmap.fromImage(self._gif_frames[0])
            if pixmap.isNull():
                print(f"[GIF setGIF] Step 5: WARNING - pixmap is null!")
                return
            self.gif_label.setPixmap(pixmap)
            self.gif_label.show()
            print(f"[GIF setGIF] Step 5: OK")

            # Start animation timer (~12 FPS for GIF)
            print(f"[GIF setGIF] Step 6: Starting animation timer...")
            self._gif_timer = QTimer(self)
            self._gif_timer.timeout.connect(self._cycle_gif_frame)
            self._gif_timer.start(83)  # ~12fps

            print(f"[GIF setGIF] DONE: {len(self._gif_frames)} frames loaded")

            if self.video_files:
                self.caption.setText(f"Video + GIF: {os.path.basename(gif_path)}")
            else:
                self.caption.setText(os.path.basename(gif_path))

            if not self.video_files:
                self.current_type = "gif"
        except Exception as e:
            import traceback
            print(f"[GIF setGIF] ERROR: {e}")
            traceback.print_exc()

    def _cycle_gif_frame(self):
        """Cycle to next GIF frame (manual animation - replaces QMovie)"""
        try:
            if not self._gif_frames:
                return
            self._gif_frame_idx = (self._gif_frame_idx + 1) % len(self._gif_frames)
            pixmap = QPixmap.fromImage(self._gif_frames[self._gif_frame_idx])
            self.gif_label.setPixmap(pixmap)
        except Exception:
            pass  # Silently skip frame on error

    def _toggle_play(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        """Play video"""
        self.is_playing = True
        self.play_btn.setText("⏸")

        if self.current_type == "gif" and self._gif_frames:
            if self._gif_timer:
                self._gif_timer.start(83)
        elif self.use_opencv:
            self._render_timer.start()
            print(f"[VIDEO PREVIEW] Playing video...")

    def pause(self):
        """Pause video"""
        self.is_playing = False
        self.play_btn.setText("▶")

        if self.current_type == "gif" and self._gif_timer:
            self._gif_timer.stop()
        elif self.use_opencv:
            self._render_timer.stop()
            print(f"[VIDEO PREVIEW] Paused video")
            
    def seek(self, position_ms: int):
        """Seek to position and render frame immediately"""
        self._current_audio_pos_ms = position_ms
        self._force_render = True
        self._render_frame()
        self._update_realtime_display(position_ms)

    def set_audio_context(self, audio_files, crossfade_sec=5):
        """Set audio context so video can sync to current track"""
        self._audio_files = audio_files
        self._crossfade_sec = crossfade_sec

    def _get_video_for_position(self, position_ms: int):
        """V4.31.3: Determine which video should play at given audio position"""
        audio_files = getattr(self, '_audio_files', None)
        if not audio_files or not self.video_files:
            return 0

        crossfade_sec = getattr(self, '_crossfade_sec', 5)
        position_sec = position_ms / 1000.0

        # Find which audio track is playing at this position
        cumulative = 0
        current_track_idx = 0
        for i, af in enumerate(audio_files):
            if i == 0:
                track_end = cumulative + af.duration
            else:
                track_end = cumulative + max(0, af.duration - crossfade_sec)

            if position_sec < track_end or i == len(audio_files) - 1:
                current_track_idx = i
                break
            cumulative = track_end

        # V4.31.3 FIX: Check if user has explicitly set different assignments
        # If all tracks have default assignment (0), use round-robin distribution
        all_default = all(getattr(af, 'video_assignment', 0) == 0 for af in audio_files)
        num_videos = len(self.video_files)

        if all_default and num_videos > 1:
            # No explicit assignment - distribute round-robin across all videos
            # Calculate tracks per video group
            tracks_per_video = max(1, len(audio_files) // num_videos)
            video_idx = min(current_track_idx // tracks_per_video, num_videos - 1)
            return video_idx
        else:
            # User has set explicit assignments - use them
            video_assignment = getattr(audio_files[current_track_idx], 'video_assignment', 0)
            if video_assignment < num_videos:
                return video_assignment
            return current_track_idx % num_videos

    def stop_playback(self):
        """Stop video playback and cleanup"""
        self._render_timer.stop()
        self.is_playing = False
        self.play_btn.setText("▶")

    def _set_playback_speed(self, speed: float):
        """Set playback speed (1x, 1.5x, 2x) and emit signal"""
        self.current_speed = speed
        self.speed_changed.emit(speed)

        print(f"[VIDEO PREVIEW] Speed set to {speed}x")

        # Update button styles to show active speed
        for speed_val, btn in self.speed_buttons.items():
            if speed_val == speed:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {Colors.ACCENT};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-size: 10px;
                    }}
                    QPushButton:hover {{
                        background: {Colors.ACCENT_DIM};
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {Colors.BG_TERTIARY};
                        color: {Colors.TEXT_SECONDARY};
                        border: none;
                        border-radius: 4px;
                        font-size: 10px;
                    }}
                    QPushButton:hover {{
                        background: {Colors.BORDER};
                    }}
                """)
    
    def _toggle_detached_window(self):
        """Toggle detached video window"""
        if self.detached_window is None or not self.detached_window.isVisible():
            self._open_detached_window()
        else:
            self._close_detached_window()

    def _open_detached_window(self):
        """Open floating detached video window"""
        self.detached_window = DetachedVideoWindow()
        self.detached_window.dock_btn.clicked.connect(self._close_detached_window)
        self.detached_window.closed.connect(self._on_detached_window_closed)
        self.detached_window.show()

        self.popout_btn.setText("Pop In ⬜")
        self.popout_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)

        # Send current frame to detached window
        self._force_render = True
        self._render_frame()

    def _close_detached_window(self):
        """Close floating detached video window"""
        if self.detached_window:
            self.detached_window.close()
            self.detached_window = None

        self.popout_btn.setText("Pop Out ⬜")
        self.popout_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)

    def _on_detached_window_closed(self):
        """Handle detached window close"""
        self._close_detached_window()

    def _update_realtime_display(self, position_ms: int):
        """Update the realtime display with current position
        V4.31.2: Shows current track name + overall progress (like CapCut)
        """
        audio_files = getattr(self, '_audio_files', None)
        crossfade_sec = getattr(self, '_crossfade_sec', 5)

        if audio_files:
            total_ms = 0
            for i, af in enumerate(audio_files):
                if i == 0:
                    total_ms += af.duration * 1000
                else:
                    total_ms += max(0, af.duration - crossfade_sec) * 1000
            total_ms = int(total_ms)
        else:
            total_ms = 0

        # Format current position
        pos_min = position_ms // 60000
        pos_sec = (position_ms % 60000) // 1000
        pos_ms = (position_ms % 1000) // 10

        # Format total duration
        total_min = total_ms // 60000
        total_sec = (total_ms % 60000) // 1000

        time_str = f"{pos_min:02d}:{pos_sec:02d}.{pos_ms:02d} / {total_min:02d}:{total_sec:02d}"

        # Add speed indicator if not 1x
        if self.current_speed != 1.0:
            time_str += f" [{self.current_speed}x]"

        self.realtime_label.setText(time_str)

        # Update caption to show current track name
        if audio_files:
            position_sec = position_ms / 1000.0
            cumulative = 0
            current_name = ""
            current_idx = 0
            for i, af in enumerate(audio_files):
                if i == 0:
                    track_end = cumulative + af.duration
                else:
                    track_end = cumulative + max(0, af.duration - crossfade_sec)

                if position_sec < track_end or i == len(audio_files) - 1:
                    current_name = af.name
                    current_idx = i
                    break
                cumulative = track_end

            # Clean up track name
            display_name = re.sub(r'^[\d]+[\.\-\s]+', '', current_name)
            display_name = os.path.splitext(display_name)[0]
            self.caption.setText(f"🎵 {current_idx + 1}/{len(audio_files)}: {display_name}")

            # Update detached window track info
            if self.detached_window:
                self.detached_window.set_track_info(display_name, time_str)


# ==================== Track Control Button ====================
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


# ==================== Track Controls Panel ====================
class TrackControlsPanel(QFrame):
    """Left panel with track controls like CapCut"""
    
    stateChanged = pyqtSignal(int, str, bool)  # track_index, property, value
    
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
        """Clear all track rows"""
        # Remove all widgets from layout
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.track_rows.clear()
        self.track_states.clear()
        
    def addTrackRow(self, track_type: str, track_name: str, track_index: int):
        """Add a track control row"""
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
        
        # Track type icon
        icon_map = {"gif": "🖼", "video": "🎬", "audio": "🎵"}
        icon_label = QLabel(icon_map.get(track_type, "📁"))
        icon_label.setFixedWidth(20)
        layout.addWidget(icon_label)
        
        # Lock button
        lock_btn = TrackControlButton("🔓", "Lock track")
        lock_btn.setIcons("🔒", "🔓")
        lock_btn.clicked.connect(lambda: self._emit_state(track_index, "locked", lock_btn.isChecked()))
        layout.addWidget(lock_btn)
        
        # Visibility button
        eye_btn = TrackControlButton("👁", "Show/Hide")
        eye_btn.setIcons("👁", "👁‍🗨")
        eye_btn.setChecked(True)
        eye_btn.clicked.connect(lambda: self._emit_state(track_index, "visible", eye_btn.isChecked()))
        layout.addWidget(eye_btn)
        
        # Mute button (for audio only)
        if track_type == "audio":
            mute_btn = TrackControlButton("🔊", "Mute/Unmute")
            mute_btn.setIcons("🔇", "🔊")
            mute_btn.clicked.connect(lambda: self._emit_state(track_index, "muted", mute_btn.isChecked()))
            layout.addWidget(mute_btn)
        else:
            spacer = QWidget()
            spacer.setFixedWidth(24)
            layout.addWidget(spacer)
        
        # More button
        more_btn = QPushButton("⋯")
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
        """Show more options menu"""
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
        """Add spacer for ruler alignment"""
        spacer = QWidget()
        spacer.setFixedHeight(height)
        self.layout.insertWidget(0, spacer)


# ==================== Thumbnail Cache (V4.31.3 - Video Filmstrip) ====================
class ThumbnailCache:
    """Extract and cache video frame thumbnails for filmstrip display on Timeline"""

    _cache: Dict[str, list] = {}  # path -> list of (QPixmap, timestamp_sec)

    @classmethod
    def get_thumbnails(cls, file_path: str, num_frames: int = 20, thumb_height: int = 35) -> list:
        """Get cached thumbnails. Returns list of QPixmap objects."""
        cache_key = f"{file_path}:{num_frames}:{thumb_height}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        thumbnails = cls._extract_thumbnails(file_path, num_frames, thumb_height)
        cls._cache[cache_key] = thumbnails
        return thumbnails

    @classmethod
    def _extract_thumbnails(cls, file_path: str, num_frames: int, thumb_height: int) -> list:
        """Extract frames from video/gif using ffmpeg, return as QPixmap list"""
        try:
            # First get duration
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", file_path
            ]
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            duration = 5.0  # default
            if probe.returncode == 0:
                import json as _json
                info = _json.loads(probe.stdout)
                duration = float(info.get("format", {}).get("duration", 5.0))

            if duration <= 0:
                duration = 5.0

            # Calculate aspect ratio for thumbnail width
            thumb_width = int(thumb_height * 16 / 9)  # assume 16:9

            # Extract frames at even intervals using ffmpeg
            interval = max(0.1, duration / max(1, num_frames))
            fps_filter = f"fps=1/{interval}"

            with tempfile.TemporaryDirectory() as tmpdir:
                out_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
                cmd = [
                    "ffmpeg", "-i", file_path,
                    "-vf", f"{fps_filter},scale={thumb_width}:{thumb_height}:force_original_aspect_ratio=decrease,pad={thumb_width}:{thumb_height}:(ow-iw)/2:(oh-ih)/2:color=black",
                    "-q:v", "8",  # lower quality for speed
                    "-v", "quiet",
                    out_pattern
                ]
                subprocess.run(cmd, timeout=60)

                # Load frames as QPixmap
                thumbnails = []
                for i in range(1, num_frames + 50):  # overshoot to catch all
                    fpath = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
                    if os.path.exists(fpath):
                        pix = QPixmap(fpath)
                        if not pix.isNull():
                            thumbnails.append(pix)
                    else:
                        break

                if thumbnails:
                    print(f"[THUMBNAIL] Extracted {len(thumbnails)} frames from {os.path.basename(file_path)}")
                    return thumbnails

        except Exception as e:
            print(f"[THUMBNAIL] ffmpeg failed for {os.path.basename(file_path)}: {e}")

        # Fallback: return empty list (will draw solid color bar instead)
        return []

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()


# ==================== Waveform Cache (V4.31.2 - Real Audio Peaks) ====================
class WaveformCache:
    """Generate and cache REAL waveform data from audio files using ffmpeg"""

    _cache: Dict[str, list] = {}

    @classmethod
    def get_waveform(cls, file_path: str, num_samples: int = 500) -> list:
        """Get cached waveform peaks. Returns list of floats 0.0-1.0"""
        cache_key = f"{file_path}:{num_samples}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        waveform = cls._generate_real_waveform(file_path, num_samples)
        cls._cache[cache_key] = waveform
        return waveform

    @classmethod
    def _generate_real_waveform(cls, file_path: str, num_samples: int) -> list:
        """Extract real audio peak data using ffmpeg showwavespic or raw PCM"""
        try:
            # Method: Extract raw 8-bit PCM mono audio, then compute peaks per chunk
            cmd = [
                "ffmpeg", "-i", file_path,
                "-ac", "1",           # mono
                "-ar", str(max(1000, num_samples * 4)),  # low sample rate for speed
                "-f", "u8",           # unsigned 8-bit
                "-acodec", "pcm_u8",
                "-v", "quiet",
                "pipe:1"
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode != 0 or len(result.stdout) < 100:
                return cls._generate_deterministic_waveform(file_path, num_samples)

            raw = result.stdout
            total_samples = len(raw)
            chunk_size = max(1, total_samples // num_samples)

            peaks = []
            for i in range(num_samples):
                start = i * chunk_size
                end = min(start + chunk_size, total_samples)
                if start >= total_samples:
                    peaks.append(0.05)
                    continue
                chunk = raw[start:end]
                # Peak = max deviation from center (128)
                peak = max(abs(b - 128) for b in chunk) if chunk else 0
                peaks.append(peak / 128.0)

            # Normalize
            max_peak = max(peaks) if peaks and max(peaks) > 0 else 1.0
            normalized = [max(0.05, min(1.0, p / max_peak)) for p in peaks]

            print(f"[WAVEFORM] ✅ Real waveform: {os.path.basename(file_path)} ({num_samples} peaks)")
            return normalized

        except Exception as e:
            print(f"[WAVEFORM] ffmpeg failed for {os.path.basename(file_path)}: {e}")
            return cls._generate_deterministic_waveform(file_path, num_samples)

    @classmethod
    def _generate_deterministic_waveform(cls, file_path: str, num_samples: int) -> list:
        """Fallback: deterministic fake waveform based on filename hash"""
        import hashlib
        seed = int(hashlib.md5(file_path.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        wave = []
        for i in range(num_samples):
            progress = i / num_samples
            if progress < 0.05:
                envelope = 0.3 + progress * 4
            elif progress > 0.95:
                envelope = 0.3 + (1.0 - progress) * 4
            elif 0.3 < progress < 0.5 or 0.6 < progress < 0.8:
                envelope = 0.85
            else:
                envelope = 0.6
            val = 0.4 + rng.uniform(0.0, 0.6) * envelope
            wave.append(max(0.05, min(1.0, val)))
        return wave

    @classmethod
    def _generate_fake_waveform(cls, num_samples: int) -> list:
        """Simple fallback waveform"""
        return [0.3 + random.Random(42).uniform(0, 0.5) for _ in range(num_samples)]

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()


# ==================== CapCut Timeline Canvas ====================
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
        
        # Track dimensions
        self.track_height = 45
        self.track_spacing = 0
        self.ruler_height = 30
        
    def _calculate_needed_height(self):
        """Calculate total height needed to display all tracks"""
        height = self.ruler_height  # 30px ruler
        height += self.track_height  # GIF track (always shown)
        height += self.track_height  # Video track (always shown)
        height += len(self.audio_tracks) * self.track_height  # Each audio track
        return max(300, height + 10)  # Minimum 300px, +10 padding

    def setData(self, audio, video, gif, total_ms, crossfade, pps):
        """Set timeline data and resize canvas to fit all tracks"""
        self.audio_tracks = audio or []
        self.video_tracks = video or []
        self.gif_tracks = gif or []
        self.total_duration_ms = total_ms
        self.crossfade_sec = crossfade
        self.pixels_per_second = pps
        # Dynamic resize: ensure canvas is tall enough for ALL tracks
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
        
        # Background
        painter.fillRect(self.rect(), QColor(Colors.BG_PRIMARY))
        
        # Draw ruler
        self._draw_ruler(painter, w)
        
        y_offset = self.ruler_height
        
        # Draw GIF track (top)
        y_offset = self._draw_track_background(painter, y_offset, w, "GIF", Colors.GIF_COLOR)
        if self.gif_tracks:
            self._draw_media_bar(painter, y_offset - self.track_height + 5, w, 
                                self.gif_tracks, Colors.GIF_COLOR, loop=True)
        
        # Draw Video track
        y_offset = self._draw_track_background(painter, y_offset, w, "Video", Colors.VIDEO_COLOR)
        if self.video_tracks:
            self._draw_media_bar(painter, y_offset - self.track_height + 5, w,
                                self.video_tracks, Colors.VIDEO_COLOR, loop=True)
        
        # Draw Audio tracks (staircase)
        for i, audio in enumerate(self.audio_tracks):
            y_offset = self._draw_track_background(painter, y_offset, w, f"Audio {i+1}", Colors.AUDIO_COLOR)
            self._draw_audio_staircase(painter, y_offset - self.track_height + 5, w, audio, i)
        
        # Draw playhead
        self._draw_playhead(painter, h)
        
    def _draw_ruler(self, painter, w):
        """Draw time ruler"""
        painter.fillRect(0, 0, w, self.ruler_height, QColor(Colors.BG_SECONDARY))
        
        painter.setPen(QColor(Colors.TEXT_TERTIARY))
        painter.setFont(QFont("Inter", 9))
        
        if self.total_duration_ms > 0:
            # Determine interval based on duration
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
                    # V4.31.3: Time label in CapCut format HH:MM:SS:00
                    hrs = sec // 3600
                    mins = (sec % 3600) // 60
                    secs = sec % 60
                    time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}:00"
                    painter.drawText(x + 2, 18, time_str)
                    
                    # Tick mark
                    painter.drawLine(x, 22, x, self.ruler_height)
                    
    def _draw_track_background(self, painter, y_start, w, label, color):
        """Draw track background stripe"""
        # Alternating background
        bg_color = Colors.BG_PRIMARY if y_start % 90 == self.ruler_height else Colors.TRACK_BG
        painter.fillRect(0, y_start, w, self.track_height, QColor(bg_color))
        
        # Bottom border
        painter.setPen(QColor(Colors.BORDER))
        painter.drawLine(0, y_start + self.track_height, w, y_start + self.track_height)
        
        return y_start + self.track_height
        
    def _draw_media_bar(self, painter, y, w, tracks, color, loop=False):
        """V4.31.3: Draw media bar with FILMSTRIP thumbnails (like CapCut)"""
        if not tracks or self.total_duration_ms <= 0:
            return

        total_track_duration = sum(t.duration for t in tracks)
        if total_track_duration <= 0:
            return

        total_sec = self.total_duration_ms / 1000
        bar_height = self.track_height - 10

        # Draw looped segments
        current_x = -self.scroll_offset
        time_pos = 0

        while time_pos < total_sec:
            for track in tracks:
                if time_pos >= total_sec:
                    break

                seg_width = int(track.duration * self.pixels_per_second)

                if current_x + seg_width > 0 and current_x < w:
                    # Draw segment
                    draw_x = max(0, current_x)
                    draw_width = min(seg_width, w - draw_x)

                    if current_x < 0:
                        draw_width = seg_width + current_x

                    # Background bar
                    painter.setBrush(QColor(color))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRoundedRect(int(draw_x), y, int(draw_width), bar_height, 4, 4)

                    # V4.31.3: Draw FILMSTRIP thumbnails on video/gif tracks
                    # SAFETY: Skip ThumbnailCache for .gif files (ffmpeg can segfault on GIF)
                    is_gif_file = track.path.lower().endswith('.gif')
                    thumbnails = []
                    if not is_gif_file:
                        thumb_width = int(bar_height * 16 / 9)  # ~62px for 35px height
                        num_thumbs_needed = max(5, seg_width // max(1, thumb_width))
                        thumbnails = ThumbnailCache.get_thumbnails(
                            track.path, num_frames=min(num_thumbs_needed, 40), thumb_height=bar_height
                        )

                    if thumbnails:
                        # Draw filmstrip: tile thumbnails across the segment
                        clip_rect = painter.clipRegion()
                        painter.save()
                        # Clip to the rounded rect area
                        try:
                            from PyQt6.QtGui import QPainterPath
                        except ImportError:
                            from PySide6.QtGui import QPainterPath
                        clip_path = QPainterPath()
                        clip_path.addRoundedRect(float(draw_x), float(y),
                                                  float(draw_width), float(bar_height), 4.0, 4.0)
                        painter.setClipPath(clip_path)

                        tx = int(current_x)  # start from segment start (not draw_x)
                        thumb_idx = 0
                        step_px = max(1, seg_width // len(thumbnails)) if len(thumbnails) > 0 else thumb_width

                        for i, pix in enumerate(thumbnails):
                            frame_x = tx + i * step_px
                            if frame_x + step_px < 0:
                                continue
                            if frame_x > w:
                                break
                            # Scale pixmap to fill the step width
                            scaled = pix.scaled(step_px, bar_height,
                                                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                Qt.TransformationMode.FastTransformation)
                            painter.drawPixmap(int(frame_x), y, step_px, bar_height, scaled)

                        # Semi-transparent color overlay for track identity
                        overlay_color = QColor(color)
                        overlay_color.setAlpha(45)
                        painter.setBrush(overlay_color)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawRect(int(draw_x), y, int(draw_width), bar_height)

                        painter.restore()
                    else:
                        if is_gif_file:
                            # GIF track: draw repeating GIF icon pattern
                            painter.setPen(QPen(QColor("#ffffff70"), 1))
                            painter.setFont(QFont("Inter", 7))
                            icon_spacing = 80
                            for gx in range(int(draw_x) + 10, int(draw_x + draw_width) - 10, icon_spacing):
                                painter.drawText(gx, y + bar_height // 2 + 4, "GIF")
                        else:
                            # Fallback: deterministic waveform pattern
                            import hashlib
                            seed = int(hashlib.md5(track.path.encode()).hexdigest()[:8], 16)
                            rng = random.Random(seed)
                            painter.setPen(QPen(QColor("#ffffff40"), 1))
                            for wx in range(int(draw_x) + 5, int(draw_x + draw_width) - 5, 3):
                                wave_h = rng.randint(5, bar_height - 10)
                                wy = y + (bar_height - wave_h) // 2
                                painter.drawLine(wx, wy, wx, wy + wave_h)

                    # Label overlay with semi-transparent background
                    if draw_width > 50:
                        # Draw label background pill
                        painter.setBrush(QColor(0, 0, 0, 140))
                        painter.setPen(Qt.PenStyle.NoPen)
                        label_text = track.name
                        # Format duration
                        dur_min = int(track.duration) // 60
                        dur_sec = int(track.duration) % 60
                        dur_str = f"{dur_min}:{dur_sec:02d}"
                        display = f"{label_text[:20]}  {dur_str}"
                        fm = painter.fontMetrics()
                        painter.setFont(QFont("Inter", 8, QFont.Weight.Bold))
                        text_w = painter.fontMetrics().horizontalAdvance(display) + 12
                        painter.drawRoundedRect(int(draw_x) + 4, y + 2, min(int(text_w), int(draw_width) - 8), 16, 3, 3)
                        # Draw text
                        painter.setPen(QColor("#ffffff"))
                        painter.drawText(int(draw_x) + 10, y + 14, display)

                current_x += seg_width
                time_pos += track.duration

            if not loop:
                break
                
    def _draw_audio_staircase(self, painter, y, w, audio, index):
        """Draw single audio track with staircase offset - V4.31.2: cached waveform"""
        if not audio or self.total_duration_ms <= 0:
            return

        bar_height = self.track_height - 10

        # Calculate start position with crossfade overlap
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

            # Draw bar background
            painter.setBrush(QColor(Colors.AUDIO_COLOR))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(int(draw_x), y, int(draw_width), bar_height, 4, 4)

            # V4.31.2: Draw cached waveform (deterministic, no flicker)
            waveform = WaveformCache.get_waveform(audio.path, max(100, seg_width // 2))
            painter.setPen(QPen(QColor("#ffffff50"), 1))

            visible_start = max(0, int(draw_x) - int(x))  # offset into the segment
            step = max(1, seg_width // len(waveform)) if len(waveform) > 0 else 2

            for wx_pixel in range(int(draw_x) + 2, int(draw_x + draw_width) - 2, 2):
                # Map pixel to waveform sample index
                pixel_offset = wx_pixel - int(x)
                if pixel_offset < 0:
                    continue
                sample_idx = int((pixel_offset / seg_width) * (len(waveform) - 1))
                sample_idx = max(0, min(sample_idx, len(waveform) - 1))

                val = waveform[sample_idx]
                wave_h = max(3, int(val * (bar_height - 6)))
                wy = y + (bar_height - wave_h) // 2
                painter.drawLine(wx_pixel, wy, wx_pixel, wy + wave_h)

            # Highlight currently playing region
            if self.playhead_position_ms > 0:
                playhead_sec = self.playhead_position_ms / 1000
                if start_sec <= playhead_sec <= start_sec + audio.duration:
                    # This track is currently playing - draw progress overlay
                    played_width = int((playhead_sec - start_sec) * self.pixels_per_second)
                    played_draw_x = max(int(draw_x), int(x))
                    played_draw_w = min(played_width, int(draw_width))
                    if played_draw_w > 0:
                        painter.setBrush(QColor(255, 165, 0, 40))  # Orange highlight
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawRoundedRect(int(played_draw_x), y, played_draw_w, bar_height, 4, 4)

            # Crossfade overlap zone indicators
            cf_px = int(self.crossfade_sec * self.pixels_per_second)
            if index > 0 and cf_px > 2:
                # Fade-in zone at start of this track
                fade_in_w = min(cf_px, int(draw_width))
                grad = QLinearGradient(int(draw_x), 0, int(draw_x) + fade_in_w, 0)
                grad.setColorAt(0, QColor(255, 136, 68, 120))
                grad.setColorAt(1, QColor(255, 136, 68, 0))
                painter.setBrush(QBrush(grad))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(int(draw_x), y, fade_in_w, bar_height, 4, 4)
            if index < len(self.audio_tracks) - 1 and cf_px > 2:
                # Fade-out zone at end of this track
                fade_out_x = int(draw_x + draw_width) - cf_px
                if fade_out_x >= int(draw_x):
                    grad = QLinearGradient(fade_out_x, 0, fade_out_x + cf_px, 0)
                    grad.setColorAt(0, QColor(255, 136, 68, 0))
                    grad.setColorAt(1, QColor(255, 136, 68, 120))
                    painter.setBrush(QBrush(grad))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRoundedRect(fade_out_x, y, cf_px, bar_height, 4, 4)

            # V4.31.3: Label with duration (like CapCut)
            if draw_width > 60:
                # Format duration
                dur_min = int(audio.duration) // 60
                dur_sec = int(audio.duration) % 60
                dur_str = f"{dur_min}:{dur_sec:02d}"
                name = audio.name[:22]
                display = f"{name}  {dur_str}"
                # Draw label background pill
                painter.setFont(QFont("Inter", 8, QFont.Weight.Bold))
                text_w = painter.fontMetrics().horizontalAdvance(display) + 12
                painter.setBrush(QColor(0, 0, 0, 140))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(int(draw_x) + 4, y + 2, min(int(text_w), int(draw_width) - 8), 16, 3, 3)
                # Draw text
                painter.setPen(QColor("#ffffff"))
                painter.drawText(int(draw_x) + 10, y + 14, display)

    def _draw_playhead(self, painter, h):
        """Draw playhead line"""
        if self.total_duration_ms <= 0:
            return
            
        x = int((self.playhead_position_ms / 1000) * self.pixels_per_second) - self.scroll_offset
        
        if 0 <= x <= self.width():
            # Playhead line
            painter.setPen(QPen(QColor(Colors.ACCENT), 2))
            painter.drawLine(x, self.ruler_height, x, h)
            
            # Playhead handle (triangle)
            painter.setBrush(QColor(Colors.ACCENT))
            painter.setPen(Qt.PenStyle.NoPen)
            
            handle = QPolygon([
                QPoint(x - 8, self.ruler_height - 12),
                QPoint(x + 8, self.ruler_height - 12),
                QPoint(x, self.ruler_height)
            ])
            painter.drawPolygon(handle)


# ==================== CapCut Timeline Widget ====================
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
        
        # Playhead animation
        self.playhead_timer = QTimer()
        self.playhead_timer.timeout.connect(self._update_playhead)
        
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scrollable timeline area (controls + canvas) - vertical scroll for many tracks
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

        # Container widget inside scroll area
        scroll_content = QWidget()
        timeline_area = QHBoxLayout(scroll_content)
        timeline_area.setSpacing(0)
        timeline_area.setContentsMargins(0, 0, 0, 0)

        # Track controls (left)
        self.track_controls = TrackControlsPanel()
        self.track_controls.addRulerSpacer(30)  # Match ruler height
        timeline_area.addWidget(self.track_controls)

        # Canvas (right)
        self.canvas = TimelineCanvas()
        self.canvas.seekRequested.connect(self.seekRequested.emit)
        timeline_area.addWidget(self.canvas, 1)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)
        
        # Horizontal scrollbar
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
        """Zoom in (increase pixels per second)"""
        if self.pixels_per_second < 5:
            self.pixels_per_second = min(5, self.pixels_per_second * 1.5)
        else:
            self.pixels_per_second = min(200, self.pixels_per_second + 5)
        self._refresh_canvas()

    def zoomOut(self):
        """Zoom out (decrease pixels per second) - can go very small for long projects"""
        if self.pixels_per_second <= 5:
            self.pixels_per_second = max(0.5, self.pixels_per_second * 0.7)
        else:
            self.pixels_per_second = max(1, self.pixels_per_second - 5)
        self._refresh_canvas()

    def zoomFit(self):
        """Fit entire timeline to visible canvas width (like CapCut Cmd+0)"""
        canvas_width = self.canvas.width() - 20  # small padding
        if self.total_duration_ms > 0 and canvas_width > 100:
            total_sec = self.total_duration_ms / 1000.0
            self.pixels_per_second = max(0.5, canvas_width / total_sec)
            print(f"[TIMELINE] Zoom to fit: {self.pixels_per_second:.1f} px/sec for {total_sec:.0f}sec in {canvas_width}px")
        self._refresh_canvas()
        
    def _refresh_canvas(self):
        """Refresh canvas with current zoom level"""
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
        """Update all tracks"""
        self.audio_tracks = audio or []
        self.video_tracks = video or []
        self.gif_tracks = gif or []
        
        # Calculate total duration with crossfade
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
            
        # Update canvas
        self.canvas.setData(
            self.audio_tracks,
            self.video_tracks,
            self.gif_tracks,
            self.total_duration_ms,
            self.crossfade_sec,
            self.pixels_per_second
        )
        
        # Update track controls
        self._update_track_controls()
        
        # Update scrollbar
        total_width = self._get_total_width()
        visible_width = self.canvas.width()
        self.scrollbar.setEnabled(total_width > visible_width)
        
    def _update_track_controls(self):
        """V4.31.3: Rebuild track controls panel with CapCut-style labels"""
        self.track_controls.clear()
        self.track_controls.addRulerSpacer(30)

        # GIF track - show first GIF filename
        gif_name = self.gif_tracks[0].name[:8] if self.gif_tracks else "GIF"
        self.track_controls.addTrackRow("gif", gif_name, 0)

        # Video track - show first video filename
        vid_name = self.video_tracks[0].name[:8] if self.video_tracks else "Video"
        self.track_controls.addTrackRow("video", vid_name, 1)

        # Audio tracks - show "Master" if single merged, or track number
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

        # V4.31.2: Smooth auto-scroll - keep playhead centered (like CapCut)
        if self.total_duration_ms > 0 and self.is_playing:
            playhead_x = int((position_ms / 1000) * self.pixels_per_second)
            canvas_w = self.canvas.width()
            visible_start = self.scroll_offset
            visible_end = self.scroll_offset + canvas_w

            # Keep playhead at ~40% from left during playback (CapCut style)
            target_x = int(canvas_w * 0.4)
            desired_scroll = playhead_x - target_x

            # Only scroll if playhead would be outside 30%-70% zone
            if playhead_x < visible_start + int(canvas_w * 0.3) or playhead_x > visible_start + int(canvas_w * 0.7):
                max_scroll = max(1, self._get_total_width() - canvas_w)
                new_scroll = max(0, min(desired_scroll, max_scroll))

                # Smooth interpolation (ease toward target)
                self.scroll_offset = int(self.scroll_offset + (new_scroll - self.scroll_offset) * 0.3)
                self.canvas.setScrollOffset(self.scroll_offset)

                self.scrollbar.blockSignals(True)
                self.scrollbar.setValue(int((self.scroll_offset / max_scroll) * 1000))
                self.scrollbar.blockSignals(False)
                
    def setPlaying(self, playing: bool):
        self.is_playing = playing
        if playing:
            self.playhead_timer.start(33)  # V4.31.2: ~30fps for smoother timeline
        else:
            self.playhead_timer.stop()
            
    def _update_playhead(self):
        self.canvas.update()


# ==================== Track List Item ====================
class TrackListItem(QFrame):
    """Single track in playlist"""
    # Signal to request preview crossfade
    previewCrossfadeRequested = pyqtSignal(int)  # track index
    
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
        
        # Index
        idx_label = QLabel(str(index + 1))
        idx_label.setFixedWidth(20)
        idx_label.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 12px; font-weight: bold;")
        layout.addWidget(idx_label)
        
        # Track info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(0)
        
        name = QLabel(track.name)
        name.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 11px;")
        info_layout.addWidget(name)
        
        details = QLabel(f"{track.duration_str} • {track.lufs:.1f} LUFS")
        details.setStyleSheet(f"color: {Colors.TEXT_TERTIARY}; font-size: 9px;")
        info_layout.addWidget(details)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # PREVIEW CROSSFADE button (only for tracks after first)
        if index > 0:
            self.preview_btn = QPushButton("🎧")
            self.preview_btn.setFixedSize(28, 28)
            self.preview_btn.setToolTip(f"Preview crossfade: Track {index} → {index + 1}")
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
        
        # Video selector - V4.31.2 FIX: Don't hardcode V1-V3, use ["V1"] as default
        # update_video_options() will set the correct count later
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
        """Update video combo options based on number of videos"""
        current = self.video_combo.currentIndex()
        self.video_combo.clear()
        options = [f"V{i+1}" for i in range(max(1, num_videos))]
        self.video_combo.addItems(options)
        # Restore selection if valid
        if current < len(options):
            self.video_combo.setCurrentIndex(current)


# ==================== Timestamp Dialog ====================
# ==================== AI DJ Dialog ====================
class AIDJDialog(QDialog):
    """Dialog for AI DJ features - smart playlist ordering"""
    orderApplied = pyqtSignal(list)  # emits new order of file paths
    
    def __init__(self, audio_files: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎧 AI DJ - Smart Playlist")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        self.audio_files = audio_files  # List of MediaFile objects
        self.ai_dj = None
        self.current_order = []
        self.track_data = []
        
        # Import AI DJ
        try:
            from ai_dj import AIDJ, AudioAnalysis
            self.ai_dj = AIDJ()
        except ImportError:
            pass
            
        self._setup_ui()
        self._analyze_tracks()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header_layout = QHBoxLayout()
        
        header = QLabel("🎧 AI DJ - Smart Playlist Ordering")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Strategy selector
        strategy_label = QLabel("Strategy:")
        strategy_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        header_layout.addWidget(strategy_label)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "🎯 Smooth Flow (Key + Energy)",
            "📈 Energy Up (Low → High)",
            "📉 Energy Down (High → Low)",
            "🎲 Smart Random"
        ])
        self.strategy_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 200px;
            }}
        """)
        header_layout.addWidget(self.strategy_combo)
        
        layout.addLayout(header_layout)
        
        # Stats panel
        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_SECONDARY};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        stats_layout = QHBoxLayout(self.stats_frame)
        
        self.best_opener_label = QLabel("🏆 Best #1: Analyzing...")
        self.best_opener_label.setStyleSheet(f"color: {Colors.ACCENT}; font-weight: bold;")
        stats_layout.addWidget(self.best_opener_label)
        
        stats_layout.addStretch()
        
        self.flow_label = QLabel("📊 Flow: --")
        self.flow_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        stats_layout.addWidget(self.flow_label)
        
        self.energy_label = QLabel("⚡ Energy: --")
        self.energy_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        stats_layout.addWidget(self.energy_label)
        
        layout.addWidget(self.stats_frame)
        
        # Buttons row
        btn_layout = QHBoxLayout()
        
        self.suggest_btn = QPushButton("🤖 AI Suggest")
        self.suggest_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        self.suggest_btn.clicked.connect(self._ai_suggest)
        btn_layout.addWidget(self.suggest_btn)
        
        self.shuffle_btn = QPushButton("🔄 Shuffle Again")
        self.shuffle_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        self.shuffle_btn.clicked.connect(self._shuffle_again)
        btn_layout.addWidget(self.shuffle_btn)
        
        self.random_btn = QPushButton("🎲 Pure Random")
        self.random_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        self.random_btn.clicked.connect(self._pure_random)
        btn_layout.addWidget(self.random_btn)
        
        btn_layout.addStretch()
        
        # Navigation buttons for shuffle history
        self.prev_btn = QPushButton("⬅️ Previous")
        self.prev_btn.setStyleSheet(self.random_btn.styleSheet())
        self.prev_btn.clicked.connect(self._previous_shuffle)
        self.prev_btn.setEnabled(False)
        btn_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("➡️ Next")
        self.next_btn.setStyleSheet(self.random_btn.styleSheet())
        self.next_btn.clicked.connect(self._next_shuffle)
        self.next_btn.setEnabled(False)
        btn_layout.addWidget(self.next_btn)
        
        layout.addLayout(btn_layout)
        
        # Preview controls row
        preview_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶️")
        self.play_btn.setFixedSize(50, 40)
        self.play_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_GREEN};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #45a049;
            }}
        """)
        self.play_btn.clicked.connect(self._toggle_play)
        preview_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹️")
        self.stop_btn.setFixedSize(50, 40)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_RED};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background: #cc3333;
            }}
        """)
        self.stop_btn.clicked.connect(self._stop_preview)
        preview_layout.addWidget(self.stop_btn)
        
        # Time display
        self.time_label = QLabel("0:00")
        self.time_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-family: 'Menlo', 'Courier New'; min-width: 45px;")
        preview_layout.addWidget(self.time_label)
        
        # Seek slider
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(1000)
        self.seek_slider.setValue(0)
        self.seek_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {Colors.BG_TERTIARY};
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {Colors.ACCENT};
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {Colors.ACCENT};
                border-radius: 4px;
            }}
        """)
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        self.seek_slider.sliderMoved.connect(self._on_seek_moved)
        preview_layout.addWidget(self.seek_slider, 1)
        
        # Duration display
        self.duration_label = QLabel("0:00")
        self.duration_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-family: 'Menlo', 'Courier New'; min-width: 45px;")
        preview_layout.addWidget(self.duration_label)
        
        layout.addLayout(preview_layout)
        
        # Now playing label
        self.now_playing_label = QLabel("Double-click track or select & press ▶️")
        self.now_playing_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-style: italic; font-size: 11px;")
        self.now_playing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.now_playing_label)
        
        # Track list
        list_header = QLabel("📋 Track Order (drag to reorder, double-click to play):")
        list_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(list_header)
        
        self.track_list = DraggableTrackListWidget()
        self.track_list.setMinimumHeight(350)
        self.track_list.orderChanged.connect(self._on_order_changed)
        self.track_list.playRequested.connect(self._play_track_at_index)
        layout.addWidget(self.track_list)
        
        # Setup audio player for preview
        self.preview_player = QMediaPlayer()
        self.preview_audio = QAudioOutput()
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.positionChanged.connect(self._on_position_changed)
        self.preview_player.durationChanged.connect(self._on_duration_changed)
        
        # Seek state
        self.is_seeking = False
        self.is_playing = False
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px 30px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(cancel_btn)
        
        # Clear All button
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: #FF4444;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #FF6666;
            }}
        """)
        clear_btn.clicked.connect(self._clear_all_tracks)
        bottom_layout.addWidget(clear_btn)
        
        bottom_layout.addStretch()
        
        # Rename button - rename files with numbering
        rename_btn = QPushButton("📝 Rename Files (01, 02...)")
        rename_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        rename_btn.clicked.connect(self._rename_files)
        bottom_layout.addWidget(rename_btn)
        
        apply_btn = QPushButton("✅ Apply Order")
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_GREEN};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #45a049;
            }}
        """)
        apply_btn.clicked.connect(self._apply_order)
        bottom_layout.addWidget(apply_btn)
        
        layout.addLayout(bottom_layout)
        
    def _analyze_tracks(self):
        """Analyze all tracks"""
        if not self.ai_dj or not self.audio_files:
            return
            
        self.track_data = []
        for af in self.audio_files:
            analysis = self.ai_dj.analyze_track(af.path)
            self.track_data.append({
                'file_path': af.path,
                'name': os.path.splitext(os.path.basename(af.path))[0],
                'duration_sec': analysis.duration_sec,
                'bpm': analysis.bpm,
                'key': analysis.key,
                'energy': analysis.energy,
                'intro_score': analysis.intro_score,
            })
            
        # Set initial order
        self.current_order = [t['file_path'] for t in self.track_data]
        self.track_list.set_tracks(self.track_data)
        
        # Update stats
        self._update_stats()
        
    def _update_stats(self):
        """Update statistics display"""
        if not self.ai_dj or not self.current_order:
            return
            
        # Find best opener
        best = self.ai_dj.get_best_opener(self.current_order, top_n=1)
        if best:
            best_name = os.path.splitext(os.path.basename(best[0][0]))[0]
            self.best_opener_label.setText(f"🏆 Best #1: {best_name} (Score: {best[0][1]:.0f})")
            
        # Get stats
        stats = self.ai_dj.get_playlist_stats(self.current_order)
        if stats:
            self.flow_label.setText(f"📊 Flow: {stats['smoothness']:.0f}% Smooth")
            self.energy_label.setText(f"⚡ Balance: {stats['energy_balance']:.0f}%")
            
    def _get_strategy(self) -> str:
        """Get selected strategy"""
        idx = self.strategy_combo.currentIndex()
        strategies = ["smooth", "energy_up", "energy_down", "random_smart"]
        return strategies[idx]
        
    def _ai_suggest(self):
        """Run AI suggestion"""
        if not self.ai_dj:
            return
            
        strategy = self._get_strategy()
        new_order = self.ai_dj.suggest_order(self.current_order, strategy)
        self._apply_new_order(new_order)
        
    def _shuffle_again(self):
        """Generate another shuffle"""
        if not self.ai_dj:
            return
            
        new_order = self.ai_dj.shuffle_again(self.current_order)
        self._apply_new_order(new_order)
        
        # Update navigation buttons
        self.prev_btn.setEnabled(self.ai_dj.current_shuffle_index > 0)
        self.next_btn.setEnabled(False)
        
    def _pure_random(self):
        """Pure random shuffle"""
        import random
        new_order = self.current_order.copy()
        random.shuffle(new_order)
        self._apply_new_order(new_order)
        
    def _previous_shuffle(self):
        """Go to previous shuffle in history"""
        if self.ai_dj:
            prev_order = self.ai_dj.get_previous_shuffle()
            if prev_order:
                self._apply_new_order(prev_order)
                self.prev_btn.setEnabled(self.ai_dj.current_shuffle_index > 0)
                self.next_btn.setEnabled(True)
                
    def _next_shuffle(self):
        """Go to next shuffle in history"""
        if self.ai_dj:
            next_order = self.ai_dj.get_next_shuffle()
            if next_order:
                self._apply_new_order(next_order)
                self.next_btn.setEnabled(self.ai_dj.current_shuffle_index < len(self.ai_dj.shuffle_history) - 1)
                self.prev_btn.setEnabled(True)
                
    def _apply_new_order(self, new_order: list):
        """Apply a new track order"""
        self.current_order = new_order
        
        # Rebuild track_data in new order
        path_to_data = {t['file_path']: t for t in self.track_data}
        reordered_data = [path_to_data[p] for p in new_order if p in path_to_data]
        
        self.track_list.set_tracks(reordered_data)
        self._update_stats()
        
    def _on_order_changed(self, new_paths: list):
        """Handle manual reorder via drag & drop"""
        self.current_order = new_paths
        self._update_stats()
        
    def _toggle_play(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.preview_player.pause()
            self.play_btn.setText("▶️")
            self.is_playing = False
        else:
            if self.preview_player.source().isEmpty():
                # No track loaded, play selected or first
                current_row = self.track_list.currentRow()
                if current_row >= 0:
                    self._play_track_at_index(current_row)
                elif self.track_list.count() > 0:
                    self._play_track_at_index(0)
            else:
                self.preview_player.play()
                self.play_btn.setText("⏸️")
                self.is_playing = True
            
    def _play_track_at_index(self, index: int):
        """Play track at specific index"""
        if index < 0 or index >= len(self.current_order):
            return
            
        file_path = self.current_order[index]
        track_name = os.path.basename(file_path)
        
        self.preview_player.stop()
        self.preview_player.setSource(QUrl.fromLocalFile(file_path))
        self.preview_player.play()
        
        self.play_btn.setText("⏸️")
        self.is_playing = True
        
        self.now_playing_label.setText(f"🎵 {track_name}")
        self.now_playing_label.setStyleSheet(f"color: {Colors.METER_GREEN}; font-weight: bold; font-size: 11px;")
        
        # Select the track in list
        self.track_list.setCurrentRow(index)
        
    def _stop_preview(self):
        """Stop preview playback"""
        self.preview_player.stop()
        self.play_btn.setText("▶️")
        self.is_playing = False
        self.seek_slider.setValue(0)
        self.time_label.setText("0:00")
        self.now_playing_label.setText("Playback stopped")
        self.now_playing_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-style: italic; font-size: 11px;")
        
    def _on_position_changed(self, position: int):
        """Handle playback position change"""
        if not self.is_seeking:
            duration = self.preview_player.duration()
            if duration > 0:
                slider_pos = int((position / duration) * 1000)
                self.seek_slider.setValue(slider_pos)
            
            # Update time label
            secs = position // 1000
            mins = secs // 60
            secs = secs % 60
            self.time_label.setText(f"{mins}:{secs:02d}")
            
    def _on_duration_changed(self, duration: int):
        """Handle duration change when new track loads"""
        secs = duration // 1000
        mins = secs // 60
        secs = secs % 60
        self.duration_label.setText(f"{mins}:{secs:02d}")
        
    def _on_seek_pressed(self):
        """User started dragging seek slider"""
        self.is_seeking = True
        
    def _on_seek_released(self):
        """User released seek slider"""
        self.is_seeking = False
        duration = self.preview_player.duration()
        if duration > 0:
            position = int((self.seek_slider.value() / 1000) * duration)
            self.preview_player.setPosition(position)
            
    def _on_seek_moved(self, value: int):
        """User is dragging seek slider"""
        duration = self.preview_player.duration()
        if duration > 0:
            position = int((value / 1000) * duration)
            secs = position // 1000
            mins = secs // 60
            secs = secs % 60
            self.time_label.setText(f"{mins}:{secs:02d}")
        
    def _apply_order(self):
        """Apply and close"""
        self._stop_preview()  # Stop playback before closing
        self.orderApplied.emit(self.current_order)
        self.accept()
    
    def _clear_all_tracks(self):
        """Clear all tracks and close dialog"""
        if not self.current_order:
            return
            
        reply = QMessageBox.question(
            self,
            "Clear All Tracks?",
            f"ลบเพลงทั้งหมด {len(self.current_order)} เพลงออกจากโปรเจค?\n\n"
            "⚠️ ไฟล์จะไม่ถูกลบ แค่จะเอาออกจาก playlist",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._stop_preview()
            self.current_order.clear()
            self.track_data.clear()
            self.track_list.clear()
            self.orderApplied.emit([])  # Emit empty list to clear in main window
            self.accept()
        
    def _rename_files(self):
        """Rename files with sequential numbering - replaces old numbering"""
        if not self.current_order:
            return
            
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Rename Files?",
            f"This will rename {len(self.current_order)} files with sequential numbering:\n\n"
            "Example:\n"
            "  01.Song Name.wav\n"
            "  02.Another Song.wav\n"
            "  ...\n\n"
            "⚠️ This will REPLACE existing numbering!\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        import re
        import shutil
        import uuid
        
        renamed_count = 0
        new_order = []
        errors = []
        
        # Phase 1: Rename all files to temporary unique names first
        temp_paths = []
        for file_path in self.current_order:
            try:
                if not os.path.exists(file_path):
                    errors.append(f"{os.path.basename(file_path)}: File not found")
                    continue
                    
                directory = os.path.dirname(file_path)
                old_name = os.path.basename(file_path)
                ext = os.path.splitext(old_name)[1]
                
                # Create unique temp name in same directory
                temp_name = f"_temp_{uuid.uuid4().hex}{ext}"
                temp_path = os.path.join(directory, temp_name)
                
                # Use shutil.move for cross-filesystem support
                shutil.move(file_path, temp_path)
                temp_paths.append((temp_path, old_name, directory))
            except Exception as e:
                errors.append(f"{os.path.basename(file_path)}: {str(e)}")
                # Keep original path for retry
                temp_paths.append((file_path, os.path.basename(file_path), os.path.dirname(file_path)))
        
        # Phase 2: Rename from temp to final numbered names
        for idx, (temp_path, old_name, directory) in enumerate(temp_paths, 1):
            try:
                # Remove existing numbering prefix (handles "01.Song", "01 Song", "01. Song", "01_Song")
                clean_name = re.sub(r'^\d+[.\s_-]+', '', old_name)
                # Also remove leading dots/spaces/underscores if any remain
                clean_name = clean_name.lstrip('. _-')
                # If nothing left, use original name
                if not clean_name:
                    clean_name = old_name
                
                # Add new numbering
                new_name = f"{idx:02d}.{clean_name}"
                new_path = os.path.join(directory, new_name)
                
                shutil.move(temp_path, new_path)
                new_order.append(new_path)
                renamed_count += 1
                
            except Exception as e:
                errors.append(f"{old_name}: {str(e)}")
                new_order.append(temp_path)
                
        # Update current order with new paths
        self.current_order = new_order
        
        # Update track_data with new paths
        for i, track in enumerate(self.track_data):
            if i < len(new_order):
                track['file_path'] = new_order[i]
                track['name'] = os.path.splitext(os.path.basename(new_order[i]))[0]
        
        # Refresh display
        self._apply_new_order(self.current_order)
        
        # Show result
        if errors:
            QMessageBox.warning(
                self,
                "Rename Completed with Errors",
                f"✅ Renamed: {renamed_count} files\n"
                f"❌ Errors: {len(errors)}\n\n"
                + "\n".join(errors[:5])
            )
        else:
            QMessageBox.information(
                self,
                "Rename Complete",
                f"✅ Successfully renamed {renamed_count} files!"
            )


# ==================== AI Video Dialog ====================
class AIVideoDialog(QDialog):
    """Dialog for AI Video ordering - shuffle/reorder videos"""
    orderApplied = pyqtSignal(list)  # emits new order of file paths
    
    def __init__(self, video_files: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎬 AI VDO - Smart Video Order")
        self.setMinimumSize(900, 600)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        self.video_files = video_files  # List of MediaFile objects
        self.current_order = []
        self.track_data = []
        self.shuffle_history = []
        self.current_shuffle_index = -1
        
        self._setup_ui()
        self._analyze_videos()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header_layout = QHBoxLayout()
        
        header = QLabel("🎬 AI VDO - Smart Video Ordering")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Strategy selector
        strategy_label = QLabel("Strategy:")
        strategy_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        header_layout.addWidget(strategy_label)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "📏 By Duration (Short → Long)",
            "📐 By Duration (Long → Short)",
            "🔤 By Name (A → Z)",
            "🎲 Smart Random"
        ])
        self.strategy_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 200px;
            }}
        """)
        header_layout.addWidget(self.strategy_combo)
        
        layout.addLayout(header_layout)
        
        # Stats panel
        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_SECONDARY};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        stats_layout = QHBoxLayout(self.stats_frame)
        
        self.video_count_label = QLabel(f"📹 Videos: {len(self.video_files)}")
        self.video_count_label.setStyleSheet(f"color: {Colors.ACCENT}; font-weight: bold;")
        stats_layout.addWidget(self.video_count_label)
        
        stats_layout.addStretch()
        
        self.total_duration_label = QLabel("⏱️ Total: --")
        self.total_duration_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        stats_layout.addWidget(self.total_duration_label)
        
        layout.addWidget(self.stats_frame)
        
        # Buttons row
        btn_layout = QHBoxLayout()
        
        self.suggest_btn = QPushButton("🤖 AI Suggest")
        self.suggest_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        self.suggest_btn.clicked.connect(self._ai_suggest)
        btn_layout.addWidget(self.suggest_btn)
        
        self.shuffle_btn = QPushButton("🔄 Shuffle Again")
        self.shuffle_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        self.shuffle_btn.clicked.connect(self._shuffle_again)
        btn_layout.addWidget(self.shuffle_btn)
        
        self.random_btn = QPushButton("🎲 Pure Random")
        self.random_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        self.random_btn.clicked.connect(self._pure_random)
        btn_layout.addWidget(self.random_btn)
        
        btn_layout.addStretch()
        
        # Navigation buttons
        self.prev_btn = QPushButton("⬅️ Previous")
        self.prev_btn.setStyleSheet(self.random_btn.styleSheet())
        self.prev_btn.clicked.connect(self._previous_shuffle)
        self.prev_btn.setEnabled(False)
        btn_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("➡️ Next")
        self.next_btn.setStyleSheet(self.random_btn.styleSheet())
        self.next_btn.clicked.connect(self._next_shuffle)
        self.next_btn.setEnabled(False)
        btn_layout.addWidget(self.next_btn)
        
        layout.addLayout(btn_layout)
        
        # Video Preview Widget (actual video display)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(200)
        self.video_widget.setStyleSheet("background: #000000; border-radius: 8px;")
        layout.addWidget(self.video_widget)
        
        # Video Preview Controls (like AI DJ audio player)
        preview_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶️")
        self.play_btn.setFixedSize(50, 40)
        self.play_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_GREEN};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #45a049;
            }}
        """)
        self.play_btn.clicked.connect(self._toggle_play)
        preview_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹️")
        self.stop_btn.setFixedSize(50, 40)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_RED};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background: #cc3333;
            }}
        """)
        self.stop_btn.clicked.connect(self._stop_preview)
        preview_layout.addWidget(self.stop_btn)
        
        # Time display
        self.time_label = QLabel("0:00")
        self.time_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-family: 'Menlo', 'Courier New'; min-width: 45px;")
        preview_layout.addWidget(self.time_label)
        
        # Video Seek slider
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(1000)
        self.seek_slider.setValue(0)
        self.seek_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {Colors.BG_TERTIARY};
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {Colors.VIDEO_COLOR};
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {Colors.VIDEO_COLOR};
                border-radius: 4px;
            }}
        """)
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        self.seek_slider.sliderMoved.connect(self._on_seek_moved)
        preview_layout.addWidget(self.seek_slider, 1)
        
        # Duration display
        self.duration_label = QLabel("0:00")
        self.duration_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-family: 'Menlo', 'Courier New'; min-width: 45px;")
        preview_layout.addWidget(self.duration_label)
        
        layout.addLayout(preview_layout)
        
        # Now playing label
        self.now_playing_label = QLabel("Double-click video or select & press ▶️")
        self.now_playing_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-style: italic; font-size: 11px;")
        self.now_playing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.now_playing_label)
        
        # Setup video player for preview
        self.preview_player = QMediaPlayer()
        self.preview_audio = QAudioOutput()
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.setVideoOutput(self.video_widget)  # Connect to video widget!
        self.preview_player.positionChanged.connect(self._on_position_changed)
        self.preview_player.durationChanged.connect(self._on_duration_changed)
        
        # Seek state
        self.is_seeking = False
        self.is_playing = False
        
        # Video list
        list_header = QLabel("📋 Video Order (drag to reorder, double-click to play):")
        list_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(list_header)
        
        self.track_list = DraggableTrackListWidget()
        self.track_list.setMinimumHeight(300)
        self.track_list.orderChanged.connect(self._on_order_changed)
        self.track_list.playRequested.connect(self._play_video_at_index)
        layout.addWidget(self.track_list)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px 30px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(cancel_btn)
        
        # Clear All button
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: #FF4444;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #FF6666;
            }}
        """)
        clear_btn.clicked.connect(self._clear_all_videos)
        bottom_layout.addWidget(clear_btn)
        
        bottom_layout.addStretch()
        
        # Rename button
        rename_btn = QPushButton("📝 Rename Files (01, 02...)")
        rename_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        rename_btn.clicked.connect(self._rename_files)
        bottom_layout.addWidget(rename_btn)
        
        apply_btn = QPushButton("✅ Apply Order")
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_GREEN};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #45a049;
            }}
        """)
        apply_btn.clicked.connect(self._apply_order)
        bottom_layout.addWidget(apply_btn)
        
        layout.addLayout(bottom_layout)
        
    def _analyze_videos(self):
        """Analyze all videos"""
        if not self.video_files:
            return
            
        self.track_data = []
        total_duration = 0
        
        for vf in self.video_files:
            duration = vf.duration if hasattr(vf, 'duration') else 5.0
            total_duration += duration
            
            self.track_data.append({
                'file_path': vf.path,
                'name': os.path.splitext(os.path.basename(vf.path))[0],
                'duration_sec': duration,
                'bpm': 0,  # Not applicable for video
                'key': 'N/A',
                'energy': 50,  # Default
                'intro_score': 50,
            })
            
        # Set initial order
        self.current_order = [t['file_path'] for t in self.track_data]
        self.track_list.set_tracks(self.track_data)
        
        # Update stats
        mins = int(total_duration) // 60
        secs = int(total_duration) % 60
        self.total_duration_label.setText(f"⏱️ Total: {mins}:{secs:02d}")
        self.video_count_label.setText(f"📹 Videos: {len(self.video_files)}")
        
    def _get_strategy(self) -> str:
        """Get selected strategy"""
        idx = self.strategy_combo.currentIndex()
        strategies = ["duration_asc", "duration_desc", "name_asc", "random"]
        return strategies[idx]
        
    def _ai_suggest(self):
        """Run AI suggestion based on strategy"""
        strategy = self._get_strategy()
        
        if strategy == "duration_asc":
            # Sort by duration (short to long)
            sorted_data = sorted(self.track_data, key=lambda x: x['duration_sec'])
            new_order = [t['file_path'] for t in sorted_data]
        elif strategy == "duration_desc":
            # Sort by duration (long to short)
            sorted_data = sorted(self.track_data, key=lambda x: x['duration_sec'], reverse=True)
            new_order = [t['file_path'] for t in sorted_data]
        elif strategy == "name_asc":
            # Sort by name
            sorted_data = sorted(self.track_data, key=lambda x: x['name'].lower())
            new_order = [t['file_path'] for t in sorted_data]
        else:
            # Random
            import random
            new_order = self.current_order.copy()
            random.shuffle(new_order)
            
        self._apply_new_order(new_order)
        self._add_to_history(new_order)
        
    def _shuffle_again(self):
        """Generate another shuffle"""
        import random
        new_order = self.current_order.copy()
        random.shuffle(new_order)
        self._apply_new_order(new_order)
        self._add_to_history(new_order)
        
    def _pure_random(self):
        """Pure random shuffle"""
        import random
        new_order = self.current_order.copy()
        random.shuffle(new_order)
        self._apply_new_order(new_order)
        self._add_to_history(new_order)
        
    def _add_to_history(self, order: list):
        """Add order to shuffle history"""
        # Truncate future history if we're not at the end
        if self.current_shuffle_index < len(self.shuffle_history) - 1:
            self.shuffle_history = self.shuffle_history[:self.current_shuffle_index + 1]
        
        self.shuffle_history.append(order.copy())
        self.current_shuffle_index = len(self.shuffle_history) - 1
        
        self.prev_btn.setEnabled(self.current_shuffle_index > 0)
        self.next_btn.setEnabled(False)
        
    def _previous_shuffle(self):
        """Go to previous shuffle"""
        if self.current_shuffle_index > 0:
            self.current_shuffle_index -= 1
            self._apply_new_order(self.shuffle_history[self.current_shuffle_index])
            self.prev_btn.setEnabled(self.current_shuffle_index > 0)
            self.next_btn.setEnabled(True)
            
    def _next_shuffle(self):
        """Go to next shuffle"""
        if self.current_shuffle_index < len(self.shuffle_history) - 1:
            self.current_shuffle_index += 1
            self._apply_new_order(self.shuffle_history[self.current_shuffle_index])
            self.next_btn.setEnabled(self.current_shuffle_index < len(self.shuffle_history) - 1)
            self.prev_btn.setEnabled(True)
            
    def _apply_new_order(self, new_order: list):
        """Apply a new track order"""
        self.current_order = new_order
        
        # Rebuild track_data in new order
        path_to_data = {t['file_path']: t for t in self.track_data}
        reordered_data = [path_to_data[p] for p in new_order if p in path_to_data]
        
        self.track_list.set_tracks(reordered_data)
        
    def _on_order_changed(self, new_paths: list):
        """Handle manual reorder via drag & drop"""
        self.current_order = new_paths
        
    # ==================== Video Preview Methods ====================
    def _toggle_play(self):
        """Toggle play/pause for video preview"""
        if self.is_playing:
            self.preview_player.pause()
            self.play_btn.setText("▶️")
            self.is_playing = False
        else:
            # If no video loaded, try to play selected
            if self.preview_player.source().isEmpty():
                selected = self.track_list.currentRow()
                if selected >= 0 and selected < len(self.current_order):
                    self._play_video_at_index(selected)
                    return
            self.preview_player.play()
            self.play_btn.setText("⏸️")
            self.is_playing = True
            
    def _stop_preview(self):
        """Stop video preview"""
        self.preview_player.stop()
        self.play_btn.setText("▶️")
        self.is_playing = False
        self.seek_slider.setValue(0)
        self.time_label.setText("0:00")
        self.now_playing_label.setText("Double-click video or select & press ▶️")
        
    def _play_video_at_index(self, index: int):
        """Play video at specific index"""
        if index < 0 or index >= len(self.current_order):
            return
            
        video_path = self.current_order[index]
        video_name = os.path.basename(video_path)
        
        self.preview_player.setSource(QUrl.fromLocalFile(video_path))
        self.preview_player.play()
        self.play_btn.setText("⏸️")
        self.is_playing = True
        self.now_playing_label.setText(f"🎬 Playing: {video_name}")
        
    def _on_position_changed(self, position: int):
        """Handle playback position change"""
        if not self.is_seeking:
            duration = self.preview_player.duration()
            if duration > 0:
                slider_pos = int((position / duration) * 1000)
                self.seek_slider.setValue(slider_pos)
                
        # Update time label
        pos_sec = position // 1000
        pos_min = pos_sec // 60
        pos_sec = pos_sec % 60
        self.time_label.setText(f"{pos_min}:{pos_sec:02d}")
        
    def _on_duration_changed(self, duration: int):
        """Handle duration change"""
        dur_sec = duration // 1000
        dur_min = dur_sec // 60
        dur_sec = dur_sec % 60
        self.duration_label.setText(f"{dur_min}:{dur_sec:02d}")
        
    def _on_seek_pressed(self):
        """Handle seek slider press"""
        self.is_seeking = True
        
    def _on_seek_released(self):
        """Handle seek slider release"""
        self.is_seeking = False
        duration = self.preview_player.duration()
        if duration > 0:
            position = int((self.seek_slider.value() / 1000) * duration)
            self.preview_player.setPosition(position)
            
    def _on_seek_moved(self, value: int):
        """Handle seek slider move"""
        duration = self.preview_player.duration()
        if duration > 0:
            position = int((value / 1000) * duration)
            pos_sec = position // 1000
            pos_min = pos_sec // 60
            pos_sec = pos_sec % 60
            self.time_label.setText(f"{pos_min}:{pos_sec:02d}")
    
    def _apply_order(self):
        """Apply and close"""
        # Stop any playing video first
        self.preview_player.stop()
        self.orderApplied.emit(self.current_order)
        self.accept()
    
    def _clear_all_videos(self):
        """Clear all videos and close dialog"""
        if not self.current_order:
            return
            
        reply = QMessageBox.question(
            self,
            "Clear All Videos?",
            f"ลบวิดีโอทั้งหมด {len(self.current_order)} ไฟล์ออกจากโปรเจค?\n\n"
            "⚠️ ไฟล์จะไม่ถูกลบ แค่จะเอาออกจาก playlist",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.preview_player.stop()
            self.current_order.clear()
            self.track_data.clear()
            self.track_list.clear()
            self.orderApplied.emit([])  # Emit empty list to clear in main window
            self.accept()
        
    def _rename_files(self):
        """Rename video files with sequential numbering - replaces old numbering"""
        if not self.current_order:
            return
            
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Rename Video Files?",
            f"This will rename {len(self.current_order)} video files with sequential numbering:\n\n"
            "Example:\n"
            "  01.Video Name.mp4\n"
            "  02.Another Video.mp4\n"
            "  ...\n\n"
            "⚠️ This will REPLACE existing numbering!\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        import re
        import shutil
        import uuid
        
        renamed_count = 0
        new_order = []
        errors = []
        
        # Phase 1: Rename all files to temporary unique names first
        temp_paths = []
        for file_path in self.current_order:
            try:
                if not os.path.exists(file_path):
                    errors.append(f"{os.path.basename(file_path)}: File not found")
                    continue
                    
                directory = os.path.dirname(file_path)
                old_name = os.path.basename(file_path)
                ext = os.path.splitext(old_name)[1]
                
                # Create unique temp name in same directory
                temp_name = f"_temp_{uuid.uuid4().hex}{ext}"
                temp_path = os.path.join(directory, temp_name)
                
                # Use shutil.move for cross-filesystem support
                shutil.move(file_path, temp_path)
                temp_paths.append((temp_path, old_name, directory))
            except Exception as e:
                errors.append(f"{os.path.basename(file_path)}: {str(e)}")
                temp_paths.append((file_path, os.path.basename(file_path), os.path.dirname(file_path)))
        
        # Phase 2: Rename from temp to final numbered names
        for idx, (temp_path, old_name, directory) in enumerate(temp_paths, 1):
            try:
                # Remove existing numbering prefix (handles "01.Song", "01 Song", "01. Song", "01_Song")
                clean_name = re.sub(r'^\d+[.\s_-]+', '', old_name)
                # Also remove leading dots/spaces/underscores if any remain
                clean_name = clean_name.lstrip('. _-')
                # If nothing left, use original name
                if not clean_name:
                    clean_name = old_name
                
                # Add new numbering
                new_name = f"{idx:02d}.{clean_name}"
                new_path = os.path.join(directory, new_name)
                
                shutil.move(temp_path, new_path)
                new_order.append(new_path)
                renamed_count += 1
                
            except Exception as e:
                errors.append(f"{old_name}: {str(e)}")
                new_order.append(temp_path)
                
        # Update current order
        self.current_order = new_order
        
        # Update track_data
        for i, track in enumerate(self.track_data):
            if i < len(new_order):
                track['file_path'] = new_order[i]
                track['name'] = os.path.splitext(os.path.basename(new_order[i]))[0]
        
        # Refresh display
        self._apply_new_order(self.current_order)
        
        # Show result
        if errors:
            QMessageBox.warning(
                self,
                "Rename Completed with Errors",
                f"✅ Renamed: {renamed_count} files\n"
                f"❌ Errors: {len(errors)}\n\n"
                + "\n".join(errors[:5])
            )
        else:
            QMessageBox.information(
                self,
                "Rename Complete",
                f"✅ Successfully renamed {renamed_count} video files!"
            )


# ==================== YouTube Generator Dialog ====================
class YouTubeGeneratorDialog(QDialog):
    """Dialog for generating YouTube metadata with Channel Presets"""
    
    # Channel Presets with full templates
    CHANNEL_PRESETS = {
        "Chillin' Vibes Official": {
            "url": "https://www.youtube.com/@ChillinVibesOfficial",
            "style": "chill_music",
            "tags": "chill music, relaxing music, study music, work music, เพลงชิลๆ, เพลงฟังสบาย, เพลงทำงาน, เพลงอ่านหนังสือ, lofi, cafe music, ambient music, peaceful music, เพลงคลายเครียด, เพลงพักผ่อน, ChillinVibes, RoadTripVibes, DrivingMusic, TravelPlaylist"
        },
        "Custom Channel": {
            "url": "",
            "style": "custom",
            "tags": ""
        }
    }
    
    # SEO Keywords Database (from research)
    SEO_KEYWORDS_DB = {
        "High Volume (34M+)": "chill music, study music, sleep music, relaxing music, lofi hip hop, jazz music, piano music, meditation music, background music, calm music",
        "Relax/Chill": "เพลงเพราะๆ ฟังสบาย, chill vibes, peaceful music, calm music, soft music, gentle music, soothing music, tranquil music, เพลงชิลๆ, เพลงผ่อนคลาย",
        "Study/Work": "focus music, productivity music, concentration music, background music for work, music for studying, deep focus, work music, office music, เพลงทำงาน, เพลงอ่านหนังสือ",
        "Sleep": "deep sleep music, เพลงก่อนนอน, sleep meditation, insomnia music, relaxing sleep music, bedtime music, night music, เพลงนอนหลับ, เพลงกล่อมนอน",
        "Cafe/Jazz": "cafe music, coffee shop music, bossa nova, jazz cafe, morning coffee music, cafe ambience, jazz lounge, เพลงร้านกาแฟ, เพลงคาเฟ่",
        "Piano": "relaxing piano, soft piano music, piano instrumental, peaceful piano, calm piano, piano for relaxation, เพลงเปียโน, เปียโนบรรเลง",
        "Thai Keywords": "เพลงชิลๆ, เพลงฟังสบาย, เพลงทำงาน, เพลงอ่านหนังสือ, เพลงคลายเครียด, เพลงพักผ่อน, เพลงนอนหลับ, เพลงร้านกาแฟ, เพลงเปียโน, เพลงบรรเลง"
    }
    
    def __init__(self, audio_files: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📺 YouTube Generator")
        self.setMinimumSize(800, 800)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        self.audio_files = audio_files
        self.yt_gen = None
        
        # Import YouTube Generator
        try:
            from ai_dj import YouTubeGenerator
            self.yt_gen = YouTubeGenerator()
        except ImportError:
            pass
            
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("📺 YouTube Metadata Generator")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # Settings row
        settings_layout = QHBoxLayout()
        
        # Channel Preset
        channel_label = QLabel("📺 Channel:")
        channel_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        settings_layout.addWidget(channel_label)
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(list(self.CHANNEL_PRESETS.keys()))
        self.channel_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 150px;
            }}
        """)
        settings_layout.addWidget(self.channel_combo)
        
        settings_layout.addSpacing(15)
        
        # Volume
        vol_label = QLabel("Volume:")
        vol_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        settings_layout.addWidget(vol_label)
        
        self.volume_spin = QSpinBox()
        self.volume_spin.setRange(1, 999)
        self.volume_spin.setValue(1)
        self.volume_spin.setPrefix("Vol. ")
        self.volume_spin.setStyleSheet(f"""
            QSpinBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 100px;
            }}
        """)
        settings_layout.addWidget(self.volume_spin)
        
        settings_layout.addSpacing(20)
        
        # Theme
        theme_label = QLabel("Theme:")
        theme_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        settings_layout.addWidget(theme_label)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems([
            "☕ Cafe & Coffee",
            "🚗 Driving & Travel",
            "🌙 Sleep & Relax",
            "💪 Workout & Energy",
            "🎯 Focus & Study",
            "🌴 Chill Vibes"
        ])
        self.theme_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 180px;
            }}
        """)
        settings_layout.addWidget(self.theme_combo)
        
        settings_layout.addSpacing(20)
        
        # SEO Keywords Preset
        seo_label = QLabel("SEO:")
        seo_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        settings_layout.addWidget(seo_label)
        
        self.seo_combo = QComboBox()
        self.seo_combo.addItems([
            "🎯 Auto (Theme-based)",
            "📈 High Volume Keywords",
            "🎵 Relax/Chill Keywords",
            "📚 Study/Work Keywords",
            "😴 Sleep Keywords",
            "☕ Cafe/Jazz Keywords",
            "🎹 Piano/Instrumental",
            "🇹🇭 Thai Keywords"
        ])
        self.seo_combo.setStyleSheet(self.theme_combo.styleSheet())
        settings_layout.addWidget(self.seo_combo)
        
        # Add SEO button
        add_seo_btn = QPushButton("+ Add")
        add_seo_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.SUCCESS};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #2ECC71;
            }}
        """)
        add_seo_btn.clicked.connect(self._add_seo_keywords_to_tags)
        settings_layout.addWidget(add_seo_btn)
        
        settings_layout.addStretch()
        
        # Generate button
        generate_btn = QPushButton("🚀 Generate All")
        generate_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        generate_btn.clicked.connect(self._generate_all)
        settings_layout.addWidget(generate_btn)
        
        layout.addLayout(settings_layout)
        
        # Title section
        title_header_layout = QHBoxLayout()
        title_header = QLabel("📝 Title:")
        title_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        title_header_layout.addWidget(title_header)
        title_header_layout.addStretch()
        
        copy_title_btn = QPushButton("📋 Copy")
        copy_title_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        copy_title_btn.clicked.connect(lambda: self._copy_text(self.title_edit.text()))
        title_header_layout.addWidget(copy_title_btn)
        layout.addLayout(title_header_layout)
        
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Click 'Generate All' to create title...")
        self.title_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }}
        """)
        layout.addWidget(self.title_edit)
        
        # Description section
        desc_header_layout = QHBoxLayout()
        desc_header = QLabel("📄 Description (with Timestamps):")
        desc_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        desc_header_layout.addWidget(desc_header)
        desc_header_layout.addStretch()
        
        copy_desc_btn = QPushButton("📋 Copy")
        copy_desc_btn.setStyleSheet(copy_title_btn.styleSheet())
        copy_desc_btn.clicked.connect(lambda: self._copy_text(self.desc_edit.toPlainText()))
        desc_header_layout.addWidget(copy_desc_btn)
        layout.addLayout(desc_header_layout)
        
        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Description with timestamps will appear here...")
        self.desc_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px;
                font-family: 'Menlo', 'Courier New';
                font-size: 12px;
            }}
        """)
        self.desc_edit.setMinimumHeight(250)
        layout.addWidget(self.desc_edit)
        
        # Tags section
        tags_header_layout = QHBoxLayout()
        self.tags_header = QLabel("🏷️ Tags (0/500):")
        self.tags_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        tags_header_layout.addWidget(self.tags_header)
        tags_header_layout.addStretch()
        
        copy_tags_btn = QPushButton("📋 Copy")
        copy_tags_btn.setStyleSheet(copy_title_btn.styleSheet())
        copy_tags_btn.clicked.connect(lambda: self._copy_text(self.tags_edit.toPlainText()))
        tags_header_layout.addWidget(copy_tags_btn)
        layout.addLayout(tags_header_layout)
        
        self.tags_edit = QTextEdit()
        self.tags_edit.setPlaceholderText("Tags will appear here (max 500 characters)...")
        self.tags_edit.setStyleSheet(self.desc_edit.styleSheet())
        self.tags_edit.setMaximumHeight(100)
        self.tags_edit.textChanged.connect(self._update_tags_count)
        layout.addWidget(self.tags_edit)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        export_btn = QPushButton("📤 Export to .txt")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        export_btn.clicked.connect(self._export_txt)
        bottom_layout.addWidget(export_btn)
        
        bottom_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px 30px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        close_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(close_btn)
        
        layout.addLayout(bottom_layout)
        
    def _get_theme_key(self) -> str:
        """Get theme key from combo selection"""
        idx = self.theme_combo.currentIndex()
        themes = ["cafe", "driving", "sleep", "workout", "focus", "chill"]
        return themes[idx]
        
    def _get_seo_type(self) -> str:
        """Get SEO type from combo selection"""
        idx = self.seo_combo.currentIndex()
        seo_types = ["auto", "high_volume", "relax_chill", "study_work", "sleep", "cafe_jazz", "piano_instrumental"]
        return seo_types[idx]
        
    def _on_channel_preset_changed(self, channel_name: str):
        """Handle channel preset change"""
        if channel_name in self.CHANNEL_PRESETS:
            preset = self.CHANNEL_PRESETS[channel_name]
            # Show preset info
            if preset.get('tags'):
                QMessageBox.information(
                    self, "📺 Channel Preset",
                    f"Selected: {channel_name}\n\n"
                    f"Click 'Generate All' to apply this channel's style."
                )
    
    def _add_seo_keywords_to_tags(self):
        """Add SEO keywords from selected preset to tags"""
        seo_text = self.seo_combo.currentText()
        
        # Map combo text to database key
        seo_map = {
            "🎯 Auto (Theme-based)": "auto",
            "📈 High Volume Keywords": "High Volume (34M+)",
            "🎵 Relax/Chill Keywords": "Relax/Chill",
            "📚 Study/Work Keywords": "Study/Work",
            "😴 Sleep Keywords": "Sleep",
            "☕ Cafe/Jazz Keywords": "Cafe/Jazz",
            "🎹 Piano/Instrumental": "Piano",
            "🇹🇭 Thai Keywords": "Thai Keywords"
        }
        
        seo_key = seo_map.get(seo_text, "auto")
        
        if seo_key == "auto":
            # Add top keywords from each category
            keywords_to_add = []
            for key, keywords in self.SEO_KEYWORDS_DB.items():
                kw_list = [k.strip() for k in keywords.split(",")]
                keywords_to_add.extend(kw_list[:2])  # Top 2 from each
            new_keywords = ", ".join(keywords_to_add[:20])  # Max 20
        elif seo_key in self.SEO_KEYWORDS_DB:
            new_keywords = self.SEO_KEYWORDS_DB[seo_key]
        else:
            return
        
        current_tags = self.tags_edit.toPlainText()
        
        if current_tags:
            # Avoid duplicates
            existing = set(t.strip().lower() for t in current_tags.split(","))
            new_kw_list = [k.strip() for k in new_keywords.split(",")]
            unique_new = [k for k in new_kw_list if k.lower() not in existing]
            if unique_new:
                self.tags_edit.setPlainText(f"{current_tags}, {', '.join(unique_new)}")
                QMessageBox.information(self, "✅ Keywords Added", 
                    f"Added {len(unique_new)} SEO keywords!")
            else:
                QMessageBox.information(self, "ℹ️ Info", "All keywords already exist in tags.")
        else:
            self.tags_edit.setPlainText(new_keywords)
            count = len([k for k in new_keywords.split(",") if k.strip()])
            QMessageBox.information(self, "✅ Keywords Added", 
                f"Added {count} SEO keywords!")
    
    def _generate_all(self):
        """Generate all YouTube metadata"""
        try:
            self._do_generate_all()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Generation failed: {str(e)}")
            
    def _do_generate_all(self):
        """Internal generate method"""
        if not self.yt_gen:
            QMessageBox.warning(self, "Error", "YouTube Generator module not available")
            return
            
        volume = self.volume_spin.value()
        theme = self._get_theme_key()
        
        # Build tracks list
        tracks = []
        total_duration = 0
        for af in self.audio_files:
            name = os.path.splitext(os.path.basename(af.path))[0]
            duration_sec = af.duration if af.duration and af.duration > 0 else 180
            tracks.append({
                'name': name,
                'duration_sec': duration_sec
            })
            total_duration += duration_sec
            
        # Generate timestamps
        timestamped = self.yt_gen.generate_timestamps(tracks)
        
        # Format duration
        duration_str = self.yt_gen.format_duration(total_duration)
        
        # Generate title
        title = self.yt_gen.generate_title(volume, theme, duration_str)
        self.title_edit.setText(title)
        
        # Generate description
        desc = self.yt_gen.generate_description(volume, theme, timestamped, duration_str)
        self.desc_edit.setPlainText(desc)
        
        # Generate tags with SEO keywords and channel preset
        seo_type = self._get_seo_type()
        tags = self.yt_gen.generate_tags(theme, seo_type)
        
        # Add channel preset tags if available
        channel_name = self.channel_combo.currentText()
        if channel_name in self.CHANNEL_PRESETS:
            preset = self.CHANNEL_PRESETS[channel_name]
            if preset.get('tags'):
                preset_tags = preset['tags']
                # Combine with generated tags, avoiding duplicates
                existing = set(t.strip().lower() for t in tags.split(","))
                new_tags = [t.strip() for t in preset_tags.split(",") if t.strip().lower() not in existing]
                if new_tags:
                    tags = f"{tags}, {', '.join(new_tags)}"
        
        self.tags_edit.setPlainText(tags)
        
        self._update_tags_count()
        
    def _update_tags_count(self):
        """Update tags character count"""
        text = self.tags_edit.toPlainText()
        count = len(text)
        color = Colors.METER_GREEN if count <= 500 else Colors.METER_RED
        self.tags_header.setText(f"🏷️ Tags ({count}/500):")
        self.tags_header.setStyleSheet(f"color: {color}; font-size: 12px;")
        
    def _copy_text(self, text: str):
        """Copy text to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "Copied", "Copied to clipboard!")
        
    def _export_txt(self):
        """Export all metadata to .txt file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export YouTube Metadata", 
            f"youtube_vol{self.volume_spin.value()}.txt",
            "Text Files (*.txt)"
        )
        
        if file_path:
            content = f"""=== YOUTUBE METADATA ===
Generated by LongPlay Studio V4.31

=== TITLE ===
{self.title_edit.text()}

=== DESCRIPTION ===
{self.desc_edit.toPlainText()}

=== TAGS ===
{self.tags_edit.toPlainText()}
"""
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            QMessageBox.information(self, "Exported", f"Saved to: {file_path}")


class TimestampDialog(QDialog):
    """Dialog to show and copy timestamps"""
    
    def __init__(self, timestamps: List[str], total_duration: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📋 YouTube Timestamps")
        self.setMinimumSize(500, 500)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel(f"📋 Generated Timestamps ({len(timestamps)} tracks)")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        
        # Info
        info = QLabel(f"Total Duration: {total_duration}")
        info.setStyleSheet(f"color: {Colors.VIDEO_COLOR}; font-size: 12px;")
        layout.addWidget(info)
        
        # Timestamp text
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText("\n".join(timestamps))
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 15px;
                font-family: 'Menlo', 'Courier New';
                font-size: 13px;
            }}
        """)
        layout.addWidget(self.text_edit)
        
        # Copy button
        copy_btn = QPushButton("📋 Copy All")
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #3A80C9;
            }}
        """)
        copy_btn.clicked.connect(self._copy_all)
        layout.addWidget(copy_btn)
        
    def _copy_all(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())
        QMessageBox.information(self, "Copied", "Timestamps copied to clipboard!")


# ==================== Video Prompt Generator Dialog ====================
class VideoPromptDialog(QDialog):
    """Dialog for generating Midjourney-style video prompts for meta.ai"""
    
    def __init__(self, video_files: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎬 Video Prompt Generator - Midjourney Style")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        self.video_files = video_files
        self.generator = VideoPromptGenerator() if VideoPromptGenerator else None
        self.current_analysis = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("🎬 Video Prompt Generator - Midjourney Style for meta.ai")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # Description
        desc = QLabel("Generate AI video prompts from your video files. Supports meta.ai and Midjourney styles.")
        desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Video selection
        video_layout = QHBoxLayout()
        video_label = QLabel("📹 Select Video:")
        video_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        video_layout.addWidget(video_label)
        
        self.video_combo = QComboBox()
        for vf in self.video_files:
            self.video_combo.addItem(vf.name, vf.path)
        self.video_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 300px;
            }}
        """)
        self.video_combo.currentIndexChanged.connect(self._on_video_changed)
        video_layout.addWidget(self.video_combo, 1)
        
        analyze_btn = QPushButton("🔍 Analyze")
        analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        analyze_btn.clicked.connect(self._analyze_video)
        video_layout.addWidget(analyze_btn)
        layout.addLayout(video_layout)
        
        # Analysis info
        self.analysis_info = QLabel("Select a video and click Analyze to begin")
        self.analysis_info.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
                background: {Colors.BG_SECONDARY};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        layout.addWidget(self.analysis_info)
        
        # Style selection
        style_layout = QHBoxLayout()
        style_label = QLabel("🎨 Prompt Style:")
        style_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        style_layout.addWidget(style_label)
        
        self.style_combo = QComboBox()
        styles = list(MIDJOURNEY_STYLES.keys()) if MIDJOURNEY_STYLES else [
            "cinematic", "anime", "documentary", "music_video", 
            "abstract", "lofi", "vaporwave", "nature"
        ]
        self.style_combo.addItems([s.replace("_", " ").title() for s in styles])
        self.style_combo.setStyleSheet(self.video_combo.styleSheet())
        self.style_combo.currentIndexChanged.connect(self._regenerate_prompt)
        style_layout.addWidget(self.style_combo)
        
        style_layout.addStretch()
        
        # Custom subject
        subject_label = QLabel("📝 Custom Subject:")
        subject_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        style_layout.addWidget(subject_label)
        
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("e.g., 'a lone figure walking through neon-lit streets'")
        self.subject_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 250px;
            }}
        """)
        self.subject_edit.textChanged.connect(self._regenerate_prompt)
        style_layout.addWidget(self.subject_edit)
        layout.addLayout(style_layout)
        
        # Generated prompt
        prompt_header = QLabel("✨ Generated Prompt (Midjourney Style):")
        prompt_header.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 14px; font-weight: bold;")
        layout.addWidget(prompt_header)
        
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Generated prompt will appear here...")
        self.prompt_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
            }}
        """)
        self.prompt_edit.setMinimumHeight(100)
        layout.addWidget(self.prompt_edit)
        
        # Meta.ai prompt
        meta_header = QLabel("🤖 meta.ai Video Prompt:")
        meta_header.setStyleSheet(f"color: {Colors.VIDEO_COLOR}; font-size: 14px; font-weight: bold;")
        layout.addWidget(meta_header)
        
        self.meta_prompt_edit = QTextEdit()
        self.meta_prompt_edit.setPlaceholderText("meta.ai optimized prompt will appear here...")
        self.meta_prompt_edit.setStyleSheet(self.prompt_edit.styleSheet())
        self.meta_prompt_edit.setMinimumHeight(80)
        layout.addWidget(self.meta_prompt_edit)
        
        # All styles preview
        all_styles_header = QLabel("📋 All Style Variations:")
        all_styles_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(all_styles_header)
        
        self.all_styles_edit = QTextEdit()
        self.all_styles_edit.setReadOnly(True)
        self.all_styles_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 10px;
                font-size: 11px;
            }}
        """)
        self.all_styles_edit.setMaximumHeight(150)
        layout.addWidget(self.all_styles_edit)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        copy_mj_btn = QPushButton("📋 Copy Midjourney Prompt")
        copy_mj_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        copy_mj_btn.clicked.connect(self._copy_mj_prompt)
        btn_layout.addWidget(copy_mj_btn)
        
        copy_meta_btn = QPushButton("📋 Copy meta.ai Prompt")
        copy_meta_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #3A80C9;
            }}
        """)
        copy_meta_btn.clicked.connect(self._copy_meta_prompt)
        btn_layout.addWidget(copy_meta_btn)
        
        export_btn = QPushButton("📤 Export All")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_GREEN};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #3D8B40;
            }}
        """)
        export_btn.clicked.connect(self._export_all)
        btn_layout.addWidget(export_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px 25px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        # Auto-analyze first video if available
        if self.video_files:
            self._analyze_video()
    
    def _on_video_changed(self, index: int):
        """Handle video selection change"""
        self.current_analysis = None
        self.analysis_info.setText("Click Analyze to process the selected video")
        
    def _analyze_video(self):
        """Analyze the selected video"""
        if not self.generator:
            QMessageBox.warning(self, "Error", "Video Prompt Generator not available")
            return
            
        if self.video_combo.count() == 0:
            QMessageBox.warning(self, "No Video", "Please add video files first")
            return
            
        video_path = self.video_combo.currentData()
        if not video_path:
            return
            
        self.analysis_info.setText("🔄 Analyzing video...")
        QApplication.processEvents()
        
        try:
            self.current_analysis = self.generator.analyze_video(video_path)
            
            # Update analysis info
            info_text = f"""
📹 <b>{self.current_analysis.filename}</b><br>
⏱️ Duration: {self.current_analysis.duration_sec:.1f}s | 
📐 Resolution: {self.current_analysis.width}x{self.current_analysis.height} | 
🎬 FPS: {self.current_analysis.fps:.1f}<br>
💡 Brightness: {self.current_analysis.brightness} | 
🎭 Motion: {self.current_analysis.motion_level} | 
🌆 Scene: {self.current_analysis.scene_type}<br>
🎨 Colors: {', '.join(self.current_analysis.dominant_colors)}
            """
            self.analysis_info.setText(info_text.strip())
            
            # Generate prompts
            self._regenerate_prompt()
            
        except Exception as e:
            self.analysis_info.setText(f"❌ Analysis failed: {str(e)}")
    
    def _regenerate_prompt(self):
        """Regenerate prompt with current settings"""
        if not self.current_analysis or not self.generator:
            return
            
        # Get selected style
        style_index = self.style_combo.currentIndex()
        styles = list(MIDJOURNEY_STYLES.keys()) if MIDJOURNEY_STYLES else [
            "cinematic", "anime", "documentary", "music_video", 
            "abstract", "lofi", "vaporwave", "nature"
        ]
        style = styles[style_index] if style_index < len(styles) else "cinematic"
        
        # Get custom subject
        custom_subject = self.subject_edit.text().strip()
        
        # Generate Midjourney prompt
        mj_prompt = self.generator.generate_prompt(
            self.current_analysis, style, custom_subject
        )
        self.prompt_edit.setPlainText(mj_prompt)
        
        # Generate meta.ai prompt
        meta_prompt = self.generator.generate_meta_ai_prompt(
            self.current_analysis, 
            duration_sec=5,
            custom_description=custom_subject
        )
        self.meta_prompt_edit.setPlainText(meta_prompt)
        
        # Generate all styles
        all_prompts = self.generator.generate_all_styles(
            self.current_analysis, custom_subject
        )
        all_text = ""
        for style_name, prompt in all_prompts.items():
            all_text += f"[{style_name.upper()}]\n{prompt}\n\n"
        self.all_styles_edit.setPlainText(all_text)
    
    def _copy_mj_prompt(self):
        """Copy Midjourney prompt to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.prompt_edit.toPlainText())
        QMessageBox.information(self, "Copied", "Midjourney prompt copied to clipboard!")
    
    def _copy_meta_prompt(self):
        """Copy meta.ai prompt to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.meta_prompt_edit.toPlainText())
        QMessageBox.information(self, "Copied", "meta.ai prompt copied to clipboard!")
    
    def _export_all(self):
        """Export all prompts to file"""
        if not self.current_analysis:
            QMessageBox.warning(self, "No Analysis", "Please analyze a video first")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video Prompts",
            f"{self.current_analysis.filename}_prompts.txt",
            "Text Files (*.txt);;Markdown Files (*.md)"
        )
        
        if file_path:
            content = f"""# Video Prompt Generator - Midjourney Style
# Generated by LongPlay Studio V4.31

## Video: {self.current_analysis.filename}
- Duration: {self.current_analysis.duration_sec:.1f}s
- Resolution: {self.current_analysis.width}x{self.current_analysis.height}
- FPS: {self.current_analysis.fps:.1f}
- Brightness: {self.current_analysis.brightness}
- Motion: {self.current_analysis.motion_level}
- Scene: {self.current_analysis.scene_type}
- Colors: {', '.join(self.current_analysis.dominant_colors)}

## Midjourney Prompt
```
{self.prompt_edit.toPlainText()}
```

## meta.ai Prompt
```
{self.meta_prompt_edit.toPlainText()}
```

## All Style Variations
{self.all_styles_edit.toPlainText()}
"""
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            QMessageBox.information(self, "Exported", f"Saved to: {file_path}")


# ==================== Hook Extractor Dialog ====================
class HookExtractorDialog(QDialog):
    """Dialog for extracting hook sections from audio files using waveform analysis"""
    
    def __init__(self, audio_files: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎵 Hook Extractor - Audio Waveform Analysis")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        self.audio_files = audio_files[:20]  # Limit to 20 files
        self.extractor = HookExtractor(hook_duration=30.0) if HookExtractor else None
        self.results: List = []
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("🎵 Hook Extractor - Audio Waveform Analysis")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # Description
        desc = QLabel(f"Extract hook sections from up to 20 audio files using energy analysis and peak detection. ({len(self.audio_files)} files loaded)")
        desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Settings row
        settings_layout = QHBoxLayout()
        
        # Hook duration
        duration_label = QLabel("⏱️ Hook Duration:")
        duration_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        settings_layout.addWidget(duration_label)
        
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(10, 60)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix(" sec")
        self.duration_spin.setStyleSheet(f"""
            QSpinBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 80px;
            }}
        """)
        settings_layout.addWidget(self.duration_spin)
        
        settings_layout.addStretch()
        
        # Analyze button
        analyze_btn = QPushButton("🔍 Analyze All")
        analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        analyze_btn.clicked.connect(self._analyze_all)
        settings_layout.addWidget(analyze_btn)
        
        # Extract button
        extract_btn = QPushButton("✂️ Extract Hooks")
        extract_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_GREEN};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #3D8B40;
            }}
        """)
        extract_btn.clicked.connect(self._extract_hooks)
        settings_layout.addWidget(extract_btn)
        
        layout.addLayout(settings_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                height: 20px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: {Colors.ACCENT};
                border-radius: 5px;
            }}
        """)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Results table header
        results_header = QLabel("📊 Analysis Results:")
        results_header.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 14px; font-weight: bold;")
        layout.addWidget(results_header)
        
        # Results list
        self.results_list = QListWidget()
        self.results_list.setStyleSheet(f"""
            QListWidget {{
                background: {Colors.BG_SECONDARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 10px;
            }}
            QListWidget::item {{
                background: {Colors.BG_TERTIARY};
                border-radius: 6px;
                padding: 10px;
                margin: 3px 0;
                color: {Colors.TEXT_PRIMARY};
            }}
            QListWidget::item:selected {{
                background: {Colors.ACCENT};
            }}
        """)
        layout.addWidget(self.results_list, 1)
        
        # Summary
        self.summary_label = QLabel("Click 'Analyze All' to start hook detection")
        self.summary_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
                background: {Colors.BG_TERTIARY};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        export_btn = QPushButton("📤 Export Report")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        export_btn.clicked.connect(self._export_report)
        btn_layout.addWidget(export_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px 25px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _analyze_all(self):
        """Analyze all audio files for hooks"""
        if not self.extractor:
            QMessageBox.warning(self, "Error", "Hook Extractor not available")
            return
            
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "Please add audio files first")
            return
        
        # Update hook duration
        self.extractor.hook_duration = self.duration_spin.value()
        
        # Show progress
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.audio_files))
        self.progress.setValue(0)
        self.results_list.clear()
        self.results = []
        
        # Analyze each file
        for i, af in enumerate(self.audio_files):
            self.progress.setValue(i)
            QApplication.processEvents()
            
            try:
                result = self.extractor.analyze_audio(af.path)
                self.results.append(result)
                
                # Add to list
                confidence_emoji = "🟢" if result.hook_confidence >= 0.7 else ("🟡" if result.hook_confidence >= 0.5 else "🔴")
                item_text = f"{confidence_emoji} {result.filename}\n"
                item_text += f"   ⏱️ Duration: {result.duration_sec:.1f}s | "
                item_text += f"🎵 Hook: {result.hook_time_str} | "
                item_text += f"🎯 Confidence: {result.hook_confidence:.0%}"
                
                item = QListWidgetItem(item_text)
                self.results_list.addItem(item)
                
            except Exception as e:
                item = QListWidgetItem(f"❌ {af.name} - Error: {str(e)}")
                self.results_list.addItem(item)
        
        self.progress.setValue(len(self.audio_files))
        self.progress.setVisible(False)
        
        # Update summary
        if self.results:
            avg_confidence = sum(r.hook_confidence for r in self.results) / len(self.results)
            high_conf = sum(1 for r in self.results if r.hook_confidence >= 0.7)
            
            summary = f"""
<b>✅ Analysis Complete!</b><br>
📊 Total files: {len(self.results)}<br>
🎯 Average confidence: {avg_confidence:.0%}<br>
🟢 High confidence (>70%): {high_conf} files<br>
<br>
Click 'Extract Hooks' to save hook sections as separate files.
            """
            self.summary_label.setText(summary.strip())
    
    def _extract_hooks(self):
        """Extract hook sections from analyzed files"""
        if not self.results:
            QMessageBox.warning(self, "No Analysis", "Please analyze files first")
            return
        
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Hooks"
        )
        
        if not output_dir:
            return
        
        # Show progress
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.results))
        self.progress.setValue(0)
        
        extracted = 0
        for i, result in enumerate(self.results):
            self.progress.setValue(i)
            QApplication.processEvents()
            
            try:
                hook_path = self.extractor.extract_hook(result.file_path, output_dir)
                if hook_path:
                    extracted += 1
            except Exception as e:
                print(f"Extract error for {result.filename}: {e}")
        
        self.progress.setValue(len(self.results))
        self.progress.setVisible(False)
        
        QMessageBox.information(
            self, "Extraction Complete",
            f"✅ Extracted {extracted}/{len(self.results)} hooks to:\n{output_dir}"
        )
    
    def _export_report(self):
        """Export analysis report"""
        if not self.results:
            QMessageBox.warning(self, "No Results", "Please analyze files first")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Hook Analysis Report",
            "hook_analysis_report.md",
            "Markdown Files (*.md);;Text Files (*.txt)"
        )
        
        if file_path:
            content = "# Hook Extractor Report\n"
            content += "# Generated by LongPlay Studio V4.31\n\n"
            content += f"Total files analyzed: {len(self.results)}\n"
            content += f"Hook duration setting: {self.duration_spin.value()}s\n\n"
            
            for result in self.results:
                confidence_emoji = "🟢" if result.hook_confidence >= 0.7 else ("🟡" if result.hook_confidence >= 0.5 else "🔴")
                content += f"## {confidence_emoji} {result.filename}\n"
                content += f"- Full Duration: {result.duration_sec:.1f}s\n"
                content += f"- Hook Time: {result.hook_time_str}\n"
                content += f"- Hook Duration: {result.hook_duration_sec:.1f}s\n"
                content += f"- Confidence: {result.hook_confidence:.0%}\n"
                if result.hook_file_path:
                    content += f"- Extracted File: {result.hook_file_path}\n"
                content += "\n"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            QMessageBox.information(self, "Exported", f"Report saved to: {file_path}")


# ==================== Main Window ====================
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
            os.path.join(os.path.dirname(__file__), "Logo.jpg"),
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
        
        self._setup_ui()
        self._setup_shortcuts()
        self._connect_signals()
        
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Spacebar = Play/Pause
        space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space_shortcut.activated.connect(self._toggle_playback)
        
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
            os.path.join(os.path.dirname(__file__), "Logo.jpg"),
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

        # Content Factory button
        factory_btn = QPushButton("🏭 Factory")
        factory_btn.setStyleSheet(f"""
            QPushButton {{
                background: #8B5CF6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #7C3AED;
            }}
        """)
        factory_btn.clicked.connect(self._show_content_factory)
        header.addWidget(factory_btn)

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
        
        # CapCut Timeline
        self.timeline = CapCutTimeline()
        self.timeline.setMinimumHeight(150)
        self.timeline.seekRequested.connect(self._on_seek)
        timeline_section.addWidget(self.timeline)

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
        self.tl_stop_btn.clicked.connect(self._on_tl_stop)
        transport_layout.addWidget(self.tl_stop_btn)

        transport_layout.addStretch()

        self.tl_position_label = QLabel("00:00 / 00:00")
        self.tl_position_label.setStyleSheet(
            f"color: {Colors.ACCENT}; font-size: 13px; font-weight: bold; "
            f"font-family: 'Menlo', monospace; letter-spacing: 1px;")
        transport_layout.addWidget(self.tl_position_label)

        timeline_section.addWidget(transport_bar)

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
        max_header_row = QHBoxLayout()
        max_header_row.setContentsMargins(0, 0, 0, 0)
        max_header_row.setSpacing(8)

        maximizer_header = QLabel("MAXIMIZER")
        maximizer_header.setStyleSheet("""
            color: #48CAE4;
            font-size: 11px;
            font-weight: bold;
            font-family: 'Menlo', monospace;
            letter-spacing: 3px;
            padding: 4px 0px 3px 0px;
        """)
        max_header_row.addWidget(maximizer_header)
        max_header_row.addStretch()

        # ── Original / Mastered toggle ──
        self._right_bypass_active = False
        bypass_container = QFrame()
        bypass_container.setFixedHeight(28)
        bypass_container.setStyleSheet(
            "QFrame { background: #1A1A1E; border: 1px solid #2A2A32; border-radius: 14px; }")
        bypass_lay = QHBoxLayout(bypass_container)
        bypass_lay.setContentsMargins(2, 2, 2, 2)
        bypass_lay.setSpacing(0)

        self.btn_original = QPushButton("Original")
        self.btn_original.setFixedSize(68, 24)
        self.btn_original.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_original.setStyleSheet(
            "QPushButton { background: transparent; color: #6B6B70; border: none; "
            "border-radius: 12px; font-size: 10px; font-weight: bold; font-family: 'Menlo', monospace; }"
            "QPushButton:hover { color: #FFB340; }")
        self.btn_original.clicked.connect(lambda: self._on_right_bypass_switch(True))
        bypass_lay.addWidget(self.btn_original)

        self.btn_mastered = QPushButton("Mastered")
        self.btn_mastered.setFixedSize(72, 24)
        self.btn_mastered.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mastered.setStyleSheet(
            "QPushButton { background: #48CAE4; color: #0A0A0C; border: none; "
            "border-radius: 12px; font-size: 10px; font-weight: bold; font-family: 'Menlo', monospace; }"
            "QPushButton:hover { background: #5AD8F2; }")
        self.btn_mastered.clicked.connect(lambda: self._on_right_bypass_switch(False))
        bypass_lay.addWidget(self.btn_mastered)

        max_header_row.addWidget(bypass_container)
        layout.addLayout(max_header_row)

        # Separator line under header
        header_line = QFrame()
        header_line.setFixedHeight(1)
        header_line.setStyleSheet("background: #00B4D8;")
        layout.addWidget(header_line)

        # ── AI MASTER + MASTER EXPORT buttons (top) ──
        action_btns_row = QHBoxLayout()
        action_btns_row.setSpacing(6)

        self.btn_ai_master = QPushButton("🤖 AI MASTER")
        self.btn_ai_master.setFixedHeight(36)
        self.btn_ai_master.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #FF9500, stop:1 #CC7700);
                color: #0A0A0C; border: none; border-radius: 5px;
                padding: 6px 10px; font-weight: bold; font-size: 11px;
                font-family: 'Menlo', monospace;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #FFB340, stop:1 #FF9500);
            }
        """)
        self.btn_ai_master.setToolTip("AI Master: วิเคราะห์ทุกเพลง + ตั้งค่าอัตโนมัติ")
        self.btn_ai_master.clicked.connect(self._run_ai_master)
        action_btns_row.addWidget(self.btn_ai_master)

        self.btn_open_full_master = QPushButton("⚡ MASTER EXPORT")
        self.btn_open_full_master.setFixedHeight(36)
        self.btn_open_full_master.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #FFB340, stop:0.5 #FF9500, stop:1 #CC7700);
                color: #1A1A1E; border: 1px solid #CC7700; border-radius: 5px;
                padding: 6px 10px; font-weight: bold; font-size: 11px;
                font-family: 'Menlo', monospace; letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #FFD060, stop:1 #FFB340);
            }
        """)
        self.btn_open_full_master.setToolTip("Master & export all tracks")
        self.btn_open_full_master.clicked.connect(self._master_export_from_right_panel)
        action_btns_row.addWidget(self.btn_open_full_master)

        layout.addLayout(action_btns_row)

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

        # ── IRC Sub-mode dropdown (IRC 3: Pumping/Balanced/Crisp/Clipping, IRC 4: Classic/Modern/Transient) ──
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

        # V5.10.5: Connect sub-mode change → re-sync chain with new sub-mode
        self.right_irc_submode.currentTextChanged.connect(self._on_right_irc_submode_changed)

        # ── WIDTH → SOOTHE → COMPRESS → GAIN — single row of 4 OzoneRotaryKnobs ──
        # NOTE: QDial + CSS border-radius breaks mouse hit area on macOS/PyQt6.
        # OzoneRotaryKnob has explicit mousePressEvent/mouseMoveEvent — always works.
        from modules.widgets.rotary_knob import OzoneRotaryKnob

        knobs_section = QHBoxLayout()
        knobs_section.setSpacing(4)

        # ── 1. WIDTH (Stereo Imager) ──
        self.right_width_dial = OzoneRotaryKnob(
            name="WIDTH", min_val=0.0, max_val=200.0, default=100.0,
            unit="%", decimals=0)
        self.right_width_dial.setFixedSize(70, 70)
        self.right_width_dial.valueChanged.connect(
            lambda v: self._on_right_width_changed(int(v)))
        knobs_section.addWidget(self.right_width_dial, alignment=Qt.AlignmentFlag.AlignCenter)

        self.right_width_display = QLabel("100%")
        self.right_width_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_width_display.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        self.right_width_display.setStyleSheet("color: #CE93D8;")
        self.right_width_display.setVisible(False)  # Value shown on knob itself

        # ── 2. SOOTHE ──
        self.right_soothe_knob = OzoneRotaryKnob(
            name="SOOTHE", min_val=0.0, max_val=100.0, default=0.0,
            unit="%", decimals=0)
        self.right_soothe_knob.setFixedSize(70, 70)
        self.right_soothe_knob.valueChanged.connect(self._on_soothe_knob_changed)
        knobs_section.addWidget(self.right_soothe_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        # ── 3. COMPRESS ──
        self.right_compress_knob = OzoneRotaryKnob(
            name="COMPRESS", min_val=0.0, max_val=100.0, default=0.0,
            unit="%", decimals=0)
        self.right_compress_knob.setFixedSize(70, 70)
        self.right_compress_knob.valueChanged.connect(self._on_compress_knob_changed)
        knobs_section.addWidget(self.right_compress_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        # ── 4. GAIN (last, next to dB readout) ──
        self.right_gain_dial = OzoneRotaryKnob(
            name="GAIN", min_val=0.0, max_val=20.0, default=0.0,
            unit="dB", decimals=1)
        self.right_gain_dial.setFixedSize(70, 70)
        self.right_gain_dial.valueChanged.connect(
            lambda v: self._on_right_gain_changed(int(v * 10)))
        knobs_section.addWidget(self.right_gain_dial, alignment=Qt.AlignmentFlag.AlignCenter)

        self.right_gain_display = QLabel("+0.0")
        self.right_gain_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_gain_display.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        self.right_gain_display.setStyleSheet("color: #48CAE4;")
        self.right_gain_display.setVisible(False)  # Value shown on knob itself

        # ── Display column (large dB readout) ──
        display_col = QVBoxLayout()
        display_col.setSpacing(0)
        display_col.addStretch()
        self.right_gain_big = QLabel("+0.0")
        self.right_gain_big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_gain_big.setFont(QFont("Menlo", 22, QFont.Weight.Bold))
        self.right_gain_big.setStyleSheet("color: #48CAE4;")
        display_col.addWidget(self.right_gain_big)
        gain_unit = QLabel("dB")
        gain_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gain_unit.setStyleSheet("color: #8E8A82; font-size: 10px; font-family: 'Menlo', monospace;")
        display_col.addWidget(gain_unit)
        display_col.addStretch()
        knobs_section.addLayout(display_col)

        layout.addLayout(knobs_section)

        # ── Right-click on knobs → popup Ozone 12-style meter panels ──
        from modules.master.meter_panels import (
            MaximizerMeterPanel, ImagerMeterPanel,
            CompressorMeterPanel, SootheMeterPanel
        )
        self._meter_panels = {}  # track open panels by name

        def _position_panel_clamped(panel, knob_widget):
            """Position panel near knob, clamped to screen bounds"""
            from PyQt6.QtWidgets import QApplication
            pos = knob_widget.mapToGlobal(knob_widget.rect().bottomLeft())
            screen = QApplication.primaryScreen()
            if screen:
                geo = screen.availableGeometry()
                x = pos.x() - 40
                if x + panel.width() > geo.right():
                    x = geo.right() - panel.width() - 10
                if x < geo.left():
                    x = geo.left() + 10
                y = pos.y() + 5
                if y + panel.height() > geo.bottom():
                    y = knob_widget.mapToGlobal(knob_widget.rect().topLeft()).y() - panel.height() - 5
                if y < geo.top():
                    y = geo.top() + 10
                panel.move(x, y)
            else:
                panel.move(pos.x() - 40, pos.y() + 8)

        def _toggle_meter_panel(knob_widget, panel_cls, panel_key):
            """Toggle popup meter panel on right-click"""
            if panel_key in self._meter_panels and self._meter_panels[panel_key].isVisible():
                self._meter_panels[panel_key].close()
                del self._meter_panels[panel_key]
                return
            panel = panel_cls(parent=self)
            _position_panel_clamped(panel, knob_widget)
            panel.closed.connect(lambda k=panel_key: self._meter_panels.pop(k, None))
            self._meter_panels[panel_key] = panel
            panel.show()
            panel.raise_()

        for knob, cls, key in [
            (self.right_width_dial,    ImagerMeterPanel,     "imager"),
            (self.right_soothe_knob,   SootheMeterPanel,     "soothe"),
            (self.right_compress_knob, CompressorMeterPanel, "compressor"),
            (self.right_gain_dial,     MaximizerMeterPanel,  "maximizer"),
        ]:
            knob.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            knob.customContextMenuRequested.connect(
                lambda _pos, w=knob, c=cls, k=key: _toggle_meter_panel(w, c, k))

        # ── Meter panel data feed timer (20fps) ──
        self._meter_panel_timer = QTimer(self)
        self._meter_panel_timer.setInterval(50)
        self._meter_panel_timer.timeout.connect(self._feed_meter_panels)
        self._meter_panel_timer.start()

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

        # (SOOTHE + COMPRESS knobs are now in knobs_section above)

        # Backward compat: keep old refs working
        self.right_res_enabled = type('Obj', (), {'isChecked': lambda s: self.right_soothe_knob.value() > 0, 'setChecked': lambda s, v: None})()
        self.right_res_depth = type('Obj', (), {'value': lambda s: int(self.right_soothe_knob.value() * 2), 'setValue': lambda s, v: self.right_soothe_knob.setValue(v / 2)})()
        self.right_res_depth_val = QLabel("0")
        self.right_res_depth_val.setVisible(False)
        self.right_res_mode = None
        self.right_res_mix = None
        self.right_res_delta = None
        self.right_dyn_enabled = type('Obj', (), {'isChecked': lambda s: self.right_compress_knob.value() > 0, 'setChecked': lambda s, v: None})()
        self.right_dyn_amount = type('Obj', (), {'value': lambda s: int(self.right_compress_knob.value()), 'setValue': lambda s, v: self.right_compress_knob.setValue(v)})()
        self.right_dyn_amount_val = QLabel("0%")
        self.right_dyn_amount_val.setVisible(False)

        # ── Gain Reduction History (Ozone 12 waveform) ──
        if _HAS_MASTER_WIDGETS:
            self.right_gr_history = GainReductionHistoryWidget()
            layout.addWidget(self.right_gr_history, alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            self.right_gr_history = None

        # ── V5.8: Logic Channel Strip BEFORE/AFTER Meters ──
        if _HAS_MASTER_WIDGETS:
            self.right_logic_meter = LogicChannelMeter(ceiling_db=-1.0)
            layout.addWidget(self.right_logic_meter, alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            self.right_logic_meter = None

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

        # V5.10: Rust Real-Time DSP Engine (like Logic Pro / Ozone)
        # Replaces QMediaPlayer when mastering controls are active
        self._rt_engine = None
        self._rt_active = False
        self._rt_file = None
        try:
            import longplay
            self._rt_engine = longplay.PyRtEngine()
            print("[RT ENGINE] Rust PyRtEngine ready")
        except Exception as e:
            print(f"[RT ENGINE] Not available: {e}")

        # RT engine position sync timer
        self._rt_pos_timer = QTimer()
        self._rt_pos_timer.setInterval(33)  # ~30fps position updates
        self._rt_pos_timer.timeout.connect(self._on_rt_position_tick)

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

        # ── Vectorscope (Stereo Width Meter) ──
        try:
            from gui.widgets.vectorscope import Vectorscope
            self.right_vectorscope = Vectorscope()
            self.right_vectorscope.setFixedSize(140, 140)
            vs_label = QLabel("STEREO FIELD")
            vs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vs_label.setStyleSheet("color: #00B4D8; font-size: 8px; font-weight: bold; font-family: 'Menlo'; letter-spacing: 2px;")
            layout.addWidget(vs_label)
            layout.addWidget(self.right_vectorscope, alignment=Qt.AlignmentFlag.AlignCenter)
            print("[VECTORSCOPE] ✅ Added to Maximizer panel")
        except Exception as e:
            self.right_vectorscope = None
            print(f"[VECTORSCOPE] Not available: {e}")

        # Separator before DJ Crossfade
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

        # V5.10.1: LUFS measurement using pyloudnorm (ITU-R BS.1770-4 compliant)
        # K-weighted loudness with proper gating, 400ms momentary, 3s short-term
        if self.meter.is_playing:
            import math

            # Initialize LUFS state
            if not hasattr(self, '_lufs_meter'):
                try:
                    import pyloudnorm as pyln
                    self._lufs_meter = pyln.Meter(48000)  # will update SR on first use
                    self._lufs_meter_sr = 48000
                except ImportError:
                    self._lufs_meter = None
                    self._lufs_meter_sr = 0
                self._lufs_mom_buf = []          # 400ms audio buffer
                self._lufs_short_buf_audio = []  # 3s audio buffer
                self._lufs_int_blocks = []       # all gated blocks for integrated
                self._lufs_short_history = []    # short-term history for LRA
                self._lufs_integrated_val = -70.0

            # Get raw audio samples from AudioAnalysisEngine
            has_audio_data = (hasattr(self, 'audio_engine') and
                             self.audio_engine._current_data is not None and
                             self.audio_engine._has_soundfile)

            levels_db = getattr(self.meter, '_last_levels_db', None)

            if has_audio_data and self._lufs_meter is not None:
                import numpy as np
                import pyloudnorm as pyln

                sr = self.audio_engine._current_sr
                data = self.audio_engine._current_data

                # Update meter sample rate if changed
                if sr != self._lufs_meter_sr and sr > 0:
                    self._lufs_meter = pyln.Meter(sr)
                    self._lufs_meter_sr = sr

                # Get current position in samples
                pos_samples = int(track_local_pos / 1000.0 * sr)

                # V5.10: Apply gain+ceiling to LUFS measurement (match actual output)
                _lufs_gain_db = getattr(self, '_right_gain_db', 0.0)
                _lufs_ceiling = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0
                def _apply_gain_to_chunk(chunk):
                    if _lufs_gain_db > 0.01:
                        chunk = chunk * np.float32(10 ** (_lufs_gain_db / 20.0))
                        ceil_lin = np.float32(10 ** (_lufs_ceiling / 20.0))
                        np.clip(chunk, -ceil_lin, ceil_lin, out=chunk)
                    return chunk

                # === Momentary LUFS (400ms window) ===
                mom_samples = int(sr * 0.4)
                mom_start = max(0, pos_samples - mom_samples)
                mom_end = min(len(data), pos_samples)
                if mom_end - mom_start > 1024:
                    mom_chunk = data[mom_start:mom_end].copy()
                    if mom_chunk.ndim == 1:
                        mom_chunk = np.column_stack([mom_chunk, mom_chunk])
                    mom_chunk = _apply_gain_to_chunk(mom_chunk)
                    try:
                        momentary_lufs = self._lufs_meter.integrated_loudness(mom_chunk)
                        if np.isinf(momentary_lufs) or np.isnan(momentary_lufs):
                            momentary_lufs = -70.0
                    except Exception:
                        momentary_lufs = -70.0
                else:
                    momentary_lufs = -70.0

                # === Short-term LUFS (3s window) ===
                short_samples = int(sr * 3.0)
                short_start = max(0, pos_samples - short_samples)
                short_end = min(len(data), pos_samples)
                if short_end - short_start > sr:
                    short_chunk = data[short_start:short_end].copy()
                    if short_chunk.ndim == 1:
                        short_chunk = np.column_stack([short_chunk, short_chunk])
                    short_chunk = _apply_gain_to_chunk(short_chunk)
                    try:
                        shortterm_lufs = self._lufs_meter.integrated_loudness(short_chunk)
                        if np.isinf(shortterm_lufs) or np.isnan(shortterm_lufs):
                            shortterm_lufs = -70.0
                    except Exception:
                        shortterm_lufs = -70.0
                else:
                    shortterm_lufs = momentary_lufs

                # === Integrated LUFS (from start, gated) ===
                # Re-measure every 3 seconds to avoid CPU overhead
                if not hasattr(self, '_lufs_int_last_pos'):
                    self._lufs_int_last_pos = 0
                if pos_samples - self._lufs_int_last_pos > sr * 3 or self._lufs_integrated_val <= -70.0:
                    int_end = min(len(data), pos_samples)
                    int_samples = min(int_end, sr * 60)  # max 60s lookback
                    int_start = max(0, int_end - int_samples)
                    if int_end - int_start > sr:
                        int_chunk = data[int_start:int_end].copy()
                        if int_chunk.ndim == 1:
                            int_chunk = np.column_stack([int_chunk, int_chunk])
                        int_chunk = _apply_gain_to_chunk(int_chunk)
                        try:
                            integrated_lufs = self._lufs_meter.integrated_loudness(int_chunk)
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

                # === LRA (10th-95th percentile of short-term history) ===
                if shortterm_lufs > -70.0:
                    self._lufs_short_history.append(shortterm_lufs)
                    # Keep last 120 entries (~2 minutes at ~1/s)
                    if len(self._lufs_short_history) > 120:
                        self._lufs_short_history.pop(0)
                if len(self._lufs_short_history) >= 4:
                    sorted_st = sorted(self._lufs_short_history)
                    n = len(sorted_st)
                    p10 = sorted_st[max(0, int(n * 0.10))]
                    p95 = sorted_st[min(n - 1, int(n * 0.95))]
                    lra = max(0.0, p95 - p10)
                else:
                    lra = 0.0

                # === True Peak (from peak dB + 0.5 dB headroom for inter-sample) ===
                if levels_db and levels_db.get("left_peak_db", -70) > -70:
                    # Approximate True Peak: sample peak + ~0.5 dB for inter-sample peaks
                    tp_l = levels_db["left_peak_db"]
                    tp_r = levels_db["right_peak_db"]
                    peak_db = max(tp_l, tp_r)
                else:
                    peak_db = -70.0
                    tp_l = -70.0
                    tp_r = -70.0

            else:
                # Fallback: approximate from RMS levels (no K-weighting)
                if levels_db and levels_db.get("left_rms_db", -70) > -70:
                    avg_rms_db = (levels_db["left_rms_db"] + levels_db["right_rms_db"]) / 2.0
                    momentary_lufs = max(-70.0, min(0.0, avg_rms_db))
                    peak_db = max(levels_db["left_peak_db"], levels_db["right_peak_db"])
                    tp_l = levels_db["left_peak_db"]
                    tp_r = levels_db["right_peak_db"]
                else:
                    avg_level = (self.meter.left_level + self.meter.right_level) / 2.0
                    momentary_lufs = 20 * math.log10(max(avg_level, 1e-10)) - 6.0
                    momentary_lufs = max(-70.0, min(0.0, momentary_lufs))
                    peak_db = max(
                        20 * math.log10(max(self.meter.peak_left, 1e-10)),
                        20 * math.log10(max(self.meter.peak_right, 1e-10)),
                    )
                    tp_l = peak_db
                    tp_r = peak_db
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

            # V5.10: Feed Vectorscope with audio samples
            if hasattr(self, 'right_vectorscope') and self.right_vectorscope is not None:
                if not self.right_vectorscope._timer.isActive():
                    self.right_vectorscope.start()
                if has_audio_data:
                    import numpy as np
                    sr = self.audio_engine._current_sr
                    data = self.audio_engine._current_data
                    pos_s = int(track_local_pos / 1000.0 * sr)
                    chunk_len = min(512, len(data) - pos_s)
                    if chunk_len > 0 and pos_s < len(data):
                        chunk = data[pos_s:pos_s + chunk_len]
                        if chunk.ndim == 2 and chunk.shape[1] >= 2:
                            self.right_vectorscope.feed_samples(chunk[:, 0], chunk[:, 1])
                        elif chunk.ndim == 1:
                            self.right_vectorscope.feed_samples(chunk, chunk)

            # V5.8: Feed Logic Channel Strip BEFORE/AFTER meters
            if hasattr(self, 'right_logic_meter') and self.right_logic_meter is not None:
                tp_l = levels_db.get("left_peak_db", -70.0) if levels_db else peak_db
                tp_r = levels_db.get("right_peak_db", -70.0) if levels_db else peak_db
                rms_l = levels_db.get("left_rms_db", -70.0) if levels_db else peak_db - 6
                rms_r = levels_db.get("right_rms_db", -70.0) if levels_db else peak_db - 6
                # BEFORE = input signal (raw peaks)
                self.right_logic_meter.set_before(
                    l_peak=tp_l, r_peak=tp_r, l_rms=rms_l, r_rms=rms_r)
                # AFTER = signal clamped by ceiling (simulates limiter output)
                gain_db_v = getattr(self, '_right_gain_db', 0.0)
                ceil_v = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0
                after_l = min(tp_l + gain_db_v, ceil_v)
                after_r = min(tp_r + gain_db_v, ceil_v)
                after_rms_l = min(rms_l + gain_db_v, ceil_v)
                after_rms_r = min(rms_r + gain_db_v, ceil_v)
                self.right_logic_meter.set_after(
                    l_peak=after_l, r_peak=after_r, l_rms=after_rms_l, r_rms=after_rms_r)

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
        if duration_ms > 0 and hasattr(self, 'timeline'):
            self.timeline.total_duration_ms = duration_ms

    # V5.5.2: Maximizer Gain — apply gain to AUDIO DATA (not QAudioOutput which caps at 1.0)
    def _on_right_gain_changed(self, value: int):
        """Gain dial changed (0-200 → 0.0-20.0 dB).
        V5.10.2: Matches V2 behavior — update meters + QAudioOutput volume instantly.
        """
        gain_db = value / 10.0
        self._right_gain_db = gain_db

        # 1. Update AudioAnalysisEngine for meters (same as V2)
        ceiling = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0
        if hasattr(self, 'audio_engine'):
            self.audio_engine.set_gain(gain_db, ceiling)

        # 2. Instant audio feedback via QAudioOutput volume
        #    Maps 0-20 dB gain to 0.5-3.0 volume (louder as gain increases)
        volume = max(0.5, min(3.0, 0.5 + gain_db / 20.0 * 2.5))
        if hasattr(self, 'audio_player') and hasattr(self.audio_player, 'audio_output'):
            self.audio_player.audio_output.setVolume(volume)

        # 3. Update display
        if gain_db < 6.0:
            color = "#00CED1"
        elif gain_db < 12.0:
            color = "#FFD700"
        elif gain_db < 16.0:
            color = "#FF8C00"
        else:
            color = "#FF4444"
        if hasattr(self, 'right_gain_display'):
            self.right_gain_display.setText(f"+{gain_db:.1f}")
            self.right_gain_display.setStyleSheet(f"color: {color};")
        if hasattr(self, 'right_gain_big'):
            self.right_gain_big.setText(f"+{gain_db:.1f}")
            self.right_gain_big.setStyleSheet(f"color: {color};")

        # 4. Update Logic meter ceiling display
        if hasattr(self, 'right_logic_meter') and self.right_logic_meter is not None:
            if hasattr(self.right_logic_meter, 'set_ceiling'):
                self.right_logic_meter.set_ceiling(ceiling)

        # 5. Process audio + hot-swap in QMediaPlayer (offline preview)
        #    This renders the gained audio to a temp WAV and swaps it in
        self._trigger_master_rerender()

    def _on_right_ceiling_changed(self, value: float):
        """Output ceiling changed — update RT engine ceiling + meters."""
        gain_db = getattr(self, '_right_gain_db', 0.0)
        if hasattr(self, 'audio_engine'):
            self.audio_engine.set_gain(gain_db, value)
        if hasattr(self, 'right_logic_meter') and self.right_logic_meter is not None:
            self.right_logic_meter.set_ceiling(value)

        # ★ RT engine ceiling
        if self._rt_engine and self._rt_active:
            self._rt_engine.set_ceiling(value)

        self._trigger_master_rerender()

    def _on_right_width_changed(self, value: int):
        """V5.10: Stereo width dial changed (0=mono, 100=normal, 200=wide)."""
        width_pct = value
        if hasattr(self, 'right_width_display'):
            if width_pct == 0:
                label = "MONO"
                color = "#FF8A65"
            elif width_pct < 100:
                label = f"{width_pct}%"
                color = "#CE93D8"
            elif width_pct == 100:
                label = "100%"
                color = "#CE93D8"
            else:
                label = f"{width_pct}%"
                color = "#E040FB"
            self.right_width_display.setText(label)
            self.right_width_display.setStyleSheet(f"color: {color};")

        # ★ RT engine width (instant audio change)
        if self._rt_engine and self._rt_active:
            self._rt_engine.set_width(width_pct)

        # Sync to chain
        chain = self._get_right_panel_chain()
        if chain and hasattr(chain, 'imager'):
            chain.imager.set_width(width_pct)

        self._trigger_master_rerender()

    # ══ Resonance Suppressor Handlers (right panel) ══

    def _on_soothe_knob_changed(self, value):
        """Soothe knob changed — enable/disable + set amount."""
        amount = value
        enabled = amount > 0
        self.right_res_depth_val.setText(f"{amount:.0f}")
        chain = self._get_right_panel_chain()
        if chain and hasattr(chain, 'soothe'):
            chain.soothe.enabled = enabled
            chain.soothe.set_params(amount=amount)
        if chain and hasattr(chain, 'resonance_suppressor'):
            chain.resonance_suppressor.enabled = enabled
            chain.resonance_suppressor.set_depth(amount / 10.0)
        try:
            if self._rt_engine and hasattr(self._rt_engine, 'set_res_bypass'):
                self._rt_engine.set_res_bypass(not enabled)
                if hasattr(self._rt_engine, 'set_res_depth'):
                    self._rt_engine.set_res_depth(amount / 10.0)
        except Exception:
            pass
        self._trigger_master_rerender()

    def _on_compress_knob_changed(self, value):
        """Compress knob changed — enable/disable + set amount."""
        amount = value
        enabled = amount > 0
        self.right_dyn_amount_val.setText(f"{amount:.0f}%")
        self._on_right_dyn_enabled(enabled)
        if enabled:
            self._on_right_dyn_amount(int(amount))

    def _on_right_res_enabled(self, checked):
        chain = self._get_right_panel_chain()
        if chain and hasattr(chain, 'resonance_suppressor'):
            chain.resonance_suppressor.enabled = checked
        try:
            if self._rt_engine and hasattr(self._rt_engine, 'set_res_bypass'):
                self._rt_engine.set_res_bypass(not checked)
        except Exception:
            pass
        self._trigger_master_rerender()

    def _on_right_res_depth(self, value):
        depth = value / 10.0
        self.right_res_depth_val.setText(f"{depth:.1f}")
        chain = self._get_right_panel_chain()
        if chain and hasattr(chain, 'resonance_suppressor'):
            chain.resonance_suppressor.set_depth(depth)
            chain.resonance_suppressor.set_selectivity(3.0 + depth * 0.3)
        try:
            if self._rt_engine and hasattr(self._rt_engine, 'set_res_depth'):
                self._rt_engine.set_res_depth(depth)
                self._rt_engine.set_res_selectivity(3.0 + depth * 0.3)
        except Exception:
            pass
        self._trigger_master_rerender()

    # ══ Dynamics Handlers (right panel — simple amount) ══

    def _on_right_dyn_enabled(self, checked):
        chain = self._get_right_panel_chain()
        if chain and hasattr(chain, 'dynamics'):
            chain.dynamics.enabled = checked
        try:
            if self._rt_engine and hasattr(self._rt_engine, 'set_dyn_bypass'):
                self._rt_engine.set_dyn_bypass(not checked)
        except Exception:
            pass
        self._trigger_master_rerender()

    def _on_right_dyn_amount(self, value):
        """Simple amount slider → maps to threshold/ratio/attack/release automatically."""
        pct = value / 100.0
        self.right_dyn_amount_val.setText(f"{value}%")
        threshold = -8.0 - pct * 22.0
        ratio = 1.5 + pct * 6.5
        attack = 20.0 - pct * 15.0
        release = 200.0 - pct * 150.0
        makeup = pct * 8.0

        chain = self._get_right_panel_chain()
        if chain and hasattr(chain, 'dynamics'):
            sb = chain.dynamics.single_band
            sb.threshold = threshold
            sb.ratio = ratio
            sb.attack = attack
            sb.release = release
            sb.makeup = makeup
        try:
            if self._rt_engine and hasattr(self._rt_engine, 'set_dyn_threshold'):
                self._rt_engine.set_dyn_threshold(threshold)
                self._rt_engine.set_dyn_ratio(ratio)
                self._rt_engine.set_dyn_attack(attack)
                self._rt_engine.set_dyn_release(release)
                self._rt_engine.set_dyn_makeup(makeup)
        except Exception:
            pass
        self._trigger_master_rerender()

    def _on_right_bypass_switch(self, is_original: bool):
        """Toggle Original/Mastered on the right panel Maximizer header.
        Original = play original file, Mastered = play processed file via chain.
        """
        self._right_bypass_active = is_original

        # Update button styles
        if is_original:
            self.btn_original.setStyleSheet(
                "QPushButton { background: #FF9500; color: #0A0A0C; border: none; "
                "border-radius: 12px; font-size: 10px; font-weight: bold; font-family: 'Menlo', monospace; }")
            self.btn_mastered.setStyleSheet(
                "QPushButton { background: transparent; color: #6B6B70; border: none; "
                "border-radius: 12px; font-size: 10px; font-weight: bold; font-family: 'Menlo', monospace; }"
                "QPushButton:hover { color: #48CAE4; }")
        else:
            self.btn_mastered.setStyleSheet(
                "QPushButton { background: #48CAE4; color: #0A0A0C; border: none; "
                "border-radius: 12px; font-size: 10px; font-weight: bold; font-family: 'Menlo', monospace; }"
                "QPushButton:hover { background: #5AD8F2; }")
            self.btn_original.setStyleSheet(
                "QPushButton { background: transparent; color: #6B6B70; border: none; "
                "border-radius: 12px; font-size: 10px; font-weight: bold; font-family: 'Menlo', monospace; }"
                "QPushButton:hover { color: #FFB340; }")

        # Get current playback state before switching
        try:
            was_playing = getattr(self.audio_player, 'is_playing', False)
            current_pos = self.audio_player.player.position() if hasattr(self.audio_player, 'player') else 0
        except Exception:
            was_playing = False
            current_pos = 0

        if is_original:
            # ── ORIGINAL: switch back to original file ──
            original_file = None
            if hasattr(self, 'audio_engine') and hasattr(self.audio_engine, '_current_file'):
                original_file = self.audio_engine._current_file
            elif hasattr(self, 'audio_player') and self.audio_player.files:
                idx = self.audio_player.current_file_index
                if 0 <= idx < len(self.audio_player.files):
                    original_file = self.audio_player.files[idx]

            if original_file and os.path.exists(original_file):
                self.audio_player.player.stop()
                self.audio_player.player.setSource(QUrl.fromLocalFile(original_file))
                # Reset QAudioOutput volume to 1.0 (no gain)
                if hasattr(self.audio_player, 'audio_output'):
                    self.audio_player.audio_output.setVolume(1.0)
                QTimer.singleShot(50, lambda: self._restore_after_gain(current_pos, was_playing))
                print(f"[BYPASS] → ORIGINAL: {os.path.basename(original_file)}")
            else:
                print("[BYPASS] ⚠️ No original file found")

        else:
            # ── MASTERED: render + play processed version ──
            gain_db = getattr(self, '_right_gain_db', 0.0)
            original_file = None
            if hasattr(self, 'audio_engine') and hasattr(self.audio_engine, '_current_file'):
                original_file = self.audio_engine._current_file
            elif hasattr(self, 'audio_player') and self.audio_player.files:
                idx = self.audio_player.current_file_index
                if 0 <= idx < len(self.audio_player.files):
                    original_file = self.audio_player.files[idx]

            if not original_file or not os.path.exists(original_file):
                print("[A/B] No audio file loaded")
                return

            # Always render mastered version (even if gain=0)
            import threading, tempfile
            try:
                import soundfile as _sf
                import numpy as np

                temp_dir = os.path.join(tempfile.gettempdir(), "longplay_ab")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, "mastered_ab.wav")

                def _render_mastered():
                    try:
                        audio, sr = _sf.read(original_file, dtype='float32')
                        if audio.ndim == 1:
                            audio = np.column_stack([audio, audio])

                        from modules.master.chain import _RealAudioProcessor
                        chain = self._get_right_panel_chain()
                        processed = audio.copy()

                        # Apply gain
                        if gain_db > 0.01:
                            processed = processed * np.float32(10 ** (gain_db / 20.0))

                        # Apply maximizer chain (soft clip + IRC limit + ceiling)
                        if chain and hasattr(chain, 'maximizer'):
                            processed = _RealAudioProcessor.process_maximizer(
                                processed, sr, chain.maximizer, 1.0)

                        # Ceiling hard clip
                        ceiling = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0
                        ceiling_lin = np.float32(10 ** (ceiling / 20.0))
                        np.clip(processed, -ceiling_lin, ceiling_lin, out=processed)

                        _sf.write(temp_path, processed, sr, subtype='FLOAT')
                        print(f"[A/B] Mastered rendered: peak={np.max(np.abs(processed)):.4f}")

                        # Hot-swap on main thread
                        def _swap():
                            try:
                                swap_pos = self.audio_player.player.position()
                                swap_playing = self.audio_player.is_playing
                                self.audio_player.player.stop()
                                self.audio_player.player.setSource(QUrl.fromLocalFile(temp_path))
                                self._gained_preview_active = True
                                self._gained_preview_path = temp_path
                                vol = max(0.5, min(3.0, 0.5 + gain_db / 20.0 * 2.5))
                                if hasattr(self.audio_player, 'audio_output'):
                                    self.audio_player.audio_output.setVolume(vol)
                                QTimer.singleShot(50, lambda: self._restore_after_gain(swap_pos, swap_playing))
                                print(f"[A/B] Switched to MASTERED")
                            except Exception as e:
                                print(f"[A/B] Swap error: {e}")
                        QTimer.singleShot(0, _swap)

                    except Exception as e:
                        print(f"[A/B] Render error: {e}")
                        import traceback; traceback.print_exc()

                threading.Thread(target=_render_mastered, daemon=True).start()
                print(f"[A/B] Rendering mastered (gain={gain_db:.1f}dB)...")

            except Exception as e:
                print(f"[A/B] Error: {e}")

        mode_str = "ORIGINAL" if is_original else "MASTERED"
        print(f"[BYPASS] Switched to {mode_str}")

    def _on_right_irc_mode_changed(self, mode_name: str):
        """V5.7: IRC mode changed — update sub-mode, sync to chain, trigger re-render."""
        if _HAS_PRESETS:
            sub_modes = get_irc_sub_modes(mode_name)
            if sub_modes:
                self.right_irc_submode.clear()
                for sm in sub_modes:
                    self.right_irc_submode.addItem(sm)
                # Set default: Balanced for IRC 3, Classic for IRC 4
                if "Balanced" in sub_modes:
                    self.right_irc_submode.setCurrentText("Balanced")
                elif "Classic" in sub_modes:
                    self.right_irc_submode.setCurrentText("Classic")
                else:
                    self.right_irc_submode.setCurrentIndex(0)
                self.right_irc_submode_widget.setVisible(True)
            else:
                self.right_irc_submode_widget.setVisible(False)

        # Sync to right panel chain + RT engine
        sub_mode = "Balanced"
        if hasattr(self, 'right_irc_submode') and hasattr(self, 'right_irc_submode_widget') and self.right_irc_submode_widget.isVisible():
            sub_mode = self.right_irc_submode.currentText() or "Balanced"

        # ★ RT engine IRC mode
        if self._rt_engine and self._rt_active:
            try:
                self._rt_engine.set_irc_mode(mode_name)
            except Exception:
                pass

        chain = self._get_right_panel_chain()
        if chain and hasattr(chain, 'maximizer'):
            try:
                chain.maximizer.set_irc_mode(mode_name, sub_mode)
            except Exception:
                pass

        # Sync to Master Module if open
        if hasattr(self, '_master_window') and self._master_window is not None:
            mw = self._master_window
            if hasattr(mw, 'chain') and hasattr(mw.chain, 'maximizer'):
                try:
                    mw.chain.maximizer.set_irc_mode(mode_name, sub_mode)
                except Exception:
                    pass

        # Trigger real-time re-render
        self._trigger_master_rerender()

    def _on_right_irc_submode_changed(self, sub_mode: str):
        """V5.10.5: IRC sub-mode changed — sync to chain + re-render."""
        if not sub_mode:
            return
        mode_name = self.right_irc_combo.currentText() if hasattr(self, 'right_irc_combo') else "IRC 2"

        # RT engine
        if self._rt_engine and self._rt_active:
            try:
                self._rt_engine.set_irc_mode(mode_name)
            except Exception:
                pass

        chain = self._get_right_panel_chain()
        if chain and hasattr(chain, 'maximizer'):
            try:
                chain.maximizer.set_irc_mode(mode_name, sub_mode)
            except Exception:
                pass

        if hasattr(self, '_master_window') and self._master_window is not None:
            mw = self._master_window
            if hasattr(mw, 'chain') and hasattr(mw.chain, 'maximizer'):
                try:
                    mw.chain.maximizer.set_irc_mode(mode_name, sub_mode)
                except Exception:
                    pass

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

    def _switch_to_rt_engine(self):
        """V5.10: Switch from QMediaPlayer to Rust RT engine for real-time DSP.
        DISABLED auto-switch: RT engine conflicts with QMediaPlayer causing choppy audio.
        Use offline preview (_trigger_master_rerender) instead.
        """
        # V5.10.1: Disabled auto-switch — QMediaPlayer + offline preview is more stable
        return

        # Find current audio file
        current_file = None
        if hasattr(self, 'audio_engine') and hasattr(self.audio_engine, '_current_file') and self.audio_engine._current_file:
            current_file = self.audio_engine._current_file
        elif hasattr(self, 'audio_player') and self.audio_player.files:
            idx = self.audio_player.current_file_index
            if 0 <= idx < len(self.audio_player.files):
                current_file = self.audio_player.files[idx]

        if not current_file or not os.path.exists(current_file):
            print("[RT ENGINE] No audio file to load")
            return

        try:
            # Get current position and playing state
            was_playing = self.audio_player.is_playing if hasattr(self, 'audio_player') else False
            pos_ms = self.audio_player.player.position() if hasattr(self, 'audio_player') else 0

            # ★ COMPLETELY STOP QMediaPlayer (release audio device)
            self.audio_player.player.stop()
            self.audio_player.audio_output.setVolume(0.0)

            # Load file into RT engine
            if self._rt_file != current_file:
                self._rt_engine.load_file(current_file)
                self._rt_file = current_file
                print(f"[RT ENGINE] Loaded: {os.path.basename(current_file)}")

            # Sync all current knob values
            gain_db = getattr(self, '_right_gain_db', 0.0)
            ceiling = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0
            width = self.right_width_dial.value() if hasattr(self, 'right_width_dial') else 100
            self._rt_engine.set_gain(gain_db)
            self._rt_engine.set_ceiling(ceiling)
            self._rt_engine.set_width(width)

            if hasattr(self, 'right_irc_combo'):
                try:
                    self._rt_engine.set_irc_mode(self.right_irc_combo.currentText())
                except Exception:
                    pass

            # Seek to current position and play
            if pos_ms > 0:
                self._rt_engine.seek(pos_ms)
            if was_playing:
                self._rt_engine.play()

            self._rt_active = True
            self._rt_pos_timer.start()  # Start position sync
            print(f"[RT ENGINE] Active — DSP playback with Gain={gain_db}dB Width={width}%")

        except Exception as e:
            print(f"[RT ENGINE] Switch failed: {e}")
            # Restore QMediaPlayer
            if hasattr(self, 'audio_player'):
                self.audio_player.audio_output.setVolume(1.0)

    def _switch_to_qmediaplayer(self):
        """V5.10: Switch back from RT engine to QMediaPlayer."""
        if not self._rt_active:
            return
        try:
            pos_ms = 0
            if self._rt_engine:
                pos_ms = self._rt_engine.get_position()
                self._rt_engine.stop()
            self._rt_pos_timer.stop()
            self._rt_active = False

            # Restore QMediaPlayer
            if hasattr(self, 'audio_player'):
                self.audio_player.audio_output.setVolume(1.0)
                self.audio_player.player.setPosition(pos_ms)
                self.audio_player.player.play()
            print("[RT ENGINE] Deactivated → QMediaPlayer resumed")
        except Exception as e:
            print(f"[RT ENGINE] Switch back failed: {e}")

    def _on_rt_position_tick(self):
        """V5.10: Sync RT engine position → timeline + meters (30fps)."""
        if not self._rt_engine or not self._rt_active:
            return
        try:
            pos_ms = self._rt_engine.get_position()

            # Update timeline
            self.timeline.setPlayheadPosition(pos_ms)

            # Update video preview
            if hasattr(self, 'video_preview') and self.video_preview:
                self.video_preview.seek(pos_ms)

            # Get RT meter data and feed to WLM meter
            meter = self._rt_engine.get_meter_data()
            if meter and hasattr(self, 'right_wlm_meter') and self.right_wlm_meter:
                import math
                pk_l = meter.get('peak_l', -70.0)
                pk_r = meter.get('peak_r', -70.0)
                rms_l = meter.get('rms_l', -70.0)
                rms_r = meter.get('rms_r', -70.0)
                gr = meter.get('gain_reduction_db', 0.0)

                # Approximate LUFS from RMS
                avg_rms = (rms_l + rms_r) / 2.0
                momentary = max(-70.0, avg_rms)

                if not hasattr(self, '_rt_short_buf'):
                    self._rt_short_buf = []
                self._rt_short_buf.append(momentary)
                if len(self._rt_short_buf) > 90:
                    self._rt_short_buf.pop(0)
                short_term = sum(self._rt_short_buf) / len(self._rt_short_buf) if self._rt_short_buf else -70.0

                self.right_wlm_meter.set_levels(
                    momentary=momentary, short_term=short_term,
                    integrated=short_term, lra=0.0,
                    tp_left=pk_l, tp_right=pk_r)
                self.right_wlm_meter.set_gr(abs(gr))

                # Feed Logic meter
                if hasattr(self, 'right_logic_meter') and self.right_logic_meter:
                    self.right_logic_meter.set_before(
                        l_peak=pk_l, r_peak=pk_r, l_rms=rms_l, r_rms=rms_r)
                    gain_db = getattr(self, '_right_gain_db', 0.0)
                    ceil = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0
                    self.right_logic_meter.set_after(
                        l_peak=min(pk_l + gain_db, ceil), r_peak=min(pk_r + gain_db, ceil),
                        l_rms=min(rms_l + gain_db, ceil), r_rms=min(rms_r + gain_db, ceil))

                if hasattr(self, 'right_gr_history') and self.right_gr_history:
                    self.right_gr_history.set_gr(abs(gr))

            # Update time display
            pos_min = pos_ms // 60000
            pos_sec = (pos_ms % 60000) // 1000
            if hasattr(self, 'time_label'):
                total = self.timeline.total_duration_ms
                t_min = total // 60000
                t_sec = (total % 60000) // 1000
                self.time_label.setText(f"{pos_min:02d}:{pos_sec:02d} / {t_min:02d}:{t_sec:02d}")

        except Exception as e:
            pass  # Don't spam errors at 30fps

    def _trigger_master_rerender(self):
        """V5.10: Trigger preview — only needed when RT engine is NOT active."""
        if self._rt_active:
            return  # RT engine handles everything in real-time
        if not hasattr(self, '_master_rerender_timer'):
            self._master_rerender_timer = QTimer()
            self._master_rerender_timer.setSingleShot(True)
            self._master_rerender_timer.timeout.connect(self._apply_realtime_preview)
        self._master_rerender_timer.start(150)

    def _on_forwarded_meter_data(self, levels: dict):
        """Receive meter data forwarded from MasterPanel._update_realtime_meters().

        This is the SAME data that drives the working LUFS/TP meters.
        Stored for pickup by _feed_meter_panels() on its 50ms timer.
        """
        self._last_meter_levels = levels

    def _install_meter_forward_hook(self):
        """Install forwarding hook on MasterPanel so popup panels get real data."""
        mw = getattr(self, '_master_window', None)
        if mw and not getattr(mw, '_popup_meter_forward', None):
            mw._popup_meter_forward = self._on_forwarded_meter_data

    def _feed_meter_panels(self):
        """Feed audio meter data to popup panels using SAME source as LUFS meter.

        Data comes from MasterPanel._update_realtime_meters() via forwarding hook.
        This ensures popup panels show the exact same data as the working meters.
        """
        if not hasattr(self, '_meter_panels') or not self._meter_panels:
            return

        # Try to install hook if not yet done
        self._install_meter_forward_hook()

        # Get forwarded data (same source as working LUFS/TP meters)
        levels = getattr(self, '_last_meter_levels', None)
        if not levels:
            return

        # Read current knob values
        gain_db = self.right_gain_dial.value() if hasattr(self, 'right_gain_dial') else 0.0
        width_val = self.right_width_dial.value() if hasattr(self, 'right_width_dial') else 100
        soothe_amt = self.right_soothe_knob.value() if hasattr(self, 'right_soothe_knob') else 0
        compress_amt = self.right_compress_knob.value() if hasattr(self, 'right_compress_knob') else 0
        ceiling = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0

        # Extract from forwarded levels dict (same keys as chain._send_meter())
        # Per-stage data available for targeted panel feeds
        stage_data = levels.get('_stage_data', {})

        # Use post_maximizer data for maximizer panel, final for general
        max_data = stage_data.get('post_maximizer', stage_data.get('final', levels))
        dyn_data = stage_data.get('post_dynamics', levels)
        img_data = stage_data.get('post_imager', levels)

        peak_l = levels.get('left_peak_db', -60.0)
        peak_r = levels.get('right_peak_db', -60.0)
        gr_db = levels.get('gain_reduction_db', 0.0)
        lufs = levels.get('lufs_integrated', levels.get('lufs', -14.0))

        # ─── Feed each panel ───

        # Maximizer panel
        p = self._meter_panels.get("maximizer")
        if p and p.isVisible():
            irc_text = "IRC 2"
            if hasattr(self, 'right_irc_combo'):
                idx = self.right_irc_combo.currentIndex() if hasattr(self.right_irc_combo, 'currentIndex') else 0
                irc_text = f"IRC {idx + 1}"
            mp_l = max_data.get('left_peak_db', peak_l)
            mp_r = max_data.get('right_peak_db', peak_r)
            mp_gr = max_data.get('gain_reduction_db', gr_db)
            p.update_meter(
                gain_reduction_db=mp_gr,
                output_peak_l=mp_l, output_peak_r=mp_r,
                lufs=lufs, ceiling=ceiling, irc_mode=irc_text,
                gain_db=gain_db,
                true_peak=max(mp_l, mp_r),
            )

        # Imager panel
        p = self._meter_panels.get("imager")
        if p and p.isVisible():
            vl = img_data.get('left_rms_db', levels.get('left_rms_db', 0.0))
            vr = img_data.get('right_rms_db', levels.get('right_rms_db', 0.0))
            corr = img_data.get('correlation', levels.get('correlation', 1.0))
            # V5.10.5: Read mono bass freq from chain imager
            mono_bass = 0
            mw = getattr(self, '_master_window', None)
            for src in [mw, self]:
                ch = getattr(src, 'chain', None) or getattr(src, '_right_chain', None)
                if ch and hasattr(ch, 'imager') and hasattr(ch.imager, 'mono_bass_freq'):
                    mono_bass = getattr(ch.imager, 'mono_bass_freq', 0)
                    break
            p.update_meter(
                width=int(width_val),
                mono_bass_freq=mono_bass,
                correlation=corr,
                vector_l=vl, vector_r=vr,
            )

        # Soothe panel
        p = self._meter_panels.get("soothe")
        if p and p.isVisible():
            reduction = None
            mw = getattr(self, '_master_window', None)
            for src in [mw, self]:
                ch = getattr(src, 'chain', None) or getattr(src, '_right_chain', None)
                if ch and hasattr(ch, 'soothe') and hasattr(ch.soothe, 'get_reduction_spectrum'):
                    try:
                        reduction = ch.soothe.get_reduction_spectrum()
                        if reduction is not None:
                            break
                    except Exception:
                        pass
            # V5.10.5: Read soothe params from chain
            s_freq_low, s_freq_high, s_speed, s_smoothing, s_delta = 2000.0, 8000.0, 50.0, 50.0, False
            for src in [mw, self]:
                ch = getattr(src, 'chain', None) or getattr(src, '_right_chain', None)
                if ch and hasattr(ch, 'soothe'):
                    s = ch.soothe
                    s_freq_low = getattr(s, 'freq_low', 2000.0)
                    s_freq_high = getattr(s, 'freq_high', 8000.0)
                    s_speed = getattr(s, 'speed', 50.0)
                    s_smoothing = getattr(s, 'smoothing', 50.0)
                    s_delta = getattr(s, 'delta', False)
                    break
            p.update_meter(
                amount=soothe_amt,
                reduction_db=reduction,
                freq_low=s_freq_low,
                freq_high=s_freq_high,
                speed=s_speed,
                smoothing=s_smoothing,
                delta=s_delta,
            )

        # Compressor panel
        p = self._meter_panels.get("compressor")
        if p and p.isVisible():
            comp_gr = dyn_data.get('gain_reduction_db', gr_db) if compress_amt > 0 else 0.0
            # V5.10.5: Read compressor params from chain dynamics
            d_threshold, d_ratio, d_attack, d_release = -18.0, 2.5, 10.0, 100.0
            mw2 = getattr(self, '_master_window', None)
            for src in [mw2, self]:
                ch = getattr(src, 'chain', None) or getattr(src, '_right_chain', None)
                if ch and hasattr(ch, 'dynamics'):
                    sb = getattr(ch.dynamics, 'single_band', None)
                    if sb:
                        d_threshold = getattr(sb, 'threshold', -18.0)
                        d_ratio = getattr(sb, 'ratio', 2.5)
                        d_attack = getattr(sb, 'attack', 10.0)
                        d_release = getattr(sb, 'release', 100.0)
                    break
            p.update_meter(
                gain_reduction_db=comp_gr,
                threshold=d_threshold,
                ratio=d_ratio,
                attack_ms=d_attack,
                release_ms=d_release,
                band_gr_low=dyn_data.get('band_gr_low', 0.0),
                band_gr_mid=dyn_data.get('band_gr_mid', 0.0),
                band_gr_high=dyn_data.get('band_gr_high', 0.0),
            )

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
                            sub_mode = mx.get("irc_sub_mode", None)
                            chain.maximizer.set_irc_mode(mx["irc_mode"], sub_mode)

    def _apply_realtime_preview(self):
        """V5.10: Smooth real-time preview using numpy + double-buffered WAV swap.
        Processes audio in background thread → seamless hot-swap without audible gap.
        """
        try:
            gain_db = getattr(self, '_right_gain_db', 0.0)
            width_pct = self.right_width_dial.value() if hasattr(self, 'right_width_dial') else 100

            # Check if anything needs processing
            needs_processing = gain_db > 0.01 or width_pct != 100
            has_preset = (hasattr(self, 'right_mastering_preset') and
                         self.right_mastering_preset.currentText() != "— None —")
            has_soothe = hasattr(self, 'right_res_enabled') and self.right_res_enabled.isChecked()
            has_compress = hasattr(self, 'right_dyn_enabled') and self.right_dyn_enabled.isChecked()
            needs_processing = needs_processing or has_preset or has_soothe or has_compress

            if not needs_processing:
                # Reset to original file
                if hasattr(self, '_gained_preview_active') and self._gained_preview_active:
                    original = self.audio_engine._current_file if hasattr(self, 'audio_engine') else None
                    if original and os.path.exists(original):
                        was_playing = self.audio_player.is_playing
                        current_pos = self.audio_player.player.position()
                        self.audio_player.player.setSource(QUrl.fromLocalFile(original))
                        self._gained_preview_active = False
                        QTimer.singleShot(50, lambda: self._restore_after_gain(current_pos, was_playing))
                return

            if not hasattr(self, 'audio_engine') or not self.audio_engine._has_soundfile:
                return
            if self.audio_engine._current_data is None:
                return
            if not hasattr(self, 'audio_player') or self.audio_player is None:
                return

            was_playing = getattr(self.audio_player, 'is_playing', False)
            current_pos = self.audio_player.player.position() if hasattr(self.audio_player, 'player') else 0

            # Get raw audio data from memory (already loaded)
            data = self.audio_engine._current_data
            sr = self.audio_engine._current_sr

            # Double-buffer: alternate between 2 temp files for seamless swap
            import tempfile
            temp_dir = os.path.join(tempfile.gettempdir(), "longplay_gain")
            os.makedirs(temp_dir, exist_ok=True)
            self._preview_buf_idx = 1 - getattr(self, '_preview_buf_idx', 0)
            temp_path = os.path.join(temp_dir, f"master_preview_{self._preview_buf_idx}.wav")

            ceiling = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0

            def _process_in_bg():
                try:
                    import numpy as np
                    processed = data.astype(np.float32)

                    # 0. SOOTHE — Resonance Suppressor (before other processing)
                    if has_soothe:
                        chain = self._get_right_panel_chain()
                        if chain and hasattr(chain, 'resonance_suppressor') and chain.resonance_suppressor.enabled:
                            processed = chain.resonance_suppressor.process(processed)

                    # 1. Apply stereo width (M/S processing)
                    if width_pct != 100 and len(processed.shape) > 1 and processed.shape[1] >= 2:
                        mid = (processed[:, 0] + processed[:, 1]) * 0.5
                        side = (processed[:, 0] - processed[:, 1]) * 0.5
                        width_factor = width_pct / 100.0
                        side = side * width_factor
                        processed[:, 0] = mid + side
                        processed[:, 1] = mid - side

                    # 2. Apply gain
                    if gain_db > 0.01:
                        gain_linear = 10 ** (gain_db / 20.0)
                        processed = processed * np.float32(gain_linear)

                    # 2.5 COMPRESS — Dynamics (after gain, before ceiling)
                    if has_compress:
                        chain = self._get_right_panel_chain()
                        if chain and hasattr(chain, 'dynamics') and chain.dynamics.enabled:
                            try:
                                from modules.master.resonance_suppressor import AutoDynamicProcessor
                                auto_dyn = AutoDynamicProcessor(sr)
                                pct = self.right_dyn_amount.value() / 100.0 if hasattr(self, 'right_dyn_amount') else 0.3
                                targets = {0.0: "gentle", 0.5: "balanced", 1.0: "aggressive"}
                                target = "gentle" if pct < 0.33 else ("balanced" if pct < 0.66 else "aggressive")
                                auto_dyn.set_target(target)
                                auto_dyn.analyze(processed, sr)
                                processed = auto_dyn.process(processed)
                            except Exception as e:
                                print(f"[COMPRESS] Error: {e}")

                    # 3. Apply ceiling limiter (brick-wall)
                    ceiling_linear = np.float32(10 ** (ceiling / 20.0))
                    np.clip(processed, -ceiling_linear, ceiling_linear, out=processed)

                    # 4. Write to temp WAV (float32 for speed)
                    sf = self.audio_engine._sf
                    sf.write(temp_path, processed, sr, subtype='FLOAT')

                    # Seamless hot-swap on main thread
                    def _swap():
                        try:
                            # Capture exact position right before swap
                            swap_pos = self.audio_player.player.position()
                            swap_playing = self.audio_player.is_playing

                            # Set new source (QMediaPlayer handles transition)
                            self.audio_player.player.setSource(QUrl.fromLocalFile(temp_path))
                            self._gained_preview_active = True
                            self._gained_preview_path = temp_path

                            # Restore position immediately
                            QTimer.singleShot(30, lambda: self._restore_after_gain(swap_pos, swap_playing))
                        except Exception as e:
                            print(f"[MASTER] Swap error: {e}")
                    QTimer.singleShot(0, _swap)

                except Exception as e:
                    print(f"[MASTER] Process error: {e}")

            import threading
            threading.Thread(target=_process_in_bg, daemon=True).start()

        except Exception as e:
            print(f"[MASTER] Preview error: {e}")

    def _apply_gain_preview(self):
        """V5.7: Legacy — redirects to fast real-time preview."""
        self._apply_realtime_preview()

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
        # V5.10: Also control RT engine if active
        if self._rt_engine and self._rt_active:
            if self._rt_engine.is_playing():
                self._rt_engine.pause()
                self.audio_player.pause()  # sync UI state
            else:
                self._rt_engine.play()
            return
        if self.audio_player.is_playing:
            self.audio_player.pause()
        else:
            self.audio_player.play()

    def _on_tl_stop(self):
        """Stop playback — handles both RT engine and QMediaPlayer."""
        if self._rt_engine and self._rt_active:
            self._rt_engine.stop()
            self._rt_active = False
            if hasattr(self, '_rt_pos_timer'):
                self._rt_pos_timer.stop()
            # Restore QMediaPlayer volume
            if hasattr(self, 'audio_player') and hasattr(self.audio_player, 'audio_output'):
                self.audio_player.audio_output.setVolume(1.0)
        self.audio_player.stop()

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
        """V5.9: Auto-sync selected audio track to Master Module.

        When user clicks a different track in the audio list, automatically
        load it into the mastering panel for seamless workflow.
        """
        if not hasattr(self, 'audio_files') or not self.audio_files:
            return
        if 0 <= row < len(self.audio_files):
            path = self.audio_files[row].path
            if os.path.exists(path):
                # Sync to master panel if open
                if hasattr(self, '_master_window') and self._master_window is not None:
                    try:
                        self._master_window.set_audio(path)
                        print(f"[SYNC] Track selected → Master: {os.path.basename(path)}")
                    except Exception as e:
                        print(f"[SYNC] Master sync error: {e}")
                # Also update the audio player for playback
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
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a *.ogg *.aiff *.aif);;All Files (*)"
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
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)"
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

    def _run_ai_master(self):
        """AI Master: วิเคราะห์ทุกเพลงใน playlist แล้วตั้งค่าอัตโนมัติ"""
        try:
            from modules.master.ai_master import AIMasterEngine
        except ImportError as e:
            QMessageBox.warning(self, "AI Master", f"Cannot load AI Master:\n{e}")
            return

        # Gather audio files from playlist
        audio_files = []
        if hasattr(self, 'audio_player') and self.audio_player.files:
            audio_files = list(self.audio_player.files)

        if not audio_files:
            QMessageBox.warning(self, "AI Master", "กรุณาเพิ่มเพลงก่อน")
            return

        # Get ceiling from right panel
        out_ceiling = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0

        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog("AI Master กำลังวิเคราะห์...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        def update_progress(pct, msg):
            progress.setValue(pct)
            progress.setLabelText(msg)
            QApplication.processEvents()

        try:
            engine = AIMasterEngine()
            settings = engine.analyze_playlist(
                audio_files,
                out_ceiling_db=out_ceiling,
                platform="youtube",
                progress_callback=update_progress
            )

            progress.close()

            # Store for later use
            self._ai_master_settings = settings
            self._ai_master_engine = engine

            # Apply preset for current track
            current_idx = max(0, getattr(self.audio_player, 'current_file_index', 0))
            if current_idx < len(settings.track_presets):
                preset = settings.track_presets[current_idx]
                self._apply_ai_preset(preset)

            QMessageBox.information(self, "AI Master",
                f"วิเคราะห์เสร็จ!\n\n"
                f"เพลงทั้งหมด: {len(settings.track_presets)}\n"
                f"Gain range: {settings.gain_range_db:.1f} dB\n"
                f"Target: {settings.target_lufs:.0f} LUFS\n\n"
                f"ค่าถูกตั้งเป็นค่าเริ่มต้นแล้ว — ปรับ manual ได้เลย")

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "AI Master Error", f"เกิดข้อผิดพลาด:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _apply_ai_preset(self, preset):
        """Apply AI preset to right panel knobs/controls."""
        # Gain
        if hasattr(self, 'right_gain_dial'):
            self.right_gain_dial.setValue(preset.pre_gain_db)
            self._on_right_gain_changed(int(preset.pre_gain_db * 10))

        # Ceiling
        if hasattr(self, 'right_ceiling_spin'):
            self.right_ceiling_spin.setValue(preset.maximizer_ceiling_db)

        # Width
        if hasattr(self, 'right_width_dial'):
            self.right_width_dial.setValue(preset.width_amount * 100)

        # IRC Mode
        if hasattr(self, 'right_irc_combo'):
            # Map IRC_IV → IRC 4
            mode_map = {"IRC_I": "IRC 1", "IRC_II": "IRC 2", "IRC_III": "IRC 3",
                        "IRC_IV": "IRC 4", "IRC_V": "IRC 5"}
            mode_name = mode_map.get(preset.irc_mode, preset.irc_mode)
            idx = self.right_irc_combo.findText(mode_name)
            if idx >= 0:
                self.right_irc_combo.setCurrentIndex(idx)

        # Soothe
        if hasattr(self, 'right_res_depth') and preset.soothe_amount > 0:
            self.right_res_depth.setValue(int(preset.soothe_amount / 10))
            if hasattr(self, 'right_res_enabled'):
                self.right_res_enabled.setChecked(True)

        print(f"[AI MASTER] Applied preset for {preset.filename}: "
              f"gain={preset.pre_gain_db:+.1f}dB, IRC={preset.irc_mode}")

    def _master_export_from_right_panel(self):
        """V5.10: Master & export all tracks using right panel settings (no separate window)."""
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "กรุณาเพิ่มเพลงก่อน (Add audio files first)")
            return

        chain = self._get_right_panel_chain()
        if chain is None:
            QMessageBox.warning(self, "Error", "Cannot create mastering chain")
            return

        # Sync all right panel settings to chain
        self._sync_right_panel_to_chain()

        # Ask for output directory
        default_dir = os.path.dirname(self.audio_files[0].path)
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Mastered Files",
            default_dir, QFileDialog.Option.ShowDirsOnly)
        if not output_dir:
            return

        # Get all track paths
        tracks = [mf.path for mf in self.audio_files if os.path.exists(mf.path)]
        if not tracks:
            QMessageBox.warning(self, "Error", "No audio files found on disk")
            return

        gain_db = getattr(self, '_right_gain_db', 0.0)
        ceiling = self.right_ceiling_spin.value() if hasattr(self, 'right_ceiling_spin') else -1.0
        irc_mode = self.right_irc_combo.currentText() if hasattr(self, 'right_irc_combo') else "IRC 2"

        reply = QMessageBox.question(
            self, "Master Export",
            f"Master {len(tracks)} track(s)?\n\n"
            f"Gain: +{gain_db:.1f} dB\n"
            f"Ceiling: {ceiling:.1f} dBTP\n"
            f"IRC Mode: {irc_mode}\n"
            f"Output: {output_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Process in background thread
        self.btn_open_full_master.setEnabled(False)
        self.btn_open_full_master.setText("⏳ MASTERING...")

        def _do_batch():
            results = []
            for i, track_path in enumerate(tracks):
                try:
                    self._sync_right_panel_to_chain()
                    chain.load_audio(track_path)
                    base_name = os.path.splitext(os.path.basename(track_path))[0]
                    out_path = os.path.join(output_dir, f"{base_name}_mastered.wav")
                    chain.output_path = out_path
                    result = chain.render(output_path=out_path)
                    if result:
                        results.append((base_name, True))
                        print(f"[MASTER EXPORT] ✅ [{i+1}/{len(tracks)}] {base_name}")
                    else:
                        results.append((base_name, False))
                except Exception as e:
                    print(f"[MASTER EXPORT] ❌ {os.path.basename(track_path)}: {e}")
                    results.append((os.path.basename(track_path), False))

            # Update UI on main thread
            def _done():
                self.btn_open_full_master.setEnabled(True)
                self.btn_open_full_master.setText("⚡  MASTER EXPORT")
                success = sum(1 for _, ok in results if ok)
                QMessageBox.information(
                    self, "Master Export Complete",
                    f"Mastered {success}/{len(results)} tracks\n"
                    f"Output: {output_dir}")
            QTimer.singleShot(0, _done)

        import threading
        threading.Thread(target=_do_batch, daemon=True).start()

    def _open_master_module_with_path(self, audio_path: str):
        """V5.5: Open MasterPanel with a specific audio file path (e.g. joined compilation)."""
        try:
            from modules.master.ui_panel import MasterPanel

            # Create or reuse master window
            if not hasattr(self, '_master_window') or self._master_window is None:
                self._master_window = MasterPanel(
                    ffmpeg_path="ffmpeg",
                    shared_audio_player=getattr(self, 'audio_player', None)
                )
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
                self._master_window = MasterPanel(
                    ffmpeg_path="ffmpeg",
                    shared_audio_player=getattr(self, 'audio_player', None)
                )
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
    
    def _show_content_factory(self):
        """Show Content Factory wizard for batch content production."""
        from gui.dialogs.content_factory import ContentFactoryDialog
        dialog = ContentFactoryDialog(self)
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
        try:
            from gui.dialogs.lipsync_dialog import LipSyncDialog
        except ImportError:
            QMessageBox.warning(self, "Unavailable", "Lip-sync module not available.")
            return

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
        # === Audio I/O Preferences ===
        audio_group = QGroupBox("🎧 Audio I/O Preferences")
        audio_group.setStyleSheet(video_group.styleSheet())
        audio_layout = QVBoxLayout(audio_group)

        try:
            from PyQt6.QtMultimedia import QMediaDevices

            # Output device
            out_row = QHBoxLayout()
            out_lbl = QLabel("Output:")
            out_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-weight: bold; min-width: 50px;")
            out_row.addWidget(out_lbl)

            self._settings_out_combo = QComboBox()
            self._settings_out_combo.setStyleSheet(f"""
                QComboBox {{
                    background: {Colors.BG_TERTIARY}; border: 1px solid {Colors.BORDER};
                    color: {Colors.TEXT_PRIMARY}; padding: 6px; border-radius: 4px; font-size: 11px;
                }}
                QComboBox QAbstractItemView {{
                    background: #141418; color: #E0DCD4; selection-background-color: #00B4D8;
                }}
            """)
            current_out = None
            if hasattr(self, 'audio_player') and hasattr(self.audio_player, 'audio_output'):
                current_out = self.audio_player.audio_output.device().description()

            out_devices = QMediaDevices.audioOutputs()
            for i, dev in enumerate(out_devices):
                self._settings_out_combo.addItem(dev.description(), dev.id())
                if current_out and dev.description() == current_out:
                    self._settings_out_combo.setCurrentIndex(i)

            def _apply_output(idx):
                devs = QMediaDevices.audioOutputs()
                if 0 <= idx < len(devs):
                    chosen = devs[idx]
                    if hasattr(self, 'audio_player') and hasattr(self.audio_player, 'audio_output'):
                        self.audio_player.audio_output.setDevice(chosen)
                        print(f"[AUDIO] Output → {chosen.description()}")

            self._settings_out_combo.currentIndexChanged.connect(_apply_output)
            out_row.addWidget(self._settings_out_combo, 1)
            audio_layout.addLayout(out_row)

            # Input device
            in_row = QHBoxLayout()
            in_lbl = QLabel("Input:")
            in_lbl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-weight: bold; min-width: 50px;")
            in_row.addWidget(in_lbl)

            self._settings_in_combo = QComboBox()
            self._settings_in_combo.setStyleSheet(self._settings_out_combo.styleSheet())

            in_devices = QMediaDevices.audioInputs()
            for dev in in_devices:
                self._settings_in_combo.addItem(dev.description(), dev.id())

            in_row.addWidget(self._settings_in_combo, 1)
            audio_layout.addLayout(in_row)

            # Sample rate info
            default_out = QMediaDevices.defaultAudioOutput()
            sr_label = QLabel(f"Default: {default_out.description()}")
            sr_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 9px; font-style: italic;")
            audio_layout.addWidget(sr_label)

        except Exception as e:
            err_lbl = QLabel(f"Audio devices unavailable: {e}")
            err_lbl.setStyleSheet(f"color: #FF4444; font-size: 10px;")
            audio_layout.addWidget(err_lbl)

        layout.addWidget(audio_group)

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
    """Check license and show activation dialog if needed
    
    Returns:
        True if license is valid, False if user dismisses dialog
    """
    from license_manager import check_license
    
    is_licensed, message, license_data = check_license()
    
    if is_licensed:
        print(f"✅ License valid: {message}")
        return True
    
    # Show license dialog - user MUST activate or app exits
    dialog = LicenseDialog()
    result = dialog.exec()
    
    # If user accepted, verify license was activated
    if result == QDialog.DialogCode.Accepted:
        is_licensed, message, license_data = check_license()
        if is_licensed:
            print(f"✅ License activated: {message}")
            return True
    
    # If we reach here, license check failed - user must exit
    print("❌ License activation failed or was dismissed")
    return False


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

    # Auto-load audio folder from command line: python3 gui.py --folder /path/to/folder
    if "--folder" in sys.argv:
        idx = sys.argv.index("--folder")
        if idx + 1 < len(sys.argv):
            folder = sys.argv[idx + 1]
            if os.path.isdir(folder):
                import glob
                audio_exts = ['*.wav', '*.mp3', '*.flac', '*.aac', '*.m4a', '*.ogg', '*.aiff']
                files = []
                for ext in audio_exts:
                    files.extend(glob.glob(os.path.join(folder, ext)))
                if files:
                    print(f"[AUTO-LOAD] Loading {len(files)} files from {folder}")
                    QTimer.singleShot(500, lambda: window._process_audio_files(sorted(files)))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
