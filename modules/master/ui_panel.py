"""
LongPlay Studio V5.0 — Master Chain UI Panel (Waves-Inspired Pro Audio Plugin)
PyQt6 integrated panel for AI Mastering.

Design: Inspired by Neve 1073, SSL 4000, analog console hardware.
Aesthetic: Gunmetal chassis, warm amber VU glow, brushed steel knobs,
          channel-strip layout, backlit panel labels, 3D beveled controls.

┌──────────────────────────────────────────────────────────────────────┐
│  TOP BAR: Platform / Genre / File / Intensity / Reset               │
├──────────────────────────────────────────────────────────────────────┤
│  MODULE CHAIN:  [EQ] → [DYN] → [IMG] → [MAX] → [AI]               │
├──────────────────────────────────────────────────┬───────────────────┤
│  ACTIVE MODULE DETAIL                            │  METERS PANEL     │
│  (Content changes based on selected module)      │  IN/OUT LUFS      │
│                                                  │  True Peak        │
│                                                  │  LRA / Target     │
├──────────────────────────────────────────────────┴───────────────────┤
│  ACTION BAR: [Analyze] [Preview] [AB Compare] [Render]  ▓▓▓░ 55%   │
└──────────────────────────────────────────────────────────────────────┘
"""

import os
import sys
import threading
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
        QPushButton, QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
        QCheckBox, QTabWidget, QProgressBar, QFrame, QGridLayout,
        QFileDialog, QMessageBox, QSizePolicy, QScrollArea,
        QStackedWidget, QSpacerItem, QMenu, QDial,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtGui import (
        QFont, QColor, QPalette, QIcon, QPainter, QPen, QBrush,
        QLinearGradient, QRadialGradient, QPainterPath, QConicalGradient,
    )
except ImportError:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
        QPushButton, QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
        QCheckBox, QTabWidget, QProgressBar, QFrame, QGridLayout,
        QFileDialog, QMessageBox, QSizePolicy, QScrollArea,
        QStackedWidget, QSpacerItem, QMenu, QDial,
    )
    from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal, QTimer, QSize, QUrl
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtGui import (
        QFont, QColor, QPalette, QIcon, QPainter, QPen, QBrush,
        QLinearGradient, QRadialGradient, QPainterPath, QConicalGradient,
    )
import math

# Import master modules — uses Rust backend if available, Python fallback otherwise
from . import MasterChain
# V5.10: Real-time audio engine (Rust + cpal) — lock-free parameter control
try:
    from longplay import PyRtEngine
    _HAS_RT_ENGINE = True
except ImportError:
    _HAS_RT_ENGINE = False
    print("[MASTER] PyRtEngine not available — using offline QMediaPlayer playback")
from .genre_profiles import (
    GENRE_PROFILES, PLATFORM_TARGETS, IRC_MODES, IRC_TOP_MODES, TONE_PRESETS,
    MASTERING_PRESETS, MASTERING_PRESET_NAMES, MASTERING_PRESET_CATEGORIES,
    get_genre_list, get_irc_sub_modes, get_mastering_preset,
)
from .equalizer import EQ_TONE_PRESETS
from .dynamics import DYNAMICS_PRESETS
from .imager import IMAGER_PRESETS


# ══════════════════════════════════════════════════
#  WAVES-INSPIRED PRO AUDIO PLUGIN DESIGN SYSTEM
# ══════════════════════════════════════════════════
#
#  Inspired by:
#    Waves SSL E-Channel — deep black, amber VU, clean layout
#    Waves Abbey Road — warm golden tones on dark metal
#    Waves CLA MixHub — console channel strip aesthetic
#    Waves Renaissance — clean dark UI with subtle glow
#    iZotope Ozone — modern dark with teal accents
#

# --- Chassis & Surface ---
# ═══════════════════════════════════════════════════════════════
#  V5.10 UNIFIED COLOR PALETTE — matches main LongPlay Studio UI
#  Dark teal/orange theme, monospace font, consistent with gui.py
# ═══════════════════════════════════════════════════════════════

# --- Chassis & Panel (Dark neutral — same as main app) ---
C_CHASSIS      = "#1a1a1a"       # Main dark background (BG_PRIMARY)
C_PANEL        = "#242424"       # Secondary panels (BG_SECONDARY)
C_PANEL_LIGHT  = "#2d2d2d"       # Raised surfaces (BG_TERTIARY)
C_PANEL_INSET  = "#141414"       # Deep inset/recessed (PANEL_DEEP)
C_FACEPLATE    = "#2a2a2a"       # Card-style containers (BG_CARD)

# --- Accent Orange (primary highlight — same as main app) ---
C_AMBER        = "#FF9500"       # Primary orange accent
C_AMBER_GLOW   = "#FFB340"       # Bright orange glow
C_AMBER_DIM    = "#CC7700"       # Dimmed orange for borders
C_GOLD         = "#FF9500"       # Labels use same accent

# --- Teal (secondary accent — same as main app) ---
C_TEAL         = "#00B4D8"       # Primary teal accent
C_TEAL_DIM     = "#0077B6"       # Dimmed teal
C_TEAL_GLOW    = "#48CAE4"       # Bright teal glow

# --- Text Colors (White/Grey — same as main app) ---
C_CREAM        = "#ffffff"       # White text (TEXT_PRIMARY)
C_CREAM_DIM    = "#aaaaaa"       # Grey secondary text
C_CREAM_DARK   = "#666666"       # Subtle/disabled text

# --- LED / Status Colors ---
C_LED_RED      = "#E53935"       # Red LED
C_LED_GREEN    = "#43A047"       # Green LED
C_LED_YELLOW   = "#FDD835"       # Yellow LED
C_LED_BLUE     = "#00B4D8"       # Teal LED (matches accent)

# --- Grooves & Borders (Dark neutral) ---
C_GROOVE       = "#111111"       # Deep groove/shadow
C_RIDGE        = "#444444"       # Ridge highlight (BORDER_LIGHT)
C_SCREW        = "#78787E"       # Chrome highlight
C_BORDER       = "#333333"       # Panel border (BORDER)

# --- Module Identity Colors (same as main app) ---
C_MOD_EQ       = "#4FC3F7"       # Light blue (SSL EQ)
C_MOD_DYN      = "#FF8A65"       # Warm orange (CLA Dynamics)
C_MOD_IMG      = "#CE93D8"       # Purple (Abbey Road Imager)
C_MOD_MAX      = "#FFD54F"       # Gold (L2 Maximizer)
C_MOD_AI       = "#00B4D8"       # Teal (AI Assist)

# ── Stylesheet Fragments ──

GLOBAL_STYLE = f"""
QWidget {{
    background-color: {C_CHASSIS};
    color: {C_CREAM};
    font-family: 'Menlo', 'Courier New', monospace;
    font-size: 12px;
}}
QLabel {{
    background: transparent;
}}
QGroupBox {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {C_PANEL_LIGHT}, stop:0.02 {C_PANEL}, stop:0.98 {C_PANEL}, stop:1 {C_GROOVE});
    border: 2px solid {C_GROOVE};
    border-top: 2px solid {C_RIDGE};
    border-radius: 6px;
    padding: 20px 12px 12px 12px;
    margin-top: 14px;
    font-weight: bold;
    color: {C_GOLD};
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: {C_GOLD};
    font-family: 'Menlo', monospace;
}}
QComboBox {{
    background: {C_PANEL_INSET};
    border: 1px solid {C_GROOVE};
    border-top: 1px solid {C_RIDGE};
    border-radius: 3px;
    padding: 5px 8px;
    color: {C_AMBER_GLOW};
    min-width: 100px;
    font-family: 'Menlo', monospace;
    font-weight: bold;
}}
QComboBox:focus {{
    border-color: {C_TEAL};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {C_PANEL};
    color: {C_CREAM};
    selection-background-color: {C_AMBER_DIM};
    selection-color: white;
    border: 1px solid {C_GROOVE};
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {C_GROOVE}, stop:0.5 {C_PANEL_INSET}, stop:1 {C_RIDGE});
    border-radius: 3px;
    border: 1px solid {C_GROOVE};
}}
QSlider::handle:horizontal {{
    width: 20px; height: 20px;
    margin: -8px 0;
    border-radius: 10px;
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #5A5A62, stop:0.3 #4A4A52, stop:0.5 #3A3A42, stop:0.7 #2A2A30, stop:1 #1a1a1a);
    border: 1px solid {C_GROOVE};
    border-top: 1px solid {C_RIDGE};
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C_AMBER_DIM}, stop:1 {C_AMBER});
    border-radius: 3px;
}}
QSlider::groove:vertical {{
    width: 6px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C_GROOVE}, stop:0.5 {C_PANEL_INSET}, stop:1 {C_RIDGE});
    border-radius: 3px;
    border: 1px solid {C_GROOVE};
}}
QSlider::handle:vertical {{
    width: 24px; height: 12px;
    margin: 0 -9px;
    border-radius: 4px;
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #5A5A62, stop:0.3 #4A4A52, stop:0.5 #3A3A42, stop:0.7 #2A2A30, stop:1 #1a1a1a);
    border: 1px solid {C_GROOVE};
    border-top: 1px solid {C_RIDGE};
}}
QDoubleSpinBox, QSpinBox {{
    background: {C_PANEL_INSET};
    border: 1px solid {C_GROOVE};
    border-top: 1px solid {C_RIDGE};
    border-radius: 3px;
    padding: 4px 8px;
    color: {C_AMBER_GLOW};
    font-family: 'Menlo', monospace;
    font-weight: bold;
}}
QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: {C_TEAL};
}}
QCheckBox {{
    spacing: 8px;
    color: {C_CREAM};
    font-size: 11px;
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border-radius: 3px;
    border: 1px solid {C_GROOVE};
    border-top: 1px solid {C_RIDGE};
    background: {C_PANEL_INSET};
}}
QCheckBox::indicator:checked {{
    background: {C_LED_GREEN};
    border-color: #1E7A33;
}}
QProgressBar {{
    background: {C_PANEL_INSET};
    border: 1px solid {C_GROOVE};
    border-radius: 2px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C_TEAL_DIM}, stop:0.5 {C_TEAL}, stop:1 {C_TEAL_GLOW});
    border-radius: 2px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
"""

# ── Button Styles ──

BTN_PRIMARY_STYLE = f"""
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {C_AMBER_GLOW}, stop:0.05 {C_AMBER}, stop:0.5 {C_AMBER_DIM}, stop:0.95 #995500, stop:1 #553300);
    color: #1A1A1E;
    border: 2px solid {C_AMBER_DIM};
    border-top: 2px solid {C_AMBER_GLOW};
    border-radius: 6px;
    padding: 10px 22px;
    font-weight: bold;
    font-size: 12px;
    font-family: 'Menlo', monospace;
    text-transform: uppercase;
    letter-spacing: 2px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {C_AMBER_GLOW}, stop:0.1 {C_AMBER}, stop:0.9 {C_AMBER_DIM}, stop:1 #995500);
    border-color: {C_AMBER_GLOW};
}}
QPushButton:pressed {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #553300, stop:0.1 {C_AMBER_DIM}, stop:0.9 {C_AMBER}, stop:1 {C_AMBER_GLOW});
    border-top: 2px solid {C_GROOVE};
    border-bottom: 2px solid {C_AMBER_GLOW};
}}
QPushButton:disabled {{
    background: {C_PANEL};
    color: {C_CREAM_DARK};
    border-color: {C_GROOVE};
}}
"""

BTN_SECONDARY_STYLE = f"""
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {C_RIDGE}, stop:0.05 {C_PANEL_LIGHT}, stop:0.5 {C_PANEL}, stop:0.95 {C_PANEL}, stop:1 {C_GROOVE});
    color: {C_CREAM};
    border: 2px solid {C_GROOVE};
    border-top: 2px solid {C_RIDGE};
    border-radius: 6px;
    padding: 9px 16px;
    font-size: 11px;
    font-family: 'Menlo', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {C_SCREW}, stop:0.1 {C_RIDGE}, stop:0.9 {C_PANEL_LIGHT}, stop:1 {C_PANEL});
    color: {C_AMBER_GLOW};
    border-color: {C_AMBER_DIM};
}}
QPushButton:pressed {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {C_GROOVE}, stop:0.1 {C_PANEL}, stop:0.9 {C_PANEL_LIGHT}, stop:1 {C_RIDGE});
    border-top: 2px solid {C_GROOVE};
    border-bottom: 2px solid {C_RIDGE};
}}
QPushButton:disabled {{
    color: {C_CREAM_DARK};
    border-color: {C_GROOVE};
}}
"""

BTN_GHOST_STYLE = f"""
QPushButton {{
    background: transparent;
    color: {C_CREAM_DIM};
    border: none;
    padding: 6px 10px;
    font-size: 11px;
    font-family: 'Menlo', monospace;
}}
QPushButton:hover {{
    color: {C_AMBER};
    background: rgba(232,168,50,0.06);
    border-radius: 3px;
}}
"""

BTN_TEAL_STYLE = f"""
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {C_TEAL_GLOW}, stop:0.05 {C_TEAL}, stop:0.5 {C_TEAL_DIM}, stop:0.95 #005577, stop:1 #003344);
    color: #FFFFFF;
    border: 2px solid {C_TEAL_DIM};
    border-top: 2px solid {C_TEAL_GLOW};
    border-radius: 6px;
    padding: 10px 22px;
    font-weight: bold;
    font-size: 13px;
    font-family: 'Menlo', monospace;
    text-transform: uppercase;
    letter-spacing: 2px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 {C_TEAL_GLOW}, stop:0.1 {C_TEAL}, stop:0.9 {C_TEAL_DIM}, stop:1 #005577);
}}
QPushButton:pressed {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #003344, stop:0.1 {C_TEAL_DIM}, stop:0.9 {C_TEAL}, stop:1 {C_TEAL_GLOW});
    border-top: 2px solid {C_GROOVE};
    border-bottom: 2px solid {C_TEAL_GLOW};
}}
QPushButton:disabled {{
    background: {C_PANEL};
    color: {C_CREAM_DARK};
    border-color: {C_GROOVE};
}}
"""


# ══════════════════════════════════════════════════
#  Module Chain Button (Hardware-Style Selector)
# ══════════════════════════════════════════════════
class ModuleChainNode(QPushButton):
    """A clickable module node styled as a hardware channel selector."""

    def __init__(self, icon_text: str, name: str, module_id: str, color: str = None, parent=None):
        super().__init__(parent)
        self.module_id = module_id
        self._active = False
        self._enabled_module = True
        self._accent_color = color or C_TEAL
        self.setFixedSize(96, 56)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setText(f"{icon_text}\n{name}")
        self._update_style()

    def set_active(self, active: bool):
        self._active = active
        self._update_style()

    def set_module_enabled(self, enabled: bool):
        self._enabled_module = enabled
        self._update_style()

    def _update_style(self):
        ac = self._accent_color
        if self._active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 {C_FACEPLATE}, stop:0.05 {C_PANEL_LIGHT},
                        stop:0.95 {C_PANEL}, stop:1 {C_GROOVE});
                    border: 1px solid {ac};
                    border-bottom: 2px solid {ac};
                    border-radius: 4px;
                    color: {ac};
                    font-size: 10px;
                    font-weight: bold;
                    font-family: 'Menlo', monospace;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 {C_PANEL_LIGHT}, stop:0.05 {C_PANEL},
                        stop:0.95 {C_PANEL_INSET}, stop:1 {C_GROOVE});
                    border: 1px solid {C_GROOVE};
                    border-top: 1px solid {C_RIDGE};
                    border-radius: 4px;
                    color: {C_CREAM_DIM};
                    font-size: 10px;
                    font-family: 'Menlo', monospace;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                QPushButton:hover {{
                    color: {C_CREAM};
                    border-color: {C_RIDGE};
                }}
            """)


# ══════════════════════════════════════════════════
#  EQ Curve Widget - iZotope Ozone Style
#  (Visual parametric EQ display with frequency response)
# ══════════════════════════════════════════════════

class EQCurveWidget(QWidget):
    """iZotope Ozone-style parametric EQ frequency response display.

    Shows a dark grid with frequency/dB axes, a glowing EQ curve,
    colored gradient fill under the curve, and draggable band nodes.
    """

    bandChanged = pyqtSignal(int, float)  # band_index, gain_db

    FREQ_LABELS = ["20", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"]
    FREQ_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    BAND_FREQS = [32, 64, 125, 250, 1000, 4000, 8000, 16000]  # 8 bands
    # Colors for gradient fill - teal on left (low freq), magenta on right (high freq)
    BAND_COLORS = [
        "#00B4D8", "#00B4D8", "#4FC3F7", "#4FC3F7",
        "#CE93D8", "#FF8A65", "#E53935", "#E53935"
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMinimumWidth(400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gains = [0.0] * 8  # dB gain for each band (-12 to +12)
        self._dragging_band = -1
        self.setMouseTracking(True)
        self._hover_band = -1

    def setGain(self, band: int, gain_db: float):
        """Set gain for a specific band"""
        if 0 <= band < 8:
            self.gains[band] = max(-12.0, min(12.0, gain_db))
            self.update()

    def _freq_to_x(self, freq: float, left: float, width: float) -> float:
        """Convert frequency to X position (logarithmic scale)"""
        if freq <= 0:
            return left
        log_min = math.log10(20)
        log_max = math.log10(20000)
        log_freq = math.log10(max(20, min(20000, freq)))
        return left + (log_freq - log_min) / (log_max - log_min) * width

    def _db_to_y(self, db: float, top: float, height: float) -> float:
        """Convert dB value to Y position"""
        # -12 dB at bottom, +12 dB at top, 0 at center
        normalized = (db + 12.0) / 24.0  # 0..1
        return top + height - normalized * height

    def _y_to_db(self, y: float, top: float, height: float) -> float:
        """Convert Y position to dB value"""
        normalized = (top + height - y) / height
        return normalized * 24.0 - 12.0

    def _get_band_pos(self, band: int, left: float, top: float, width: float, height: float):
        """Get screen position for a band node"""
        x = self._freq_to_x(self.BAND_FREQS[band], left, width)
        y = self._db_to_y(self.gains[band], top, height)
        return x, y

    def _interpolate_curve(self, x: float, left: float, width: float) -> float:
        """Interpolate the EQ curve at a given x position using smooth spline-like interpolation"""
        # Convert x to frequency
        log_min = math.log10(20)
        log_max = math.log10(20000)
        if width <= 0:
            return 0.0
        ratio = (x - left) / width
        log_freq = log_min + ratio * (log_max - log_min)
        freq = 10 ** log_freq

        # Sum contributions from all bands (bell filter response approximation)
        total_db = 0.0
        for i, (bf, gain) in enumerate(zip(self.BAND_FREQS, self.gains)):
            if abs(gain) < 0.01:
                continue
            # Approximate a bell curve response
            q = 1.5  # Quality factor
            log_ratio = math.log2(freq / bf) if bf > 0 and freq > 0 else 0
            response = gain * math.exp(-0.5 * (log_ratio * q) ** 2)
            total_db += response

        return max(-12.0, min(12.0, total_db))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            margin = 50
            left = margin
            top = 30
            width = self.width() - margin - 20
            height = self.height() - top - 30

            mx, my = event.position().x(), event.position().y()
            # Find closest band node
            closest = -1
            closest_dist = 20  # Max grab distance
            for i in range(8):
                bx, by = self._get_band_pos(i, left, top, width, height)
                dist = ((mx - bx)**2 + (my - by)**2) ** 0.5
                if dist < closest_dist:
                    closest_dist = dist
                    closest = i

            if closest >= 0:
                self._dragging_band = closest

    def mouseMoveEvent(self, event):
        margin = 50
        left = margin
        top = 30
        width = self.width() - margin - 20
        height = self.height() - top - 30
        mx, my = event.position().x(), event.position().y()

        if self._dragging_band >= 0:
            new_db = self._y_to_db(my, top, height)
            new_db = max(-12.0, min(12.0, new_db))
            self.gains[self._dragging_band] = new_db
            self.bandChanged.emit(self._dragging_band, new_db)
            self.update()
        else:
            # Hover detection
            old_hover = self._hover_band
            self._hover_band = -1
            for i in range(8):
                bx, by = self._get_band_pos(i, left, top, width, height)
                dist = ((mx - bx)**2 + (my - by)**2) ** 0.5
                if dist < 15:
                    self._hover_band = i
                    break
            if old_hover != self._hover_band:
                self.update()

    def mouseReleaseEvent(self, event):
        self._dragging_band = -1

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Layout constants
        margin_left = 50
        margin_right = 20
        margin_top = 30
        margin_bottom = 30

        graph_left = margin_left
        graph_top = margin_top
        graph_width = w - margin_left - margin_right
        graph_height = h - margin_top - margin_bottom

        # ── Background: deep black recessed panel ──
        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0.0, QColor(C_GROOVE))
        bg_grad.setColorAt(0.02, QColor(C_PANEL_INSET))
        bg_grad.setColorAt(0.98, QColor(C_PANEL_INSET))
        bg_grad.setColorAt(1.0, QColor(C_GROOVE))
        painter.fillRect(0, 0, w, h, bg_grad)

        # ── Chrome bezel border ──
        painter.setPen(QPen(QColor(C_GROOVE), 2))
        painter.drawRect(1, 1, w - 2, h - 2)
        painter.setPen(QPen(QColor(C_RIDGE), 1))
        painter.drawLine(2, 2, w - 3, 2)  # Top highlight
        painter.drawLine(2, 2, 2, h - 3)  # Left highlight

        # ── Graph background ──
        painter.fillRect(int(graph_left), int(graph_top),
                        int(graph_width), int(graph_height),
                        QColor("#080810"))

        # ── Grid lines (subtle) ──
        # Vertical lines at frequency markers
        painter.setPen(QPen(QColor(40, 40, 48), 1, Qt.PenStyle.DotLine))
        for freq in self.FREQ_VALUES:
            x = self._freq_to_x(freq, graph_left, graph_width)
            painter.drawLine(int(x), int(graph_top), int(x), int(graph_top + graph_height))

        # Horizontal lines at dB markers
        for db in [-12, -6, 0, 6, 12]:
            y = self._db_to_y(db, graph_top, graph_height)
            if db == 0:
                painter.setPen(QPen(QColor(60, 60, 70), 1))  # 0dB line brighter
            else:
                painter.setPen(QPen(QColor(40, 40, 48), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(graph_left), int(y), int(graph_left + graph_width), int(y))

        # ── Frequency labels (bottom) ──
        painter.setFont(QFont("Menlo", 7))
        painter.setPen(QColor(C_CREAM_DARK))
        for freq, label in zip(self.FREQ_VALUES, self.FREQ_LABELS):
            x = self._freq_to_x(freq, graph_left, graph_width)
            painter.drawText(int(x - 12), int(graph_top + graph_height + 15), label)

        # ── dB labels (left) ──
        for db in [-12, -6, 0, 6, 12]:
            y = self._db_to_y(db, graph_top, graph_height)
            text = f"{db:+d}" if db != 0 else " 0"
            painter.drawText(5, int(y + 4), f"{text} dB")

        # ── Title ──
        painter.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        painter.setPen(QColor(C_MOD_EQ))
        painter.drawText(int(graph_left + 5), int(graph_top - 8), "PARAMETRIC EQ — FREQUENCY RESPONSE")

        # ── Build EQ curve path ──
        num_points = 200
        from PyQt6.QtCore import QPointF

        curve_path = QPainterPath()
        fill_path = QPainterPath()

        zero_y = self._db_to_y(0, graph_top, graph_height)
        first_x = graph_left

        points = []
        for i in range(num_points + 1):
            x = graph_left + (i / num_points) * graph_width
            db = self._interpolate_curve(x, graph_left, graph_width)
            y = self._db_to_y(db, graph_top, graph_height)
            points.append((x, y))

        # Build curve path
        if points:
            curve_path.moveTo(points[0][0], points[0][1])
            for px, py in points[1:]:
                curve_path.lineTo(px, py)

            # Build fill path (curve + baseline)
            fill_path.moveTo(points[0][0], zero_y)
            for px, py in points:
                fill_path.lineTo(px, py)
            fill_path.lineTo(points[-1][0], zero_y)
            fill_path.closeSubpath()

        # ── Fill under curve with gradient (teal left → magenta right) ──
        fill_grad = QLinearGradient(graph_left, 0, graph_left + graph_width, 0)
        fill_grad.setColorAt(0.0, QColor(0, 180, 216, 40))    # Teal transparent
        fill_grad.setColorAt(0.3, QColor(79, 195, 247, 50))    # Light blue
        fill_grad.setColorAt(0.5, QColor(206, 147, 216, 40))   # Purple
        fill_grad.setColorAt(0.7, QColor(255, 138, 101, 50))   # Orange
        fill_grad.setColorAt(1.0, QColor(229, 57, 53, 40))     # Red transparent

        painter.setBrush(QBrush(fill_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        # ── Draw curve line with glow ──
        # Glow (wider, semi-transparent)
        glow_pen = QPen(QColor(79, 195, 247, 80), 5)
        painter.setPen(glow_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(curve_path)

        # Main curve line
        curve_grad = QLinearGradient(graph_left, 0, graph_left + graph_width, 0)
        curve_grad.setColorAt(0.0, QColor(C_TEAL))
        curve_grad.setColorAt(0.3, QColor(C_MOD_EQ))
        curve_grad.setColorAt(0.6, QColor(C_MOD_IMG))
        curve_grad.setColorAt(1.0, QColor(C_LED_RED))

        painter.setPen(QPen(QBrush(curve_grad), 2.5))
        painter.drawPath(curve_path)

        # ── Draw band nodes ──
        freq_labels = ["32", "64", "125", "250", "1K", "4K", "8K", "16K"]
        for i in range(8):
            bx, by = self._get_band_pos(i, graph_left, graph_top, graph_width, graph_height)

            is_active = abs(self.gains[i]) > 0.1
            is_hover = (self._hover_band == i) or (self._dragging_band == i)

            node_color = QColor(self.BAND_COLORS[i])

            # Node glow
            if is_hover or is_active:
                glow = QRadialGradient(bx, by, 18)
                glow.setColorAt(0.0, QColor(node_color.red(), node_color.green(), node_color.blue(), 100))
                glow.setColorAt(1.0, QColor(0, 0, 0, 0))
                painter.setBrush(QBrush(glow))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(bx - 18), int(by - 18), 36, 36)

            # Node circle
            radius = 8 if is_hover else 6
            painter.setBrush(QBrush(node_color))
            painter.setPen(QPen(QColor("#FFFFFF" if is_hover else C_PANEL_INSET), 2))
            painter.drawEllipse(int(bx - radius), int(by - radius), radius * 2, radius * 2)

            # Band label near node
            if is_hover or is_active:
                painter.setFont(QFont("Menlo", 7, QFont.Weight.Bold))
                painter.setPen(node_color)
                label_text = f"{freq_labels[i]} {self.gains[i]:+.1f}dB"
                painter.drawText(int(bx - 20), int(by - 14), label_text)

        painter.end()


# ══════════════════════════════════════════════════
#  Vintage Rotary Knob Widget (SSL G-Master Style)
#  Big chunky rotary dial with 3D metallic body, pointer, ticks
# ══════════════════════════════════════════════════

class VintageKnobWidget(QWidget):
    """SSL/Waves-style vintage rotary knob control.

    Features:
    - 3D metallic knob body with conical gradient
    - Pointer indicator line showing current position
    - Tick marks around perimeter
    - Value display below knob
    - Label above knob
    - Mouse drag to adjust value (up/down)

    Signals:
    - valueChanged(float): Emitted when value changes interactively
    """

    valueChanged = pyqtSignal(float)

    def __init__(self, label="", min_val=0.0, max_val=100.0, default=50.0,
                 suffix="", step=0.5, decimals=1, color=None, parent=None):
        """
        Initialize vintage knob widget.

        Args:
            label: Text label above knob (e.g., "THRESHOLD")
            min_val: Minimum value
            max_val: Maximum value
            default: Initial value
            suffix: Unit suffix (e.g., " dB", ":1", " ms")
            step: Value change per pixel dragged
            decimals: Decimal places for display
            color: Accent color for pointer (default C_AMBER)
            parent: Parent widget
        """
        super().__init__(parent)

        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value_data = default
        self.suffix = suffix
        self.step = step
        self.decimals = decimals
        self.color = color or C_AMBER

        # Interaction state
        self._dragging = False
        self._drag_start_y = 0

        # Appearance: fixed size with space for label above, value below
        self.setFixedSize(80, 110)
        self.setMouseTracking(True)

    def value(self) -> float:
        """Get current value without emitting signal."""
        return self.value_data

    def setValue(self, val: float):
        """Set value without emitting signal (used for preset loading)."""
        self.value_data = max(self.min_val, min(self.max_val, val))
        self.update()

    def setValueEmit(self, val: float):
        """Set value and emit signal (used for interactive changes)."""
        clamped = max(self.min_val, min(self.max_val, val))
        if abs(clamped - self.value_data) > 1e-6:
            self.value_data = clamped
            self.valueChanged.emit(clamped)
            self.update()

    def paintEvent(self, event):
        """Paint the vintage knob with 3D metallic appearance."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Widget dimensions
        w, h = self.width(), self.height()
        knob_size = 60
        knob_x = (w - knob_size) // 2  # 10
        knob_y = 20  # below label area
        knob_cx = knob_x + knob_size // 2  # 40
        knob_cy = knob_y + knob_size // 2  # 50

        # ── Draw Label (above knob, small, uppercase, GOLD)
        painter.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        painter.setPen(QColor(C_GOLD))
        painter.drawText(0, 0, w, 16, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter, self.label)

        # ── Draw Outer Chrome Ring (subtle highlight)
        ring_gradient = QConicalGradient(knob_cx, knob_cy, 0)
        ring_gradient.setColorAt(0.0, QColor("#505058"))
        ring_gradient.setColorAt(0.25, QColor("#707078"))
        ring_gradient.setColorAt(0.5, QColor("#505058"))
        ring_gradient.setColorAt(0.75, QColor("#606068"))
        ring_gradient.setColorAt(1.0, QColor("#505058"))
        painter.setBrush(QBrush(ring_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(knob_x - 2, knob_y - 2, knob_size + 4, knob_size + 4)

        # ── Draw Main Knob Body (conical gradient for 3D metal look)
        body_gradient = QConicalGradient(knob_cx, knob_cy, 0)
        body_gradient.setColorAt(0.0, QColor("#2A2A30"))
        body_gradient.setColorAt(0.25, QColor("#3E3E46"))
        body_gradient.setColorAt(0.5, QColor("#2A2A30"))
        body_gradient.setColorAt(0.75, QColor("#383840"))
        body_gradient.setColorAt(1.0, QColor("#2A2A30"))
        painter.setBrush(QBrush(body_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(knob_x, knob_y, knob_size, knob_size)

        # ── Draw Inner Recessed Center Plate (dark, slightly beveled)
        inner_size = 48
        inner_x = knob_x + (knob_size - inner_size) // 2
        inner_y = knob_y + (knob_size - inner_size) // 2
        inner_cx = inner_x + inner_size // 2
        inner_cy = inner_y + inner_size // 2

        inner_gradient = QRadialGradient(inner_cx - 3, inner_cy - 3, inner_size // 2)
        inner_gradient.setColorAt(0.0, QColor("#1E1E22"))
        inner_gradient.setColorAt(0.5, QColor("#0E0E10"))
        inner_gradient.setColorAt(1.0, QColor("#0A0A0C"))
        painter.setBrush(QBrush(inner_gradient))
        painter.setPen(QPen(QColor("#0A0A0C"), 1))
        painter.drawEllipse(inner_x, inner_y, inner_size, inner_size)

        # ── Draw Tick Marks (11 ticks, from 225° to -45° = 270° range, SSL style)
        start_angle = 225  # 7 o'clock
        end_angle = -45     # 5 o'clock (equivalent to 315°)
        angle_range = 270   # degrees
        num_ticks = 11

        painter.setPen(QPen(QColor("#4A4A52"), 1.5))
        for i in range(num_ticks):
            angle_frac = i / (num_ticks - 1)  # 0.0 to 1.0
            angle_deg = start_angle - (angle_frac * angle_range)
            angle_rad = math.radians(angle_deg)

            # Tick outer and inner radius
            tick_outer_r = knob_size // 2
            tick_inner_r = knob_size // 2 - 6

            tick_x1 = knob_cx + tick_outer_r * math.cos(angle_rad)
            tick_y1 = knob_cy + tick_outer_r * math.sin(angle_rad)
            tick_x2 = knob_cx + tick_inner_r * math.cos(angle_rad)
            tick_y2 = knob_cy + tick_inner_r * math.sin(angle_rad)

            painter.drawLine(int(tick_x1), int(tick_y1), int(tick_x2), int(tick_y2))

        # ── Draw Pointer Line (from center outward, showing current value position)
        value_frac = (self.value_data - self.min_val) / (self.max_val - self.min_val)
        value_frac = max(0.0, min(1.0, value_frac))
        pointer_angle_deg = start_angle - (value_frac * angle_range)
        pointer_angle_rad = math.radians(pointer_angle_deg)

        pointer_r_inner = 8
        pointer_r_outer = 26

        pointer_x1 = knob_cx + pointer_r_inner * math.cos(pointer_angle_rad)
        pointer_y1 = knob_cy + pointer_r_inner * math.sin(pointer_angle_rad)
        pointer_x2 = knob_cx + pointer_r_outer * math.cos(pointer_angle_rad)
        pointer_y2 = knob_cy + pointer_r_outer * math.sin(pointer_angle_rad)

        # Draw pointer with glow effect
        painter.setPen(QPen(QColor(self.color), 2.5))
        painter.drawLine(int(pointer_x1), int(pointer_y1), int(pointer_x2), int(pointer_y2))

        # Subtle glow around pointer
        painter.setPen(QPen(QColor(self.color + "40"), 1))
        painter.drawLine(int(pointer_x1 - 1), int(pointer_y1 - 1), int(pointer_x2 - 1), int(pointer_y2 - 1))

        # ── Draw Value Text Below Knob
        value_str = f"{self.value_data:.{self.decimals}f}{self.suffix}"
        painter.setFont(QFont("Menlo", 9, QFont.Weight.Bold))
        painter.setPen(QColor(C_AMBER_GLOW))
        value_y = knob_y + knob_size + 8
        painter.drawText(0, value_y, w, h - value_y, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop, value_str)

        painter.end()

    def mousePressEvent(self, event):
        """Start drag interaction."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_y = event.y()
            event.accept()

    def mouseMoveEvent(self, event):
        """Update value based on vertical drag."""
        if self._dragging:
            delta_y = self._drag_start_y - event.y()  # up = positive
            delta_value = delta_y * self.step
            new_value = self.value_data + delta_value
            self.setValueEmit(new_value)
            self._drag_start_y = event.y()
            event.accept()

    def mouseReleaseEvent(self, event):
        """End drag interaction."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()

    def wheelEvent(self, event):
        """Handle mouse wheel for fine adjustment."""
        delta = event.angleDelta().y()
        steps = (delta / 120) * self.step * 3
        new_value = self.value_data + steps
        self.setValueEmit(new_value)
        event.accept()


# ══════════════════════════════════════════════════
#  Dynamics Compressor Curve Widget (Waves-Style)
#  Shows transfer curve, gain reduction meter, and info readouts
# ══════════════════════════════════════════════════

class DynamicsCurveWidget(QWidget):
    """Waves-style Dynamics Compressor visualization widget.

    Left side: Transfer curve with grid, threshold line, knee highlight
    Right side: Gain reduction LED meter (0 to -20 dB)
    Bottom: Parameter readout (THR, RATIO, ATK, REL, GR, MU)

    Interactive: drag threshold line, drag curve to adjust ratio
    """

    thresholdChanged = pyqtSignal(float)  # emits new threshold dB
    ratioChanged = pyqtSignal(float)      # emits new ratio value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setMinimumWidth(500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Compressor parameters
        self.threshold = -16.0     # dB
        self.ratio = 2.5           # :1
        self.knee = 6.0            # dB
        self.attack = 10.0         # ms
        self.release = 100.0       # ms
        self.makeup = 0.0          # dB
        self.gain_reduction = 0.0  # dB (0 to -20)

        # Interactive state
        self._dragging_threshold = False
        self._dragging_ratio = False
        self._drag_start_x = 0

        self.setMouseTracking(True)

    # ── Property Setters ──
    def setThreshold(self, db: float):
        """Set threshold in dB (-60 to 0)"""
        self.threshold = max(-60, min(0, db))
        self.update()

    def setRatio(self, ratio: float):
        """Set compression ratio (1.0 to 20.0)"""
        self.ratio = max(1.0, min(20.0, ratio))
        self.update()

    def setKnee(self, db: float):
        """Set soft knee in dB (0 to 20)"""
        self.knee = max(0, min(20, db))
        self.update()

    def setAttack(self, ms: float):
        """Set attack time in ms (display only)"""
        self.attack = max(0.1, ms)
        self.update()

    def setRelease(self, ms: float):
        """Set release time in ms (display only)"""
        self.release = max(10, ms)
        self.update()

    def setMakeup(self, db: float):
        """Set makeup gain in dB (display only)"""
        self.makeup = db
        self.update()

    def setGainReduction(self, db: float):
        """Set gain reduction for meter (-20 to 0 dB)"""
        self.gain_reduction = max(-20, min(0, db))
        self.update()

    # ── Coordinate Conversion ──
    def _db_to_x(self, db: float, left: float, width: float) -> float:
        """Convert dB level to X position (linear scale -60 to 0)"""
        # -60 dB at left, 0 dB at right
        normalized = (db + 60.0) / 60.0
        return left + normalized * width

    def _db_to_y(self, db: float, top: float, height: float) -> float:
        """Convert dB level to Y position (linear scale -60 to 0)"""
        # -60 dB at bottom, 0 dB at top
        normalized = (db + 60.0) / 60.0
        return top + height - normalized * height

    def _x_to_db(self, x: float, left: float, width: float) -> float:
        """Convert X position to dB"""
        if width <= 0:
            return -60
        normalized = (x - left) / width
        return -60 + normalized * 60.0

    def _y_to_db(self, y: float, top: float, height: float) -> float:
        """Convert Y position to dB"""
        if height <= 0:
            return -60
        normalized = (top + height - y) / height
        return -60 + normalized * 60.0

    # ── Compressor Transfer Curve ──
    def _compute_compression(self, input_db: float) -> float:
        """Compute output dB given input dB using compression parameters."""
        if input_db < self.threshold - self.knee / 2:
            # Below knee: no compression
            return input_db
        elif input_db > self.threshold + self.knee / 2:
            # Above knee: full compression
            excess = input_db - self.threshold
            output = self.threshold + excess / self.ratio
            return output
        else:
            # In knee region: smooth transition
            knee_start = self.threshold - self.knee / 2
            knee_end = self.threshold + self.knee / 2
            alpha = (input_db - knee_start) / self.knee  # 0..1

            # Linear interpolation between uncompressed and compressed
            uncompressed = input_db
            excess = input_db - self.threshold
            compressed = self.threshold + excess / self.ratio

            output = uncompressed * (1 - alpha) + compressed * alpha
            return output

    # ── Mouse Events ──
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        # Layout dimensions
        margin_left = 60
        margin_right = 100
        margin_top = 25
        margin_bottom = 80

        graph_left = margin_left
        graph_top = margin_top
        graph_width = self.width() * 0.6
        graph_height = self.height() - margin_top - margin_bottom

        mx = event.position().x()
        my = event.position().y()

        # Check if clicking on threshold line
        thr_x = self._db_to_x(self.threshold, graph_left, graph_width)
        if abs(mx - thr_x) < 8 and graph_top <= my <= graph_top + graph_height:
            self._dragging_threshold = True
            self._drag_start_x = mx
            return

        # Check if clicking on curve (for ratio adjustment)
        # Approximate hit region around curve
        for test_db in range(-60, 1):
            curve_y = self._db_to_y(self._compute_compression(test_db), graph_top, graph_height)
            if abs(my - curve_y) < 8 and graph_left <= mx <= graph_left + graph_width:
                self._dragging_ratio = True
                self._drag_start_x = mx
                return

    def mouseMoveEvent(self, event):
        mx = event.position().x()

        margin_left = 60
        margin_right = 100
        margin_top = 25
        margin_bottom = 80

        graph_left = margin_left
        graph_width = self.width() * 0.6

        if self._dragging_threshold:
            # Constrain threshold within visible range
            new_threshold = self._x_to_db(mx, graph_left, graph_width)
            new_threshold = max(-60, min(0, new_threshold))
            if abs(new_threshold - self.threshold) > 0.1:
                self.threshold = new_threshold
                self.thresholdChanged.emit(self.threshold)
                self.update()

        elif self._dragging_ratio:
            # Adjust ratio based on drag distance
            delta_x = mx - self._drag_start_x
            ratio_change = delta_x * 0.05  # Sensitivity: 5% per pixel
            new_ratio = max(1.0, min(20.0, self.ratio + ratio_change))
            if abs(new_ratio - self.ratio) > 0.05:
                self.ratio = new_ratio
                self.ratioChanged.emit(self.ratio)
                self._drag_start_x = mx
                self.update()

    def mouseReleaseEvent(self, event):
        self._dragging_threshold = False
        self._dragging_ratio = False

    # ── Paint Event ──
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Layout constants
        margin_left = 60
        margin_right = 100
        margin_top = 25
        margin_bottom = 80

        graph_left = margin_left
        graph_top = margin_top
        graph_width = w * 0.6
        graph_height = h - margin_top - margin_bottom

        meter_left = graph_left + graph_width + 15
        meter_width = 25
        meter_height = graph_height

        # ── Background: deep black panel ──
        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0.0, QColor(C_GROOVE))
        bg_grad.setColorAt(0.02, QColor(C_PANEL_INSET))
        bg_grad.setColorAt(0.98, QColor(C_PANEL_INSET))
        bg_grad.setColorAt(1.0, QColor(C_GROOVE))
        painter.fillRect(0, 0, w, h, bg_grad)

        # ── Graph Area Background ──
        painter.fillRect(int(graph_left), int(graph_top),
                        int(graph_width), int(graph_height),
                        QColor("#0A0A0C"))

        # Chrome bezel (groove + ridge)
        painter.setPen(QPen(QColor(C_GROOVE), 2))
        painter.drawRect(int(graph_left), int(graph_top),
                        int(graph_width), int(graph_height))
        painter.setPen(QPen(QColor(C_RIDGE), 1))
        painter.drawLine(int(graph_left + 1), int(graph_top + 1),
                        int(graph_left + graph_width - 1), int(graph_top + 1))
        painter.drawLine(int(graph_left + 1), int(graph_top + 1),
                        int(graph_left + 1), int(graph_top + graph_height - 1))

        # ── Grid Lines ──
        painter.setPen(QPen(QColor(40, 40, 48), 1, Qt.PenStyle.DotLine))
        for db in [-48, -36, -24, -12, -6]:
            y = self._db_to_y(db, graph_top, graph_height)
            painter.drawLine(int(graph_left), int(y),
                           int(graph_left + graph_width), int(y))

        for db in [-48, -36, -24, -12, -6]:
            x = self._db_to_x(db, graph_left, graph_width)
            painter.drawLine(int(x), int(graph_top),
                           int(x), int(graph_top + graph_height))

        # Brighter 0 dB lines
        painter.setPen(QPen(QColor(60, 60, 70), 1))
        y_zero = self._db_to_y(0, graph_top, graph_height)
        x_zero = self._db_to_x(0, graph_left, graph_width)
        painter.drawLine(int(graph_left), int(y_zero),
                        int(graph_left + graph_width), int(y_zero))
        painter.drawLine(int(x_zero), int(graph_top),
                        int(x_zero), int(graph_top + graph_height))

        # ── Reference 1:1 Line (dashed) ──
        painter.setPen(QPen(QColor(C_CREAM_DIM), 1, Qt.PenStyle.DashLine))
        ref_start_x = graph_left
        ref_start_y = self._db_to_y(0, graph_top, graph_height)  # 0 dB input → 0 dB output
        ref_end_x = graph_left + graph_width
        ref_end_y = self._db_to_y(-60, graph_top, graph_height)  # -60 dB input → -60 dB output
        painter.drawLine(int(ref_start_x), int(ref_start_y),
                        int(ref_end_x), int(ref_end_y))

        # ── Compression Curve Path ──
        curve_path = QPainterPath()
        fill_path = QPainterPath()

        num_points = 150
        points = []
        for i in range(num_points + 1):
            input_db = -60 + (i / num_points) * 60
            output_db = self._compute_compression(input_db)
            x = self._db_to_x(input_db, graph_left, graph_width)
            y = self._db_to_y(output_db, graph_top, graph_height)
            points.append((x, y, input_db, output_db))

        if points:
            # Curve path
            curve_path.moveTo(points[0][0], points[0][1])
            for x, y, _, _ in points[1:]:
                curve_path.lineTo(x, y)

            # Fill path: curve + reference line
            ref_start_y = self._db_to_y(0, graph_top, graph_height)
            fill_path.moveTo(points[0][0], ref_start_y)
            for x, y, _, _ in points:
                fill_path.lineTo(x, y)
            fill_path.lineTo(points[-1][0], ref_start_y)
            fill_path.closeSubpath()

        # Fill under curve (gain reduction area)
        fill_grad = QLinearGradient(0, 0, 0, graph_top + graph_height)
        fill_grad.setColorAt(0.0, QColor(255, 138, 101, 60))   # Orange transparent
        fill_grad.setColorAt(1.0, QColor(229, 57, 53, 40))     # Red transparent
        painter.setBrush(QBrush(fill_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        # Draw curve with glow
        glow_pen = QPen(QColor(255, 138, 101, 100), 5)
        painter.setPen(glow_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(curve_path)

        # Main curve line
        curve_grad = QLinearGradient(graph_left, 0, graph_left + graph_width, 0)
        curve_grad.setColorAt(0.0, QColor(C_MOD_DYN))
        curve_grad.setColorAt(1.0, QColor(C_LED_RED))
        painter.setPen(QPen(QBrush(curve_grad), 2.5))
        painter.drawPath(curve_path)

        # ── Knee Highlight (subtle zone) ──
        knee_top_db = self.threshold + self.knee / 2
        knee_bot_db = self.threshold - self.knee / 2
        knee_x_bot = self._db_to_x(knee_bot_db, graph_left, graph_width)
        knee_x_top = self._db_to_x(knee_top_db, graph_left, graph_width)
        knee_width = knee_x_top - knee_x_bot

        painter.fillRect(int(knee_x_bot), int(graph_top),
                        int(knee_width), int(graph_height),
                        QColor(255, 138, 101, 15))  # Subtle orange highlight

        # ── Threshold Line (vertical dashed) ──
        thr_x = self._db_to_x(self.threshold, graph_left, graph_width)
        painter.setPen(QPen(QColor(C_AMBER_GLOW), 1.5, Qt.PenStyle.DashLine))
        painter.drawLine(int(thr_x), int(graph_top),
                        int(thr_x), int(graph_top + graph_height))

        # Threshold label
        painter.setFont(QFont("Menlo", 7, QFont.Weight.Bold))
        painter.setPen(QColor(C_AMBER_GLOW))
        painter.drawText(int(thr_x - 15), int(graph_top - 3),
                        f"THR")

        # ── Axis Labels ──
        painter.setFont(QFont("Menlo", 7))
        painter.setPen(QColor(C_CREAM_DIM))

        # X-axis labels (input dB)
        for db in [-60, -48, -36, -24, -12, 0]:
            x = self._db_to_x(db, graph_left, graph_width)
            painter.drawText(int(x - 10), int(graph_top + graph_height + 12),
                           f"{db}")

        # Y-axis labels (output dB)
        for db in [-60, -48, -36, -24, -12, 0]:
            y = self._db_to_y(db, graph_top, graph_height)
            painter.drawText(int(graph_left - 30), int(y + 3),
                           f"{db}")

        # Axis titles
        painter.setFont(QFont("Menlo", 7, QFont.Weight.Bold))
        painter.setPen(QColor(C_GOLD))
        painter.drawText(int(graph_left + graph_width / 2 - 30),
                        int(graph_top + graph_height + 25),
                        "INPUT (dB)")

        # ── Gain Reduction Meter (right side) ──
        meter_top = graph_top
        meter_left_pos = graph_left + graph_width + 15

        # Meter background
        painter.fillRect(int(meter_left_pos - 5), int(meter_top),
                        int(meter_width + 10), int(meter_height),
                        QColor(C_PANEL_INSET))

        # GR label
        painter.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        painter.setPen(QColor(C_GOLD))
        painter.drawText(int(meter_left_pos - 8), int(meter_top - 10),
                        "GR")

        # Meter segments (20 segments for -20 dB range)
        num_segments = 20
        segment_height = meter_height / num_segments

        for i in range(num_segments):
            segment_db = -(20 - i)  # 0 at top, -20 at bottom
            y = meter_top + i * segment_height

            # Determine color based on GR
            if segment_db >= self.gain_reduction:
                if segment_db >= -3:
                    color = QColor(C_LED_GREEN)
                elif segment_db >= -8:
                    color = QColor(C_LED_YELLOW)
                elif segment_db >= -14:
                    color = QColor(C_MOD_DYN)  # Orange
                else:
                    color = QColor(C_LED_RED)
            else:
                color = QColor(50, 50, 60)  # Dimmed

            # Draw rounded segment
            from PyQt6.QtCore import QRectF
            segment_rect = QRectF(meter_left_pos, y, meter_width, segment_height - 1)
            painter.fillRect(segment_rect, color)
            painter.setPen(QPen(QColor(30, 30, 40), 0.5))
            painter.drawRect(segment_rect)

        # Meter scale labels (right side)
        painter.setFont(QFont("Menlo", 6))
        painter.setPen(QColor(C_CREAM_DIM))
        scale_labels = [(0, "0"), (-3, "-3"), (-6, "-6"), (-10, "-10"), (-15, "-15"), (-20, "-20")]
        for db, label in scale_labels:
            y = self._db_to_y(db, meter_top, meter_height)
            painter.drawText(int(meter_left_pos + meter_width + 5), int(y + 2),
                           label)

        # Current GR value display
        painter.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        painter.setPen(QColor(C_AMBER_GLOW))
        gr_text = f"{self.gain_reduction:.1f}"
        painter.drawText(int(meter_left_pos - 5), int(meter_top + meter_height + 25),
                        gr_text)

        # ── Parameter Readouts (bottom-right) ──
        readout_left = graph_left + graph_width - 140
        readout_top = graph_top + graph_height + 40

        painter.setFont(QFont("Menlo", 8))
        painter.setPen(QColor(C_AMBER_GLOW))

        readouts = [
            f"THR: {self.threshold:.1f} dB",
            f"RATIO: {self.ratio:.2f}:1",
            f"ATK: {self.attack:.0f} ms",
            f"REL: {self.release:.0f} ms",
            f"GR: {self.gain_reduction:.1f} dB",
            f"MU: {self.makeup:.1f} dB",
        ]

        for i, text in enumerate(readouts):
            y = readout_top + i * 14
            painter.drawText(int(readout_left), int(y), text)

        # ── Title ──
        painter.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        painter.setPen(QColor(C_MOD_DYN))
        painter.drawText(int(graph_left + 5), int(graph_top - 8),
                        "COMPRESSOR — TRANSFER CURVE")

        painter.end()


# ══════════════════════════════════════════════════
#  Analog Hardware Decoration Widget
#  (VU Arcs, Signal Flow, Rack Screws, Engraved Plates)
# ══════════════════════════════════════════════════

class HardwareDecoration(QWidget):
    """
    Custom-painted analog hardware decoration panel.
    Fills empty space with authentic-looking console elements:
    - Rack screw holes at corners
    - Signal flow diagram with module chain
    - Decorative VU meter arcs
    - Brushed-metal texture
    - Engraved brand plate
    - LED dot indicators
    """

    # Decoration modes
    MODE_SIGNAL_FLOW = "signal_flow"      # Full signal flow diagram (AI Assist)
    MODE_VU_PANEL    = "vu_panel"         # Decorative VU arcs (EQ, Imager)
    MODE_RACK_PLATE  = "rack_plate"       # Minimal rack plate (Dynamics, Maximizer)

    def __init__(self, mode=MODE_SIGNAL_FLOW, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setMinimumHeight(120 if mode == self.MODE_RACK_PLATE else 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # ── Background: brushed metal panel ──
        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0.0, QColor(C_PANEL_INSET))
        bg_grad.setColorAt(0.03, QColor(C_PANEL))
        bg_grad.setColorAt(0.5, QColor(C_PANEL_LIGHT))
        bg_grad.setColorAt(0.97, QColor(C_PANEL))
        bg_grad.setColorAt(1.0, QColor(C_GROOVE))
        painter.fillRect(0, 0, w, h, bg_grad)

        # ── Top groove line ──
        painter.setPen(QPen(QColor(C_GROOVE), 2))
        painter.drawLine(0, 1, w, 1)
        painter.setPen(QPen(QColor(C_RIDGE), 1))
        painter.drawLine(0, 3, w, 3)

        # ── Bottom groove line ──
        painter.setPen(QPen(QColor(C_GROOVE), 2))
        painter.drawLine(0, h - 2, w, h - 2)
        painter.setPen(QPen(QColor(C_RIDGE), 1))
        painter.drawLine(0, h - 4, w, h - 4)

        # ── Rack screws at corners ──
        self._draw_rack_screws(painter, w, h)

        # ── Horizontal brushed metal lines (subtle texture) ──
        painter.setPen(QPen(QColor(58, 58, 62, 30), 1))
        for y_line in range(10, h - 10, 3):
            painter.drawLine(16, y_line, w - 16, y_line)

        # ── Mode-specific content ──
        if self.mode == self.MODE_SIGNAL_FLOW:
            self._draw_signal_flow(painter, w, h)
        elif self.mode == self.MODE_VU_PANEL:
            self._draw_vu_arcs(painter, w, h)
        elif self.mode == self.MODE_RACK_PLATE:
            self._draw_rack_plate(painter, w, h)

        painter.end()

    def _draw_rack_screws(self, painter, w, h):
        """Draw Phillips-head rack screws at corners."""
        screw_positions = [
            (14, 14), (w - 14, 14),
            (14, h - 14), (w - 14, h - 14),
        ]
        for sx, sy in screw_positions:
            # Screw body — radial gradient for 3D look
            screw_grad = QRadialGradient(sx - 1, sy - 1, 7)
            screw_grad.setColorAt(0.0, QColor(100, 100, 105))
            screw_grad.setColorAt(0.5, QColor(75, 75, 80))
            screw_grad.setColorAt(0.85, QColor(55, 55, 60))
            screw_grad.setColorAt(1.0, QColor(35, 35, 38))
            painter.setBrush(QBrush(screw_grad))
            painter.setPen(QPen(QColor(30, 30, 32), 1))
            painter.drawEllipse(sx - 6, sy - 6, 12, 12)

            # Phillips cross
            painter.setPen(QPen(QColor(40, 40, 44), 1.5))
            painter.drawLine(sx - 3, sy, sx + 3, sy)
            painter.drawLine(sx, sy - 3, sx, sy + 3)

    def _draw_signal_flow(self, painter, w, h):
        """Draw the master chain signal flow diagram."""
        cy = h // 2 + 5
        modules = ["INPUT", "EQ", "DYN", "IMG", "MAX", "OUTPUT"]
        n = len(modules)
        spacing = (w - 80) / (n - 1)
        start_x = 40

        # ── "SIGNAL FLOW" engraved title ──
        title_font = QFont("Menlo", 8, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(C_CREAM_DARK), 1))
        painter.drawText(start_x, cy - 42, "SIGNAL  FLOW  —  MASTER  CHAIN")

        # ── Brand plate ──
        plate_y = cy + 38
        plate_w = 200
        plate_x = (w - plate_w) // 2
        plate_h = 22
        plate_grad = QLinearGradient(plate_x, plate_y, plate_x, plate_y + plate_h)
        plate_grad.setColorAt(0.0, QColor(50, 50, 54))
        plate_grad.setColorAt(0.5, QColor(62, 62, 67))
        plate_grad.setColorAt(1.0, QColor(45, 45, 49))
        painter.setBrush(QBrush(plate_grad))
        painter.setPen(QPen(QColor(C_GROOVE), 1))
        painter.drawRoundedRect(plate_x, plate_y, plate_w, plate_h, 3, 3)

        brand_font = QFont("Menlo", 8, QFont.Weight.Bold)
        painter.setFont(brand_font)
        painter.setPen(QPen(QColor(C_GOLD), 1))
        from PyQt6.QtCore import QRectF
        painter.drawText(
            QRectF(plate_x, plate_y, plate_w, plate_h),
            Qt.AlignmentFlag.AlignCenter,
            "L O N G P L A Y   S T U D I O"
        )

        # ── Draw connecting wire ──
        wire_y = cy
        x_first = start_x
        x_last = start_x + (n - 1) * spacing

        # Shadow wire
        painter.setPen(QPen(QColor(C_GROOVE), 3))
        painter.drawLine(int(x_first + 20), wire_y + 1, int(x_last - 20), wire_y + 1)
        # Main wire
        painter.setPen(QPen(QColor(C_AMBER_DIM), 2))
        painter.drawLine(int(x_first + 20), wire_y, int(x_last - 20), wire_y)

        # ── Draw module blocks ──
        module_colors = {
            "EQ": C_MOD_EQ, "DYN": C_MOD_DYN, "IMG": C_MOD_IMG,
            "MAX": C_MOD_MAX, "INPUT": C_CREAM_DIM, "OUTPUT": C_CREAM_DIM
        }
        module_led_colors = {
            "EQ": C_MOD_EQ, "DYN": C_MOD_DYN, "IMG": C_MOD_IMG, "MAX": C_MOD_MAX
        }

        for i, name in enumerate(modules):
            cx = int(start_x + i * spacing)

            # Block background
            block_w, block_h = 38, 26
            bx = cx - block_w // 2
            by = cy - block_h // 2

            if name in ("INPUT", "OUTPUT"):
                # Terminal blocks — darker, round
                term_grad = QRadialGradient(cx, cy, 18)
                term_grad.setColorAt(0.0, QColor(C_PANEL_LIGHT))
                term_grad.setColorAt(1.0, QColor(C_PANEL_INSET))
                painter.setBrush(QBrush(term_grad))
                painter.setPen(QPen(QColor(C_RIDGE), 1.5))
                painter.drawRoundedRect(bx, by, block_w, block_h, 12, 12)
                txt_color = QColor(C_CREAM_DIM)
            else:
                # Module blocks — beveled metal
                blk_grad = QLinearGradient(bx, by, bx, by + block_h)
                blk_grad.setColorAt(0.0, QColor(C_RIDGE))
                blk_grad.setColorAt(0.15, QColor(C_FACEPLATE))
                blk_grad.setColorAt(0.85, QColor(C_PANEL))
                blk_grad.setColorAt(1.0, QColor(C_GROOVE))
                painter.setBrush(QBrush(blk_grad))
                painter.setPen(QPen(QColor(C_BORDER), 1))
                painter.drawRoundedRect(bx, by, block_w, block_h, 4, 4)
                txt_color = QColor(module_colors.get(name, C_AMBER))

            # LED dot above module
            if name not in ("INPUT", "OUTPUT"):
                led_color = QColor(module_led_colors.get(name, C_LED_GREEN))
                led_glow = QRadialGradient(cx, cy - block_h // 2 - 6, 5)
                led_glow.setColorAt(0.0, led_color.lighter(150))
                led_glow.setColorAt(0.5, led_color)
                led_glow.setColorAt(1.0, QColor(0, 0, 0, 0))
                painter.setBrush(QBrush(led_glow))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(cx - 4, cy - block_h // 2 - 9, 8, 8)
                # LED center
                painter.setBrush(QBrush(led_color))
                painter.drawEllipse(cx - 2, cy - block_h // 2 - 7, 4, 4)

            # Arrow between modules (small triangle)
            if i > 0 and i < n:
                ax = int(start_x + (i - 0.5) * spacing)
                painter.setBrush(QBrush(QColor(C_AMBER_DIM)))
                painter.setPen(Qt.PenStyle.NoPen)
                arrow_path = QPainterPath()
                arrow_path.moveTo(ax - 4, cy - 4)
                arrow_path.lineTo(ax + 4, cy)
                arrow_path.lineTo(ax - 4, cy + 4)
                arrow_path.closeSubpath()
                painter.drawPath(arrow_path)

            # Module label
            label_font = QFont("Menlo", 7, QFont.Weight.Bold)
            painter.setFont(label_font)
            painter.setPen(QPen(txt_color, 1))
            from PyQt6.QtCore import QRectF
            painter.drawText(
                QRectF(bx, by, block_w, block_h),
                Qt.AlignmentFlag.AlignCenter,
                name
            )

        # ── Small decorative LED strip at bottom left ──
        led_x_start = 40
        led_y = h - 28
        led_colors = [C_LED_GREEN, C_LED_GREEN, C_LED_GREEN, C_LED_YELLOW,
                      C_LED_YELLOW, C_LED_RED, C_CREAM_DARK, C_CREAM_DARK]
        for i, lc in enumerate(led_colors):
            lx = led_x_start + i * 10
            painter.setBrush(QBrush(QColor(lc)))
            painter.setPen(QPen(QColor(C_GROOVE), 1))
            painter.drawEllipse(lx, led_y, 5, 5)

        # "LEVEL" label next to LEDs
        painter.setFont(QFont("Menlo", 6))
        painter.setPen(QPen(QColor(C_CREAM_DARK), 1))
        painter.drawText(led_x_start + len(led_colors) * 10 + 4, led_y + 5, "LEVEL")

        # ── Small LED strip at bottom right ──
        rled_x = w - 130
        for i, lc in enumerate(led_colors):
            lx = rled_x + i * 10
            painter.setBrush(QBrush(QColor(lc)))
            painter.setPen(QPen(QColor(C_GROOVE), 1))
            painter.drawEllipse(lx, led_y, 5, 5)

        painter.setFont(QFont("Menlo", 6))
        painter.setPen(QPen(QColor(C_CREAM_DARK), 1))
        painter.drawText(rled_x + len(led_colors) * 10 + 4, led_y + 5, "OUTPUT")

    def _draw_vu_arcs(self, painter, w, h):
        """Draw decorative VU meter arcs — two side by side."""
        cy = h // 2 + 8
        arc_r = min(w // 5, h // 2 - 20)

        for side, label in enumerate(["L", "R"]):
            cx = w // 4 + side * w // 2

            # VU background circle
            vu_bg = QRadialGradient(cx, cy, arc_r + 5)
            vu_bg.setColorAt(0.0, QColor(C_PANEL_INSET))
            vu_bg.setColorAt(0.8, QColor(C_PANEL))
            vu_bg.setColorAt(1.0, QColor(C_GROOVE))
            painter.setBrush(QBrush(vu_bg))
            painter.setPen(QPen(QColor(C_GROOVE), 2))
            painter.drawEllipse(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2)

            # Scale arc (from ~220 deg to ~320 deg)
            painter.setPen(QPen(QColor(C_CREAM_DIM), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            from PyQt6.QtCore import QRectF
            arc_rect = QRectF(
                cx - arc_r + 10, cy - arc_r + 10,
                (arc_r - 10) * 2, (arc_r - 10) * 2
            )
            painter.drawArc(arc_rect, 210 * 16, -120 * 16)

            # Scale tick marks
            for tick in range(13):
                angle_deg = 210 - tick * 10
                angle_rad = math.radians(angle_deg)
                inner_r = arc_r - 14
                outer_r = arc_r - 8
                x1 = cx + inner_r * math.cos(angle_rad)
                y1 = cy - inner_r * math.sin(angle_rad)
                x2 = cx + outer_r * math.cos(angle_rad)
                y2 = cy - outer_r * math.sin(angle_rad)

                if tick >= 10:
                    painter.setPen(QPen(QColor(C_LED_RED), 1.5))
                elif tick >= 7:
                    painter.setPen(QPen(QColor(C_LED_YELLOW), 1))
                else:
                    painter.setPen(QPen(QColor(C_CREAM_DIM), 1))
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            # VU needle — positioned at ~-6 dB area
            needle_angle = 190 - side * 15  # slightly different per channel
            needle_rad = math.radians(needle_angle)
            needle_len = arc_r - 18
            nx = cx + needle_len * math.cos(needle_rad)
            ny = cy - needle_len * math.sin(needle_rad)

            # Needle shadow
            painter.setPen(QPen(QColor(0, 0, 0, 80), 2))
            painter.drawLine(cx + 1, cy + 1, int(nx) + 1, int(ny) + 1)
            # Needle
            painter.setPen(QPen(QColor(C_AMBER), 2.5))
            painter.drawLine(cx, cy, int(nx), int(ny))

            # Amber glow at pivot
            pivot_glow = QRadialGradient(cx, cy, 8)
            pivot_glow.setColorAt(0.0, QColor(C_AMBER_GLOW))
            pivot_glow.setColorAt(0.5, QColor(C_AMBER))
            pivot_glow.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setBrush(QBrush(pivot_glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(cx - 6, cy - 6, 12, 12)

            # Needle pivot
            pivot_grad = QRadialGradient(cx, cy, 5)
            pivot_grad.setColorAt(0.0, QColor(C_SCREW))
            pivot_grad.setColorAt(1.0, QColor(C_GROOVE))
            painter.setBrush(QBrush(pivot_grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(cx - 4, cy - 4, 8, 8)

            # Channel label
            painter.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(C_CREAM_DARK), 1))
            painter.drawText(cx - 4, cy + arc_r - 4, label)

        # "VU" label centered
        painter.setFont(QFont("Menlo", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(C_GOLD), 1))
        painter.drawText(w // 2 - 6, cy - arc_r + 20, "VU")

        # Brand text
        painter.setFont(QFont("Menlo", 6))
        painter.setPen(QPen(QColor(C_CREAM_DARK), 1))
        painter.drawText(w // 2 - 40, h - 18, "LONGPLAY STUDIO  ·  V5.0")

    def _draw_rack_plate(self, painter, w, h):
        """Draw a minimal rack-mount plate with model info."""
        cy = h // 2

        # Center plate
        plate_w = min(w - 60, 320)
        plate_h = 36
        plate_x = (w - plate_w) // 2
        plate_y = cy - plate_h // 2

        plate_grad = QLinearGradient(plate_x, plate_y, plate_x, plate_y + plate_h)
        plate_grad.setColorAt(0.0, QColor(58, 58, 63))
        plate_grad.setColorAt(0.15, QColor(68, 68, 74))
        plate_grad.setColorAt(0.85, QColor(52, 52, 57))
        plate_grad.setColorAt(1.0, QColor(40, 40, 44))
        painter.setBrush(QBrush(plate_grad))
        painter.setPen(QPen(QColor(C_GROOVE), 1))
        painter.drawRoundedRect(plate_x, plate_y, plate_w, plate_h, 4, 4)

        # Inner bevel
        painter.setPen(QPen(QColor(C_RIDGE), 0.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(plate_x + 2, plate_y + 2, plate_w - 4, plate_h - 4, 3, 3)

        # Model text
        from PyQt6.QtCore import QRectF
        painter.setFont(QFont("Menlo", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(C_GOLD), 1))
        painter.drawText(
            QRectF(plate_x, plate_y, plate_w, plate_h),
            Qt.AlignmentFlag.AlignCenter,
            "LONGPLAY  STUDIO  —  AI  MASTER  V5.0"
        )

        # Small decorative screws flanking the plate
        for sx in [plate_x - 16, plate_x + plate_w + 6]:
            screw_grad = QRadialGradient(sx + 5, cy, 5)
            screw_grad.setColorAt(0.0, QColor(90, 90, 95))
            screw_grad.setColorAt(1.0, QColor(40, 40, 44))
            painter.setBrush(QBrush(screw_grad))
            painter.setPen(QPen(QColor(30, 30, 32), 1))
            painter.drawEllipse(sx, cy - 5, 10, 10)
            # Cross
            painter.setPen(QPen(QColor(45, 45, 48), 1))
            painter.drawLine(sx + 3, cy, sx + 7, cy)
            painter.drawLine(sx + 5, cy - 2, sx + 5, cy + 2)

        # LED dots on both sides
        for side in [0, 1]:
            base_x = 36 if side == 0 else w - 66
            for j in range(3):
                lx = base_x + j * 10
                led_c = QColor(C_LED_GREEN) if j < 2 else QColor(C_LED_YELLOW)
                painter.setBrush(QBrush(led_c))
                painter.setPen(QPen(QColor(C_GROOVE), 1))
                painter.drawEllipse(lx, cy - 2, 5, 5)


# ══════════════════════════════════════════════════
#  Waves WLM Plus-Style Loudness Meter
# ══════════════════════════════════════════════════
class WavesWLMMeter(QWidget):
    """
    V5.7 Waves WLM Plus-style loudness meter — matches the official Waves WLM Plus layout.

    Layout (top to bottom):
    ┌─────────────────────────────────────────┐
    │  SHORT TERM │  LONG TERM  │   RANGE     │  ← 3 LCD boxes with big numbers
    │    -15      │    -14      │     5       │
    │  -14.9 LKFS│  -13.9 LKFS │    LU       │
    ├─────────────────────────────────────────┤
    │  Momentary  ████████████░░░░░  -9.3 LKFS│  ← horizontal bar + readout
    ├─────────────────────────────────────────┤
    │  True Peak  ██████████████░░░░  0.0 dBTP│  ← horizontal bar + readout
    ├─────────────────────────────────────────┤
    │  True Peak Limiter           GR: -1.9 dB│  ← limiter GR readout
    └─────────────────────────────────────────┘
    """

    def __init__(self, parent=None, target_lufs=-14.0):
        super().__init__(parent)
        self.setFixedSize(270, 320)
        self._target = target_lufs

        # Current values
        self._mom = -70.0
        self._short = -70.0
        self._int = -70.0
        self._lra = 0.0
        self._tp_l = -70.0
        self._tp_r = -70.0
        self._gr = 0.0  # Gain reduction dB
        self._max_mom = -70.0
        self._max_short = -70.0
        self._max_tp = -70.0

    def set_levels(self, momentary=-70.0, short_term=-70.0, integrated=-70.0,
                   lra=0.0, tp_left=-70.0, tp_right=-70.0):
        self._mom = max(-60.0, min(6.0, momentary))
        self._short = max(-60.0, min(6.0, short_term))
        self._int = max(-60.0, min(6.0, integrated))
        self._lra = max(0.0, min(30.0, lra))
        self._tp_l = max(-60.0, min(6.0, tp_left))
        self._tp_r = max(-60.0, min(6.0, tp_right))
        if self._mom > self._max_mom:
            self._max_mom = self._mom
        if self._short > self._max_short:
            self._max_short = self._short
        tp_max = max(self._tp_l, self._tp_r)
        if tp_max > self._max_tp:
            self._max_tp = tp_max
        self.update()

    def set_gr(self, gr_db: float):
        """Set gain reduction (positive value)."""
        self._gr = max(0.0, min(20.0, abs(gr_db)))
        self.update()

    def set_target(self, target_lufs=-14.0):
        self._target = target_lufs
        self.update()

    def reset(self):
        self._mom = self._short = self._int = -70.0
        self._lra = 0.0
        self._tp_l = self._tp_r = -70.0
        self._gr = 0.0
        self._max_mom = self._max_short = self._max_tp = -70.0
        self.update()

    def _lufs_color(self, lufs):
        if lufs > self._target + 2.0:
            return QColor("#E53935")
        elif lufs > self._target + 0.5:
            return QColor("#F39C12")
        elif lufs > self._target - 2.0:
            return QColor("#43A047")
        elif lufs > self._target - 6.0:
            return QColor("#26A69A")
        else:
            return QColor("#42A5F5")

    def _tp_color(self, tp_db):
        if tp_db > -0.5:
            return QColor(C_LED_RED)
        elif tp_db > -1.5:
            return QColor(C_LED_YELLOW)
        else:
            return QColor(C_LED_GREEN)

    def _db_to_bar_ratio(self, db, lo=-60.0, hi=6.0):
        return max(0.0, min(1.0, (db - lo) / (hi - lo)))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        # ── Background (Waves dark gray) ──
        p.fillRect(0, 0, w, h, QColor("#2A2A2E"))
        # Inner bevel
        p.setPen(QPen(QColor("#1A1A1E"), 1))
        p.drawRect(1, 1, w - 3, h - 3)
        p.setPen(QPen(QColor("#3E3E44"), 1))
        p.drawLine(2, 2, w - 3, 2)

        # ═══ TOP: 3 LCD BOXES (Short Term | Long Term | Range) ═══
        box_h = 68
        box_y = 6
        gap = 4
        box_w = (w - 12 - gap * 2) // 3
        boxes = [
            ("SHORT TERM", self._short, "LKFS"),
            ("LONG TERM", self._int, "LKFS"),
            ("RANGE", self._lra, "LU"),
        ]

        for i, (label, val, unit) in enumerate(boxes):
            bx = 6 + i * (box_w + gap)

            # V5.8 C-1: Color-coded LCD background based on target proximity
            if label == "RANGE":
                box_bg = QColor("#080808")
            else:
                is_active = val > -69
                if is_active:
                    diff = abs(val - self._target)
                    if diff <= 2.0:
                        box_bg = QColor(67, 160, 71, 40)    # green tint
                    elif diff <= 5.0:
                        box_bg = QColor(253, 216, 53, 30)    # yellow tint
                    else:
                        box_bg = QColor(229, 57, 53, 35)     # red tint
                else:
                    box_bg = QColor("#080808")

            p.fillRect(bx, box_y, box_w, box_h, box_bg)
            p.setPen(QPen(QColor("#1A1A1E"), 1))
            p.drawRect(bx, box_y, box_w, box_h)

            # Label
            p.setFont(QFont("Menlo", 7))
            p.setPen(QColor("#888888"))
            p.drawText(bx, box_y + 2, box_w, 12, Qt.AlignmentFlag.AlignCenter, label)

            # Big number — Waves WLM Plus style (bright red/green LCD)
            if label == "RANGE":
                val_text = f"{val:.0f}" if val > 0.01 else "0"
                val_color = QColor("#E0E0E0")
            else:
                is_active = val > -69
                val_text = f"{val:.0f}" if is_active else "--"
                if is_active:
                    diff = val - self._target
                    if diff > 2.0:
                        val_color = QColor("#FF1744")   # bright red (too loud)
                    elif diff > 0.5:
                        val_color = QColor("#FF6D00")   # orange (slightly loud)
                    elif diff > -2.0:
                        val_color = QColor("#00E676")   # bright green (on target)
                    else:
                        val_color = QColor("#00BCD4")   # cyan (too quiet)
                else:
                    val_color = QColor("#555555")

            p.setFont(QFont("Menlo", 28, QFont.Weight.Bold))
            p.setPen(val_color)
            p.drawText(bx, box_y + 12, box_w, 36, Qt.AlignmentFlag.AlignCenter, val_text)

            # Precise value + unit
            if label != "RANGE":
                precise = f"{val:.1f}" if val > -69 else "--.-"
                p.setFont(QFont("Menlo", 8))
                p.setPen(QColor("#888888"))
                p.drawText(bx, box_y + box_h - 16, box_w, 14, Qt.AlignmentFlag.AlignCenter,
                           f"{precise} {unit}")
            else:
                p.setFont(QFont("Menlo", 9))
                p.setPen(QColor("#888888"))
                p.drawText(bx, box_y + box_h - 16, box_w, 14, Qt.AlignmentFlag.AlignCenter, unit)

        y = box_y + box_h + 8

        # ═══ MOMENTARY BAR ═══
        # Section background (darker inset like Waves WLM)
        p.fillRect(4, y - 2, w - 8, 40, QColor("#1A1A1E"))
        p.setPen(QPen(QColor("#2A2A30"), 1))
        p.drawRect(4, y - 2, w - 8, 40)

        p.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        p.setPen(QColor("#CCCCCC"))
        p.drawText(8, y, "Momentary")
        # Readout (right side) — bright colored like Waves WLM
        mom_text = f"{self._mom:.1f}" if self._mom > -69 else "--.-"
        if self._mom > -69:
            mom_col = QColor("#FF1744") if self._mom > self._target + 1 else (
                QColor("#00E676") if self._mom > self._target - 3 else QColor("#00BCD4"))
        else:
            mom_col = QColor("#555555")
        p.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        p.setPen(mom_col)
        p.drawText(w - 80, y, 74, 14, Qt.AlignmentFlag.AlignRight, f"{mom_text}")
        p.setFont(QFont("Menlo", 7))
        p.setPen(QColor("#999999"))
        p.drawText(w - 80, y, 74, 14, Qt.AlignmentFlag.AlignLeft, "LKFS")
        y += 14

        bar_x, bar_w, bar_h = 6, w - 12, 20
        self._draw_meter_bar(p, bar_x, y, bar_w, bar_h, self._mom, self._max_mom, -53, 0)

        # Scale labels
        y += bar_h + 1
        p.setFont(QFont("Menlo", 6))
        p.setPen(QColor("#666666"))
        for sv in [-48, -36, -24, -18, -14, -9, -6, 0]:
            ratio = self._db_to_bar_ratio(sv, -53, 0)
            sx = bar_x + int(ratio * bar_w)
            p.drawText(sx - 6, y + 8, f"{sv}")
        y += 14

        # ═══ TRUE PEAK BAR ═══
        # Section background
        p.fillRect(4, y - 2, w - 8, 40, QColor("#1A1A1E"))
        p.setPen(QPen(QColor("#2A2A30"), 1))
        p.drawRect(4, y - 2, w - 8, 40)

        p.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        p.setPen(QColor("#CCCCCC"))
        p.drawText(8, y, "True Peak")
        tp_max = max(self._tp_l, self._tp_r)
        tp_text = f"{tp_max:.1f}" if tp_max > -69 else "--.-"
        if tp_max > -69:
            tp_col = QColor("#FF1744") if tp_max > -0.5 else (
                QColor("#FFD600") if tp_max > -1.5 else QColor("#00E676"))
        else:
            tp_col = QColor("#555555")
        p.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        p.setPen(tp_col)
        p.drawText(w - 80, y, 74, 14, Qt.AlignmentFlag.AlignRight, tp_text)
        p.setFont(QFont("Menlo", 7))
        p.setPen(QColor("#999999"))
        p.drawText(w - 80, y, 74, 14, Qt.AlignmentFlag.AlignLeft, "dBTP")
        y += 14

        self._draw_meter_bar(p, bar_x, y, bar_w, bar_h, tp_max, self._max_tp, -58, 3, is_tp=True)
        y += bar_h + 6

        # ═══ TRUE PEAK LIMITER / GR DISPLAY ═══
        p.fillRect(6, y, w - 12, 40, QColor("#181818"))
        p.setPen(QPen(QColor("#2A2A30"), 1))
        p.drawRect(6, y, w - 12, 40)

        p.setFont(QFont("Menlo", 8))
        p.setPen(QColor("#AAAAAA"))
        p.drawText(12, y + 4, "True Peak Limiter")

        # GR readout
        gr_text = f"-{self._gr:.1f}" if self._gr > 0.05 else "0.0"
        gr_col = QColor(C_LED_RED) if self._gr > 6 else (
            QColor(C_AMBER_GLOW) if self._gr > 3 else (
                QColor(C_LED_GREEN) if self._gr > 0.1 else QColor("#555555")))
        p.setFont(QFont("Menlo", 7))
        p.setPen(QColor("#888888"))
        p.drawText(w - 90, y + 4, "GR dB")
        p.setFont(QFont("Menlo", 16, QFont.Weight.Bold))
        p.setPen(gr_col)
        p.drawText(w - 80, y + 14, 68, 22, Qt.AlignmentFlag.AlignRight, gr_text)

        # GR mini bar
        gr_bar_x = 12
        gr_bar_w = w - 100
        gr_bar_y = y + 22
        p.fillRect(gr_bar_x, gr_bar_y, gr_bar_w, 10, QColor("#0A0A0A"))
        if self._gr > 0.05:
            gr_fill = min(1.0, self._gr / 20.0)
            fill_w = int(gr_fill * gr_bar_w)
            grad = QLinearGradient(gr_bar_x, 0, gr_bar_x + gr_bar_w, 0)
            grad.setColorAt(0.0, QColor("#43A047"))
            grad.setColorAt(0.5, QColor("#F39C12"))
            grad.setColorAt(1.0, QColor("#E53935"))
            p.fillRect(gr_bar_x, gr_bar_y, fill_w, 10, grad)
        p.setPen(QPen(QColor("#2A2A30"), 1))
        p.drawRect(gr_bar_x, gr_bar_y, gr_bar_w, 10)

        y += 44

        # ═══ TARGET BAR (bottom) ═══
        p.fillRect(6, y, w - 12, 18, QColor("#111116"))
        p.setPen(QPen(QColor("#2A2A30"), 1))
        p.drawRect(6, y, w - 12, 18)
        p.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        p.setPen(QColor(C_TEAL_GLOW))
        p.drawText(6, y, w - 12, 18, Qt.AlignmentFlag.AlignCenter,
                   f"TARGET: {self._target:.0f} LUFS")

        p.end()

    def _draw_meter_bar(self, p, x, y, w, h, value, peak, lo, hi, is_tp=False):
        """Draw a Waves WLM Plus-style horizontal meter bar with segmented LED look."""
        # Recessed background
        p.fillRect(x, y, w, h, QColor("#050508"))
        p.setPen(QPen(QColor("#1A1A1E"), 1))
        p.drawRect(x, y, w, h)

        inner_x = x + 1
        inner_w = w - 2
        inner_y = y + 1
        inner_h = h - 2

        # Waves WLM scale points for the bar
        if is_tp:
            scale_pts = [-58, -48, -40, -30, -25, -20, -15, -10, -5, 0, 2]
        else:
            scale_pts = [-53, -48, -43, -38, -33, -28, -23, -18, -13, -8]

        # Draw segmented LED blocks
        num_segments = 50
        seg_w = max(1, (inner_w - num_segments + 1) // num_segments)
        seg_gap = 1

        if value > lo:
            fill_ratio = self._db_to_bar_ratio(value, lo, hi)
            lit_segments = int(fill_ratio * num_segments)

            for s in range(num_segments):
                sx = inner_x + s * (seg_w + seg_gap)
                if sx + seg_w > inner_x + inner_w:
                    break
                seg_ratio = s / float(num_segments)

                # Waves WLM color scheme: blue → green → yellow → red
                if seg_ratio < 0.35:
                    # Blue-green zone (very low levels)
                    r = 30 + int(seg_ratio / 0.35 * 30)
                    g = 80 + int(seg_ratio / 0.35 * 120)
                    b = 180 - int(seg_ratio / 0.35 * 80)
                    col = QColor(r, g, b)
                    col_dim = QColor(r // 5, g // 5, b // 5)
                elif seg_ratio < 0.65:
                    # Green zone
                    t = (seg_ratio - 0.35) / 0.30
                    r = 60 + int(t * 190)
                    g = 200 - int(t * 20)
                    b = 100 - int(t * 80)
                    col = QColor(r, g, b)
                    col_dim = QColor(r // 5, g // 5, b // 5)
                elif seg_ratio < 0.85:
                    # Yellow zone
                    t = (seg_ratio - 0.65) / 0.20
                    r = 250
                    g = 200 - int(t * 80)
                    b = 20 + int(t * 10)
                    col = QColor(r, g, b)
                    col_dim = QColor(r // 5, g // 5, b // 5)
                else:
                    # Red zone (loud / over)
                    t = (seg_ratio - 0.85) / 0.15
                    r = 250 - int(t * 20)
                    g = 60 - int(t * 40)
                    b = 30
                    col = QColor(r, g, b)
                    col_dim = QColor(r // 5, g // 5, b // 5)

                if s < lit_segments:
                    p.fillRect(sx, inner_y, seg_w, inner_h, col)
                    # Subtle glow on lit segments
                    glow = QColor(col)
                    glow.setAlpha(40)
                    p.fillRect(sx, inner_y, seg_w, inner_h // 3, glow)
                else:
                    p.fillRect(sx, inner_y, seg_w, inner_h, col_dim)

        # Peak hold marker (bright white line)
        if peak > lo:
            pk_ratio = self._db_to_bar_ratio(peak, lo, hi)
            pk_x = inner_x + int(pk_ratio * inner_w)
            pk_col = QColor("#FF4444") if peak > hi - 1 else QColor("#FFFFFF")
            p.setPen(QPen(pk_col, 2))
            p.drawLine(pk_x, inner_y, pk_x, inner_y + inner_h)

        # Target line (teal, only for LUFS bars)
        if not is_tp:
            tgt_ratio = self._db_to_bar_ratio(self._target, lo, hi)
            tgt_x = inner_x + int(tgt_ratio * inner_w)
            p.setPen(QPen(QColor("#00E5FF"), 2))
            p.drawLine(tgt_x, y, tgt_x, y + h)

        # Draw scale tick marks on top of bar
        p.setFont(QFont("Menlo", 5))
        for sv in scale_pts:
            ratio = self._db_to_bar_ratio(sv, lo, hi)
            sx = inner_x + int(ratio * inner_w)
            p.setPen(QPen(QColor(255, 255, 255, 60), 1))
            p.drawLine(sx, y + h - 3, sx, y + h)


# ══════════════════════════════════════════════════
#  Gain Reduction History — Ozone 12 Style
# ══════════════════════════════════════════════════
class GainReductionHistoryWidget(QWidget):
    """
    V5.6 Ozone 12-style Gain Reduction display.

    Features:
    - Scrolling waveform-style GR history (newest on right)
    - Teal filled area showing GR depth over time
    - Current GR value in large bold text
    - Peak GR indicator with hold
    - Scale: 0 dB (top) to -20 dB (bottom)
    - Smooth animation with gradient fill
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(260, 100)

        self._current_gr = 0.0      # Current GR in positive dB
        self._peak_gr = 0.0         # Peak GR hold
        self._history = [0.0] * 120  # GR history (120 samples)
        self._max_gr = 20.0         # Scale max

    def set_gr(self, gr_db: float):
        """Set current gain reduction (positive value, e.g., 3.5 means -3.5 dB)."""
        self._current_gr = max(0.0, min(self._max_gr, abs(gr_db)))
        if self._current_gr > self._peak_gr:
            self._peak_gr = self._current_gr
        self._history.append(self._current_gr)
        if len(self._history) > 120:
            self._history.pop(0)
        self.update()

    def reset(self):
        self._current_gr = 0.0
        self._peak_gr = 0.0
        self._history = [0.0] * 120
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        # ── Background ──
        p.fillRect(0, 0, w, h, QColor("#0C0C0E"))
        p.setPen(QPen(QColor("#1A1A20"), 1))
        p.drawRect(1, 1, w - 3, h - 3)

        # Layout
        label_w = 50
        graph_x = label_w
        graph_w = w - label_w - 8
        graph_y = 4
        graph_h = h - 8

        # ── dB scale (left) ──
        p.setFont(QFont("Menlo", 7))
        scale_vals = [0, -3, -6, -10, -15, -20]
        for db in scale_vals:
            ratio = abs(db) / self._max_gr
            y = graph_y + int(ratio * graph_h)
            if y > graph_y + graph_h:
                continue
            p.setPen(QColor(C_CREAM_DIM))
            p.drawText(2, y + 3, f"{db:>3}")
            # Grid line (subtle)
            p.setPen(QPen(QColor("#1A1A20"), 1, Qt.PenStyle.DotLine))
            p.drawLine(graph_x, y, graph_x + graph_w, y)

        # ── GR waveform (filled area) ──
        num_points = len(self._history)
        if num_points > 1 and graph_w > 0:
            step_x = graph_w / (num_points - 1) if num_points > 1 else graph_w

            # Build path
            path = QPainterPath()
            path.moveTo(graph_x, graph_y)  # top-left (0 dB)

            for i, gr_val in enumerate(self._history):
                x = graph_x + i * step_x
                ratio = gr_val / self._max_gr
                y = graph_y + ratio * graph_h
                if i == 0:
                    path.moveTo(x, graph_y)
                    path.lineTo(x, y)
                else:
                    path.lineTo(x, y)

            # Close path back to top
            path.lineTo(graph_x + (num_points - 1) * step_x, graph_y)
            path.closeSubpath()

            # Fill with gradient
            grad = QLinearGradient(0, graph_y, 0, graph_y + graph_h)
            grad.setColorAt(0.0, QColor(0, 180, 216, 60))     # Teal transparent
            grad.setColorAt(0.3, QColor(0, 180, 216, 120))    # More opaque
            grad.setColorAt(1.0, QColor(0, 119, 182, 180))    # Deep teal
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(path)

            # Outline of waveform (teal glow)
            outline = QPainterPath()
            for i, gr_val in enumerate(self._history):
                x = graph_x + i * step_x
                ratio = gr_val / self._max_gr
                y = graph_y + ratio * graph_h
                if i == 0:
                    outline.moveTo(x, y)
                else:
                    outline.lineTo(x, y)
            p.setPen(QPen(QColor(C_TEAL_GLOW), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(outline)

        # ── Peak GR line (dashed) ──
        if self._peak_gr > 0.1:
            peak_ratio = self._peak_gr / self._max_gr
            peak_y = graph_y + peak_ratio * graph_h
            p.setPen(QPen(QColor(C_LED_RED), 1, Qt.PenStyle.DashLine))
            p.drawLine(graph_x, int(peak_y), graph_x + graph_w, int(peak_y))
            # Peak label
            p.setFont(QFont("Menlo", 7, QFont.Weight.Bold))
            p.setPen(QColor(C_LED_RED))
            p.drawText(graph_x + graph_w - 40, int(peak_y) - 2,
                       f"-{self._peak_gr:.1f}")

        # ── Current GR overlay (large text, bottom-right) ──
        p.setFont(QFont("Menlo", 20, QFont.Weight.Bold))
        if self._current_gr > 6.0:
            gr_color = QColor(C_LED_RED)
        elif self._current_gr > 3.0:
            gr_color = QColor(C_AMBER_GLOW)
        elif self._current_gr > 0.1:
            gr_color = QColor(C_TEAL_GLOW)
        else:
            gr_color = QColor(C_CREAM_DIM)
        gr_text = f"-{self._current_gr:.1f}" if self._current_gr > 0.05 else "0.0"
        p.setPen(gr_color)
        p.drawText(graph_x + graph_w - 80, graph_h - 8, f"{gr_text} dB")

        # ── Label ──
        p.setFont(QFont("Menlo", 7, QFont.Weight.Bold))
        p.setPen(QColor(C_TEAL))
        p.drawText(graph_x + 2, graph_y + 10, "GAIN REDUCTION")

        p.end()


# ══════════════════════════════════════════════════
#  Ozone 12 / Logic Pro Level Meter Widget
# ══════════════════════════════════════════════════
class OzoneLevelMeter(QWidget):
    """
    V5.5 SSL/Neve Vintage Dual L/R Level Meter.
    Warm amber VU-style segments with peak hold — inspired by SSL 4000/Neve 8816.
    Bigger, bolder, easier to read on dark backgrounds.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(240, 200)
        self.l_peak = -60.0
        self.r_peak = -60.0
        self.l_rms = -60.0
        self.r_rms = -60.0
        self.l_peak_hold = -60.0
        self.r_peak_hold = -60.0
        self._db_labels = [0, -3, -6, -12, -24, -48]

    def set_levels(self, l_peak=-60.0, r_peak=-60.0, l_rms=-60.0, r_rms=-60.0):
        self.l_rms = max(-60.0, min(3.0, l_rms))
        self.r_rms = max(-60.0, min(3.0, r_rms))
        self.l_peak = max(-60.0, min(3.0, l_peak))
        self.r_peak = max(-60.0, min(3.0, r_peak))
        if self.l_peak > self.l_peak_hold:
            self.l_peak_hold = self.l_peak
        if self.r_peak > self.r_peak_hold:
            self.r_peak_hold = self.r_peak
        self.update()

    def reset(self):
        self.l_peak = self.r_peak = -60.0
        self.l_rms = self.r_rms = -60.0
        self.l_peak_hold = self.r_peak_hold = -60.0
        self.update()

    def _db_to_ratio(self, db):
        db = max(-60.0, min(3.0, db))
        return (db + 60.0) / 63.0

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()

        # ── Vintage recessed panel background ──
        p.fillRect(0, 0, w, h, QColor("#0A0A0C"))
        # Inner bevel
        p.setPen(QPen(QColor("#1A1A1E"), 1))
        p.drawRect(1, 1, w - 3, h - 3)

        # ── Layout: dB scale | L bar | gap | R bar | dB scale ──
        bar_w = 32
        gap = 8
        scale_w = 28
        total_bars = 2 * bar_w + gap
        bar_start_x = (w - total_bars) // 2
        l_x = bar_start_x
        r_x = bar_start_x + bar_w + gap
        top_y = 16
        bar_h = h - 32

        # ── dB scale (left side) ──
        p.setFont(QFont("Menlo", 8))
        for db_val in self._db_labels:
            ratio = self._db_to_ratio(db_val)
            y = top_y + int(bar_h * (1.0 - ratio))
            # Label
            p.setPen(QColor(C_CREAM_DIM))
            p.drawText(l_x - scale_w - 2, y + 4, f"{db_val:>3}")
            # Tick mark
            p.setPen(QPen(QColor(C_RIDGE), 1))
            p.drawLine(l_x - 4, y, l_x - 1, y)
            # Right tick
            p.drawLine(r_x + bar_w + 1, y, r_x + bar_w + 4, y)

        # ── Draw bars ──
        self._draw_bar(p, l_x, top_y, bar_w, bar_h,
                        self.l_rms, self.l_peak, self.l_peak_hold)
        self._draw_bar(p, r_x, top_y, bar_w, bar_h,
                        self.r_rms, self.r_peak, self.r_peak_hold)

        # ── L / R labels (SSL style — bold gold) ──
        p.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
        p.setPen(QColor(C_GOLD))
        p.drawText(l_x + bar_w // 2 - 4, h - 2, "L")
        p.drawText(r_x + bar_w // 2 - 4, h - 2, "R")

        # ── Peak dB readout (above each bar) ──
        p.setFont(QFont("Menlo", 9, QFont.Weight.Bold))
        l_pk_txt = f"{self.l_peak:.1f}" if self.l_peak > -59.0 else "---"
        r_pk_txt = f"{self.r_peak:.1f}" if self.r_peak > -59.0 else "---"
        l_color = QColor(C_LED_RED) if self.l_peak > -1.0 else QColor(C_AMBER_GLOW)
        r_color = QColor(C_LED_RED) if self.r_peak > -1.0 else QColor(C_AMBER_GLOW)
        p.setPen(l_color)
        lm = p.fontMetrics().boundingRect(l_pk_txt)
        p.drawText(l_x + (bar_w - lm.width()) // 2, 12, l_pk_txt)
        p.setPen(r_color)
        rm = p.fontMetrics().boundingRect(r_pk_txt)
        p.drawText(r_x + (bar_w - rm.width()) // 2, 12, r_pk_txt)

        p.end()

    def _draw_bar(self, p, x, y_top, w, h, rms_db, peak_db, peak_hold_db):
        """Draw a single SSL-style segmented metering bar (warm amber palette)."""
        seg_h = 3
        seg_gap = 1
        seg_step = seg_h + seg_gap
        num_seg = h // seg_step

        # SSL/Neve warm palette: green → amber → orange → red
        col_green = QColor("#2ECC71")
        col_amber = QColor(C_AMBER_GLOW)
        col_orange = QColor("#F39C12")
        col_red = QColor("#E74C3C")

        col_green_dim = QColor("#0D3D1E")
        col_amber_dim = QColor("#332200")
        col_orange_dim = QColor("#3D2200")
        col_red_dim = QColor("#3D0A0A")

        # Bar background (recessed look)
        p.fillRect(x, y_top, w, h, QColor("#08080A"))

        for i in range(num_seg):
            seg_y = y_top + h - (i + 1) * seg_step
            seg_db = -60.0 + (i / max(1, num_seg - 1)) * 63.0

            if seg_db > -1.0:
                col_on, col_dim = col_red, col_red_dim
            elif seg_db > -6.0:
                col_on, col_dim = col_orange, col_orange_dim
            elif seg_db > -12.0:
                col_on, col_dim = col_amber, col_amber_dim
            else:
                col_on, col_dim = col_green, col_green_dim

            if seg_db <= rms_db:
                p.fillRect(x + 2, seg_y, w - 4, seg_h, col_on)
                # Glow effect
                glow = QColor(col_on)
                glow.setAlpha(30)
                p.fillRect(x + 1, seg_y - 1, w - 2, seg_h + 1, glow)
            elif seg_db <= peak_db:
                p.fillRect(x + 2, seg_y, w - 4, seg_h, col_dim)
            else:
                p.fillRect(x + 2, seg_y, w - 4, seg_h, QColor("#101012"))

        # Peak hold line (bright amber for SSL feel)
        if peak_hold_db > -59.0:
            ratio = self._db_to_ratio(peak_hold_db)
            y_hold = y_top + int(h * (1.0 - ratio))
            hold_color = QColor(C_LED_RED) if peak_hold_db > -1.0 else QColor(C_AMBER_GLOW)
            p.setPen(QPen(hold_color, 2))
            p.drawLine(x + 1, y_hold, x + w - 2, y_hold)


# ══════════════════════════════════════════════════
#  Logic Channel Strip Meters — BEFORE / AFTER
# ══════════════════════════════════════════════════

class LogicChannelMeter(QWidget):
    """
    V5.8 Logic Pro X Channel Strip Meters — BEFORE / AFTER (QPainter).

    Two pairs of tall vertical L/R bars side-by-side:
      BEFORE (input signal) | AFTER (post-maximizer output)

    - Gradient: green (< -12 dB), yellow (-12 to -3 dB), red (-3 to 0 dB)
    - Peak hold line (white, 2s hold, 20 dB/s decay)
    - Numeric peak at top, clip indicator (red dot) above ceiling
    - Scale markings: 0, -3, -6, -12, -24, -48 dB
    - AFTER meter NEVER exceeds Output Ceiling — proves limiter works
    """

    DB_MIN = -60.0
    DB_MAX = 3.0
    DB_RANGE = DB_MAX - DB_MIN  # 63 dB
    PEAK_HOLD_TIME_MS = 2000
    PEAK_DECAY_RATE = 20.0  # dB per second

    def __init__(self, ceiling_db: float = -1.0, parent=None):
        super().__init__(parent)
        self.setFixedSize(260, 260)
        self.ceiling_db = ceiling_db

        # BEFORE levels
        self.before_l_peak = -60.0
        self.before_r_peak = -60.0
        self.before_l_rms = -60.0
        self.before_r_rms = -60.0
        # AFTER levels
        self.after_l_peak = -60.0
        self.after_r_peak = -60.0
        self.after_l_rms = -60.0
        self.after_r_rms = -60.0

        # Peak hold state: [current_hold_db, timestamp_ms]
        import time as _time
        self._time = _time
        self._peak_holds = {
            'bl': [self.DB_MIN, 0.0], 'br': [self.DB_MIN, 0.0],
            'al': [self.DB_MIN, 0.0], 'ar': [self.DB_MIN, 0.0],
        }
        # Clip indicators (sticky until reset)
        self._clip = {'bl': False, 'br': False, 'al': False, 'ar': False}

        # dB scale ticks
        self._db_ticks = [0, -3, -6, -12, -24, -48]

        # Colors
        self._col_green = QColor("#22C55E")
        self._col_yellow = QColor("#FACC15")
        self._col_red = QColor("#EF4444")
        self._col_green_dim = QColor("#0B3D1E")
        self._col_yellow_dim = QColor("#3D3200")
        self._col_red_dim = QColor("#3D0A0A")
        self._col_bg = QColor("#08080A")
        self._col_peak_hold = QColor("#FFFFFF")
        self._col_clip = QColor("#FF1744")

    def set_ceiling(self, ceiling_db: float):
        self.ceiling_db = ceiling_db

    def set_before(self, l_peak=-60.0, r_peak=-60.0, l_rms=-60.0, r_rms=-60.0):
        self.before_l_peak = max(self.DB_MIN, min(self.DB_MAX, l_peak))
        self.before_r_peak = max(self.DB_MIN, min(self.DB_MAX, r_peak))
        self.before_l_rms = max(self.DB_MIN, min(self.DB_MAX, l_rms))
        self.before_r_rms = max(self.DB_MIN, min(self.DB_MAX, r_rms))
        self._update_hold('bl', self.before_l_peak)
        self._update_hold('br', self.before_r_peak)
        if self.before_l_peak > self.ceiling_db:
            self._clip['bl'] = True
        if self.before_r_peak > self.ceiling_db:
            self._clip['br'] = True
        self.update()

    def set_after(self, l_peak=-60.0, r_peak=-60.0, l_rms=-60.0, r_rms=-60.0):
        self.after_l_peak = max(self.DB_MIN, min(self.DB_MAX, l_peak))
        self.after_r_peak = max(self.DB_MIN, min(self.DB_MAX, r_peak))
        self.after_l_rms = max(self.DB_MIN, min(self.DB_MAX, l_rms))
        self.after_r_rms = max(self.DB_MIN, min(self.DB_MAX, r_rms))
        self._update_hold('al', self.after_l_peak)
        self._update_hold('ar', self.after_r_peak)
        if self.after_l_peak > self.ceiling_db:
            self._clip['al'] = True
        if self.after_r_peak > self.ceiling_db:
            self._clip['ar'] = True
        self.update()

    def reset(self):
        for k in self._peak_holds:
            self._peak_holds[k] = [self.DB_MIN, 0.0]
        for k in self._clip:
            self._clip[k] = False
        self.before_l_peak = self.before_r_peak = self.DB_MIN
        self.before_l_rms = self.before_r_rms = self.DB_MIN
        self.after_l_peak = self.after_r_peak = self.DB_MIN
        self.after_l_rms = self.after_r_rms = self.DB_MIN
        self.update()

    def _update_hold(self, key, peak_db):
        now = self._time.time() * 1000.0
        hold_db, hold_ts = self._peak_holds[key]
        if peak_db >= hold_db:
            self._peak_holds[key] = [peak_db, now]
        else:
            elapsed_s = (now - hold_ts) / 1000.0
            if elapsed_s > (self.PEAK_HOLD_TIME_MS / 1000.0):
                decay = self.PEAK_DECAY_RATE * (elapsed_s - self.PEAK_HOLD_TIME_MS / 1000.0)
                new_hold = hold_db - decay
                if new_hold < self.DB_MIN:
                    new_hold = self.DB_MIN
                self._peak_holds[key][0] = new_hold

    def _db_to_y(self, db, top_y, bar_h):
        db = max(self.DB_MIN, min(self.DB_MAX, db))
        ratio = (db - self.DB_MIN) / self.DB_RANGE
        return top_y + int(bar_h * (1.0 - ratio))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor("#0A0A0C"))
        p.setPen(QPen(QColor("#1A1A1E"), 1))
        p.drawRect(1, 1, w - 3, h - 3)

        # Layout constants
        bar_w = 18
        pair_gap = 4       # gap between L and R within a pair
        section_gap = 16   # gap between BEFORE and AFTER sections
        scale_w = 26
        label_h = 16       # bottom label height
        top_margin = 22    # space for peak readout + clip dot
        bottom_margin = label_h + 4

        pair_w = 2 * bar_w + pair_gap
        total_w = scale_w + pair_w + section_gap + pair_w + scale_w
        start_x = (w - total_w) // 2

        top_y = top_margin
        bar_h = h - top_margin - bottom_margin

        # Positions
        scale1_x = start_x
        before_l_x = scale1_x + scale_w
        before_r_x = before_l_x + bar_w + pair_gap
        after_l_x = before_r_x + bar_w + section_gap
        after_r_x = after_l_x + bar_w + pair_gap
        scale2_x = after_r_x + bar_w + 2

        # ── dB Scale (left) ──
        p.setFont(QFont("Menlo", 7))
        p.setPen(QColor(C_CREAM_DIM))
        for db_val in self._db_ticks:
            y = self._db_to_y(db_val, top_y, bar_h)
            p.drawText(scale1_x, y + 3, f"{db_val:>3}")
            # Tick marks
            p.setPen(QPen(QColor(C_RIDGE), 1))
            p.drawLine(before_l_x - 3, y, before_l_x - 1, y)
            p.setPen(QColor(C_CREAM_DIM))

        # ── dB Scale (right) ──
        for db_val in self._db_ticks:
            y = self._db_to_y(db_val, top_y, bar_h)
            p.drawText(scale2_x, y + 3, f"{db_val:>3}")
            p.setPen(QPen(QColor(C_RIDGE), 1))
            p.drawLine(after_r_x + bar_w + 1, y, after_r_x + bar_w + 3, y)
            p.setPen(QColor(C_CREAM_DIM))

        # ── Ceiling line (dashed, across AFTER section only) ──
        ceiling_y = self._db_to_y(self.ceiling_db, top_y, bar_h)
        pen_ceil = QPen(QColor(C_TEAL_GLOW), 1, Qt.PenStyle.DashLine)
        p.setPen(pen_ceil)
        p.drawLine(after_l_x - 2, ceiling_y, after_r_x + bar_w + 2, ceiling_y)
        p.setFont(QFont("Menlo", 7))
        p.setPen(QColor(C_TEAL_GLOW))
        p.drawText(after_r_x + bar_w + 5, ceiling_y + 3,
                   f"{self.ceiling_db:.1f}")

        # ── Draw meter bars ──
        self._draw_bar(p, before_l_x, top_y, bar_w, bar_h,
                       self.before_l_rms, self.before_l_peak, 'bl')
        self._draw_bar(p, before_r_x, top_y, bar_w, bar_h,
                       self.before_r_rms, self.before_r_peak, 'br')
        self._draw_bar(p, after_l_x, top_y, bar_w, bar_h,
                       self.after_l_rms, self.after_l_peak, 'al')
        self._draw_bar(p, after_r_x, top_y, bar_w, bar_h,
                       self.after_r_rms, self.after_r_peak, 'ar')

        # ── Peak numeric readout (top) ──
        p.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        before_max = max(self.before_l_peak, self.before_r_peak)
        after_max = max(self.after_l_peak, self.after_r_peak)
        b_txt = f"{before_max:.1f}" if before_max > -59 else "---"
        a_txt = f"{after_max:.1f}" if after_max > -59 else "---"
        b_col = QColor(self._col_red) if before_max > -1.0 else QColor(C_AMBER_GLOW)
        a_col = QColor(self._col_red) if after_max > self.ceiling_db else QColor(C_LED_GREEN)
        p.setPen(b_col)
        bm = p.fontMetrics().boundingRect(b_txt)
        bx_center = before_l_x + (pair_w - bm.width()) // 2
        p.drawText(bx_center, 12, b_txt)
        p.setPen(a_col)
        am = p.fontMetrics().boundingRect(a_txt)
        ax_center = after_l_x + (pair_w - am.width()) // 2
        p.drawText(ax_center, 12, a_txt)

        # ── Clip dots (red circle above bars if clipped) ──
        for key, cx in [('bl', before_l_x + bar_w // 2),
                        ('br', before_r_x + bar_w // 2),
                        ('al', after_l_x + bar_w // 2),
                        ('ar', after_r_x + bar_w // 2)]:
            if self._clip[key]:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(self._col_clip))
                p.drawEllipse(cx - 3, top_y - 8, 6, 6)

        # ── Bottom labels ──
        p.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        p.setPen(QColor(C_GOLD))
        b_lbl_x = before_l_x + (pair_w - p.fontMetrics().boundingRect("BEFORE").width()) // 2
        p.drawText(b_lbl_x, h - 4, "BEFORE")
        p.setPen(QColor(C_TEAL_GLOW))
        a_lbl_x = after_l_x + (pair_w - p.fontMetrics().boundingRect("AFTER").width()) // 2
        p.drawText(a_lbl_x, h - 4, "AFTER")

        # ── L/R sub-labels ──
        p.setFont(QFont("Menlo", 6))
        p.setPen(QColor(C_CREAM_DIM))
        p.drawText(before_l_x + bar_w // 2 - 2, h - 14, "L")
        p.drawText(before_r_x + bar_w // 2 - 2, h - 14, "R")
        p.drawText(after_l_x + bar_w // 2 - 2, h - 14, "L")
        p.drawText(after_r_x + bar_w // 2 - 2, h - 14, "R")

        p.end()

    def _draw_bar(self, p, x, y_top, bw, bh, rms_db, peak_db, hold_key):
        """Draw a single vertical metering bar with Logic Pro gradient."""
        seg_h = 2
        seg_gap = 1
        seg_step = seg_h + seg_gap
        num_seg = bh // seg_step

        # Bar background
        p.fillRect(x, y_top, bw, bh, self._col_bg)

        for i in range(num_seg):
            seg_y = y_top + bh - (i + 1) * seg_step
            seg_db = self.DB_MIN + (i / max(1, num_seg - 1)) * self.DB_RANGE

            # Color selection: green → yellow → red
            if seg_db > -3.0:
                col_on, col_dim = self._col_red, self._col_red_dim
            elif seg_db > -12.0:
                col_on, col_dim = self._col_yellow, self._col_yellow_dim
            else:
                col_on, col_dim = self._col_green, self._col_green_dim

            if seg_db <= rms_db:
                p.fillRect(x + 1, seg_y, bw - 2, seg_h, col_on)
            elif seg_db <= peak_db:
                p.fillRect(x + 1, seg_y, bw - 2, seg_h, col_dim)
            else:
                p.fillRect(x + 1, seg_y, bw - 2, seg_h, QColor("#101012"))

        # Peak hold line (white)
        hold_db = self._peak_holds[hold_key][0]
        if hold_db > self.DB_MIN + 1:
            hold_y = self._db_to_y(hold_db, y_top, bh)
            p.setPen(QPen(self._col_peak_hold, 2))
            p.drawLine(x + 1, hold_y, x + bw - 2, hold_y)


# ══════════════════════════════════════════════════
#  Loudness Meter Bars — Logic Pro X / Ozone 12 Style
# ══════════════════════════════════════════════════
class LoudnessMeterBars(QWidget):
    """
    V5.5 Logic Pro X / Ozone 12 style LUFS Meter.

    Features matching Logic Pro Loudness Meter:
    - Range: +6 LUFS (top) to -60 LUFS (bottom) — supports positive values!
    - 3 independent bars: M (Momentary), S (Short-term), I (Integrated)
    - Color gradient: red (>0) → orange (-5→0) → yellow (-10→-5) → green (-36→-10) → blue (<-36)
    - Large numeric readout above each bar
    - Peak hold per bar with 2s decay
    - Target line (dashed teal) at -14 LUFS
    - LU Range + Integrated readout below bars
    - dB scale on left: every 6 dB from +6 to -60
    """

    def __init__(self, parent=None, target_lufs=-14.0):
        super().__init__(parent)
        self.setFixedSize(240, 320)  # Tall like Logic Pro
        self._mom = -70.0
        self._short = -70.0
        self._int = -70.0
        self._target = target_lufs
        self._lu_range = 0.0

        # Range: +6 to -60 (like Logic Pro Loudness Meter)
        self._min_db = -60.0
        self._max_db = 6.0
        self._labels = ["M", "S", "I"]

        # Peak hold per bar (with 2s decay)
        self._peak_hold = [-70.0, -70.0, -70.0]
        self._peak_hold_time = [0.0, 0.0, 0.0]
        self._peak_hold_decay = 2.0  # seconds

    def set_levels(self, momentary=-70.0, short_term=-70.0, integrated=-70.0):
        self._mom = max(self._min_db, min(self._max_db, momentary))
        self._short = max(self._min_db, min(self._max_db, short_term))
        self._int = max(self._min_db, min(self._max_db, integrated))

        # Update peak hold
        import time as _t
        now = _t.time()
        vals = [self._mom, self._short, self._int]
        for i, v in enumerate(vals):
            if v > self._peak_hold[i]:
                self._peak_hold[i] = v
                self._peak_hold_time[i] = now
            elif now - self._peak_hold_time[i] > self._peak_hold_decay:
                # Decay peak hold
                self._peak_hold[i] = max(v, self._peak_hold[i] - 1.0)

        self.update()

    def set_target(self, target_lufs=-14.0):
        self._target = target_lufs
        self.update()

    def set_lu_range(self, lu_range=0.0):
        self._lu_range = lu_range
        self.update()

    def reset(self):
        self._mom = -70.0
        self._short = -70.0
        self._int = -70.0
        self._peak_hold = [-70.0, -70.0, -70.0]
        self._lu_range = 0.0
        self.update()

    def _db_to_ratio(self, db):
        """Map dB to 0.0-1.0 ratio within +6 to -60 range."""
        if db <= self._min_db:
            return 0.0
        if db >= self._max_db:
            return 1.0
        return (db - self._min_db) / (self._max_db - self._min_db)

    def _get_color(self, db):
        """Logic Pro X color gradient: blue→green→yellow→orange→red."""
        if db > 0.0:
            return QColor("#E53935")       # Red — clipping zone
        elif db > -5.0:
            return QColor("#F39C12")       # Orange — very loud
        elif db > -10.0:
            return QColor("#FDD835")       # Yellow — loud
        elif db > -18.0:
            return QColor("#66BB6A")       # Green — normal
        elif db > -36.0:
            return QColor("#26A69A")       # Teal-green — moderate
        else:
            return QColor("#42A5F5")       # Blue — quiet

    def _get_dim_color(self, db):
        """Dimmed version for unlit segments."""
        if db > 0.0:
            return QColor("#3D0A0A")
        elif db > -5.0:
            return QColor("#3D2200")
        elif db > -10.0:
            return QColor("#3D3300")
        elif db > -18.0:
            return QColor("#0D3D1E")
        elif db > -36.0:
            return QColor("#0A2D2D")
        else:
            return QColor("#0A1A3D")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()

        # ── Dark recessed panel background ──
        p.fillRect(0, 0, w, h, QColor("#0A0A0C"))
        p.setPen(QPen(QColor("#1A1A1E"), 1))
        p.drawRect(1, 1, w - 3, h - 3)

        # ── Title ──
        p.setFont(QFont("Menlo", 8, QFont.Weight.Bold))
        p.setPen(QColor(C_CREAM_DIM))
        p.drawText(0, 0, w, 16, Qt.AlignmentFlag.AlignCenter, "LUFS")

        values = [self._mom, self._short, self._int]
        labels = self._labels

        # ── Layout ──
        bar_w = 36
        gap = 14
        total_w = 3 * bar_w + 2 * gap
        start_x = (w - total_w) // 2 + 12  # shift right for scale
        num_readout_h = 20
        top_y = 36
        bar_h = h - 36 - num_readout_h - 52  # room for labels + LU readout
        bottom_y = top_y + bar_h

        # ── dB scale (left side — every 6 dB from +6 to -60) ──
        p.setFont(QFont("Menlo", 7))
        scale_ticks = [6, 3, 0, -3, -6, -12, -18, -24, -30, -36, -42, -48, -54, -60]
        for db_val in scale_ticks:
            ratio = self._db_to_ratio(db_val)
            y = top_y + int(bar_h * (1.0 - ratio))
            if y < top_y or y > bottom_y:
                continue
            # Label
            if db_val == int(self._target):
                p.setPen(QColor(C_TEAL_GLOW))
            elif db_val == 0:
                p.setPen(QColor(C_CREAM))
            else:
                p.setPen(QColor(C_CREAM_DIM))
            p.drawText(2, y + 3, f"{db_val:>3}")
            # Tick
            p.setPen(QPen(QColor(C_RIDGE), 1))
            p.drawLine(start_x - 4, y, start_x - 1, y)

        # ── Draw each bar ──
        num_segs = 50  # More segments for smoother appearance
        seg_h = max(1, bar_h // num_segs)

        for i, (val, label) in enumerate(zip(values, labels)):
            x = start_x + i * (bar_w + gap)

            # ── Numeric readout above bar ──
            p.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
            if val > -69.0:
                val_color = self._get_color(val)
                val_text = f"{val:.1f}"
            else:
                val_color = QColor(C_CREAM_DIM)
                val_text = "--.-"
            p.setPen(val_color)
            tm = p.fontMetrics().boundingRect(val_text)
            p.drawText(x + (bar_w - tm.width()) // 2, top_y - 4, val_text)

            # ── Bar background (recessed) ──
            p.fillRect(x, top_y, bar_w, bar_h, QColor("#08080A"))

            # ── Segmented filled bar ──
            ratio = self._db_to_ratio(val)
            filled_segs = int(ratio * num_segs)

            for s in range(num_segs):
                seg_y = top_y + bar_h - (s + 1) * seg_h
                if seg_y < top_y:
                    break
                seg_db = self._min_db + (s / num_segs) * (self._max_db - self._min_db)

                if s < filled_segs:
                    color = self._get_color(seg_db)
                    p.fillRect(x + 2, seg_y + 1, bar_w - 4, seg_h - 1, color)
                    # Subtle glow
                    glow = QColor(color)
                    glow.setAlpha(20)
                    p.fillRect(x + 1, seg_y, bar_w - 2, seg_h, glow)
                else:
                    p.fillRect(x + 2, seg_y + 1, bar_w - 4, seg_h - 1, QColor("#101012"))

            # ── Peak hold line ──
            pk = self._peak_hold[i]
            if pk > -69.0:
                pk_ratio = self._db_to_ratio(pk)
                pk_y = top_y + int(bar_h * (1.0 - pk_ratio))
                if top_y <= pk_y <= bottom_y:
                    pk_color = self._get_color(pk)
                    p.setPen(QPen(pk_color, 2))
                    p.drawLine(x + 1, pk_y, x + bar_w - 2, pk_y)

            # ── Label below bar ──
            p.setFont(QFont("Menlo", 9, QFont.Weight.Bold))
            p.setPen(QColor(C_GOLD))
            lm = p.fontMetrics().boundingRect(label)
            p.drawText(x + (bar_w - lm.width()) // 2, bottom_y + 14, label)

        # ── Target line (dashed teal, spans all bars) ──
        target_ratio = self._db_to_ratio(self._target)
        if 0.0 < target_ratio < 1.0:
            target_y = top_y + int(bar_h * (1.0 - target_ratio))
            pen = QPen(QColor(C_TEAL_GLOW), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(start_x - 2, target_y,
                       start_x + total_w + 2, target_y)
            # Target label (right side)
            p.setFont(QFont("Menlo", 7, QFont.Weight.Bold))
            p.setPen(QColor(C_TEAL_GLOW))
            p.drawText(start_x + total_w + 4, target_y + 3,
                       f"{self._target:.0f}")

        # ── LU Range + Integrated readout (below bars, like Logic Pro) ──
        readout_y = bottom_y + 22
        p.setFont(QFont("Menlo", 9))

        # LU Range
        p.setPen(QColor(C_CREAM_DIM))
        p.drawText(start_x - 8, readout_y, "LU Range")
        p.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        p.setPen(QColor(C_AMBER_GLOW))
        lu_text = f"{self._lu_range:.1f}" if self._lu_range > 0.01 else "---"
        p.drawText(start_x - 8, readout_y + 16, lu_text)

        # Integrated
        int_x = start_x + bar_w + gap + 10
        p.setFont(QFont("Menlo", 9))
        p.setPen(QColor(C_CREAM_DIM))
        p.drawText(int_x, readout_y, "Integrated")
        p.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        int_val = self._int
        if int_val > -69.0:
            p.setPen(QColor(C_TEAL_GLOW))
            int_text = f"{int_val:.1f}"
        else:
            p.setPen(QColor(C_CREAM_DIM))
            int_text = "---"
        p.drawText(int_x, readout_y + 16, int_text)

        p.end()


# ══════════════════════════════════════════════════
#  Background Worker Thread
# ══════════════════════════════════════════════════
class MasterWorker(QThread):
    """Background thread for mastering operations."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, chain: MasterChain, mode: str = "render",
                 genre: str = "", platform: str = "", intensity: int = 50):
        super().__init__()
        self.chain = chain
        self.mode = mode
        self.preview_start = 0
        self.preview_duration = 30
        # V5.0 FIX: Store analyze params to prevent AttributeError
        self._genre = genre
        self._platform = platform
        self._intensity = intensity

    def run(self):
        try:
            if self.mode == "preview":
                result = self.chain.preview(
                    start_sec=self.preview_start,
                    duration_sec=self.preview_duration,
                    callback=self._callback,
                )
                self.finished.emit(result) if result else self.error.emit("Preview failed")

            elif self.mode == "render":
                result = self.chain.render(callback=self._callback)
                self.finished.emit(result) if result else self.error.emit("Render failed")

            elif self.mode == "analyze":
                rec = self.chain.ai_recommend(
                    genre=self._genre,
                    platform=self._platform,
                    intensity=self._intensity,
                )
                self.finished.emit("analysis_done") if rec else self.error.emit("Analysis failed")

        except Exception as e:
            self.error.emit(str(e))

    def _callback(self, percent, status):
        self.progress.emit(percent, status)


# ══════════════════════════════════════════════════
#  V5.11.0: Stats for Nerds (YouTube-style loudness penalty)
# ══════════════════════════════════════════════════
class StatsForNerdsWidget(QFrame):
    """YouTube 'Stats for Nerds' style widget showing loudness penalty per platform.

    Shows:
    - Your LUFS vs Platform Target
    - Penalty (how much the platform will turn you down)
    - True Peak status
    - LRA (Loudness Range)
    - Color-coded status (green/yellow/red)
    """

    PLATFORMS = {
        "YouTube":      {"lufs": -14.0, "tp": -1.0},
        "Spotify":      {"lufs": -14.0, "tp": -1.0},
        "Apple Music":  {"lufs": -16.0, "tp": -1.0},
        "Tidal":        {"lufs": -14.0, "tp": -1.0},
        "Amazon":       {"lufs": -14.0, "tp": -2.0},
        "SoundCloud":   {"lufs": -14.0, "tp": -1.0},
        "CD":           {"lufs": -9.0,  "tp": -0.3},
        "Radio (EBU)":  {"lufs": -23.0, "tp": -1.0},
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(250)
        self.setStyleSheet(f"""
            QFrame {{
                background: #0C0E12;
                border: 1px solid #1E1E22;
                border-radius: 4px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)

        # Title
        title = QLabel("📊 STATS FOR NERDS")
        title.setStyleSheet(f"color:{C_TEAL}; font-size:8px; font-weight:bold; "
                           f"letter-spacing:2px; border:none; background:transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Platform selector
        plat_row = QHBoxLayout()
        plat_row.setSpacing(4)
        plat_lbl = QLabel("PLATFORM")
        plat_lbl.setStyleSheet(f"color:#6B6B70; font-size:7px; font-weight:bold; "
                              f"letter-spacing:1px; border:none; background:transparent;")
        plat_row.addWidget(plat_lbl)
        self.platform_combo = QComboBox()
        self.platform_combo.addItems(list(self.PLATFORMS.keys()))
        self.platform_combo.setCurrentText("YouTube")
        self.platform_combo.setStyleSheet(f"""
            QComboBox {{
                background: #141418; color: {C_TEAL_GLOW};
                border: 1px solid #2A2A30; border-radius: 2px;
                padding: 1px 4px; font-size: 8px; font-weight: bold;
                font-family: 'Menlo', monospace;
            }}
            QComboBox::drop-down {{ border: none; }}
        """)
        self.platform_combo.currentTextChanged.connect(lambda _: self._refresh())
        plat_row.addWidget(self.platform_combo, 1)
        layout.addLayout(plat_row)

        # Stats rows
        row_style = f"color:{C_CREAM_DIM}; font-size:8px; font-family:'Menlo',monospace; " \
                    f"border:none; background:transparent;"
        val_style = f"font-size:9px; font-weight:bold; font-family:'Menlo',monospace; " \
                    f"border:none; background:transparent;"

        def stat_row(label_text):
            row = QHBoxLayout()
            row.setSpacing(2)
            lbl = QLabel(label_text)
            lbl.setStyleSheet(row_style)
            lbl.setFixedWidth(100)
            row.addWidget(lbl)
            val = QLabel("—")
            val.setStyleSheet(val_style + f"color:{C_TEAL_GLOW};")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(val)
            layout.addLayout(row)
            return val

        self.val_your_lufs = stat_row("Your LUFS:")
        self.val_target = stat_row("Target:")
        self.val_penalty = stat_row("Penalty:")
        self.val_tp = stat_row("True Peak:")
        self.val_lra = stat_row("LRA:")
        self.val_crest = stat_row("Crest Factor:")

        # Status bar
        self.status_label = QLabel("⏸ ยังไม่มีข้อมูล")
        self.status_label.setStyleSheet(
            f"color:#6B6B70; font-size:8px; font-style:italic; "
            f"border:none; background:transparent; padding-top:2px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # State
        self._lufs_int = -70.0
        self._tp_left = -70.0
        self._tp_right = -70.0
        self._lra = 0.0
        self._lufs_mom = -70.0

    def update_stats(self, lufs_integrated=-70.0, tp_left=-70.0, tp_right=-70.0,
                     lra=0.0, lufs_momentary=-70.0):
        """Feed loudness data — called from meter update loop."""
        self._lufs_int = lufs_integrated
        self._tp_left = tp_left
        self._tp_right = tp_right
        self._lra = lra
        self._lufs_mom = lufs_momentary
        self._refresh()

    def _refresh(self):
        platform = self.platform_combo.currentText()
        target = self.PLATFORMS.get(platform, {"lufs": -14.0, "tp": -1.0})
        target_lufs = target["lufs"]
        target_tp = target["tp"]
        tp_max = max(self._tp_left, self._tp_right)

        # Your LUFS
        if self._lufs_int > -60:
            self.val_your_lufs.setText(f"{self._lufs_int:.1f} LUFS")
            self.val_your_lufs.setStyleSheet(
                f"color:{C_TEAL_GLOW}; font-size:9px; font-weight:bold; "
                f"font-family:'Menlo',monospace; border:none; background:transparent;")
        else:
            self.val_your_lufs.setText("—")

        # Target
        self.val_target.setText(f"{target_lufs:.1f} LUFS")

        # Penalty
        if self._lufs_int > -60:
            penalty = target_lufs - self._lufs_int
            if penalty >= 0:
                penalty_text = f"0.0 dB ✓"
                penalty_color = C_LED_GREEN
                status = f"🟢 ดีเยี่ยม! {platform} จะไม่กดเสียงลง"
            elif penalty > -3:
                penalty_text = f"{penalty:+.1f} dB"
                penalty_color = C_LED_YELLOW
                status = f"🟡 ดังไปนิด — {platform} จะกดลง {abs(penalty):.1f} dB"
            else:
                penalty_text = f"{penalty:+.1f} dB"
                penalty_color = C_LED_RED
                status = f"🔴 ดังเกิน! {platform} จะกดลง {abs(penalty):.1f} dB\nลอง Gain ลง หรือเปลี่ยน IRC mode"

            self.val_penalty.setText(penalty_text)
            self.val_penalty.setStyleSheet(
                f"color:{penalty_color}; font-size:10px; font-weight:bold; "
                f"font-family:'Menlo',monospace; border:none; background:transparent;")
            self.status_label.setText(status)
        else:
            self.val_penalty.setText("—")
            self.status_label.setText("⏸ กด Play เพื่อดูข้อมูล")

        # True Peak
        if tp_max > -60:
            tp_ok = tp_max <= target_tp
            tp_color = C_LED_GREEN if tp_ok else C_LED_RED
            self.val_tp.setText(f"{tp_max:.1f} dBTP {'✓' if tp_ok else '⚠'}")
            self.val_tp.setStyleSheet(
                f"color:{tp_color}; font-size:9px; font-weight:bold; "
                f"font-family:'Menlo',monospace; border:none; background:transparent;")
        else:
            self.val_tp.setText("—")

        # LRA
        if self._lra > 0:
            self.val_lra.setText(f"{self._lra:.1f} LU")
        else:
            self.val_lra.setText("—")

        # Crest Factor
        if self._lufs_int > -60 and tp_max > -60:
            crest = tp_max - self._lufs_int
            self.val_crest.setText(f"{crest:.1f} dB")
        else:
            self.val_crest.setText("—")


# ══════════════════════════════════════════════════
#  VU-Style Meters Panel (Right Side)
# ══════════════════════════════════════════════════
class MetersPanel(QFrame):
    """
    V5.4 UX Redesign: Dual-mode Live Metering Panel.

    Replaces the old static INPUT/OUTPUT panel with:
    - LIVE mode: Animated L/R bars + True Peak + LUFS + GR meter + Stage indicator
    - RESULT mode: Before/After comparison + TP accuracy + Gain delta

    Uses QStackedWidget to switch between modes.
    Backward compatible: update_input(), update_output(), set_status() still work.
    """

    MODE_LIVE = 0
    MODE_RESULT = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {C_GROOVE}, stop:0.02 {C_PANEL_INSET},
                    stop:0.98 {C_PANEL_INSET}, stop:1 {C_GROOVE});
                border-left: 1px solid {C_GROOVE};
            }}
        """)

        # Store Before/After data for RESULT mode
        self._input_data = {"lufs": None, "tp": None, "lra": None}
        self._output_data = {"lufs": None, "tp": None, "lra": None}
        self._target_tp = -1.0

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(6)

        # --- Mode header ---
        self.mode_header = QLabel("⚡ REAL-TIME METERING")
        self.mode_header.setStyleSheet(
            f"font-size:10px; color:{C_TEAL_GLOW}; letter-spacing:2px; font-weight:bold;")
        self.mode_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self.mode_header)

        # --- Stacked Widget: LIVE / RESULT ---
        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack, stretch=1)

        # === PAGE 0: LIVE MODE ===
        self._build_live_page()

        # === PAGE 1: RESULT MODE ===
        self._build_result_page()

        self.stack.setCurrentIndex(self.MODE_LIVE)

        # --- Target Indicator (always visible) ---
        root_layout.addWidget(self._groove_divider())
        self.target_frame = QFrame()
        self.target_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {C_PANEL_INSET}, stop:0.5 #1E1E20, stop:1 {C_PANEL_INSET});
                border: 1px solid {C_GROOVE};
                border-top: 1px solid {C_RIDGE};
                border-radius: 4px;
                padding: 6px;
            }}
        """)
        tf_layout = QVBoxLayout(self.target_frame)
        tf_layout.setContentsMargins(6, 4, 6, 4)
        tf_layout.setSpacing(2)
        target_lbl = QLabel("TARGET")
        target_lbl.setStyleSheet(f"font-size:8px; color:{C_GOLD}; letter-spacing:2px; background:transparent;")
        target_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf_layout.addWidget(target_lbl)
        self.target_val = QLabel("-14.0 LUFS")
        self.target_val.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        self.target_val.setStyleSheet(f"color: {C_TEAL_GLOW}; background:transparent;")
        self.target_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf_layout.addWidget(self.target_val)
        self.target_platform = QLabel("YouTube")
        self.target_platform.setStyleSheet(f"font-size:9px; color:{C_CREAM_DIM}; background:transparent;")
        self.target_platform.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf_layout.addWidget(self.target_platform)
        root_layout.addWidget(self.target_frame)

        # --- Status LED (always visible) ---
        self.status_row = self._info_row("STATUS", "READY")
        root_layout.addLayout(self.status_row[0])

    # ─── LIVE PAGE ────────────────────────────
    def _build_live_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── V5.8: LOGIC CHANNEL STRIP METERS (BEFORE / AFTER) ──
        self.live_logic_meter = LogicChannelMeter(ceiling_db=-1.0)
        layout.addWidget(self.live_logic_meter, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._groove_divider())

        # L/R Level Meter (reuse OzoneLevelMeter)
        self.live_ozone_meter = OzoneLevelMeter()
        layout.addWidget(self.live_ozone_meter, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._groove_divider())

        # ── V5.6: WAVES WLM-STYLE LOUDNESS METER ──
        self.live_wlm_meter = WavesWLMMeter(target_lufs=-14.0)
        layout.addWidget(self.live_wlm_meter, alignment=Qt.AlignmentFlag.AlignCenter)

        # ── V5.11.0: STATS FOR NERDS (YouTube-style loudness penalty info) ──
        self.stats_nerd = StatsForNerdsWidget()
        layout.addWidget(self.stats_nerd, alignment=Qt.AlignmentFlag.AlignCenter)

        # Backward-compat references
        self.live_tp = None       # Now inside WLM meter
        self.live_tp_led = None   # Now inside WLM meter
        self.live_lufs_bars = None  # Replaced by WLM meter
        self.live_lufs_mom = QLabel("--.-")
        self.live_lufs_mom.setVisible(False)
        self.live_lufs_short = QLabel("--.-")
        self.live_lufs_short.setVisible(False)
        self.live_lufs_int = QLabel("--.-")
        self.live_lufs_int.setVisible(False)

        layout.addWidget(self._groove_divider())

        # ── V5.6: GAIN REDUCTION HISTORY (Ozone 12 style) ──
        self.live_gr_history = GainReductionHistoryWidget()
        layout.addWidget(self.live_gr_history, alignment=Qt.AlignmentFlag.AlignCenter)

        # Backward-compat references
        self.live_gr_bar = None   # Replaced by history widget
        self.live_gr_val = None   # Replaced by history widget

        layout.addWidget(self._groove_divider())

        # ── V5.8 C-3: LOUDNESS HISTORY TIMELINE ──
        from modules.widgets.loudness_history import LoudnessHistoryWidget
        self.live_loudness_history = LoudnessHistoryWidget(target_lufs=-14.0)
        layout.addWidget(self.live_loudness_history, alignment=Qt.AlignmentFlag.AlignCenter)

        # ── V5.9: TONAL BALANCE CONTROL ──
        from modules.widgets.tonal_balance import TonalBalanceWidget
        self.live_tonal_balance = TonalBalanceWidget()
        self.live_tonal_balance.setFixedHeight(140)
        layout.addWidget(self.live_tonal_balance)

        layout.addWidget(self._groove_divider())

        # ── STAGE indicator (V5.5 — larger, more prominent) ──
        self.live_stage = QLabel("STAGE: IDLE")
        self.live_stage.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
        self.live_stage.setStyleSheet(
            f"color:{C_TEAL_GLOW}; letter-spacing:2px;")
        self.live_stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.live_stage)

        # ── V5.8 C-4: Gain Trim + TP Limiter + CSV Export ──
        c4_row = QHBoxLayout()
        c4_row.setSpacing(4)

        from modules.widgets.rotary_knob import OzoneRotaryKnob
        self.wlm_gain_trim = OzoneRotaryKnob(
            name="TRIM", min_val=-18.0, max_val=18.0, default=0.0,
            unit="dB", decimals=1)
        c4_row.addWidget(self.wlm_gain_trim)

        ctrl_col = QVBoxLayout()
        ctrl_col.setSpacing(2)
        self.wlm_tp_limiter = QCheckBox("TP LIMIT")
        self.wlm_tp_limiter.setChecked(True)
        self.wlm_tp_limiter.setStyleSheet(f"color:#48CAE4; font-size:8px; font-weight:bold;")
        ctrl_col.addWidget(self.wlm_tp_limiter)

        self.wlm_csv_btn = QPushButton("CSV")
        self.wlm_csv_btn.setFixedSize(40, 22)
        self.wlm_csv_btn.setStyleSheet(f"""
            QPushButton {{ background:#1A1A1E; border:1px solid #2A2A30; border-radius:3px;
                color:#8E8A82; font-size:8px; font-weight:bold; }}
            QPushButton:hover {{ color:#48CAE4; border-color:#0077B6; }}
        """)
        self.wlm_csv_btn.clicked.connect(self._export_loudness_csv)
        ctrl_col.addWidget(self.wlm_csv_btn)
        c4_row.addLayout(ctrl_col)
        layout.addLayout(c4_row)

        layout.addStretch()
        self.stack.addWidget(page)

    def _export_loudness_csv(self):
        """Export loudness history to CSV file."""
        if not hasattr(self, 'live_loudness_history'):
            return
        hist = self.live_loudness_history
        if len(hist._momentary) == 0:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Loudness CSV", "loudness_log.csv", "CSV Files (*.csv)")
        if not path:
            return
        import csv
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["sample", "momentary_lufs", "short_term_lufs", "integrated_lufs"])
            for i in range(len(hist._momentary)):
                m = hist._momentary[i] if i < len(hist._momentary) else -70.0
                s = hist._short_term[i] if i < len(hist._short_term) else -70.0
                it = hist._integrated[i] if i < len(hist._integrated) else -70.0
                writer.writerow([i, f"{m:.1f}", f"{s:.1f}", f"{it:.1f}"])

    # ─── RESULT PAGE ──────────────────────────
    def _build_result_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header
        result_hdr = QLabel("BEFORE  →  AFTER")
        result_hdr.setStyleSheet(f"font-size:10px; color:{C_GOLD}; letter-spacing:2px; font-weight:bold;")
        result_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(result_hdr)

        layout.addWidget(self._groove_divider())

        # LUFS comparison
        self._add_section_header(layout, "INTEGRATED LUFS")
        lufs_row = QHBoxLayout()
        self.result_lufs_in = QLabel("--.-")
        self.result_lufs_in.setFont(QFont("Menlo", 16, QFont.Weight.Bold))
        self.result_lufs_in.setStyleSheet(f"color: {C_CREAM_DIM};")
        self.result_lufs_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lufs_row.addWidget(self.result_lufs_in)
        arrow = QLabel("→")
        arrow.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        arrow.setStyleSheet(f"color: {C_GOLD};")
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lufs_row.addWidget(arrow)
        self.result_lufs_out = QLabel("--.-")
        self.result_lufs_out.setFont(QFont("Menlo", 16, QFont.Weight.Bold))
        self.result_lufs_out.setStyleSheet(f"color: {C_AMBER_GLOW};")
        self.result_lufs_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lufs_row.addWidget(self.result_lufs_out)
        layout.addLayout(lufs_row)

        layout.addWidget(self._groove_divider())

        # True Peak comparison
        self._add_section_header(layout, "TRUE PEAK")
        tp_row = QHBoxLayout()
        self.result_tp_in = QLabel("--.-")
        self.result_tp_in.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        self.result_tp_in.setStyleSheet(f"color: {C_CREAM_DIM};")
        self.result_tp_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tp_row.addWidget(self.result_tp_in)
        arrow2 = QLabel("→")
        arrow2.setFont(QFont("Menlo", 12, QFont.Weight.Bold))
        arrow2.setStyleSheet(f"color: {C_GOLD};")
        arrow2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tp_row.addWidget(arrow2)
        self.result_tp_out = QLabel("--.-")
        self.result_tp_out.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        self.result_tp_out.setStyleSheet(f"color: {C_LED_GREEN};")
        self.result_tp_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tp_row.addWidget(self.result_tp_out)
        layout.addLayout(tp_row)

        # TP Accuracy indicator
        self.result_tp_accuracy = QLabel("")
        self.result_tp_accuracy.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
        self.result_tp_accuracy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_tp_accuracy)

        layout.addWidget(self._groove_divider())

        # LRA + Gain delta
        self.lra_row = self._info_row("LRA", "-- LU")
        layout.addLayout(self.lra_row[0])
        self.short_term_row = self._info_row("SHORT", "-- LUFS")
        layout.addLayout(self.short_term_row[0])
        self.gain_row = self._info_row("GAIN \u0394", "--")
        layout.addLayout(self.gain_row[0])

        layout.addStretch()
        self.stack.addWidget(page)

    # ─── Public API (backward compatible) ─────

    def show_live_mode(self):
        """Switch to LIVE metering mode (during processing)."""
        self.stack.setCurrentIndex(self.MODE_LIVE)
        self.mode_header.setText("⚡ REAL-TIME METERING")
        self.mode_header.setStyleSheet(
            f"font-size:10px; color:{C_TEAL_GLOW}; letter-spacing:2px; font-weight:bold;")

    def show_result_mode(self):
        """Switch to RESULT mode (after processing complete)."""
        self.stack.setCurrentIndex(self.MODE_RESULT)
        self.mode_header.setText("✅ MASTERING RESULT")
        self.mode_header.setStyleSheet(
            f"font-size:10px; color:{C_LED_GREEN}; letter-spacing:2px; font-weight:bold;")
        # Populate Before/After
        if self._input_data["lufs"] is not None:
            self.result_lufs_in.setText(f"{self._input_data['lufs']:.1f}")
            self.result_tp_in.setText(f"{self._input_data['tp']:.1f}")
        if self._output_data["lufs"] is not None:
            self.result_lufs_out.setText(f"{self._output_data['lufs']:.1f}")
            self.result_tp_out.setText(f"{self._output_data['tp']:.1f}")
            self._set_result_tp_color(self._output_data['tp'])
            self._show_tp_accuracy(self._output_data['tp'])

    def update_live_levels(self, levels: dict):
        """Update live meter from chain meter callback data."""
        l_peak = levels.get("left_peak_db", -60.0)
        r_peak = levels.get("right_peak_db", -60.0)
        l_rms = levels.get("left_rms_db", -60.0)
        r_rms = levels.get("right_rms_db", -60.0)
        stage = levels.get("stage", "")

        # V5.8: Logic Channel Strip BEFORE/AFTER Meters
        if hasattr(self, 'live_logic_meter') and self.live_logic_meter is not None:
            if stage == "pre_chain":
                self.live_logic_meter.set_before(
                    l_peak=l_peak, r_peak=r_peak, l_rms=l_rms, r_rms=r_rms)
            elif stage in ("final", "post_loudnorm", "post_maximizer"):
                self.live_logic_meter.set_after(
                    l_peak=l_peak, r_peak=r_peak, l_rms=l_rms, r_rms=r_rms)

        # L/R Level Meter (SSL/Neve style)
        self.live_ozone_meter.set_levels(
            l_peak=l_peak, r_peak=r_peak, l_rms=l_rms, r_rms=r_rms)

        # V5.6: Waves WLM Meter — full loudness data
        lufs_m = levels.get("lufs_momentary", -70.0)
        lufs_s = levels.get("lufs_short_term", -70.0)
        lufs_i = levels.get("lufs_integrated", -70.0)
        lu_range = levels.get("lu_range", 0.0)

        if hasattr(self, 'live_wlm_meter') and self.live_wlm_meter is not None:
            self.live_wlm_meter.set_levels(
                momentary=lufs_m, short_term=lufs_s, integrated=lufs_i,
                lra=lu_range, tp_left=l_peak, tp_right=r_peak)

        # Backward-compat hidden labels
        self.live_lufs_mom.setText(f"{lufs_m:.1f}")
        self.live_lufs_short.setText(f"{lufs_s:.1f}")
        self.live_lufs_int.setText(f"{lufs_i:.1f}")

        # V5.6: GR History Widget
        gr = abs(levels.get("gain_reduction_db", 0.0))
        if hasattr(self, 'live_gr_history') and self.live_gr_history is not None:
            self.live_gr_history.set_gr(gr)

        # V5.8 C-3: Loudness History Timeline
        if hasattr(self, 'live_loudness_history') and self.live_loudness_history is not None:
            if stage not in ("pre_chain",):  # Only feed post-processing data
                self.live_loudness_history.append_levels(lufs_m, lufs_s, lufs_i)

        # Stage
        stage_names = {
            "pre_chain": "INPUT", "post_eq": "EQ", "post_dynamics": "DYNAMICS",
            "post_imager": "IMAGER", "post_maximizer": "MAXIMIZER",
            "post_loudnorm": "LOUDNESS", "final": "FINAL",
            "realtime": "REAL-TIME DSP", "realtime_lufs": "REAL-TIME DSP",
            "playback": "PLAYBACK",
        }
        if stage in stage_names:
            self.live_stage.setText(f"STAGE: ⚡ {stage_names[stage]}")

    def update_input(self, lufs, tp, lra):
        """Store input analysis data (backward compatible)."""
        self._input_data = {"lufs": lufs, "tp": tp, "lra": lra}
        self.lra_row[1].setText(f"{lra:.1f} LU")

    def update_output(self, lufs, tp, lra):
        """Store output analysis data and switch to RESULT mode (backward compatible)."""
        self._output_data = {"lufs": lufs, "tp": tp, "lra": lra}
        self.lra_row[1].setText(f"{lra:.1f} LU")
        # Calculate and show gain delta
        if self._input_data["lufs"] is not None:
            delta = lufs - self._input_data["lufs"]
            sign = "+" if delta >= 0 else ""
            self.gain_row[1].setText(f"{sign}{delta:.1f} dB")
        # Switch to RESULT mode
        self.show_result_mode()

    def update_target(self, lufs, tp, platform):
        self.target_val.setText(f"{lufs:.1f} LUFS")
        self.target_platform.setText(platform)
        self._target_tp = tp
        # V5.8: Sync ceiling to Logic Channel Meter
        if hasattr(self, 'live_logic_meter') and self.live_logic_meter is not None:
            self.live_logic_meter.set_ceiling(tp)

    def set_status(self, text, color=None):
        self.status_row[1].setText(text)
        if color:
            self.status_row[1].setStyleSheet(
                f"color:{color}; font-family:'Menlo',monospace; font-weight:bold;")

    # ─── Internal helpers ─────────────────────

    def _show_tp_accuracy(self, tp):
        """Show True Peak accuracy indicator."""
        error = abs(tp - self._target_tp)
        if error <= 0.3:
            self.result_tp_accuracy.setText(f"✅ TP: {tp:.1f} dBTP (error: {error:.1f} dB)")
            self.result_tp_accuracy.setStyleSheet(f"color: {C_LED_GREEN}; font-size: 9px;")
        elif error <= 1.0:
            self.result_tp_accuracy.setText(f"⚠️ TP: {tp:.1f} dBTP (error: {error:.1f} dB)")
            self.result_tp_accuracy.setStyleSheet(f"color: {C_LED_YELLOW}; font-size: 9px;")
        else:
            self.result_tp_accuracy.setText(f"❌ TP: {tp:.1f} dBTP (error: {error:.1f} dB)")
            self.result_tp_accuracy.setStyleSheet(f"color: {C_LED_RED}; font-size: 9px;")

    def _set_result_tp_color(self, tp):
        if tp > -1.0:
            self.result_tp_out.setStyleSheet(f"color: {C_LED_RED};")
        elif tp > -3.0:
            self.result_tp_out.setStyleSheet(f"color: {C_LED_YELLOW};")
        else:
            self.result_tp_out.setStyleSheet(f"color: {C_LED_GREEN};")

    def _add_section_header(self, layout, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"font-size:8px; color:{C_GOLD}; letter-spacing:2px; font-weight:bold;")
        layout.addWidget(lbl)

    def _groove_divider(self):
        line = QFrame()
        line.setFixedHeight(2)
        line.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {C_GROOVE}, stop:1 {C_RIDGE});
            }}
        """)
        return line

    def _info_row(self, label, value):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(f"font-size:10px; color:{C_GOLD}; letter-spacing:1px;")
        val = QLabel(value)
        val.setStyleSheet(
            f"font-family:'Menlo',monospace; font-weight:bold; font-size:11px; color:{C_AMBER_GLOW};")
        val.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(lbl)
        row.addWidget(val)
        return (row, val)


# ══════════════════════════════════════════════════
#  Preview Transport Bar — Logic Pro X-style
# ══════════════════════════════════════════════════
class PreviewTransportBar(QFrame):
    """Playback transport with play/pause, seek slider, time display, and bypass toggle."""

    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    seek_requested = pyqtSignal(int)   # position_ms
    bypass_toggled = pyqtSignal(bool)  # is_bypassed

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_playing = False
        self._is_bypassed = False
        self._duration_ms = 0
        self._slider_pressed = False

        self.setFixedHeight(52)
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {C_GROOVE}, stop:0.05 {C_PANEL_INSET},
                    stop:0.95 {C_PANEL_INSET}, stop:1 {C_GROOVE});
                border-top: 1px solid {C_RIDGE};
                border-bottom: 1px solid {C_GROOVE};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(10)

        # ── PLAY / PAUSE button ──
        self.btn_play = QPushButton("▶  PLAY")
        self.btn_play.setFixedSize(90, 36)
        self.btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_play.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #3A3A42, stop:1 #28282E);
                color: {C_AMBER_GLOW};
                font-family: 'Menlo', 'Courier New';
                font-size: 11px;
                font-weight: bold;
                border: 1px solid {C_RIDGE};
                border-radius: 4px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #4A4A52, stop:1 #38383E);
                color: {C_AMBER_GLOW};
            }}
            QPushButton:pressed {{
                background: {C_PANEL_INSET};
            }}
        """)
        self.btn_play.clicked.connect(self._on_play_clicked)
        layout.addWidget(self.btn_play)

        # ── STOP button ──
        self.btn_stop = QPushButton("■")
        self.btn_stop.setFixedSize(36, 36)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #3A3A42, stop:1 #28282E);
                color: {C_CREAM_DIM};
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {C_RIDGE};
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: #4A4A52; color: {C_CREAM}; }}
        """)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self.btn_stop)

        # ── Time current ──
        self.time_current = QLabel("0:00")
        self.time_current.setFixedWidth(42)
        self.time_current.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.time_current.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
        self.time_current.setStyleSheet(f"color: {C_AMBER_GLOW}; background: transparent;")
        layout.addWidget(self.time_current)

        # ── Seek Slider ──
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 10000)
        self.seek_slider.setValue(0)
        self.seek_slider.setFixedHeight(22)
        self.seek_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: {C_PANEL_INSET};
                border: 1px solid {C_GROOVE};
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {C_AMBER_DIM}, stop:1 {C_AMBER});
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: 14px;
                height: 14px;
                margin: -5px 0;
                background: qradialgradient(cx:0.5,cy:0.5,r:0.5,
                    stop:0 {C_AMBER_GLOW}, stop:0.7 {C_AMBER}, stop:1 {C_AMBER_DIM});
                border: 1px solid {C_GROOVE};
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: qradialgradient(cx:0.5,cy:0.5,r:0.5,
                    stop:0 #FFF, stop:0.5 {C_AMBER_GLOW}, stop:1 {C_AMBER});
            }}
        """)
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        layout.addWidget(self.seek_slider, stretch=1)

        # ── Time duration ──
        self.time_duration = QLabel("0:00")
        self.time_duration.setFixedWidth(42)
        self.time_duration.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.time_duration.setFont(QFont("Menlo", 10))
        self.time_duration.setStyleSheet(f"color: {C_CREAM_DIM}; background: transparent;")
        layout.addWidget(self.time_duration)

        # ── Separator ──
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(30)
        sep.setStyleSheet(f"background: {C_RIDGE};")
        layout.addWidget(sep)

        # ── BYPASS toggle ──
        self.btn_bypass = QPushButton("● MASTERED")
        self.btn_bypass.setFixedSize(120, 36)
        self.btn_bypass.setCheckable(True)
        self.btn_bypass.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_bypass_style(False)
        self.btn_bypass.toggled.connect(self._on_bypass_toggled)
        layout.addWidget(self.btn_bypass)

    def _on_play_clicked(self):
        if self._is_playing:
            self._is_playing = False
            self.btn_play.setText("▶  PLAY")
            self.pause_clicked.emit()
        else:
            self._is_playing = True
            self.btn_play.setText("⏸  PAUSE")
            self.play_clicked.emit()

    def _on_slider_pressed(self):
        self._slider_pressed = True

    def _on_slider_released(self):
        self._slider_pressed = False
        if self._duration_ms > 0:
            pos_ms = int(self.seek_slider.value() / 10000.0 * self._duration_ms)
            self.seek_requested.emit(pos_ms)

    def _on_slider_moved(self, value):
        if self._duration_ms > 0:
            pos_ms = int(value / 10000.0 * self._duration_ms)
            self.time_current.setText(self._format_time(pos_ms))

    def _on_bypass_toggled(self, checked):
        self._is_bypassed = checked
        self._update_bypass_style(checked)
        self.bypass_toggled.emit(checked)

    def _update_bypass_style(self, bypassed):
        if bypassed:
            self.btn_bypass.setText("● ORIGINAL")
            self.btn_bypass.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 #332200, stop:1 #221100);
                    color: {C_AMBER_GLOW};
                    font-family: 'Menlo', 'Courier New';
                    font-size: 10px;
                    font-weight: bold;
                    border: 2px solid {C_AMBER};
                    border-radius: 4px;
                    letter-spacing: 1px;
                }}
            """)
        else:
            self.btn_bypass.setText("● MASTERED")
            self.btn_bypass.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 #1A3A1A, stop:1 #102A10);
                    color: {C_LED_GREEN};
                    font-family: 'Menlo', 'Courier New';
                    font-size: 10px;
                    font-weight: bold;
                    border: 2px solid {C_LED_GREEN};
                    border-radius: 4px;
                    letter-spacing: 1px;
                }}
            """)

    def update_position(self, position_ms):
        """Update slider and time display from player position."""
        if not self._slider_pressed and self._duration_ms > 0:
            self.seek_slider.setValue(int(position_ms / self._duration_ms * 10000))
        self.time_current.setText(self._format_time(position_ms))

    def update_duration(self, duration_ms):
        """Set total duration."""
        self._duration_ms = duration_ms
        self.time_duration.setText(self._format_time(duration_ms))

    def set_stopped(self):
        """Reset to stopped state."""
        self._is_playing = False
        self.btn_play.setText("▶  PLAY")
        self.seek_slider.setValue(0)
        self.time_current.setText("0:00")

    def set_disabled_state(self):
        """V5.5: Show transport in disabled/dimmed state (no audio loaded yet)."""
        self.btn_play.setEnabled(False)
        self.btn_play.setText("▶  LOAD AUDIO TO PLAY")
        self.btn_play.setStyleSheet(f"""
            QPushButton {{
                background: {C_PANEL_INSET};
                color: {C_CREAM_DIM};
                font-family: 'Menlo', 'Courier New';
                font-size: 10px; font-weight: bold;
                border: 1px solid {C_GROOVE};
                border-radius: 4px; letter-spacing: 1px;
            }}
        """)
        self.btn_stop.setEnabled(False)
        self.seek_slider.setEnabled(False)
        self.btn_bypass.setEnabled(False)

    def set_enabled_state(self):
        """V5.5: Enable transport (audio loaded, ready for RT playback)."""
        self.btn_play.setEnabled(True)
        self.btn_play.setText("▶  PLAY")
        self.btn_play.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #3A3A42, stop:1 #28282E);
                color: {C_AMBER_GLOW};
                font-family: 'Menlo', 'Courier New';
                font-size: 11px; font-weight: bold;
                border: 1px solid {C_RIDGE};
                border-radius: 4px; letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #4A4A52, stop:1 #38383E);
                color: {C_AMBER_GLOW};
            }}
        """)
        self.btn_stop.setEnabled(True)
        self.seek_slider.setEnabled(True)
        self.btn_bypass.setEnabled(True)

    @staticmethod
    def _format_time(ms):
        total_sec = max(0, ms // 1000)
        m = total_sec // 60
        s = total_sec % 60
        return f"{m}:{s:02d}"


# ══════════════════════════════════════════════════
#  Main Master Panel — Neve/SSL Console
# ══════════════════════════════════════════════════
class MasterPanel(QWidget):
    """
    AI Master Panel — Neve/SSL Vintage Console design.
    Opens as separate window from LongPlay Studio main app.
    """

    master_complete = pyqtSignal(str)
    master_closed = pyqtSignal()          # V5.5: emitted when Master Module window closes
    # V5.5 REMOVED: position_changed signal (no RT engine → no position sync needed)
    request_audio_path = pyqtSignal()
    _call_on_main = pyqtSignal(object)  # thread-safe main-thread dispatch

    def __init__(self, parent=None, ffmpeg_path: str = "ffmpeg", shared_audio_player=None):
        super().__init__(parent)
        self.chain = MasterChain(ffmpeg_path)
        self.worker = None
        # V5.10: Shared audio player from main app (unified playback)
        self._shared_audio_player = shared_audio_player
        # V5.0 FIX: Initialize all instance variables to prevent AttributeError
        self._current_audio_path = None
        self._last_rec = None
        self._last_master_rec = None

        # V5.8 E-3: Undo/Redo
        from modules.master.undo import CommandHistory
        self._cmd_history = CommandHistory()

        # V5.8 E-4: AutoSave timer (60s)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(60000)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()

        # V5.4 FIX: Batch Processing State
        self.track_queue = []           # List of all track paths to process
        self.current_track_index = 0    # Current position in queue (0-based)
        self.batch_mode = False         # True when processing batch
        self.batch_output_dir = None    # User-selected output directory
        self._batch_results = []        # List of {path, success, lufs, tp, error}

        # V5.4 FIX: Realtime Meter System (thread-safe)
        self._meter_buffer = []           # List of recent meter data dicts
        self._meter_lock = threading.Lock()
        self._meter_max_buffer_size = 20
        self._meter_keep_count = 10

        # V5.10: Real-Time Engine (Rust + cpal) — lock-free DSP parameter control
        # When available, knob changes are heard instantly without re-rendering.
        self._rt_engine = None
        self._rt_playback_active = False
        if _HAS_RT_ENGINE:
            try:
                self._rt_engine = PyRtEngine()
                print("[MASTER] Real-time engine initialized (Rust + cpal)")
            except Exception as e:
                print(f"[MASTER] RT engine init failed: {e}")
                self._rt_engine = None

        # V5.2: Preview playback with bypass (dual-player for A/B)
        self._preview_path = None
        self._original_path = None
        self._is_preview_loaded = False
        self._is_bypass_mode = False
        self._preview_player = QMediaPlayer()
        self._bypass_player = QMediaPlayer()
        self._audio_output_master = QAudioOutput()
        self._audio_output_bypass = QAudioOutput()
        self._audio_output_master.setVolume(0.85)
        self._audio_output_bypass.setVolume(0.85)
        self._preview_player.setAudioOutput(self._audio_output_master)
        self._bypass_player.setAudioOutput(self._audio_output_bypass)
        self._preview_player.positionChanged.connect(self._on_player_position)
        self._preview_player.durationChanged.connect(self._on_player_duration)
        self._preview_player.mediaStatusChanged.connect(self._on_player_status)
        self._bypass_player.positionChanged.connect(self._on_player_position)
        self._bypass_player.durationChanged.connect(self._on_player_duration)
        self._bypass_player.mediaStatusChanged.connect(self._on_player_status)
        # V5.1 FIX: Safe main-thread dispatch — catch errors from deleted widgets
        def _safe_dispatch(fn):
            try:
                fn()
            except RuntimeError as e:
                # "wrapped C/C++ object has been deleted" — widget was closed
                print(f"[MASTER UI] Widget deleted during callback: {e}")
            except Exception as e:
                print(f"[MASTER UI] Main-thread dispatch error: {e}")
                import traceback
                traceback.print_exc()
        self._call_on_main.connect(_safe_dispatch)
        self.setStyleSheet(GLOBAL_STYLE)
        self._setup_ui()

        # V5.8: QTimer for realtime meter UI updates (30 Hz — smooth Logic meters)
        self._meter_update_timer = QTimer(self)
        self._meter_update_timer.setInterval(33)  # 33ms = ~30 Hz
        self._meter_update_timer.timeout.connect(self._update_realtime_meters)

        # Connect meter callback from chain (called from MasterWorker thread)
        self.chain.set_meter_callback(self._on_meter_data)

        # V5.5 REMOVED: RT position timer (no RT engine anymore)
        # Position tracking now via QMediaPlayer.positionChanged signal

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── TOP BAR ──
        root.addWidget(self._build_top_bar())

        # ── MODULE CHAIN ──
        root.addWidget(self._build_module_chain())

        # ── MAIN CONTENT (detail + meters) ──
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        # Module detail area (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.module_stack = QStackedWidget()
        self.module_stack.addWidget(self._build_resonance_view()) # 0 — RES (Soothe2)
        self.module_stack.addWidget(self._build_eq_view())        # 1
        self.module_stack.addWidget(self._build_dynamics_view())  # 2
        self.module_stack.addWidget(self._build_imager_view())    # 3
        self.module_stack.addWidget(self._build_maximizer_view()) # 4
        self.module_stack.addWidget(self._build_ai_view())        # 5
        self.module_stack.setCurrentIndex(5)  # Default to AI view
        scroll.setWidget(self.module_stack)
        content.addWidget(scroll, stretch=1)

        # Meters panel
        self.meters = MetersPanel()
        content.addWidget(self.meters)

        content_widget = QWidget()
        content_widget.setLayout(content)
        root.addWidget(content_widget, stretch=1)

        # ── TRANSPORT BAR removed (V5.5.1: Play/Stop moved to Timeline Preview) ──
        # Playback controls are now in the main Timeline transport bar.
        # Preview players kept internally for mastered audio preview if needed.

        # ── ACTION BAR ──
        root.addWidget(self._build_action_bar())

    # ─── TOP BAR ─────────────────────────────
    def _build_top_bar(self):
        bar = QFrame()
        bar.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {C_PANEL_LIGHT}, stop:0.1 {C_PANEL},
                    stop:0.9 {C_PANEL_INSET}, stop:1 {C_GROOVE});
                border-bottom: 1px solid {C_GROOVE};
                padding: 6px 16px;
            }}
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(10)

        # ← BACK TO EDITOR button
        btn_back = QPushButton("← BACK TO EDITOR")
        btn_back.setFixedHeight(32)
        btn_back.setStyleSheet("""
            QPushButton {
                background: #FF8844;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 14px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #FF9955;
            }
        """)
        btn_back.clicked.connect(self.close)
        layout.addWidget(btn_back)

        # Platform
        layout.addWidget(self._panel_label("PLATFORM"))
        self.platform_combo = QComboBox()
        for name, data in PLATFORM_TARGETS.items():
            self.platform_combo.addItem(f"{name} ({data['target_lufs']} LUFS)")
        self.platform_combo.setCurrentIndex(0)
        self.platform_combo.currentIndexChanged.connect(self._on_platform_changed)
        layout.addWidget(self.platform_combo)

        # Genre
        layout.addWidget(self._panel_label("GENRE"))
        self.genre_combo = QComboBox()
        genres = get_genre_list()
        for category, genre_names in sorted(genres.items()):
            for name in sorted(genre_names):
                self.genre_combo.addItem(f"{name}", userData=name)
        layout.addWidget(self.genre_combo)

        # Mastering Preset (one-click: Dynamics + Imager + Maximizer)
        layout.addWidget(self._panel_label("PRESET"))
        self.mastering_preset_combo = QComboBox()
        self.mastering_preset_combo.addItem("— Custom —")
        for cat, names in MASTERING_PRESET_CATEGORIES.items():
            for name in names:
                desc = MASTERING_PRESETS[name]["description"]
                self.mastering_preset_combo.addItem(f"{name}", userData=name)
        self.mastering_preset_combo.setToolTip("One-click mastering preset (Dynamics + Imager + Maximizer)")
        self.mastering_preset_combo.currentIndexChanged.connect(self._on_mastering_preset_changed)
        layout.addWidget(self.mastering_preset_combo)

        # File indicator
        layout.addWidget(self._panel_label("FILE"))
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet(f"color: {C_AMBER}; font-size: 11px; font-weight: bold;")
        layout.addWidget(self.file_label)

        # Browse button — allows user to select audio manually
        self.btn_browse = QPushButton("BROWSE")
        self.btn_browse.setFixedWidth(70)
        self.btn_browse.setStyleSheet(f"""
            QPushButton {{
                background: {C_PANEL_LIGHT};
                color: {C_AMBER};
                border: 1px solid {C_GROOVE};
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {C_GROOVE};
            }}
        """)
        self.btn_browse.clicked.connect(self._on_browse_audio)
        layout.addWidget(self.btn_browse)

        layout.addStretch()

        # Intensity
        layout.addWidget(self._panel_label("DRIVE"))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(65)
        self.intensity_slider.setFixedWidth(120)
        self.intensity_slider.valueChanged.connect(self._on_intensity_changed)
        layout.addWidget(self.intensity_slider)

        self.intensity_value = QLabel("65%")
        self.intensity_value.setFont(QFont("Menlo", 13, QFont.Weight.Bold))
        self.intensity_value.setStyleSheet(f"color: {C_AMBER_GLOW};")
        self.intensity_value.setFixedWidth(40)
        layout.addWidget(self.intensity_value)

        # Reset
        btn_reset = QPushButton("RESET")
        btn_reset.setStyleSheet(BTN_GHOST_STYLE)
        btn_reset.clicked.connect(self._on_reset_all)
        layout.addWidget(btn_reset)

        # ✕ Close button
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(32, 32)
        btn_close.setStyleSheet("""
            QPushButton {
                background: #EF4444;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #F87171;
            }
        """)
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

        return bar

    # ─── MODULE CHAIN BAR ────────────────────
    def _build_module_chain(self):
        bar = QFrame()
        bar.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {C_PANEL}, stop:0.5 {C_PANEL_INSET}, stop:1 {C_GROOVE});
                border-bottom: 2px solid {C_GROOVE};
            }}
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.chain_nodes = []
        modules = [
            ("RES", "SOOTHE", "resonance", C_LED_RED),
            ("EQ", "EQUALIZER", "eq", C_MOD_EQ),
            ("DYN", "DYNAMICS", "dynamics", C_MOD_DYN),
            ("IMG", "IMAGER", "imager", C_MOD_IMG),
            ("MAX", "MAXIMIZER", "maximizer", C_MOD_MAX),
            ("AI", "ASSIST", "ai", C_MOD_AI),
        ]

        for i, (abbr, name, mid, color) in enumerate(modules):
            if i > 0:
                arrow = QLabel("\u25B6")
                arrow.setStyleSheet(f"color: {C_AMBER_DIM}; font-size: 10px;")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(arrow)

            node = ModuleChainNode(abbr, name, mid, color=color)
            node.clicked.connect(lambda checked, idx=i: self._switch_module(idx))
            self.chain_nodes.append(node)
            layout.addWidget(node)

        # Set AI (index 5) as active by default
        self.chain_nodes[5].set_active(True)

        return bar

    # ─── RESONANCE SUPPRESSOR VIEW (Soothe2-style) ──────────────────
    def _build_resonance_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Title + Enable
        title_row = QHBoxLayout()
        title_row.addWidget(self._module_title("RESONANCE SUPPRESSOR"))
        self.res_enabled = QCheckBox("ACTIVE")
        self.res_enabled.setChecked(True)
        self.res_enabled.setStyleSheet(f"color:{C_LED_RED}; font-weight:bold;")
        self.res_enabled.toggled.connect(self._on_res_enabled)
        title_row.addWidget(self.res_enabled)
        layout.addLayout(title_row)

        # Description
        desc = QLabel("Auto-detects and suppresses harsh resonances (Soothe2-style spectral dynamic EQ)")
        desc.setStyleSheet(f"color:{C_AMBER_DIM}; font-size:9px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Mode selector (Soft / Hard)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self._panel_label("MODE"))
        self.res_mode_combo = QComboBox()
        self.res_mode_combo.addItems(["soft", "hard"])
        self.res_mode_combo.setStyleSheet(f"""
            QComboBox {{
                background: {C_PANEL_INSET}; color: {C_CREAM};
                border: 1px solid {C_GROOVE}; border-radius: 3px;
                padding: 2px 8px; font-size: 10px; font-weight: bold;
            }}
        """)
        self.res_mode_combo.currentTextChanged.connect(self._on_res_mode_changed)
        mode_row.addWidget(self.res_mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ── KNOB CONTROLS ──
        knob_style = f"""
            QDial {{
                background: {C_PANEL_INSET};
            }}
        """
        value_style = f"color:{C_TEAL_GLOW}; font-family:'Menlo'; font-size:10px; font-weight:bold;"
        label_style = f"color:{C_GOLD}; font-size:8px; font-weight:bold; letter-spacing:1px;"

        def make_knob_col(name, min_val, max_val, default, decimals, suffix, callback):
            col = QVBoxLayout()
            col.setSpacing(2)
            col.setAlignment(Qt.AlignmentFlag.AlignCenter)

            lbl = QLabel(name)
            lbl.setStyleSheet(label_style)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(lbl)

            dial = QDial()
            dial.setMinimum(int(min_val * (10 ** decimals)))
            dial.setMaximum(int(max_val * (10 ** decimals)))
            dial.setValue(int(default * (10 ** decimals)))
            dial.setFixedSize(48, 48)
            dial.setStyleSheet(knob_style)
            dial.setWrapping(False)
            dial.setNotchesVisible(True)
            col.addWidget(dial, alignment=Qt.AlignmentFlag.AlignCenter)

            val_label = QLabel(f"{default:.{decimals}f}{suffix}")
            val_label.setStyleSheet(value_style)
            val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(val_label)

            def on_change(v):
                real_val = v / (10 ** decimals)
                val_label.setText(f"{real_val:.{decimals}f}{suffix}")
                callback(real_val)
            dial.valueChanged.connect(on_change)

            return col, dial, val_label

        # Row 1: DEPTH, SHARPNESS, SELECTIVITY
        row1 = QHBoxLayout()
        row1.setSpacing(16)
        depth_col, self.res_depth_dial, self.res_depth_val = make_knob_col(
            "DEPTH", 0, 20, 5.0, 1, " dB", self._on_res_depth)
        row1.addLayout(depth_col)

        sharp_col, self.res_sharp_dial, self.res_sharp_val = make_knob_col(
            "SHARPNESS", 1, 10, 4.0, 1, "", self._on_res_sharpness)
        row1.addLayout(sharp_col)

        sel_col, self.res_sel_dial, self.res_sel_val = make_knob_col(
            "SELECTIVITY", 1, 10, 3.5, 1, "", self._on_res_selectivity)
        row1.addLayout(sel_col)
        layout.addLayout(row1)

        # Row 2: ATTACK, RELEASE, MIX, TRIM
        row2 = QHBoxLayout()
        row2.setSpacing(16)
        att_col, self.res_att_dial, self.res_att_val = make_knob_col(
            "ATTACK", 0.5, 50, 5.0, 1, " ms", self._on_res_attack)
        row2.addLayout(att_col)

        rel_col, self.res_rel_dial, self.res_rel_val = make_knob_col(
            "RELEASE", 5, 500, 50.0, 0, " ms", self._on_res_release)
        row2.addLayout(rel_col)

        mix_col, self.res_mix_dial, self.res_mix_val = make_knob_col(
            "MIX", 0, 100, 100, 0, " %", self._on_res_mix)
        row2.addLayout(mix_col)

        trim_col, self.res_trim_dial, self.res_trim_val = make_knob_col(
            "TRIM", -12, 12, 0.0, 1, " dB", self._on_res_trim)
        row2.addLayout(trim_col)
        layout.addLayout(row2)

        # ── DELTA + BYPASS ──
        option_row = QHBoxLayout()
        self.res_delta_cb = QCheckBox("DELTA (listen to removed signal)")
        self.res_delta_cb.setStyleSheet(f"color:{C_LED_YELLOW}; font-size:9px; font-weight:bold;")
        self.res_delta_cb.toggled.connect(self._on_res_delta)
        option_row.addWidget(self.res_delta_cb)
        option_row.addStretch()
        layout.addLayout(option_row)

        # ── REDUCTION DISPLAY ──
        self.res_reduction_label = QLabel("Reduction: — dB")
        self.res_reduction_label.setStyleSheet(
            f"color:{C_LED_RED}; font-family:'Menlo'; font-size:11px; font-weight:bold;")
        layout.addWidget(self.res_reduction_label)

        # V5.10.6: Spectrum Analyzer (Ozone 12 style — live FFT behind controls)
        from modules.widgets.spectrum_analyzer import SpectrumAnalyzerWidget
        self.res_spectrum = SpectrumAnalyzerWidget()
        self.res_spectrum.setFixedHeight(160)
        self.res_spectrum.setMinimumWidth(380)
        layout.addWidget(self.res_spectrum)

        layout.addStretch()
        return widget

    # ── Resonance Suppressor Handlers (all functional) ──

    def _on_res_enabled(self, checked):
        self.chain.resonance_suppressor.enabled = checked
        if self._rt_engine is not None and _HAS_RT_ENGINE:
            self._rt_engine.set_res_bypass(not checked)

    def _on_res_mode_changed(self, mode):
        self.chain.resonance_suppressor.set_mode(mode)
        if self._rt_engine is not None and _HAS_RT_ENGINE:
            self._rt_engine.set_res_mode(mode)

    def _on_res_depth(self, val):
        self.chain.resonance_suppressor.set_depth(val)
        if self._rt_engine is not None and _HAS_RT_ENGINE:
            self._rt_engine.set_res_depth(val)

    def _on_res_sharpness(self, val):
        self.chain.resonance_suppressor.set_sharpness(val)
        if self._rt_engine is not None and _HAS_RT_ENGINE:
            self._rt_engine.set_res_sharpness(val)

    def _on_res_selectivity(self, val):
        self.chain.resonance_suppressor.set_selectivity(val)
        if self._rt_engine is not None and _HAS_RT_ENGINE:
            self._rt_engine.set_res_selectivity(val)

    def _on_res_attack(self, val):
        self.chain.resonance_suppressor.speed_attack = val
        if self._rt_engine is not None and _HAS_RT_ENGINE:
            self._rt_engine.set_res_attack(val)

    def _on_res_release(self, val):
        self.chain.resonance_suppressor.speed_release = val
        if self._rt_engine is not None and _HAS_RT_ENGINE:
            self._rt_engine.set_res_release(val)

    def _on_res_mix(self, val):
        self.chain.resonance_suppressor.mix = val / 100.0
        if self._rt_engine is not None and _HAS_RT_ENGINE:
            self._rt_engine.set_res_mix(val)

    def _on_res_trim(self, val):
        self.chain.resonance_suppressor.set_trim(val)
        if self._rt_engine is not None and _HAS_RT_ENGINE:
            self._rt_engine.set_res_trim(val)

    def _on_res_delta(self, checked):
        self.chain.resonance_suppressor.set_delta(checked)
        if self._rt_engine is not None and _HAS_RT_ENGINE:
            self._rt_engine.set_res_delta(checked)

    # ─── EQ VIEW ──────────────────────────────
    def _build_eq_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Title + Enable
        title_row = QHBoxLayout()
        title_row.addWidget(self._module_title("PARAMETRIC EQUALIZER"))
        self.eq_enabled = QCheckBox("ACTIVE")
        self.eq_enabled.setChecked(True)
        self.eq_enabled.toggled.connect(lambda v: setattr(self.chain.equalizer, 'enabled', v))
        title_row.addWidget(self.eq_enabled)
        layout.addLayout(title_row)

        # Preset
        preset_row = QHBoxLayout()
        preset_row.addWidget(self._panel_label("TONE PRESET"))
        self.eq_preset_combo = QComboBox()
        self.eq_preset_combo.addItems(list(EQ_TONE_PRESETS.keys()))
        self.eq_preset_combo.currentTextChanged.connect(self._on_eq_preset_changed)
        preset_row.addWidget(self.eq_preset_combo)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        # V5.8 A-6: Analog/Digital mode + Spectrum overlay
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self.eq_analog_mode = QCheckBox("ANALOG")
        self.eq_analog_mode.setStyleSheet(f"color:{C_AMBER}; font-size:9px; font-weight:bold;")
        self.eq_analog_mode.setToolTip("Analog mode adds subtle harmonic coloring")
        self.eq_analog_mode.toggled.connect(
            lambda v: setattr(self.chain.equalizer, 'analog_mode', v))
        mode_row.addWidget(self.eq_analog_mode)

        self.eq_spectrum_overlay = QCheckBox("SPECTRUM")
        self.eq_spectrum_overlay.setChecked(True)
        self.eq_spectrum_overlay.setStyleSheet(f"color:{C_TEAL}; font-size:9px; font-weight:bold;")
        self.eq_spectrum_overlay.setToolTip("Show live FFT spectrum behind EQ curve")
        mode_row.addWidget(self.eq_spectrum_overlay)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ═══ Visual EQ Curve Display (iZotope Ozone style) ═══
        self.eq_curve = EQCurveWidget()
        self.eq_curve.setMinimumHeight(200)
        self.eq_curve.bandChanged.connect(self._on_eq_curve_band_changed)
        layout.addWidget(self.eq_curve)

        # V5.8 A-6: Spectrum Analyzer (overlaid on EQ curve area)
        from modules.widgets.spectrum_analyzer import SpectrumAnalyzerWidget
        self.eq_spectrum = SpectrumAnalyzerWidget()
        self.eq_spectrum.setFixedSize(self.eq_curve.width() or 400, 180)
        layout.addWidget(self.eq_spectrum)

        # Band sliders (below the curve)
        bands_group = QGroupBox("EQ BANDS")
        bands_layout = QGridLayout()
        bands_layout.setSpacing(6)

        self.eq_band_sliders = []
        freq_labels = ["32", "64", "125", "250", "1K", "4K", "8K", "16K"]

        for i in range(8):
            # Freq label
            flbl = QLabel(freq_labels[i])
            flbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            flbl.setStyleSheet(f"font-size:10px; color:{C_GOLD}; letter-spacing:1px;")
            bands_layout.addWidget(flbl, 0, i, Qt.AlignmentFlag.AlignCenter)

            # Vertical slider
            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(-120, 120)
            slider.setValue(0)
            slider.setFixedHeight(100)
            slider.valueChanged.connect(lambda v, idx=i: self._on_eq_band_changed(idx, v / 10.0))
            bands_layout.addWidget(slider, 1, i, Qt.AlignmentFlag.AlignCenter)

            # Value label
            val_label = QLabel("0.0")
            val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_label.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
            val_label.setStyleSheet(f"color: {C_AMBER};")
            bands_layout.addWidget(val_label, 2, i, Qt.AlignmentFlag.AlignCenter)

            self.eq_band_sliders.append((slider, val_label))

        # dB labels
        plus_lbl = QLabel("+12")
        plus_lbl.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:9px;")
        bands_layout.addWidget(plus_lbl, 1, 8, Qt.AlignmentFlag.AlignTop)
        zero_lbl = QLabel("  0")
        zero_lbl.setStyleSheet(f"color:{C_CREAM_DARK}; font-size:9px;")
        bands_layout.addWidget(zero_lbl, 1, 8, Qt.AlignmentFlag.AlignVCenter)
        minus_lbl = QLabel("-12")
        minus_lbl.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:9px;")
        bands_layout.addWidget(minus_lbl, 1, 8, Qt.AlignmentFlag.AlignBottom)

        bands_group.setLayout(bands_layout)
        layout.addWidget(bands_group)

        # V5.8 E-2: Match EQ section
        match_group = QGroupBox("MATCH EQ")
        match_layout = QHBoxLayout()
        match_layout.setSpacing(6)

        self.match_ref_btn = QPushButton("LOAD REFERENCE")
        self.match_ref_btn.setFixedHeight(26)
        self.match_ref_btn.setStyleSheet(f"""
            QPushButton {{ background:#1A1A1E; border:1px solid {C_TEAL_DIM}; border-radius:3px;
                color:{C_TEAL}; font-weight:bold; font-size:9px; padding:3px 8px; }}
            QPushButton:hover {{ background:{C_TEAL_DIM}; color:#FFF; }}
        """)
        self.match_ref_btn.clicked.connect(self._on_match_eq_load)
        match_layout.addWidget(self.match_ref_btn)

        self.match_ref_label = QLabel("No reference")
        self.match_ref_label.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:9px;")
        match_layout.addWidget(self.match_ref_label)

        self.match_strength = QSlider(Qt.Orientation.Horizontal)
        self.match_strength.setRange(0, 100)
        self.match_strength.setValue(50)
        self.match_strength.setFixedWidth(80)
        match_layout.addWidget(self.match_strength)

        self.match_strength_val = QLabel("50%")
        self.match_strength_val.setStyleSheet(f"color:{C_TEAL_GLOW}; font-family:'Menlo'; font-size:9px;")
        self.match_strength.valueChanged.connect(
            lambda v: self.match_strength_val.setText(f"{v}%"))
        match_layout.addWidget(self.match_strength_val)

        self.match_apply_btn = QPushButton("APPLY")
        self.match_apply_btn.setFixedHeight(26)
        self.match_apply_btn.setEnabled(False)
        self.match_apply_btn.setStyleSheet(f"""
            QPushButton {{ background:#1A1A1E; border:1px solid {C_TEAL_DIM}; border-radius:3px;
                color:{C_TEAL}; font-weight:bold; font-size:9px; padding:3px 8px; }}
            QPushButton:hover {{ background:{C_TEAL_DIM}; color:#FFF; }}
            QPushButton:disabled {{ color:#4A4844; border-color:#2A2A30; }}
        """)
        self.match_apply_btn.clicked.connect(self._on_match_eq_apply)
        match_layout.addWidget(self.match_apply_btn)

        match_group.setLayout(match_layout)
        layout.addWidget(match_group)

        return widget

    def _on_match_eq_load(self):
        """Load reference WAV for Match EQ."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference Track", "", "Audio Files (*.wav *.flac *.mp3 *.aif *.aiff)")
        if path:
            import os
            self._match_ref_path = path
            self.match_ref_label.setText(os.path.basename(path))
            self.match_apply_btn.setEnabled(True)

    def _on_match_eq_apply(self):
        """Apply Match EQ correction from reference track."""
        ref_path = getattr(self, '_match_ref_path', None)
        src_path = self.chain.input_path
        if not ref_path or not src_path:
            return

        try:
            import soundfile as sf
            import numpy as np

            # Load only first 4096 frames for spectral comparison (avoid loading entire files)
            ref_data, ref_sr = sf.read(ref_path, dtype='float64', frames=4096)
            src_data, src_sr = sf.read(src_path, dtype='float64', frames=4096)

            if ref_data.ndim > 1:
                ref_data = ref_data.mean(axis=1)
            if src_data.ndim > 1:
                src_data = src_data.mean(axis=1)

            # FFT of both
            n = 4096
            ref_fft = np.abs(np.fft.rfft(ref_data[:n] * np.hanning(min(n, len(ref_data))), n=n))
            src_fft = np.abs(np.fft.rfft(src_data[:n] * np.hanning(min(n, len(src_data))), n=n))

            # Correction = ref - src in dB
            ref_db = 20 * np.log10(np.maximum(ref_fft, 1e-10))
            src_db = 20 * np.log10(np.maximum(src_fft, 1e-10))
            diff_db = ref_db - src_db

            # Apply strength
            strength = self.match_strength.value() / 100.0

            # Map to 8 EQ bands
            freqs = np.fft.rfftfreq(n, 1.0 / ref_sr)
            band_centers = [32, 64, 125, 250, 1000, 4000, 8000, 16000]
            for i, fc in enumerate(band_centers):
                # Find nearest bin
                bin_idx = np.argmin(np.abs(freqs - fc))
                # Average a few bins around center
                lo = max(0, bin_idx - 3)
                hi = min(len(diff_db), bin_idx + 4)
                correction = float(np.mean(diff_db[lo:hi])) * strength
                correction = max(-12.0, min(12.0, correction))

                # Apply to EQ band
                if i < len(self.chain.equalizer.bands):
                    self.chain.equalizer.bands[i].gain = correction
                    if i < len(self.eq_band_sliders):
                        slider, label = self.eq_band_sliders[i]
                        slider.blockSignals(True)
                        slider.setValue(int(correction * 10))
                        slider.blockSignals(False)
                        label.setText(f"{correction:.1f}")

            self.meters.set_status("MATCH EQ APPLIED", C_LED_GREEN)
        except Exception as e:
            print(f"[MATCH EQ] Error: {e}")
            self.meters.set_status("MATCH EQ FAILED", C_LED_RED)

    # ─── DYNAMICS VIEW ────────────────────────
    def _build_dynamics_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Title + Enable
        title_row = QHBoxLayout()
        title_row.addWidget(self._module_title("DYNAMICS COMPRESSOR"))
        self.dyn_enabled = QCheckBox("ACTIVE")
        self.dyn_enabled.setChecked(True)
        def _on_dyn_enabled(v):
            self.chain.dynamics.enabled = v
            if self._rt_engine is not None and _HAS_RT_ENGINE:
                self._rt_engine.set_dyn_bypass(not v)
        self.dyn_enabled.toggled.connect(_on_dyn_enabled)
        title_row.addWidget(self.dyn_enabled)
        layout.addLayout(title_row)

        # Preset
        preset_row = QHBoxLayout()
        preset_row.addWidget(self._panel_label("PRESET"))
        self.dyn_preset_combo = QComboBox()
        self.dyn_preset_combo.addItems(list(DYNAMICS_PRESETS.keys()))
        self.dyn_preset_combo.setCurrentText("Standard Master")
        self.dyn_preset_combo.currentTextChanged.connect(self._on_dyn_preset_changed)
        preset_row.addWidget(self.dyn_preset_combo)

        self.dyn_multiband = QCheckBox("MULTIBAND (3-BAND)")
        self.dyn_multiband.toggled.connect(lambda v: setattr(self.chain.dynamics, 'multiband', v))
        preset_row.addWidget(self.dyn_multiband)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        # Dynamics Curve Widget
        self.dyn_curve = DynamicsCurveWidget()
        self.dyn_curve.setMinimumHeight(220)
        self.dyn_curve.thresholdChanged.connect(self._on_dyn_curve_threshold_changed)
        self.dyn_curve.ratioChanged.connect(self._on_dyn_curve_ratio_changed)
        layout.addWidget(self.dyn_curve)

        # V5.10.6: Spectrum Analyzer (Ozone 12 style — live FFT)
        from modules.widgets.spectrum_analyzer import SpectrumAnalyzerWidget
        self.dyn_spectrum = SpectrumAnalyzerWidget()
        self.dyn_spectrum.setFixedHeight(140)
        self.dyn_spectrum.setMinimumWidth(380)
        layout.addWidget(self.dyn_spectrum)

        # Controls: Vintage Rotary Knobs
        ctrl_group = QGroupBox("COMPRESSOR CONTROLS")
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(8)
        ctrl_layout.setContentsMargins(12, 12, 12, 12)

        # Define knob parameters: (label, min, max, default, suffix, attr, step)
        knob_specs = [
            ("THRESHOLD", -40, 0, -16, " dB", "threshold", 0.5),
            ("RATIO", 1.0, 20.0, 2.5, ":1", "ratio", 0.1),
            ("ATTACK", 0.1, 100, 10, " ms", "attack", 0.5),
            ("RELEASE", 10, 1000, 100, " ms", "release", 5.0),
            ("MAKEUP", -10, 20, 2.0, " dB", "makeup", 0.2),
            ("KNEE", 0, 20, 6, " dB", "knee", 0.5),
        ]

        self.dyn_knobs = {}
        for name, mn, mx, default, suffix, attr, step in knob_specs:
            knob = VintageKnobWidget(
                label=name,
                min_val=mn,
                max_val=mx,
                default=default,
                suffix=suffix,
                step=step,
                decimals=1,
                color=C_AMBER,
                parent=self
            )

            # Connect knob value changes to both curve widget and chain dynamics
            def make_knob_callback(attribute, curve_widget):
                def callback(value):
                    # Update curve widget
                    if attribute == "threshold":
                        curve_widget.setThreshold(value)
                    elif attribute == "ratio":
                        curve_widget.setRatio(value)
                    elif attribute == "knee":
                        curve_widget.setKnee(value)
                    elif attribute == "attack":
                        curve_widget.setAttack(value)
                    elif attribute == "release":
                        curve_widget.setRelease(value)
                    elif attribute == "makeup":
                        curve_widget.setMakeup(value)
                    # Update chain dynamics
                    setattr(self.chain.dynamics.single_band, attribute, value)
                    # V5.10: Forward dynamics params to RT engine
                    if self._rt_engine is not None and _HAS_RT_ENGINE:
                        _rt_dyn_map = {
                            "threshold": self._rt_engine.set_dyn_threshold,
                            "ratio": self._rt_engine.set_dyn_ratio,
                            "attack": self._rt_engine.set_dyn_attack,
                            "release": self._rt_engine.set_dyn_release,
                            "makeup": self._rt_engine.set_dyn_makeup,
                            "knee": self._rt_engine.set_dyn_knee,
                        }
                        if attribute in _rt_dyn_map:
                            _rt_dyn_map[attribute](value)
                    # V5.8 A-4: Also update TransferCurve widget
                    if hasattr(self, 'dyn_transfer_curve'):
                        sb = self.chain.dynamics.single_band
                        self.dyn_transfer_curve.set_params(
                            threshold=sb.threshold, ratio=sb.ratio,
                            knee=sb.knee, makeup=sb.makeup)
                return callback

            knob.valueChanged.connect(make_knob_callback(attr, self.dyn_curve))
            ctrl_layout.addWidget(knob)
            self.dyn_knobs[attr] = knob

        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        # V5.8 A-4: Transfer Curve + Detection Mode + Parallel Mix row
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)

        # Transfer Curve widget
        from modules.widgets.transfer_curve import TransferCurveWidget
        self.dyn_transfer_curve = TransferCurveWidget()
        self.dyn_transfer_curve.set_params(threshold=-16.0, ratio=2.5, knee=6.0, makeup=2.0)
        bottom_row.addWidget(self.dyn_transfer_curve)

        # Right column: Detection + Parallel Mix
        right_col = QVBoxLayout()
        right_col.setSpacing(6)

        # Detection mode
        det_row = QHBoxLayout()
        det_row.setSpacing(4)
        det_lbl = QLabel("DETECTION")
        det_lbl.setStyleSheet(f"color:{C_TEAL}; font-size:8px; letter-spacing:1px; font-weight:bold;")
        det_row.addWidget(det_lbl)
        self.dyn_detection = QComboBox()
        self.dyn_detection.addItems(["Peak", "Envelope", "RMS"])
        self.dyn_detection.setCurrentText("RMS")
        self.dyn_detection.currentTextChanged.connect(
            lambda t: setattr(self.chain.dynamics, 'detection_mode', t.lower()))
        det_row.addWidget(self.dyn_detection)
        right_col.addLayout(det_row)

        # Band Solo/Mute (for multiband)
        band_ctrl = QHBoxLayout()
        band_ctrl.setSpacing(4)
        band_ctrl.addWidget(QLabel("SOLO"))
        self.dyn_solo_btns = []
        for band_idx, band_name in enumerate(["LOW", "MID", "HIGH"]):
            btn = QPushButton(band_name[0])
            btn.setCheckable(True)
            btn.setFixedSize(24, 22)
            btn.setStyleSheet(f"""
                QPushButton {{ background:#1A1A1E; border:1px solid #2A2A30; border-radius:3px;
                    color:#6B6B70; font-size:8px; font-weight:bold; }}
                QPushButton:checked {{ background:{C_TEAL_DIM}; color:#FFF; border-color:{C_TEAL}; }}
            """)
            # V5.9 FIX: Connect solo buttons to enable/disable dynamics bands
            def make_solo_cb(idx):
                def cb(checked):
                    # Solo = only this band is enabled; unchecked = all enabled
                    any_checked = any(b.isChecked() for b in self.dyn_solo_btns)
                    for i, band in enumerate(self.chain.dynamics.bands):
                        band.enabled = (i == idx) if any_checked else True
                    self.chain.dynamics.multiband = any_checked
                    self._schedule_auto_preview()
                return cb
            btn.clicked.connect(make_solo_cb(band_idx))
            self.dyn_solo_btns.append(btn)
            band_ctrl.addWidget(btn)
        right_col.addLayout(band_ctrl)

        # Parallel Mix
        mix_row = QHBoxLayout()
        mix_row.setSpacing(4)
        dry_lbl = QLabel("DRY")
        dry_lbl.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:9px;")
        mix_row.addWidget(dry_lbl)
        self.dyn_mix_slider = QSlider(Qt.Orientation.Horizontal)
        self.dyn_mix_slider.setRange(0, 100)
        self.dyn_mix_slider.setValue(100)
        mix_row.addWidget(self.dyn_mix_slider)
        wet_lbl = QLabel("WET")
        wet_lbl.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:9px;")
        mix_row.addWidget(wet_lbl)
        self.dyn_mix_val = QLabel("100%")
        self.dyn_mix_val.setStyleSheet(f"color:{C_AMBER_GLOW}; font-family:'Menlo',monospace; font-weight:bold;")
        self.dyn_mix_slider.valueChanged.connect(self._on_dyn_mix_changed)
        mix_row.addWidget(self.dyn_mix_val)
        right_col.addLayout(mix_row)

        right_col.addStretch()
        bottom_row.addLayout(right_col)
        layout.addLayout(bottom_row)

        return widget

    # ─── IMAGER VIEW ──────────────────────────
    def _build_imager_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Title + Enable
        title_row = QHBoxLayout()
        title_row.addWidget(self._module_title("STEREO IMAGER"))
        self.img_enabled = QCheckBox("ACTIVE")
        self.img_enabled.setChecked(True)
        self.img_enabled.toggled.connect(lambda v: setattr(self.chain.imager, 'enabled', v))
        title_row.addWidget(self.img_enabled)
        layout.addLayout(title_row)

        # Preset
        preset_row = QHBoxLayout()
        preset_row.addWidget(self._panel_label("PRESET"))
        self.img_preset_combo = QComboBox()
        self.img_preset_combo.addItems(list(IMAGER_PRESETS.keys()))
        self.img_preset_combo.currentTextChanged.connect(self._on_img_preset_changed)
        preset_row.addWidget(self.img_preset_combo)

        self.img_multiband = QCheckBox("MULTIBAND")
        self.img_multiband.toggled.connect(lambda v: setattr(self.chain.imager, 'multiband', v))
        preset_row.addWidget(self.img_multiband)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        # Width value display — large VU-style readout
        width_display = QHBoxLayout()
        width_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_width_value = QLabel("100")
        self.img_width_value.setFont(QFont("Menlo", 40, QFont.Weight.Bold))
        self.img_width_value.setStyleSheet(f"color: {C_AMBER_GLOW};")
        self.img_width_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        width_display.addWidget(self.img_width_value)
        pct_lbl = QLabel("%")
        pct_lbl.setFont(QFont("Menlo", 16))
        pct_lbl.setStyleSheet(f"color: {C_GOLD};")
        width_display.addWidget(pct_lbl)
        layout.addLayout(width_display)

        # Width slider
        width_group = QGroupBox("STEREO WIDTH")
        width_layout = QHBoxLayout()
        mono_lbl = QLabel("MONO")
        mono_lbl.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:10px; letter-spacing:1px;")
        width_layout.addWidget(mono_lbl)
        self.img_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.img_width_slider.setRange(0, 200)
        self.img_width_slider.setValue(100)
        self.img_width_slider.valueChanged.connect(self._on_img_width_changed)
        width_layout.addWidget(self.img_width_slider)
        wide_lbl = QLabel("WIDE")
        wide_lbl.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:10px; letter-spacing:1px;")
        width_layout.addWidget(wide_lbl)
        width_group.setLayout(width_layout)
        layout.addWidget(width_group)

        # Mono Bass
        bass_group = QGroupBox("OPTIONS")
        bass_layout = QVBoxLayout()

        mono_row = QHBoxLayout()
        self.img_mono_bass = QCheckBox("MONO BASS BELOW")
        self.img_mono_bass.toggled.connect(self._on_img_mono_bass_toggled)
        mono_row.addWidget(self.img_mono_bass)
        self.img_mono_freq = QSpinBox()
        self.img_mono_freq.setRange(0, 300)
        self.img_mono_freq.setValue(200)
        self.img_mono_freq.setSuffix(" Hz")
        self.img_mono_freq.valueChanged.connect(
            lambda v: setattr(self.chain.imager, 'mono_bass_freq', v if self.img_mono_bass.isChecked() else 0))
        mono_row.addWidget(self.img_mono_freq)
        mono_row.addStretch()
        bass_layout.addLayout(mono_row)

        # Balance
        bal_row = QHBoxLayout()
        l_lbl = QLabel("L")
        l_lbl.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:10px;")
        bal_row.addWidget(l_lbl)
        self.img_balance = QSlider(Qt.Orientation.Horizontal)
        self.img_balance.setRange(-100, 100)
        self.img_balance.setValue(0)
        self.img_balance.valueChanged.connect(
            lambda v: setattr(self.chain.imager, 'balance', v / 100.0))
        bal_row.addWidget(self.img_balance)
        r_lbl = QLabel("R")
        r_lbl.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:10px;")
        bal_row.addWidget(r_lbl)
        bass_layout.addLayout(bal_row)

        bass_group.setLayout(bass_layout)
        layout.addWidget(bass_group)

        # V5.8 A-5: 4-band width sliders
        band_group = QGroupBox("MULTIBAND WIDTH")
        band_layout = QGridLayout()
        band_layout.setSpacing(4)
        self.img_band_sliders = []
        band_names = ["LOW", "LOW-MID", "HIGH-MID", "HIGH"]
        for i, bname in enumerate(band_names):
            lbl = QLabel(bname)
            lbl.setStyleSheet(f"color:{C_TEAL}; font-size:7px; letter-spacing:1px; font-weight:bold;")
            band_layout.addWidget(lbl, i, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 200)
            slider.setValue(100)
            band_layout.addWidget(slider, i, 1)
            val_lbl = QLabel("100%")
            val_lbl.setFixedWidth(36)
            val_lbl.setStyleSheet(f"color:{C_TEAL_GLOW}; font-family:'Menlo'; font-size:9px;")
            band_layout.addWidget(val_lbl, i, 2)

            def make_band_cb(idx, vlbl):
                def cb(v):
                    vlbl.setText(f"{v}%")
                    # V5.9 FIX: Sync multiband width to chain imager bands
                    # UI has 4 bands, chain has 3 — map: 0→Low, 1+2→Mid(avg), 3→High
                    bands = self.chain.imager.bands
                    if idx == 0 and len(bands) > 0:
                        bands[0].width = v
                    elif idx == 1 and len(bands) > 1:
                        bands[1].width = v
                    elif idx == 2 and len(bands) > 1:
                        # HIGH-MID also affects mid band (average with LOW-MID)
                        bands[1].width = (self.img_band_sliders[1][0].value() + v) // 2
                    elif idx == 3 and len(bands) > 2:
                        bands[2].width = v
                    self.chain.imager.multiband = True
                    self._schedule_auto_preview()
                return cb
            slider.valueChanged.connect(make_band_cb(i, val_lbl))
            self.img_band_sliders.append((slider, val_lbl))
        band_group.setLayout(band_layout)
        layout.addWidget(band_group)

        # V5.10.6: Spectrum Analyzer (Ozone 12 style — live FFT)
        from modules.widgets.spectrum_analyzer import SpectrumAnalyzerWidget
        self.img_spectrum = SpectrumAnalyzerWidget()
        self.img_spectrum.setFixedHeight(140)
        self.img_spectrum.setMinimumWidth(380)
        layout.addWidget(self.img_spectrum)

        # V5.8 A-5: Vectorscope + Stereoize + Correlation
        vis_row = QHBoxLayout()
        vis_row.setSpacing(8)

        from modules.widgets.vectorscope import VectorscopeWidget
        self.img_vectorscope = VectorscopeWidget()
        vis_row.addWidget(self.img_vectorscope)

        vis_ctrl = QVBoxLayout()
        vis_ctrl.setSpacing(4)
        self.img_stereoize_i = QCheckBox("STEREOIZE I")
        self.img_stereoize_i.setStyleSheet(f"color:{C_TEAL}; font-size:9px; font-weight:bold;")
        vis_ctrl.addWidget(self.img_stereoize_i)
        self.img_stereoize_ii = QCheckBox("STEREOIZE II")
        self.img_stereoize_ii.setStyleSheet(f"color:{C_TEAL}; font-size:9px; font-weight:bold;")
        vis_ctrl.addWidget(self.img_stereoize_ii)

        # Correlation display
        corr_lbl = QLabel("CORRELATION")
        corr_lbl.setStyleSheet(f"color:{C_TEAL}; font-size:7px; letter-spacing:1px; font-weight:bold;")
        vis_ctrl.addWidget(corr_lbl)
        self.img_corr_value = QLabel("+1.00")
        self.img_corr_value.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        self.img_corr_value.setStyleSheet(f"color:{C_LED_GREEN};")
        vis_ctrl.addWidget(self.img_corr_value)
        vis_ctrl.addStretch()
        vis_row.addLayout(vis_ctrl)
        layout.addLayout(vis_row)

        return widget

    # ─── MAXIMIZER VIEW — OZONE 12 STYLE ──────────────────────
    def _build_maximizer_view(self):
        from modules.widgets.rotary_knob import OzoneRotaryKnob

        widget = QWidget()
        widget.setStyleSheet(f"background: #111114;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # Auto-measure timer (debounced — fires 600ms after last parameter change)
        self._auto_measure_timer = QTimer()
        self._auto_measure_timer.setSingleShot(True)
        self._auto_measure_timer.setInterval(600)
        self._auto_measure_timer.timeout.connect(self._run_auto_measure)

        # V5.9: Auto-preview timer (debounced — re-renders preview 1.5s after last knob change)
        # This gives instant audible feedback like iZotope Ozone 12's real-time preview
        self._auto_preview_timer = QTimer()
        self._auto_preview_timer.setSingleShot(True)
        self._auto_preview_timer.setInterval(1500)
        self._auto_preview_timer.timeout.connect(self._run_auto_preview)
        self._auto_preview_enabled = True  # User can toggle this off

        # ═══ Title bar: MAXIMIZER + ACTIVE + IRC dropdown ═══
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        lbl_max = QLabel("MAXIMIZER")
        lbl_max.setStyleSheet(f"color:{C_TEAL_GLOW}; font-size:11px; font-weight:bold; letter-spacing:2px;")
        top_row.addWidget(lbl_max)

        self.max_enabled = QCheckBox("ACTIVE")
        self.max_enabled.setChecked(True)
        self.max_enabled.setStyleSheet(f"color:{C_TEAL}; font-size:9px; font-weight:bold;")
        self.max_enabled.toggled.connect(lambda v: setattr(self.chain.maximizer, 'enabled', v))
        top_row.addWidget(self.max_enabled)

        # ── BYPASS button: toggle between MASTERED / ORIGINAL ──
        self.max_bypass_btn = QPushButton("● MASTERED")
        self.max_bypass_btn.setFixedSize(120, 28)
        self.max_bypass_btn.setCheckable(True)
        self.max_bypass_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.max_bypass_btn.setStyleSheet(
            f"QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1A3A1A, stop:1 #102A10); "
            f"border: 2px solid {C_LED_GREEN}; border-radius: 5px; color: {C_LED_GREEN}; "
            f"font-size: 10px; font-weight: bold; font-family: 'SF Pro Display', 'Menlo', monospace; }}")
        self.max_bypass_btn.toggled.connect(self._on_max_bypass_toggled)
        top_row.addWidget(self.max_bypass_btn)

        top_row.addStretch()

        # IRC Mode dropdown button — teal, Ozone 12 style
        self.irc_mode_btn = QPushButton("IRC 3 — Balanced  ▾")
        self.irc_mode_btn.setFixedHeight(32)
        self.irc_mode_btn.setMinimumWidth(200)
        self.irc_mode_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1E1E26, stop:1 #14141A);
                border: 1px solid {C_TEAL_DIM};
                border-radius: 5px;
                color: {C_TEAL_GLOW};
                font-weight: bold;
                font-size: 11px;
                font-family: 'SF Pro Display', 'Menlo', monospace;
                padding: 5px 14px;
                text-align: left;
            }}
            QPushButton:hover {{
                border-color: {C_TEAL};
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #222230, stop:1 #1A1A24);
                color: #FFFFFF;
            }}
            QPushButton:pressed {{
                background: {C_TEAL_DIM};
                color: #FFFFFF;
            }}
        """)
        self.irc_mode_btn.clicked.connect(self._show_irc_menu)
        top_row.addWidget(self.irc_mode_btn)
        layout.addLayout(top_row)

        # V5.8 A-1: IRC Sub-Mode dropdown (visible only for IRC III/IV)
        sub_row = QHBoxLayout()
        sub_row.setSpacing(6)
        self.irc_submode_label = QLabel("SUB-MODE")
        self.irc_submode_label.setStyleSheet(f"color:{C_TEAL}; font-size:8px; letter-spacing:1px; font-weight:bold;")
        sub_row.addWidget(self.irc_submode_label)
        self.irc_submode_combo = QComboBox()
        self.irc_submode_combo.setStyleSheet(f"""
            QComboBox {{
                background: #1A1A1E; border: 1px solid {C_TEAL_DIM}; border-radius: 3px;
                color: {C_TEAL_GLOW}; font-family: 'Menlo'; font-size: 10px;
                font-weight: bold; padding: 2px 8px; min-width: 120px;
            }}
        """)
        self.irc_submode_combo.currentTextChanged.connect(self._on_irc_submode_changed)
        sub_row.addWidget(self.irc_submode_combo)
        sub_row.addStretch()
        self.irc_submode_label.setVisible(False)
        self.irc_submode_combo.setVisible(False)
        layout.addLayout(sub_row)

        # IRC description (subtle)
        self.irc_desc_label = QLabel(IRC_MODES.get("IRC 3 - Balanced", {}).get("description", ""))
        self.irc_desc_label.setWordWrap(True)
        self.irc_desc_label.setStyleSheet(
            f"color: #6B6B70; font-style: italic; font-size: 8px; padding: 0 0 2px 0;")
        layout.addWidget(self.irc_desc_label)

        # ═══ MAIN AREA: Knobs Row — Gain (large) + Ceiling + Character ═══
        knobs_row = QHBoxLayout()
        knobs_row.setSpacing(8)

        # Gain knob (large)
        self.max_gain_knob = OzoneRotaryKnob(
            name="GAIN", min_val=0.0, max_val=20.0, default=0.0,
            unit="dB", decimals=1, large=True)
        self.max_gain_knob.valueChanged.connect(self._on_gain_knob_changed)
        knobs_row.addWidget(self.max_gain_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        # Ceiling knob
        self.max_ceiling_knob = OzoneRotaryKnob(
            name="CEILING", min_val=-3.0, max_val=-0.1, default=-1.0,
            unit="dBTP", decimals=2)
        self.max_ceiling_knob.valueChanged.connect(self._on_ceiling_knob_changed)
        knobs_row.addWidget(self.max_ceiling_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        # Character knob
        self.max_character_knob = OzoneRotaryKnob(
            name="CHARACTER", min_val=0.0, max_val=10.0, default=3.0,
            unit="", decimals=1)
        self.max_character_knob.valueChanged.connect(self._on_character_knob_changed)
        knobs_row.addWidget(self.max_character_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        # Upward Compress knob
        self.max_upward_knob = OzoneRotaryKnob(
            name="UPWARD", min_val=0.0, max_val=12.0, default=0.0,
            unit="dB", decimals=1)
        self.max_upward_knob.valueChanged.connect(self._on_upward_knob_changed)
        knobs_row.addWidget(self.max_upward_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        # Soft Clip knob
        self.max_softclip_knob = OzoneRotaryKnob(
            name="SOFT CLIP", min_val=0.0, max_val=100.0, default=0.0,
            unit="%", decimals=0)
        self.max_softclip_knob.valueChanged.connect(self._on_softclip_knob_changed)
        knobs_row.addWidget(self.max_softclip_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(knobs_row)

        # ═══ Controls Row: True Peak + Transient Emphasis + Stereo Independence ═══
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(12)

        self.max_true_peak = QCheckBox("TRUE PEAK")
        self.max_true_peak.setChecked(True)
        self.max_true_peak.setStyleSheet(f"color:{C_TEAL}; font-size:9px; font-weight:bold;")
        self.max_true_peak.toggled.connect(
            lambda v: setattr(self.chain.maximizer, 'true_peak', v))
        ctrl_row.addWidget(self.max_true_peak)

        # Transient Emphasis: H/M/L buttons
        te_lbl = QLabel("TRANSIENT")
        te_lbl.setStyleSheet(f"color:{C_TEAL}; font-size:8px; letter-spacing:1px; font-weight:bold;")
        ctrl_row.addWidget(te_lbl)
        self.max_band_btns = []
        for band in ["H", "M", "L"]:
            btn = QPushButton(band)
            btn.setCheckable(True)
            btn.setFixedSize(26, 22)
            btn.clicked.connect(lambda checked, b=band: self._on_transient_band(b))
            self.max_band_btns.append(btn)
            ctrl_row.addWidget(btn)
        self.max_band_btns[1].setChecked(True)
        self._update_band_button_styles()

        # Stereo Independence checkbox
        self.max_stereo_ind = QCheckBox("STEREO IND")
        self.max_stereo_ind.setStyleSheet(f"color:{C_TEAL}; font-size:9px; font-weight:bold;")
        self.max_stereo_ind.toggled.connect(
            lambda v: setattr(self.chain.maximizer, 'stereo_independence', v))
        ctrl_row.addWidget(self.max_stereo_ind)
        ctrl_row.addStretch()

        # True Peak dBTP display
        self.max_meter_db = QLabel("—  dB")
        self.max_meter_db.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.max_meter_db.setStyleSheet(
            f"color:{C_TEAL_GLOW}; font-family:'Menlo'; font-size:9px; font-weight:bold;")
        ctrl_row.addWidget(self.max_meter_db)

        layout.addLayout(ctrl_row)

        # ═══ Tone chip buttons ═══
        tone_row = QHBoxLayout()
        tone_row.setSpacing(3)
        tone_lbl = QLabel("TONE")
        tone_lbl.setStyleSheet(f"color:{C_TEAL}; font-size:7px; letter-spacing:1.5px; font-weight:bold;")
        tone_row.addWidget(tone_lbl)
        self.tone_buttons = []
        tone_names = list(TONE_PRESETS.keys())
        for i, name in enumerate(tone_names):
            btn = QPushButton(name[:4].upper())
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setFixedWidth(42)
            btn.clicked.connect(lambda checked, idx=i: self._on_tone_clicked(idx))
            self.tone_buttons.append(btn)
            tone_row.addWidget(btn)
        self.tone_buttons[0].setChecked(True)
        self._update_tone_button_styles()
        layout.addLayout(tone_row)

        # V5.10.6: Spectrum Analyzer (Ozone 12 style — live FFT behind Maximizer)
        from modules.widgets.spectrum_analyzer import SpectrumAnalyzerWidget
        self.max_spectrum = SpectrumAnalyzerWidget()
        self.max_spectrum.setFixedHeight(160)
        self.max_spectrum.setMinimumWidth(380)
        layout.addWidget(self.max_spectrum)

        # ═══ GAIN REDUCTION — Ozone 12 Style History Widget ═══
        self.max_gr_history = GainReductionHistoryWidget()
        layout.addWidget(self.max_gr_history)

        # Keep legacy references for backward compat
        self.max_gr_bar = None
        self.max_gr_label = None
        # Legacy compat: create hidden references for old handler code
        self.max_gain_display = QLabel("+0.0")
        self.max_gain_display.setVisible(False)
        self.max_char_val = QLabel("3.00")
        self.max_char_val.setVisible(False)
        self.max_lufs_label = QLabel("— LUFS")
        self.max_lufs_label.setStyleSheet(
            f"color:{C_TEAL_GLOW}; font-family:'Menlo'; font-weight:bold; font-size:10px;")

        # Learn + Measure row
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.btn_learn_gain = QPushButton("LEARN INPUT GAIN")
        self.btn_learn_gain.setFixedHeight(26)
        self.btn_learn_gain.setStyleSheet(f"""
            QPushButton {{
                background: #1A1A1E; border: 1px solid {C_TEAL_DIM};
                border-radius: 3px; color: {C_TEAL}; font-weight: bold;
                font-size: 9px; padding: 3px 10px;
            }}
            QPushButton:hover {{ background: {C_TEAL_DIM}; color: #FFFFFF; }}
        """)
        self.btn_learn_gain.clicked.connect(self._on_learn_gain)
        action_row.addWidget(self.btn_learn_gain)
        action_row.addWidget(self.max_lufs_label)

        self.btn_measure = QPushButton("⟳  MEASURE")
        self.btn_measure.setFixedHeight(26)
        self.btn_measure.setStyleSheet(f"""
            QPushButton {{
                background: #1A1A1E; border: 1px solid {C_TEAL_DIM};
                border-radius: 3px; color: {C_TEAL}; font-weight: bold;
                font-size: 9px; padding: 3px 10px;
            }}
            QPushButton:hover {{ background: {C_TEAL_DIM}; color: #FFFFFF; }}
        """)
        self.btn_measure.clicked.connect(self._on_measure_levels)
        action_row.addWidget(self.btn_measure)
        action_row.addStretch()
        layout.addLayout(action_row)

        return widget

    # V5.8 A-3: Rotary knob handlers (float-based, replacing int-based)
    def _on_gain_knob_changed(self, gain_db: float):
        """OzoneRotaryKnob Gain changed (0.0-20.0 dB direct)."""
        self.chain.maximizer.set_gain(gain_db)
        self.max_gain_display.setText(f"+{gain_db:.1f}")

        # V5.10: Forward to RT engine for instant real-time feedback
        if self._rt_engine is not None:
            self._rt_engine.set_gain(gain_db)
        else:
            # Fallback: Instant volume feedback via QAudioOutput
            volume = min(3.0, 0.5 + gain_db / 20.0 * 2.5)
            if hasattr(self, '_audio_output_master'):
                self._audio_output_master.setVolume(volume)
            if hasattr(self, '_audio_output_bypass'):
                self._audio_output_bypass.setVolume(volume)

        # Teal → Yellow → Orange → Red color based on gain push
        if gain_db < 6.0:
            color = C_TEAL_GLOW
        elif gain_db < 12.0:
            color = C_LED_YELLOW
        elif gain_db < 16.0:
            color = "#FF8C00"  # Orange
        else:
            color = C_LED_RED
        self.max_gain_display.setStyleSheet(f"color: {color};")
        self._schedule_auto_measure()
        self._schedule_auto_preview()

    def _on_ceiling_knob_changed(self, value: float):
        """OzoneRotaryKnob Ceiling changed."""
        self.chain.maximizer.set_ceiling(value)
        # V5.10: Forward to RT engine for instant real-time feedback
        if self._rt_engine is not None:
            self._rt_engine.set_ceiling(value)

    def _on_character_knob_changed(self, value: float):
        """OzoneRotaryKnob Character changed (0.0-10.0 direct)."""
        self.chain.maximizer.set_character(value)
        self.max_char_val.setText(f"{value:.2f}")

    def _on_upward_knob_changed(self, value: float):
        """OzoneRotaryKnob Upward Compress changed."""
        self.chain.maximizer.set_upward_compress(value)

    def _on_softclip_knob_changed(self, value: float):
        """OzoneRotaryKnob Soft Clip changed (0-100%)."""
        pct = int(value)
        self.chain.maximizer.set_soft_clip(pct > 0, pct)

    def _on_irc_submode_changed(self, text: str):
        """IRC sub-mode combo changed."""
        if text and hasattr(self.chain.maximizer, 'set_irc_sub_mode'):
            self.chain.maximizer.set_irc_sub_mode(text)

    def _ozone_label(self, text):
        """Create a small teal label for Ozone 12 style."""
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{C_TEAL}; font-size:8px; letter-spacing:1px; font-weight:bold; border:none;")
        return lbl

    def _schedule_auto_preview(self):
        """Schedule auto-preview re-render after parameter change (debounced 1.5s).

        V5.9: Like iZotope Ozone 12 — adjusting any knob triggers an automatic
        short preview render so the user hears the change immediately.
        """
        if not getattr(self, '_auto_preview_enabled', False):
            return
        if not self.chain.input_path:
            return
        if getattr(self, '_auto_preview_timer', None):
            self._auto_preview_timer.start()  # restart debounce timer

    def _run_auto_preview(self):
        """Auto-preview: render a short preview and auto-play it."""
        if not self.chain.input_path:
            return
        # Don't overlap with existing processing
        with self.chain._processing_lock:
            if self.chain.is_processing:
                self._schedule_auto_preview()  # re-queue
                return

        self.meters.set_status("AUTO-PREVIEW...", C_LED_BLUE)

        def _do_preview():
            try:
                preview_path = self.chain.preview(duration_sec=15)
                if preview_path and os.path.exists(preview_path):
                    # Analyze output loudness
                    try:
                        self.chain.output_analysis = self.chain.loudness_meter.analyze(preview_path)
                    except Exception:
                        pass
                    self._call_on_main.emit(lambda p=preview_path: self._on_auto_preview_done(p))
                else:
                    self._call_on_main.emit(lambda: self.meters.set_status("READY", C_TEAL))
            except Exception as e:
                print(f"[AUTO-PREVIEW] Error: {e}")
                self._call_on_main.emit(lambda: self.meters.set_status("READY", C_TEAL))

        threading.Thread(target=_do_preview, daemon=True).start()

    def _on_auto_preview_done(self, preview_path):
        """Auto-preview completed — load and auto-play."""
        try:
            self._load_preview_for_playback(preview_path)
            # Auto-play the preview
            self._on_transport_play()
            self.meters.set_status("▶ AUTO-PREVIEW", C_LED_GREEN)
            # Update meters if we have output analysis
            if self.chain.output_analysis and self.chain.input_analysis:
                try:
                    self.meters.update_output(
                        self.chain.output_analysis.integrated_lufs,
                        self.chain.output_analysis.true_peak_dbtp,
                        getattr(self.chain.output_analysis, 'lra', 0.0),
                    )
                except Exception:
                    pass
        except Exception as e:
            print(f"[AUTO-PREVIEW] Play error: {e}")

    def _schedule_auto_measure(self):
        """Schedule an auto-measure after parameter change (debounced 600ms)."""
        if hasattr(self, '_auto_measure_timer'):
            self._auto_measure_timer.start()  # restart timer

    # ─── AI ASSIST VIEW ──────────────────────
    def _build_ai_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        layout.addWidget(self._module_title("MASTER ASSISTANT"))

        # V5.8 D-3: Auto-detect genre button
        detect_row = QHBoxLayout()
        detect_row.setSpacing(6)
        self.btn_auto_detect = QPushButton("AUTO-DETECT GENRE")
        self.btn_auto_detect.setFixedHeight(28)
        self.btn_auto_detect.setStyleSheet(f"""
            QPushButton {{ background:#1A1A1E; border:1px solid {C_AMBER_DIM}; border-radius:3px;
                color:{C_AMBER}; font-weight:bold; font-size:9px; padding:3px 10px; }}
            QPushButton:hover {{ background:{C_AMBER_DIM}; color:#FFF; }}
        """)
        self.btn_auto_detect.clicked.connect(self._on_auto_detect_genre)
        detect_row.addWidget(self.btn_auto_detect)

        self.ai_detected_label = QLabel("")
        self.ai_detected_label.setStyleSheet(f"color:{C_AMBER_GLOW}; font-size:10px; font-weight:bold;")
        detect_row.addWidget(self.ai_detected_label)
        detect_row.addStretch()
        layout.addLayout(detect_row)

        # Genre + Platform (compact row)
        config_row = QHBoxLayout()
        config_row.addWidget(self._panel_label("GENRE"))
        self.ai_genre = QComboBox()
        self.ai_genre.addItems(sorted(GENRE_PROFILES.keys()))
        config_row.addWidget(self.ai_genre)

        config_row.addWidget(self._panel_label("PLATFORM"))
        self.platform_combo_ai = QComboBox()
        for name, data in PLATFORM_TARGETS.items():
            self.platform_combo_ai.addItem(f"{name} ({data['target_lufs']} LUFS)")
        config_row.addWidget(self.platform_combo_ai)
        config_row.addStretch()
        layout.addLayout(config_row)

        # Mastering Drive intensity
        ai_int_group = QGroupBox("MASTERING DRIVE")
        ai_int_layout = QHBoxLayout()
        cons_lbl = QLabel("CONSERVATIVE")
        cons_lbl.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:9px; letter-spacing:1px;")
        ai_int_layout.addWidget(cons_lbl)
        self.ai_intensity = QSlider(Qt.Orientation.Horizontal)
        self.ai_intensity.setRange(0, 100)
        self.ai_intensity.setValue(65)
        ai_int_layout.addWidget(self.ai_intensity)
        agg_lbl = QLabel("AGGRESSIVE")
        agg_lbl.setStyleSheet(f"color:{C_CREAM_DIM}; font-size:9px; letter-spacing:1px;")
        ai_int_layout.addWidget(agg_lbl)
        self.ai_intensity_label = QLabel("65%")
        self.ai_intensity_label.setStyleSheet(
            f"color:{C_AMBER_GLOW}; font-family:'Menlo',monospace; font-weight:bold;")
        self.ai_intensity.valueChanged.connect(
            lambda v: self.ai_intensity_label.setText(f"{v}%"))
        ai_int_layout.addWidget(self.ai_intensity_label)
        ai_int_group.setLayout(ai_int_layout)
        layout.addWidget(ai_int_group)

        # BIG MASTER BUTTON
        self.btn_master = QPushButton("⚡  M A S T E R")
        self.btn_master.setFixedHeight(80)
        self.btn_master.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {C_TEAL_GLOW}, stop:0.1 {C_TEAL}, stop:0.9 {C_TEAL_DIM}, stop:1 #004466);
                color: #FFFFFF;
                border: 2px solid {C_TEAL_DIM};
                border-top: 2px solid {C_TEAL_GLOW};
                border-radius: 6px;
                padding: 16px;
                font-size: 18px;
                font-weight: bold;
                font-family: 'Menlo', monospace;
                letter-spacing: 3px;
                text-transform: uppercase;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #5AD8F0, stop:0.1 {C_TEAL_GLOW},
                    stop:0.9 {C_TEAL}, stop:1 {C_TEAL_DIM});
                border-color: {C_TEAL_GLOW};
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #004466, stop:0.1 {C_TEAL_DIM},
                    stop:0.9 {C_TEAL}, stop:1 {C_TEAL_GLOW});
                border-top: 2px solid #004466;
            }}
            QPushButton:disabled {{
                background: {C_PANEL};
                color: {C_CREAM_DARK};
                border-color: {C_GROOVE};
            }}
        """)
        self.btn_master.clicked.connect(self._on_one_button_master)
        layout.addWidget(self.btn_master)

        # Master progress bar
        self.master_progress = QProgressBar()
        self.master_progress.setRange(0, 100)
        self.master_progress.setValue(0)
        self.master_progress.setFixedHeight(12)
        self.master_progress.setTextVisible(False)
        self.master_progress.setVisible(False)
        self.master_progress.setStyleSheet(f"""
            QProgressBar {{
                background: {C_PANEL_INSET};
                border: 1px solid {C_GROOVE};
                border-radius: 6px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {C_TEAL_DIM}, stop:0.5 {C_TEAL}, stop:1 {C_TEAL_GLOW});
                border-radius: 5px;
            }}
        """)
        layout.addWidget(self.master_progress)

        # Master status
        self.master_status = QLabel("READY")
        self.master_status.setStyleSheet(f"color:{C_AMBER}; font-size:11px; font-weight:bold; letter-spacing:1px;")
        self.master_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.master_status)

        # Results frame (appears after mastering)
        self.master_results_frame = QGroupBox("RESULTS")
        self.master_results_frame.setVisible(False)
        res_layout = QVBoxLayout()

        # Before/After LUFS display
        self.master_lufs_before = QLabel("BEFORE: --.- LUFS")
        self.master_lufs_before.setStyleSheet(f"color:{C_CREAM}; font-size:11px; font-weight:bold;")
        res_layout.addWidget(self.master_lufs_before)

        self.master_lufs_after = QLabel("AFTER: --.- LUFS")
        self.master_lufs_after.setStyleSheet(f"color:{C_TEAL_GLOW}; font-size:11px; font-weight:bold;")
        res_layout.addWidget(self.master_lufs_after)

        self.master_confidence = QLabel("CONFIDENCE: --% MEETS TARGET")
        self.master_confidence.setStyleSheet(f"color:{C_AMBER}; font-size:11px; font-weight:bold;")
        res_layout.addWidget(self.master_confidence)

        self.master_results_frame.setLayout(res_layout)
        layout.addWidget(self.master_results_frame)

        # Post-mastering controls frame
        self.post_master_frame = QGroupBox("ADJUST")
        self.post_master_frame.setVisible(False)
        adjust_layout = QVBoxLayout()

        # EQ Amount
        eq_row = QHBoxLayout()
        eq_row.addWidget(QLabel("EQ AMOUNT"))
        self.post_eq_slider = QSlider(Qt.Orientation.Horizontal)
        self.post_eq_slider.setRange(0, 100)
        self.post_eq_slider.setValue(100)
        eq_row.addWidget(self.post_eq_slider)
        self.post_eq_label = QLabel("100%")
        self.post_eq_label.setStyleSheet(f"color:{C_AMBER_GLOW}; font-weight:bold; min-width:35px;")
        self.post_eq_slider.valueChanged.connect(self._on_post_eq_changed)
        eq_row.addWidget(self.post_eq_label)
        adjust_layout.addLayout(eq_row)

        # Dynamics Amount
        dyn_row = QHBoxLayout()
        dyn_row.addWidget(QLabel("DYNAMICS"))
        self.post_dyn_slider = QSlider(Qt.Orientation.Horizontal)
        self.post_dyn_slider.setRange(0, 100)
        self.post_dyn_slider.setValue(100)
        dyn_row.addWidget(self.post_dyn_slider)
        self.post_dyn_label = QLabel("100%")
        self.post_dyn_label.setStyleSheet(f"color:{C_AMBER_GLOW}; font-weight:bold; min-width:35px;")
        self.post_dyn_slider.valueChanged.connect(self._on_post_dyn_changed)
        dyn_row.addWidget(self.post_dyn_label)
        adjust_layout.addLayout(dyn_row)

        # Stereo Width
        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("WIDTH"))
        self.post_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.post_width_slider.setRange(50, 200)
        self.post_width_slider.setValue(100)
        width_row.addWidget(self.post_width_slider)
        self.post_width_label = QLabel("100%")
        self.post_width_label.setStyleSheet(f"color:{C_AMBER_GLOW}; font-weight:bold; min-width:35px;")
        self.post_width_slider.valueChanged.connect(self._on_post_width_changed)
        width_row.addWidget(self.post_width_label)
        adjust_layout.addLayout(width_row)

        # Ceiling
        ceil_row = QHBoxLayout()
        ceil_row.addWidget(QLabel("CEILING"))
        self.post_ceiling_slider = QSlider(Qt.Orientation.Horizontal)
        self.post_ceiling_slider.setRange(-30, -1)
        self.post_ceiling_slider.setValue(-10)
        ceil_row.addWidget(self.post_ceiling_slider)
        self.post_ceiling_label = QLabel("-1.0 dB")
        self.post_ceiling_label.setStyleSheet(f"color:{C_AMBER_GLOW}; font-weight:bold; min-width:50px;")
        self.post_ceiling_slider.valueChanged.connect(self._on_post_ceiling_changed)
        ceil_row.addWidget(self.post_ceiling_label)
        adjust_layout.addLayout(ceil_row)

        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_remaster = QPushButton("RE-MASTER")
        self.btn_remaster.setStyleSheet(BTN_TEAL_STYLE)
        self.btn_remaster.clicked.connect(self._on_remaster)
        btn_row.addWidget(self.btn_remaster)

        self.btn_render_full = QPushButton("RENDER FULL")
        self.btn_render_full.setStyleSheet(BTN_PRIMARY_STYLE)
        self.btn_render_full.clicked.connect(self._on_render_full_from_master)
        btn_row.addWidget(self.btn_render_full)

        adjust_layout.addLayout(btn_row)

        self.post_master_frame.setLayout(adjust_layout)
        layout.addWidget(self.post_master_frame)

        # Hardware decoration — signal flow diagram fills empty space
        layout.addWidget(HardwareDecoration(HardwareDecoration.MODE_SIGNAL_FLOW))
        return widget

    # ─── ACTION BAR ──────────────────────────
    def _build_action_bar(self):
        bar = QFrame()
        bar.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {C_RIDGE}, stop:0.05 {C_PANEL},
                    stop:0.95 {C_PANEL_INSET}, stop:1 {C_GROOVE});
                border-top: 1px solid {C_GROOVE};
            }}
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(8)

        # V5.10: ANALYZE and PREVIEW removed — auto-analyze on load, direct play
        self.btn_analyze = QPushButton("ANALYZE")
        self.btn_analyze.hide()  # kept for compatibility but hidden
        self.btn_preview = QPushButton("PREVIEW 30s")
        self.btn_preview.hide()  # kept for compatibility but hidden

        self.btn_ab = QPushButton("A/B COMPARE")
        self.btn_ab.setStyleSheet(BTN_SECONDARY_STYLE)
        self.btn_ab.clicked.connect(self._on_ab_compare)
        self.btn_ab.setEnabled(False)
        layout.addWidget(self.btn_ab)

        self.btn_render = QPushButton("MASTER & EXPORT")
        self.btn_render.setStyleSheet(BTN_PRIMARY_STYLE)
        self.btn_render.clicked.connect(self._on_render)
        layout.addWidget(self.btn_render)

        # V5.4 FIX: Batch Master button
        self.btn_batch_render = QPushButton("⚡ BATCH MASTER ALL")
        self.btn_batch_render.setStyleSheet(
            "QPushButton { background: #008B8B; color: white; font-weight: bold; "
            "font-size: 11px; padding: 8px 16px; border-radius: 4px; } "
            "QPushButton:hover { background: #00AAAA; }")
        self.btn_batch_render.setToolTip("Master all tracks in the current playlist")
        self.btn_batch_render.clicked.connect(self._on_batch_render)
        layout.addWidget(self.btn_batch_render)

        # V5.4 FIX: Live stage label for meter display
        self.live_stage_label = QLabel("STAGE: IDLE")
        self.live_stage_label.setStyleSheet(
            f"font-size: 9px; color: {C_AMBER}; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(self.live_stage_label)

        # Progress area
        progress_layout = QHBoxLayout()
        progress_layout.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.progress_text = QLabel("READY")
        self.progress_text.setStyleSheet(
            f"font-size:10px; color:{C_AMBER}; font-weight:bold; letter-spacing:1px;")
        self.progress_text.setMinimumWidth(120)
        progress_layout.addWidget(self.progress_text)

        layout.addLayout(progress_layout, stretch=1)

        return bar

    # ═══════════════════════════════════════════
    #  Event Handlers
    # ═══════════════════════════════════════════

    def _switch_module(self, index: int):
        for i, node in enumerate(self.chain_nodes):
            node.set_active(i == index)
        self.module_stack.setCurrentIndex(index)

    def _on_platform_changed(self, index: int):
        platform_names = list(PLATFORM_TARGETS.keys())
        if 0 <= index < len(platform_names):
            platform = platform_names[index]
            self.chain.set_platform(platform)
            target = PLATFORM_TARGETS[platform]
            self.meters.update_target(target["target_lufs"], target["true_peak"], platform)
            if hasattr(self, 'max_ceiling_knob'):
                self.max_ceiling_knob.setValue(target["true_peak"])

    def _on_intensity_changed(self, value: int):
        self.chain.intensity = value
        self.intensity_value.setText(f"{value}%")
        self._schedule_auto_preview()

    def _on_reset_all(self):
        self.chain = MasterChain(self.chain.ffmpeg_path)
        self.chain.set_meter_callback(self._on_meter_data)
        if self._current_audio_path:
            self.chain.load_audio(self._current_audio_path)
        self._sync_ui_from_chain()

    def _on_mastering_preset_changed(self, index: int):
        """Apply a one-click mastering preset (Dynamics + Imager + Maximizer)."""
        if index <= 0:  # "— Custom —" selected
            return
        name = self.mastering_preset_combo.itemData(index)
        if not name or name not in MASTERING_PRESETS:
            return
        preset = MASTERING_PRESETS[name]
        print(f"[PRESET] Applying mastering preset: {name}")

        # --- Apply Dynamics ---
        dyn = preset["dynamics"]
        band = self.chain.dynamics.single_band
        band.threshold = dyn["threshold"]
        band.ratio = dyn["ratio"]
        band.attack = dyn["attack"]
        band.release = dyn["release"]
        band.makeup = dyn["makeup"]
        band.knee = dyn.get("knee", 4)
        self._sync_dynamics_ui()

        # --- Apply Imager ---
        img = preset["imager"]
        self.chain.imager.set_width(img["width"])
        if img.get("mono_bass_freq", 0) > 0:
            self.chain.imager.mono_bass_freq = img["mono_bass_freq"]
        self._sync_imager_ui()

        # --- Apply Maximizer ---
        mx = preset["maximizer"]
        self.chain.maximizer.set_gain(mx["gain_db"])
        self.chain.maximizer.set_ceiling(mx["ceiling"])
        self.chain.maximizer.set_character(mx["character"])
        self.chain.maximizer.set_irc_mode(mx["irc_mode"], mx.get("irc_sub_mode"))
        self.chain.maximizer.transient_emphasis_pct = mx.get("transient_emphasis_pct", 0)
        self.chain.maximizer.upward_compress_db = mx.get("upward_compress_db", 0)
        self.chain.maximizer.soft_clip_enabled = mx.get("soft_clip_enabled", False)
        self.chain.maximizer.soft_clip_pct = mx.get("soft_clip_pct", 0)
        self._sync_maximizer_ui()

        # Schedule auto-measure and auto-preview if audio loaded
        self._schedule_auto_measure()
        self._schedule_auto_preview()

    def _sync_dynamics_ui(self):
        """Sync dynamics UI knobs + curve from chain state."""
        band = self.chain.dynamics.single_band
        if hasattr(self, 'dyn_knobs'):
            knob_vals = {
                "threshold": band.threshold, "ratio": band.ratio,
                "attack": band.attack, "release": band.release,
                "makeup": band.makeup, "knee": band.knee,
            }
            for attr, val in knob_vals.items():
                if attr in self.dyn_knobs:
                    self.dyn_knobs[attr].setValue(val)
        # Update curve widget
        if hasattr(self, 'dyn_curve'):
            self.dyn_curve.setThreshold(band.threshold)
            self.dyn_curve.setRatio(band.ratio)
            self.dyn_curve.setKnee(band.knee)
            self.dyn_curve.setAttack(band.attack)
            self.dyn_curve.setRelease(band.release)
            self.dyn_curve.setMakeup(band.makeup)

    def _sync_imager_ui(self):
        """Sync imager UI controls from chain state."""
        img = self.chain.imager
        if hasattr(self, 'img_width_slider'):
            self.img_width_slider.blockSignals(True)
            self.img_width_slider.setValue(img.width)
            self.img_width_slider.blockSignals(False)

    def _sync_maximizer_ui(self):
        """Sync maximizer UI controls from chain state."""
        mx = self.chain.maximizer
        # V5.8: Sync OzoneRotaryKnob widgets
        if hasattr(self, 'max_gain_knob'):
            self.max_gain_knob.blockSignals(True)
            self.max_gain_knob.setValue(mx.gain_db)
            self.max_gain_knob.blockSignals(False)
        if hasattr(self, 'max_gain_display'):
            self.max_gain_display.setText(f"+{mx.gain_db:.1f}")
        if hasattr(self, 'max_ceiling_knob'):
            self.max_ceiling_knob.blockSignals(True)
            self.max_ceiling_knob.setValue(mx.ceiling)
            self.max_ceiling_knob.blockSignals(False)
        if hasattr(self, 'max_character_knob'):
            self.max_character_knob.blockSignals(True)
            self.max_character_knob.setValue(mx.character)
            self.max_character_knob.blockSignals(False)
        if hasattr(self, 'max_char_val'):
            self.max_char_val.setText(f"{mx.character:.2f}")
        if hasattr(self, 'irc_mode_btn'):
            mode = mx.irc_mode
            sub = mx.irc_sub_mode
            if sub:
                self.irc_mode_btn.setText(f"{mode} — {sub}  ▾")
            else:
                self.irc_mode_btn.setText(f"{mode}  ▾")
        if hasattr(self, 'max_upward_knob'):
            self.max_upward_knob.blockSignals(True)
            self.max_upward_knob.setValue(mx.upward_compress_db)
            self.max_upward_knob.blockSignals(False)
        if hasattr(self, 'max_softclip_knob'):
            self.max_softclip_knob.blockSignals(True)
            self.max_softclip_knob.setValue(mx.soft_clip_pct)
            self.max_softclip_knob.blockSignals(False)

    def _on_eq_preset_changed(self, preset: str):
        self.chain.equalizer.load_tone_preset(preset)
        for i, band in enumerate(self.chain.equalizer.bands):
            if i < len(self.eq_band_sliders):
                slider, label = self.eq_band_sliders[i]
                slider.blockSignals(True)
                slider.setValue(int(band.gain * 10))
                slider.blockSignals(False)
                label.setText(f"{band.gain:.1f}")
        self._schedule_auto_preview()

    def _on_eq_band_changed(self, index: int, gain: float):
        try:
            self.chain.equalizer.set_band(index, gain=gain)
            self.chain.equalizer.preset_mode = False
            if index < len(self.eq_band_sliders):
                _, label = self.eq_band_sliders[index]
                label.setText(f"{gain:.1f}")
            # Sync EQ curve display
            if hasattr(self, 'eq_curve') and self.eq_curve is not None:
                self.eq_curve.setGain(index, gain)
            # V5.10: Real-time engine — instant EQ change
            if self._rt_engine is not None:
                self._rt_engine.set_eq_gain(index, gain)
            self._schedule_auto_preview()
        except Exception as e:
            print(f"[EQ] Band change error: {e}")

    def _on_eq_curve_band_changed(self, band: int, gain_db: float):
        """Called when user drags a band node on the EQ curve widget."""
        try:
            gain_db = round(gain_db, 1)
            # Update the corresponding slider
            if 0 <= band < len(self.eq_band_sliders):
                slider, val_label = self.eq_band_sliders[band]
                slider.blockSignals(True)
                slider.setValue(int(gain_db * 10))
                slider.blockSignals(False)
                val_label.setText(f"{gain_db:.1f}")
            # Update the chain
            self._on_eq_band_changed(band, gain_db)
        except Exception as e:
            print(f"[EQ] Curve band change error: {e}")

    def _on_dyn_mix_changed(self, value: int):
        """Parallel Mix slider → updates dry/wet mix parameter."""
        mix = value / 100.0
        self.chain.dynamics.single_band.parallel_mix = mix
        self.dyn_mix_val.setText(f"{value}%")
        self._schedule_auto_preview()

    def _on_dyn_preset_changed(self, preset: str):
        self.chain.dynamics.load_preset(preset)
        band = self.chain.dynamics.single_band
        for attr, knob in self.dyn_knobs.items():
            knob.blockSignals(True)
            knob.setValue(getattr(band, attr, knob.value()))
            knob.blockSignals(False)
        # Update curve widget from preset
        if hasattr(self, 'dyn_curve'):
            self.dyn_curve.blockSignals(True)
            self.dyn_curve.setThreshold(band.threshold)
            self.dyn_curve.setRatio(band.ratio)
            self.dyn_curve.setKnee(band.knee)
            self.dyn_curve.setAttack(band.attack)
            self.dyn_curve.setRelease(band.release)
            self.dyn_curve.setMakeup(band.makeup)
            self.dyn_curve.blockSignals(False)
        self._schedule_auto_preview()

    def _on_dyn_curve_threshold_changed(self, db: float):
        """Called when user drags threshold line on curve widget"""
        try:
            if 'threshold' in self.dyn_knobs:
                self.dyn_knobs['threshold'].blockSignals(True)
                self.dyn_knobs['threshold'].setValue(db)
                self.dyn_knobs['threshold'].blockSignals(False)
            if hasattr(self.chain, 'dynamics') and hasattr(self.chain.dynamics, 'single_band'):
                setattr(self.chain.dynamics.single_band, 'threshold', db)
            # V5.10: Forward to RT engine
            if self._rt_engine is not None and _HAS_RT_ENGINE:
                self._rt_engine.set_dyn_threshold(db)
            self._schedule_auto_preview()
        except Exception as e:
            print(f"[MASTER UI] _on_dyn_curve_threshold_changed error: {e}")

    def _on_dyn_curve_ratio_changed(self, ratio: float):
        """Called when user drags curve to adjust ratio"""
        try:
            if 'ratio' in self.dyn_knobs:
                self.dyn_knobs['ratio'].blockSignals(True)
                self.dyn_knobs['ratio'].setValue(ratio)
                self.dyn_knobs['ratio'].blockSignals(False)
            if hasattr(self.chain, 'dynamics') and hasattr(self.chain.dynamics, 'single_band'):
                setattr(self.chain.dynamics.single_band, 'ratio', ratio)
            # V5.10: Forward to RT engine
            if self._rt_engine is not None and _HAS_RT_ENGINE:
                self._rt_engine.set_dyn_ratio(ratio)
            self._schedule_auto_preview()
        except Exception as e:
            print(f"[MASTER UI] _on_dyn_curve_ratio_changed error: {e}")

    def _on_img_preset_changed(self, preset: str):
        self.chain.imager.load_preset(preset)
        self.img_width_slider.blockSignals(True)
        self.img_width_slider.setValue(self.chain.imager.width)
        self.img_width_slider.blockSignals(False)
        self.img_width_value.setText(f"{self.chain.imager.width}")
        self._schedule_auto_preview()

    def _on_img_width_changed(self, value: int):
        self.chain.imager.set_width(value)
        self.img_width_value.setText(f"{value}")
        # V5.10: Real-time engine — instant width change
        if self._rt_engine is not None:
            self._rt_engine.set_width(float(value))
        self._schedule_auto_preview()

    def _on_img_mono_bass_toggled(self, checked: bool):
        if checked:
            self.chain.imager.mono_bass_freq = self.img_mono_freq.value()
        else:
            self.chain.imager.mono_bass_freq = 0
        self._schedule_auto_preview()

    # ═══ MAXIMIZER HANDLERS — Ozone 12 Style ═══

    def _on_max_bypass_toggled(self, checked):
        """Toggle BYPASS on Maximizer header — switch between MASTERED / ORIGINAL."""
        self._is_bypass_mode = checked
        self._on_transport_bypass(checked)
        if checked:
            self.max_bypass_btn.setText("● ORIGINAL")
            self.max_bypass_btn.setStyleSheet(
                f"QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #332200, stop:1 #221100); "
                f"border: 2px solid {C_AMBER}; border-radius: 5px; color: {C_AMBER_GLOW}; "
                f"font-size: 10px; font-weight: bold; font-family: 'SF Pro Display', 'Menlo', monospace; }}")
        else:
            self.max_bypass_btn.setText("● MASTERED")
            self.max_bypass_btn.setStyleSheet(
                f"QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1A3A1A, stop:1 #102A10); "
                f"border: 2px solid {C_LED_GREEN}; border-radius: 5px; color: {C_LED_GREEN}; "
                f"font-size: 10px; font-weight: bold; font-family: 'SF Pro Display', 'Menlo', monospace; }}")
        # Sync with A/B compare button if exists
        if hasattr(self, 'btn_ab'):
            if checked:
                self.btn_ab.setText("▶ ORIGINAL")
                self.btn_ab.setStyleSheet(
                    f"QPushButton {{ background: #332200; color: {C_AMBER_GLOW}; font-weight: bold; "
                    f"font-size: 10px; padding: 6px 12px; border-radius: 3px; border: 1px solid {C_AMBER}; }}")
            else:
                self.btn_ab.setText("A/B COMPARE")
                self.btn_ab.setStyleSheet(
                    f"QPushButton {{ background: #1A1A1E; color: {C_TEAL_GLOW}; font-weight: bold; "
                    f"font-size: 10px; padding: 6px 12px; border-radius: 3px; border: 1px solid {C_TEAL_DIM}; }}")

    def _show_irc_menu(self):
        """Show IRC mode dropdown menu (Ozone 12 style).
        6 modes: IRC 1-5 + IRC LL
        Sub-modes only on IRC 3 & IRC 4 (Pumping/Balanced/Crisp)"""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background: #12121A;
                border: 1px solid {C_TEAL_DIM};
                border-radius: 6px;
                color: {C_CREAM};
                font-family: 'SF Pro Display', 'Menlo', monospace;
                font-size: 11px;
                padding: 6px 2px;
            }}
            QMenu::item {{
                padding: 8px 16px;
                border-radius: 4px;
                margin: 1px 4px;
            }}
            QMenu::item:selected {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C_TEAL_DIM}, stop:1 {C_TEAL});
                color: #FFFFFF;
            }}
            QMenu::separator {{
                height: 1px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 transparent, stop:0.5 {C_TEAL_DIM}, stop:1 transparent);
                margin: 4px 12px;
            }}
        """)

        # IRC mode descriptions for menu items
        mode_icons = {
            "IRC 1": "◇",   # Transparent
            "IRC 2": "◈",   # Adaptive
            "IRC 3": "◆",   # Multi-band (most popular)
            "IRC 4": "⬥",   # Aggressive
            "IRC 5": "⬢",   # Maximum Density
            "IRC LL": "⚡",  # Low Latency
        }

        for mode_key in IRC_TOP_MODES:
            mode_data = IRC_MODES.get(mode_key, {})
            sub_modes = mode_data.get("sub_modes", [])
            icon = mode_icons.get(mode_key, "●")

            if sub_modes:
                sub_menu = menu.addMenu(f"{icon}  {mode_data.get('name', mode_key)}")
                sub_menu.setStyleSheet(menu.styleSheet())
                for sub in sub_modes:
                    sub_key = f"{mode_key} - {sub}"
                    sub_desc = IRC_MODES.get(sub_key, {}).get("description", "")
                    # Shorten description for menu
                    short_desc = sub_desc.split("—")[1].strip() if "—" in sub_desc else sub_desc.split(".")[0]
                    action = sub_menu.addAction(f"{sub}  ·  {short_desc}")
                    action.triggered.connect(
                        lambda checked, m=mode_key, s=sub: self._select_irc(m, s))
            else:
                # No sub-modes
                short_desc = mode_data.get("description", "").split("—")[1].strip() if "—" in mode_data.get("description", "") else ""
                action = menu.addAction(f"{icon}  {mode_data.get('name', mode_key)}  ·  {short_desc}")
                action.triggered.connect(
                    lambda checked, m=mode_key: self._select_irc(m, ""))

            # Add separator after IRC 2 and IRC 5
            if mode_key in ("IRC 2", "IRC 5"):
                menu.addSeparator()

        menu.exec(self.irc_mode_btn.mapToGlobal(
            self.irc_mode_btn.rect().bottomLeft()))

    def _select_irc(self, mode: str, sub_mode: str):
        """Apply selected IRC mode + sub-mode to engine and update UI."""
        self.chain.maximizer.set_irc_mode(mode, sub_mode if sub_mode else None)
        # V5.10: Real-time engine — instant IRC mode change
        if self._rt_engine is not None:
            self._rt_engine.set_irc_mode(mode)
        if sub_mode:
            self.irc_mode_btn.setText(f"{mode} — {sub_mode}  ▾")
            key = f"{mode} - {sub_mode}"
        else:
            display_name = IRC_MODES.get(mode, {}).get("name", mode)
            self.irc_mode_btn.setText(f"{display_name}  ▾")
            key = mode
        desc = IRC_MODES.get(key, IRC_MODES.get(mode, {})).get("description", "")
        self.irc_desc_label.setText(desc)

        # V5.8 A-1: Update IRC sub-mode dropdown
        mode_data = IRC_MODES.get(mode, {})
        sub_modes = mode_data.get("sub_modes", [])
        if sub_modes and hasattr(self, 'irc_submode_combo'):
            self.irc_submode_combo.blockSignals(True)
            self.irc_submode_combo.clear()
            self.irc_submode_combo.addItems(sub_modes)
            if sub_mode and sub_mode in sub_modes:
                self.irc_submode_combo.setCurrentText(sub_mode)
            self.irc_submode_combo.blockSignals(False)
            self.irc_submode_label.setVisible(True)
            self.irc_submode_combo.setVisible(True)
        elif hasattr(self, 'irc_submode_combo'):
            self.irc_submode_label.setVisible(False)
            self.irc_submode_combo.setVisible(False)

        print(f"[MAXIMIZER UI] IRC set: {mode} / {sub_mode}")
        self._schedule_auto_preview()

    def _on_tone_clicked(self, index: int):
        for i, btn in enumerate(self.tone_buttons):
            btn.setChecked(i == index)
        self._update_tone_button_styles()
        tone_names = list(TONE_PRESETS.keys())
        if 0 <= index < len(tone_names):
            self.chain.maximizer.tone = tone_names[index]
            self._schedule_auto_measure()
            self._schedule_auto_preview()

    def _update_tone_button_styles(self):
        for btn in self.tone_buttons:
            if btn.isChecked():
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {C_TEAL}; border: none; border-radius: 10px;
                        color: #FFFFFF; font-weight: bold; padding: 2px 6px;
                        font-family: 'Menlo'; font-size: 8px; letter-spacing: 0.5px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: #1A1A1E; border: 1px solid #2A2A30; border-radius: 10px;
                        color: #6B6B70; padding: 2px 6px;
                        font-family: 'Menlo'; font-size: 8px;
                    }}
                    QPushButton:hover {{ color: {C_TEAL}; border-color: {C_TEAL_DIM}; }}
                """)

    def _on_gain_changed(self, value: int):
        """Gain dial changed (0-200 → 0.0-20.0 dB). Auto-measure triggered.

        V5.5.1: Extended range from 12 dB → 20 dB for aggressive signal push.
        GAIN knob does TWO things:
        1. Sets chain.maximizer.gain_db for next offline render
        2. V5.10: Forwards to RT engine for INSTANT real-time feedback
        """
        gain_db = value / 10.0
        self.chain.maximizer.set_gain(gain_db)
        self.max_gain_display.setText(f"+{gain_db:.1f}")

        # V5.10: Real-time engine — instant parameter change (lock-free)
        if self._rt_engine is not None:
            self._rt_engine.set_gain(gain_db)
        else:
            # Fallback: Instant volume feedback via QAudioOutput
            volume = min(3.0, 0.5 + gain_db / 20.0 * 2.5)
            if hasattr(self, '_audio_output_master'):
                self._audio_output_master.setVolume(volume)
            if hasattr(self, '_audio_output_bypass'):
                self._audio_output_bypass.setVolume(volume)

        # Teal → Yellow → Orange → Red color based on gain push
        if gain_db < 6.0:
            color = C_TEAL_GLOW
        elif gain_db < 12.0:
            color = C_LED_YELLOW
        elif gain_db < 16.0:
            color = "#FF8C00"  # Orange
        else:
            color = C_LED_RED
        self.max_gain_display.setStyleSheet(f"color: {color};")
        self._schedule_auto_measure()
        self._schedule_auto_preview()

    def _on_ceiling_changed(self, value: float):
        self.chain.maximizer.set_ceiling(value)
        # V5.10: Real-time engine — instant ceiling change
        if self._rt_engine is not None:
            self._rt_engine.set_ceiling(value)
        self._schedule_auto_measure()
        self._schedule_auto_preview()

    def _on_character_changed(self, value: int):
        """Character slider changed (0-100 → 0.0-10.0)."""
        char_val = value / 10.0
        self.chain.maximizer.set_character(char_val)
        self.max_char_val.setText(f"{char_val:.2f}")
        self._schedule_auto_measure()
        self._schedule_auto_preview()

    def _on_learn_gain(self):
        """Run Learn Input Gain analysis (threaded)."""
        audio_path = getattr(self, '_current_audio_path', None)
        if not audio_path:
            self.max_lufs_label.setText("NO FILE")
            return

        self.btn_learn_gain.setText("ANALYZING...")
        self.btn_learn_gain.setEnabled(False)

        def _do_learn():
            suggested = self.chain.maximizer.learn_input_gain(audio_path)
            lufs = self.chain.maximizer.get_learned_lufs()

            def _update_ui():
                self.btn_learn_gain.setText("LEARN INPUT GAIN")
                self.btn_learn_gain.setEnabled(True)
                if lufs is not None:
                    self.max_lufs_label.setText(f"{lufs:.1f} LUFS")
                if suggested is not None:
                    if hasattr(self, 'max_gain_knob'):
                        self.max_gain_knob.setValue(suggested)
                    elif hasattr(self, 'max_gain_dial'):
                        self.max_gain_dial.setValue(int(suggested * 10))
                    print(f"[MAXIMIZER UI] Learned: {lufs:.1f} LUFS → suggested gain +{suggested:.1f} dB")

            QTimer.singleShot(0, _update_ui)

        threading.Thread(target=_do_learn, daemon=True).start()

    def _auto_learn_input_gain(self, audio_path: str):
        """V5.5: Auto-analyze LUFS when audio is loaded (background thread).
        Updates the LUFS label next to LEARN INPUT GAIN automatically.
        Does NOT change the Gain dial — only shows the measured LUFS."""
        def _analyze():
            try:
                import soundfile as sf_mod
                import pyloudnorm as pyln_mod
                data, sr = sf_mod.read(audio_path, dtype='float64')
                if data.ndim == 1:
                    data = np.column_stack([data, data])
                meter = pyln_mod.Meter(sr)
                lufs = meter.integrated_loudness(data)
                if lufs == float('-inf') or lufs < -70:
                    lufs = -70.0
                self.chain.maximizer._learned_lufs = lufs

                def _update():
                    if hasattr(self, 'max_lufs_label'):
                        self.max_lufs_label.setText(f"{lufs:.1f} LUFS")
                        print(f"[MAXIMIZER] Auto-LUFS: {lufs:.1f} LUFS ({os.path.basename(audio_path)})")
                QTimer.singleShot(0, _update)
            except Exception as e:
                print(f"[MAXIMIZER] Auto-LUFS error: {e}")
                def _fallback():
                    if hasattr(self, 'max_lufs_label'):
                        self.max_lufs_label.setText("— LUFS")
                QTimer.singleShot(0, _fallback)

        threading.Thread(target=_analyze, daemon=True).start()

    def _on_softclip_toggled(self, checked: bool):
        pct = getattr(self.chain.maximizer, 'soft_clip_pct', 50)
        self.chain.maximizer.set_soft_clip(checked, pct)
        self._schedule_auto_preview()

    def _on_softclip_pct_changed(self, value: int):
        enabled = getattr(self.chain.maximizer, 'soft_clip', False)
        self.chain.maximizer.set_soft_clip(enabled, value)
        self._schedule_auto_preview()

    def _on_transient_changed(self, value: int):
        self.chain.maximizer.set_transient_emphasis(value, self.chain.maximizer.transient_band)
        self._schedule_auto_preview()

    def _on_transient_band(self, band: str):
        for btn in self.max_band_btns:
            btn.setChecked(btn.text() == band)
        self._update_band_button_styles()
        current_pct = getattr(self.chain.maximizer, 'transient_emphasis_pct', 0)
        self.chain.maximizer.set_transient_emphasis(current_pct, band)
        self._schedule_auto_preview()

    def _update_band_button_styles(self):
        for btn in self.max_band_btns:
            if btn.isChecked():
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {C_TEAL_DIM}; border: 1px solid {C_TEAL};
                        border-radius: 3px; color: #FFFFFF; font-weight: bold; font-size: 9px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: #1A1A1E; border: 1px solid #2A2A30;
                        border-radius: 3px; color: #6B6B70; font-size: 9px;
                    }}
                    QPushButton:hover {{ color: {C_TEAL}; }}
                """)

    def _on_stereo_ind_changed(self):
        trans = getattr(self.chain.maximizer, 'stereo_ind_transient', 50)
        sust = getattr(self.chain.maximizer, 'stereo_ind_sustain', 50)
        self.chain.maximizer.set_stereo_independence(trans, sust)
        self._schedule_auto_measure()
        self._schedule_auto_preview()

    def _run_auto_measure(self):
        """Auto-measure triggered by parameter change (runs in background thread)."""
        audio_path = getattr(self, '_current_audio_path', None)
        if not audio_path:
            return
        # Avoid running multiple measures simultaneously
        if getattr(self, '_measuring', False):
            self._schedule_auto_measure()  # re-queue
            return
        self._measuring = True

        def _do():
            result = self.chain.maximizer.measure_levels(audio_path)

            def _update_ui():
                self._measuring = False
                if result:
                    self._apply_meter_result(result)
            QTimer.singleShot(0, _update_ui)

        threading.Thread(target=_do, daemon=True).start()

    def _apply_meter_result(self, result):
        """Apply measurement results to the Ozone meter + GR display."""
        l_peak = result.get("l_peak", result["peak_db"])
        r_peak = result.get("r_peak", result["peak_db"])
        peak = result["peak_db"]
        gr = result["gain_reduction"]

        # Peak dB text with color
        if peak > -1.0:
            col = C_LED_RED
        elif peak > -6.0:
            col = C_LED_YELLOW
        else:
            col = C_TEAL_GLOW
        self.max_meter_db.setStyleSheet(
            f"color:{col}; font-family:'Menlo'; font-size:9px; font-weight:bold;")
        self.max_meter_db.setText(f"{peak:.1f} dB")

        # Gain reduction — update history widget
        if hasattr(self, 'max_gr_history'):
            self.max_gr_history.set_gr(gr)

        print(f"[MAXIMIZER UI] Measured: L={l_peak:.1f} R={r_peak:.1f} peak={peak:.1f}dB GR={gr:.1f}dB")

    def _on_measure_levels(self):
        """Manual measure button — same as auto but with button feedback."""
        audio_path = getattr(self, '_current_audio_path', None)
        if not audio_path:
            self.max_meter_db.setText("NO FILE")
            return

        self.btn_measure.setText("⟳  MEASURING...")
        self.btn_measure.setEnabled(False)

        def _do_measure():
            result = self.chain.maximizer.measure_levels(audio_path)

            def _update_ui():
                self.btn_measure.setText("⟳  MEASURE LEVELS")
                self.btn_measure.setEnabled(True)
                if result:
                    self._apply_meter_result(result)
                else:
                    self.max_meter_db.setText("ERR")

            QTimer.singleShot(0, _update_ui)

        threading.Thread(target=_do_measure, daemon=True).start()

    # ═══════════════════════════════════════════
    #  Actions
    # ═══════════════════════════════════════════

    def set_audio(self, path: str):
        """Set audio file to master (called from main app or browse)."""
        if not path:
            print("[MASTER PANEL] set_audio called with empty path")
            self.file_label.setText("❌ No path provided")
            self.file_label.setStyleSheet(f"color: #FF4444; font-size: 11px; font-weight: bold;")
            return False

        print(f"[MASTER PANEL] set_audio called with: {path}")
        print(f"[MASTER PANEL] os.path.exists = {os.path.exists(path)}")

        if not os.path.exists(path):
            print(f"[MASTER PANEL] ❌ File not found: {path}")
            self.file_label.setText(f"❌ NOT FOUND: {os.path.basename(path)}")
            self.file_label.setStyleSheet(f"color: #FF4444; font-size: 11px; font-weight: bold;")
            self.progress_text.setText(f"ERROR: File not found — use BROWSE button")
            self.meters.set_status("FILE NOT FOUND", C_LED_RED)
            return False

        try:
            load_result = self.chain.load_audio(path)
            print(f"[MASTER PANEL] chain.load_audio result = {load_result}")

            if load_result:
                self._current_audio_path = path
                self.file_label.setText(os.path.basename(path))
                self.file_label.setStyleSheet(f"color: {C_AMBER}; font-size: 11px; font-weight: bold;")
                self.progress_text.setText(f"LOADED: {os.path.basename(path)}")
                self.meters.set_status("READY", C_LED_GREEN)
                print(f"[MASTER PANEL] ✅ Audio loaded: {path}")

                # V5.5: Auto-analyze LUFS on load (like Ozone 12 Learn Input Gain)
                self._auto_learn_input_gain(path)

                # V5.10: Load into RT engine for real-time DSP playback
                if self._rt_engine is not None:
                    try:
                        self._rt_engine.load_file(path)
                        self._rt_playback_active = True
                        print(f"[MASTER] RT engine loaded: {os.path.basename(path)}")
                    except Exception as e:
                        print(f"[MASTER] RT engine load failed: {e}")
                        self._rt_playback_active = False

                return True
            else:
                print(f"[MASTER PANEL] ❌ chain.load_audio returned False for: {path}")
                self.file_label.setText(f"❌ LOAD FAILED: {os.path.basename(path)}")
                self.file_label.setStyleSheet(f"color: #FF4444; font-size: 11px; font-weight: bold;")
                self.progress_text.setText(f"ERROR: Could not load audio — use BROWSE button")
                self.meters.set_status("LOAD FAILED", C_LED_RED)
                return False
        except Exception as e:
            print(f"[MASTER PANEL] ❌ Exception in set_audio: {e}")
            import traceback
            traceback.print_exc()
            self.file_label.setText(f"❌ ERROR: {e}")
            self.file_label.setStyleSheet(f"color: #FF4444; font-size: 11px; font-weight: bold;")
            return False

    def _on_browse_audio(self):
        """Browse for audio file manually."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a *.ogg);;All Files (*)"
        )
        if path:
            self.set_audio(path)

    def _on_analyze(self):
        if not self.chain.input_path:
            QMessageBox.warning(self, "No Audio", "Please load audio first")
            return

        self.btn_analyze.setEnabled(False)
        self.progress_text.setText("ANALYZING...")
        self.meters.set_status("ANALYZING...", C_LED_BLUE)

        def _do():
            try:
                analysis = self.chain.loudness_meter.analyze(self.chain.input_path)
                if analysis:
                    self.chain.input_analysis = analysis
                    self._call_on_main.emit(lambda a=analysis: self._on_analyze_done(a))
                else:
                    self._call_on_main.emit(lambda: self._on_analyze_done(None))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._call_on_main.emit(lambda: self._on_analyze_done(None))

        threading.Thread(target=_do, daemon=True).start()

    def _on_analyze_done(self, analysis):
        try:
            self.btn_analyze.setEnabled(True)
            if analysis and hasattr(analysis, 'integrated_lufs'):
                self.meters.update_input(
                    getattr(analysis, 'integrated_lufs', -24.0),
                    getattr(analysis, 'true_peak_dbtp', -6.0),
                    getattr(analysis, 'lra', 0.0),
                )
                self.progress_text.setText("ANALYSIS COMPLETE")
                self.meters.set_status("ANALYZED", C_LED_GREEN)
            else:
                self.progress_text.setText("ANALYSIS FAILED")
                self.meters.set_status("FAILED", C_LED_RED)
        except Exception as e:
            print(f"[MASTER UI] _on_analyze_done error: {e}")
            self.progress_text.setText("ANALYSIS ERROR")
            self.meters.set_status("ERROR", C_LED_RED)

    def _on_preview(self):
        if not self.chain.input_path:
            QMessageBox.warning(self, "No Audio", "Please load audio first")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._set_buttons_enabled(False)

        # V5.5: Start meter timer + switch to LIVE mode for rendering progress
        self._start_meter_timer()
        self.meters.set_status("⏳ RENDERING...", C_LED_BLUE)
        if hasattr(self.meters, 'show_live_mode'):
            self.meters.show_live_mode()

        self.worker = MasterWorker(self.chain, "preview")
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_preview_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_render(self):
        if not self.chain.input_path:
            QMessageBox.warning(self, "No Audio", "Please load audio first")
            return

        # V5.5: Ask user to select output folder
        default_dir = os.path.dirname(self.chain.input_path)
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Mastered File",
            default_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        if not output_dir:
            return  # User cancelled

        # Set output path in chosen directory
        base_name = os.path.splitext(os.path.basename(self.chain.input_path))[0]
        self.chain.output_path = os.path.join(output_dir, f"{base_name}_mastered.wav")

        reply = QMessageBox.question(
            self, "Master & Export",
            f"Master and export?\n\n"
            f"File: {os.path.basename(self.chain.input_path)}\n"
            f"Output: {self.chain.output_path}\n"
            f"Platform: {self.chain.platform}\n"
            f"Target: {self.chain.target_lufs} LUFS / {self.chain.target_tp} dBTP\n"
            f"Intensity: {self.chain.intensity}%",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._set_buttons_enabled(False)

        # V5.5: Start meter timer + switch to LIVE mode for rendering progress
        self._start_meter_timer()
        self.meters.set_status("⏳ RENDERING...", C_LED_BLUE)
        if hasattr(self.meters, 'show_live_mode'):
            self.meters.show_live_mode()

        self.worker = MasterWorker(self.chain, "render")
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_render_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    # V5.0 FIX: Removed orphaned _on_ai_assist(), _on_ai_done(), _on_ai_apply()
    # These methods referenced UI elements (btn_ai_start, ai_results_frame, etc.)
    # that were removed during refactor to one-button master flow.
    # The new flow uses _on_one_button_master() instead.

    def _on_ab_compare(self):
        """V5.8 E-1: Toggle A/B compare — switch between ORIGINAL and MASTERED playback."""
        self._is_bypass_mode = not self._is_bypass_mode
        # Sync Maximizer bypass button (avoid recursive signal)
        if hasattr(self, 'max_bypass_btn'):
            self.max_bypass_btn.blockSignals(True)
            self.max_bypass_btn.setChecked(self._is_bypass_mode)
            self.max_bypass_btn.blockSignals(False)
            self._on_max_bypass_toggled(self._is_bypass_mode)
        else:
            self._on_transport_bypass(self._is_bypass_mode)
            if self._is_bypass_mode:
                self.btn_ab.setText("▶ ORIGINAL")
                self.btn_ab.setStyleSheet(
                    f"QPushButton {{ background: #332200; color: {C_AMBER_GLOW}; font-weight: bold; "
                    f"font-size: 10px; padding: 6px 12px; border-radius: 3px; border: 1px solid {C_AMBER}; }}")
            else:
                self.btn_ab.setText("A/B COMPARE")
                self.btn_ab.setStyleSheet(
                    f"QPushButton {{ background: #1A1A1E; color: {C_TEAL_GLOW}; font-weight: bold; "
                    f"font-size: 10px; padding: 6px 12px; border-radius: 3px; border: 1px solid {C_TEAL_DIM}; }}")

    def _on_progress(self, percent, status):
        if percent >= 0:
            self.progress_bar.setValue(percent)
        self.progress_text.setText(status.upper())

    def _on_preview_done(self, path):
        # V5.5 FIX: Stop realtime meter timer after preview
        self._stop_meter_timer()
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)
        self.progress_text.setText(f"PREVIEW READY: {os.path.basename(path)}")
        self.meters.set_status("PREVIEW OK", C_LED_GREEN)
        # V5.2: Auto-load for playback
        try:
            self._load_preview_for_playback(path)
            # V5.9: Auto-play after preview completes
            self._on_transport_play()
        except Exception as pe:
            print(f"[TRANSPORT] Preview load error (non-fatal): {pe}")

    def _on_render_done(self, path):
        # V5.4 FIX: Stop realtime meter timer
        self._stop_meter_timer()

        # V5.4: Handle batch mode — continue to next track
        if self.batch_mode:
            # Record success
            result_entry = {
                "path": path,
                "success": True,
                "lufs": getattr(self.chain.output_analysis, 'integrated_lufs', None) if self.chain.output_analysis else None,
                "tp": getattr(self.chain.output_analysis, 'true_peak_dbtp', None) if self.chain.output_analysis else None,
                "error": None,
            }
            self._batch_results.append(result_entry)
            self.current_track_index += 1
            self._batch_process_next()
            return

        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)
        self.btn_ab.setEnabled(True)

        if self.chain.output_analysis:
            self.meters.update_output(
                self.chain.output_analysis.integrated_lufs,
                self.chain.output_analysis.true_peak_dbtp,
                self.chain.output_analysis.lra,
            )
            self.progress_text.setText(
                f"MASTERED: {self.chain.output_analysis.integrated_lufs:.1f} LUFS"
            )
        else:
            self.progress_text.setText(f"MASTERED: {os.path.basename(path)}")

        self.meters.set_status("COMPLETE", C_LED_GREEN)
        self._update_tonal_balance(path)
        self.master_complete.emit(path)

        QMessageBox.information(self, "Mastering Complete",
                                f"Master complete:\n{path}")

    def _on_error(self, error_msg):
        # V5.4 FIX: Stop realtime meter timer on error
        self._stop_meter_timer()

        # V5.4: Handle batch mode errors — record failure and continue
        if self.batch_mode:
            current_path = self.track_queue[self.current_track_index] if self.current_track_index < len(self.track_queue) else "unknown"
            self._batch_results.append({
                "path": current_path,
                "success": False,
                "lufs": None,
                "tp": None,
                "error": error_msg,
            })
            self.current_track_index += 1
            self._batch_process_next()
            return

        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)
        self.progress_text.setText(f"ERROR: {error_msg}")
        self.meters.set_status("ERROR", C_LED_RED)
        QMessageBox.critical(self, "Error", f"Error:\n{error_msg}")

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_analyze.setEnabled(enabled)
        self.btn_preview.setEnabled(enabled)
        self.btn_render.setEnabled(enabled)
        # V5.0 FIX: btn_ai_start was removed in refactor, use btn_master instead
        if hasattr(self, 'btn_master'):
            self.btn_master.setEnabled(enabled)

    # ══════════════════════════════════════════════════
    #  V5.2: Preview Playback + Bypass Methods
    # ══════════════════════════════════════════════════
    def _load_preview_for_playback(self, preview_path):
        """Load mastered preview and original audio into players."""
        try:
            if not preview_path or not os.path.exists(preview_path):
                print(f"[TRANSPORT] Preview path invalid: {preview_path}")
                return

            self._preview_path = preview_path
            self._original_path = self.chain.input_path

            # Stop any current playback
            self._preview_player.stop()
            self._bypass_player.stop()

            # Load mastered preview
            self._preview_player.setSource(QUrl.fromLocalFile(preview_path))

            # Load original audio for bypass
            if self._original_path and os.path.exists(self._original_path):
                self._bypass_player.setSource(QUrl.fromLocalFile(self._original_path))
            else:
                print(f"[TRANSPORT] Original path not available for bypass")

            self._is_preview_loaded = True
            self._is_bypass_mode = False

            # V5.5.1: Transport bar removed — no UI transport to update

            print(f"[TRANSPORT] Preview loaded: {os.path.basename(preview_path)}")
        except Exception as e:
            print(f"[TRANSPORT] Load error: {e}")
            import traceback
            traceback.print_exc()

    def _get_active_player(self):
        """Return the currently active QMediaPlayer (mastered or bypass)."""
        return self._bypass_player if self._is_bypass_mode else self._preview_player

    def _get_inactive_player(self):
        """Return the inactive QMediaPlayer."""
        return self._preview_player if self._is_bypass_mode else self._bypass_player

    def _on_transport_play(self):
        """V5.10: Play via RT engine (instant DSP) or shared player or QMediaPlayer."""
        # V5.10: Prefer real-time engine for mastered playback
        if self._rt_engine is not None and self._rt_playback_active:
            self._rt_engine.play()
            self.meters.set_status("▶ RT PLAYING", C_LED_GREEN)
            if hasattr(self.meters, 'show_live_mode'):
                self.meters.show_live_mode()
            self._start_meter_timer("STAGE: ▶ RT LIVE")
            return
        if self._shared_audio_player:
            self._shared_audio_player.play()
            self.meters.set_status("▶ PLAYING", C_LED_GREEN)
            if hasattr(self.meters, 'show_live_mode'):
                self.meters.show_live_mode()
            self._start_meter_timer("STAGE: ▶ LIVE")
            return
        if not self._is_preview_loaded:
            return
        player = self._get_active_player()
        inactive = self._get_inactive_player()
        inactive.pause()
        player.play()
        self.meters.set_status("▶ PLAYING", C_LED_GREEN)

    def _on_transport_pause(self):
        """Pause playback."""
        if self._rt_engine is not None and self._rt_playback_active:
            self._rt_engine.pause()
            self.meters.set_status("⏸ PAUSED", C_AMBER)
            return
        if self._shared_audio_player:
            self._shared_audio_player.pause()
            self.meters.set_status("⏸ PAUSED", C_AMBER)
            return
        self._preview_player.pause()
        self._bypass_player.pause()
        self.meters.set_status("⏸ PAUSED", C_AMBER)

    def _on_transport_stop(self):
        """Stop playback and reset position."""
        # V5.10: Reset RT LUFS buffers on stop
        if hasattr(self, '_rt_lufs_buf'):
            self._rt_lufs_buf.clear()
        if hasattr(self, '_rt_lufs_int_acc'):
            self._rt_lufs_int_acc.clear()
        if self._rt_engine is not None and self._rt_playback_active:
            self._rt_engine.stop()
            self.meters.set_status("⏹ STOPPED", C_AMBER_DIM)
            return
        if self._shared_audio_player:
            self._shared_audio_player.stop()
            self.meters.set_status("⏹ STOPPED", C_AMBER_DIM)
            return
        self._preview_player.stop()
        self._bypass_player.stop()
        self.meters.set_status("⏹ STOPPED", C_AMBER_DIM)

    def _on_transport_seek(self, position_ms):
        """Seek to position."""
        if self._rt_engine is not None and self._rt_playback_active:
            self._rt_engine.seek(position_ms)
            return
        self._preview_player.setPosition(position_ms)
        self._bypass_player.setPosition(position_ms)

    def _on_transport_bypass(self, is_bypassed):
        """Toggle between MASTERED and ORIGINAL audio (A/B Compare)."""
        was_playing = (self._get_active_player().playbackState() ==
                       QMediaPlayer.PlaybackState.PlayingState)
        current_pos = self._get_active_player().position()
        self._get_active_player().pause()
        self._is_bypass_mode = is_bypassed
        new_player = self._get_active_player()
        new_player.setPosition(current_pos)
        if was_playing:
            new_player.play()
        mode_str = "● ORIGINAL" if is_bypassed else "● MASTERED"
        print(f"[TRANSPORT] Switched to {mode_str}")
        self.meters.set_status(
            "● ORIGINAL" if is_bypassed else "● MASTERED",
            C_AMBER if is_bypassed else C_LED_GREEN)

    def _on_player_position(self, position):
        """Update from player position signal (V5.5.1: transport bar removed)."""
        pass  # Transport bar removed — position tracking not needed in MasterPanel

    def _on_player_duration(self, duration):
        """Update from player duration (V5.5.1: transport bar removed)."""
        pass  # Transport bar removed — duration tracking not needed in MasterPanel

    def _on_player_status(self, status):
        """Handle end of media and errors (V5.5.1: transport bar removed)."""
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._preview_player.stop()
            self._bypass_player.stop()
            self.meters.set_status("⏹ ENDED", C_AMBER_DIM)

    def keyPressEvent(self, event):
        """V5.9 Comprehensive keyboard shortcuts for mastering workflow."""
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if key == Qt.Key.Key_Escape:
            self.close()
        # ── Playback ──
        elif key == Qt.Key.Key_Space:
            self._on_preview()
        # ── A/B Compare ──
        elif key == Qt.Key.Key_B:
            if self.btn_ab.isEnabled():
                self._on_ab_compare()
        # ── Undo / Redo ──
        elif key == Qt.Key.Key_Z and ctrl:
            self._redo() if shift else self._undo()
        # ── Render / Export ──
        elif key == Qt.Key.Key_R and ctrl:
            if self.btn_render.isEnabled():
                self._on_render()
        # ── Master Assistant (AI) ──
        elif key == Qt.Key.Key_A and ctrl:
            if hasattr(self, 'btn_ai_master') and self.btn_ai_master.isEnabled():
                self._on_ai_master()
        # ── Module navigation (1-5 keys) ──
        elif key == Qt.Key.Key_1:
            self._select_module(0)  # EQ
        elif key == Qt.Key.Key_2:
            self._select_module(1)  # Dynamics
        elif key == Qt.Key.Key_3:
            self._select_module(2)  # Imager
        elif key == Qt.Key.Key_4:
            self._select_module(3)  # Maximizer
        elif key == Qt.Key.Key_5:
            self._select_module(4)  # AI
        # ── Bypass toggle ──
        elif key == Qt.Key.Key_0:
            self._toggle_bypass()
        # ── Preset navigation ──
        elif key == Qt.Key.Key_BracketLeft:
            self._prev_preset()
        elif key == Qt.Key.Key_BracketRight:
            self._next_preset()
        # ── Intensity +/- ──
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            if hasattr(self, 'intensity_slider'):
                v = min(100, self.intensity_slider.value() + 5)
                self.intensity_slider.setValue(v)
        elif key == Qt.Key.Key_Minus:
            if hasattr(self, 'intensity_slider'):
                v = max(0, self.intensity_slider.value() - 5)
                self.intensity_slider.setValue(v)
        else:
            super().keyPressEvent(event)

    def _select_module(self, index: int):
        """Select a module tab by index."""
        if hasattr(self, 'module_stack') and 0 <= index < self.module_stack.count():
            self.module_stack.setCurrentIndex(index)
            if hasattr(self, 'module_btns') and index < len(self.module_btns):
                for i, btn in enumerate(self.module_btns):
                    btn.setChecked(i == index)

    def _toggle_bypass(self):
        """Toggle all processing bypass."""
        self.chain.bypass = not getattr(self.chain, 'bypass', False)
        state = "BYPASSED" if self.chain.bypass else "ACTIVE"
        self.meters.set_status(f"CHAIN {state}", C_AMBER if self.chain.bypass else C_LED_GREEN)
        self._schedule_auto_preview()

    def _prev_preset(self):
        """Navigate to previous mastering preset."""
        if hasattr(self, 'preset_combo'):
            idx = max(0, self.preset_combo.currentIndex() - 1)
            self.preset_combo.setCurrentIndex(idx)

    def _next_preset(self):
        """Navigate to next mastering preset."""
        if hasattr(self, 'preset_combo'):
            idx = min(self.preset_combo.count() - 1, self.preset_combo.currentIndex() + 1)
            self.preset_combo.setCurrentIndex(idx)

    def _on_auto_detect_genre(self):
        """V5.8 D-3: Auto-detect genre from first 5 seconds of audio."""
        audio_path = getattr(self, '_current_audio_path', None) or self.chain.input_path
        if not audio_path:
            self.ai_detected_label.setText("No audio loaded")
            return

        self.btn_auto_detect.setText("ANALYZING...")
        self.btn_auto_detect.setEnabled(False)

        def _detect():
            try:
                import soundfile as sf
                import numpy as np

                # Read only first 5 seconds for genre detection (avoid loading entire file)
                info = sf.info(audio_path)
                sr = info.samplerate
                frames_5s = sr * 5
                data, sr = sf.read(audio_path, dtype='float64', frames=frames_5s)
                n_samples = len(data)
                data = data[:n_samples]

                if data.ndim > 1:
                    mono = data.mean(axis=1)
                else:
                    mono = data

                # Simple spectral analysis for genre detection
                fft = np.abs(np.fft.rfft(mono[:4096] * np.hanning(min(4096, len(mono)))))
                freqs = np.fft.rfftfreq(4096, 1.0 / sr)

                # Energy in bass (20-200 Hz), mid (200-2000 Hz), high (2000-20000 Hz)
                bass = np.mean(fft[(freqs >= 20) & (freqs < 200)])
                mid = np.mean(fft[(freqs >= 200) & (freqs < 2000)])
                high = np.mean(fft[(freqs >= 2000) & (freqs < 20000)])
                total = bass + mid + high + 1e-10

                bass_ratio = bass / total
                high_ratio = high / total

                # RMS level
                rms = np.sqrt(np.mean(mono ** 2))
                rms_db = 20 * np.log10(max(rms, 1e-10))

                # Simple heuristic genre detection
                if rms_db > -8 and bass_ratio > 0.5:
                    genre = "EDM"
                elif rms_db > -10 and bass_ratio > 0.4:
                    genre = "Hip-Hop"
                elif high_ratio > 0.3 and rms_db > -12:
                    genre = "Pop"
                elif rms_db < -16:
                    genre = "Ambient"
                elif bass_ratio > 0.35:
                    genre = "Rock"
                elif high_ratio > 0.25:
                    genre = "Jazz"
                else:
                    genre = "All-Purpose Mastering"

                confidence = min(85, int(50 + abs(rms_db) * 2))

                def _update():
                    self.btn_auto_detect.setText("AUTO-DETECT GENRE")
                    self.btn_auto_detect.setEnabled(True)
                    self.ai_detected_label.setText(f"{genre} ({confidence}%)")
                    # Set the combo box
                    idx = self.ai_genre.findText(genre)
                    if idx >= 0:
                        self.ai_genre.setCurrentIndex(idx)

                self._call_on_main.emit(_update)

            except Exception as e:
                def _err():
                    self.btn_auto_detect.setText("AUTO-DETECT GENRE")
                    self.btn_auto_detect.setEnabled(True)
                    self.ai_detected_label.setText(f"Error: {e}")
                self._call_on_main.emit(_err)

        threading.Thread(target=_detect, daemon=True).start()

    def _on_one_button_master(self):
        """One-button master: analyze → recommend → apply → preview."""
        if not self.chain.input_path:
            QMessageBox.warning(self, "No Audio", "Please load audio first")
            return

        genre = self.ai_genre.currentText()
        platform_names = list(PLATFORM_TARGETS.keys())
        platform_idx = self.platform_combo_ai.currentIndex()
        platform = platform_names[platform_idx] if 0 <= platform_idx < len(platform_names) else "YouTube"
        intensity = self.ai_intensity.value()

        # Disable button, show progress
        self.btn_master.setEnabled(False)
        self.btn_master.setText("⚡ MASTERING...")
        self.master_progress.setVisible(True)
        self.master_progress.setValue(0)
        self.master_status.setText("ANALYZING AUDIO...")
        self.master_status.setStyleSheet(f"color:{C_LED_BLUE}; font-size:11px; font-weight:bold;")

        # Hide post-mastering controls during processing
        self.post_master_frame.setVisible(False)
        self.master_results_frame.setVisible(False)

        def _do_master():
            try:
                # Step 1: Analyze + Recommend (30%) — non-fatal if fails
                self._call_on_main.emit(lambda: self._update_master_progress(10, "ANALYZING AUDIO..."))

                rec = None
                try:
                    rec = self.chain.ai_recommend(genre, platform, intensity)
                except Exception as e:
                    print(f"[MASTER] ai_recommend failed (non-fatal): {e}")

                if rec:
                    self._last_master_rec = rec
                    # Step 2: Apply recommendations
                    self._call_on_main.emit(lambda: self._update_master_progress(40, "APPLYING SETTINGS..."))
                    try:
                        self.chain.apply_recommendation(rec)
                    except Exception as e:
                        print(f"[MASTER] apply_recommendation failed (non-fatal): {e}")
                else:
                    # V5.8: Skip AI recommend — use current manual settings
                    print(f"[MASTER] AI recommend unavailable — using current settings")
                    self._call_on_main.emit(lambda: self._update_master_progress(40, "USING CURRENT SETTINGS..."))

                # Step 3: Render 30s preview (100%)
                self._call_on_main.emit(lambda: self._update_master_progress(60, "RENDERING PREVIEW..."))
                preview_path = None
                try:
                    preview_path = self.chain.preview()
                except Exception as e:
                    print(f"[MASTER] Preview error: {e}")

                if not preview_path:
                    # Try render instead of preview
                    try:
                        import tempfile
                        preview_path = os.path.join(tempfile.gettempdir(), "longplay_preview.wav")
                        result = self.chain.render(output_path=preview_path)
                        if not result or not os.path.exists(preview_path):
                            preview_path = None
                    except Exception as e:
                        print(f"[MASTER] Fallback render error: {e}")

                if not preview_path:
                    self._call_on_main.emit(lambda: self._on_master_error("Render failed — check audio file"))
                    return

                # V5.2 FIX: Analyze LUFS before/after so meters can display values
                self._call_on_main.emit(lambda: self._update_master_progress(85, "ANALYZING LOUDNESS..."))
                try:
                    self.chain.input_analysis = self.chain.loudness_meter.analyze(self.chain.input_path)
                    self.chain.output_analysis = self.chain.loudness_meter.analyze(preview_path)
                    print(f"[MASTER] Input LUFS: {self.chain.input_analysis.integrated_lufs:.1f}" if self.chain.input_analysis else "[MASTER] Input analysis failed")
                    print(f"[MASTER] Output LUFS: {self.chain.output_analysis.integrated_lufs:.1f}" if self.chain.output_analysis else "[MASTER] Output analysis failed")
                except Exception as e:
                    print(f"[MASTER] LUFS analysis error (non-fatal): {e}")

                self._call_on_main.emit(lambda r=rec, p=preview_path: self._on_master_done(r, p))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._call_on_main.emit(lambda msg=str(e): self._on_master_error(msg))

        threading.Thread(target=_do_master, daemon=True).start()

    def _update_master_progress(self, percent, text):
        """Update master progress bar and status."""
        self.master_progress.setValue(percent)
        self.master_status.setText(text)

    def _on_master_done(self, rec, preview_path):
        """Show results + reveal post-mastering controls."""
        try:
            # V5.1 FIX: Guard against widget deletion (thread callback race)
            if not self.isVisible():
                print("[MASTER UI] Widget not visible, skipping _on_master_done")
                return

            # V5.8: rec can be None if AI recommend was skipped — that's OK

            # Analyze output to get before/after LUFS — V5.1 FIX: safe attribute access
            before_lufs = None
            after_lufs = None
            try:
                if self.chain.input_analysis and hasattr(self.chain.input_analysis, 'integrated_lufs'):
                    before_lufs = self.chain.input_analysis.integrated_lufs
                if self.chain.output_analysis and hasattr(self.chain.output_analysis, 'integrated_lufs'):
                    after_lufs = self.chain.output_analysis.integrated_lufs
            except Exception:
                pass

            # Update results display
            if before_lufs is not None:
                self.master_lufs_before.setText(f"BEFORE: {before_lufs:.1f} LUFS")
            if after_lufs is not None:
                self.master_lufs_after.setText(f"AFTER: {after_lufs:.1f} LUFS")

            confidence = rec.confidence if rec and hasattr(rec, 'confidence') else 0
            self.master_confidence.setText(f"CONFIDENCE: {confidence}% ✓ MEETS TARGET" if confidence else "✓ MASTERED")

            # V5.2: Update right-side Meters Panel with before/after LUFS
            try:
                if before_lufs is not None and hasattr(self, 'meters'):
                    in_tp = getattr(self.chain.input_analysis, 'true_peak_dbtp', 0.0) if self.chain.input_analysis else 0.0
                    in_lra = getattr(self.chain.input_analysis, 'lra', 0.0) if self.chain.input_analysis else 0.0
                    self.meters.update_input(before_lufs, in_tp, in_lra)
                if after_lufs is not None and hasattr(self, 'meters'):
                    out_tp = getattr(self.chain.output_analysis, 'true_peak_dbtp', 0.0) if self.chain.output_analysis else 0.0
                    out_lra = getattr(self.chain.output_analysis, 'lra', 0.0) if self.chain.output_analysis else 0.0
                    self.meters.update_output(after_lufs, out_tp, out_lra)
                    # Update gain delta
                    if before_lufs is not None:
                        delta = after_lufs - before_lufs
                        sign = "+" if delta >= 0 else ""
                        self.meters.gain_row[1].setText(f"{sign}{delta:.1f} dB")
            except Exception as me:
                print(f"[MASTER UI] Meters update error (non-fatal): {me}")

            # Show results
            self.master_results_frame.setVisible(True)

            # Show post-mastering controls
            self.post_master_frame.setVisible(True)

            # Reset sliders
            self.post_eq_slider.blockSignals(True)
            self.post_eq_slider.setValue(100)
            self.post_eq_slider.blockSignals(False)

            self.post_dyn_slider.blockSignals(True)
            self.post_dyn_slider.setValue(100)
            self.post_dyn_slider.blockSignals(False)

            self.post_width_slider.blockSignals(True)
            self.post_width_slider.setValue(100)
            self.post_width_slider.blockSignals(False)

            self.post_ceiling_slider.blockSignals(True)
            # V5.0 FIX: Safe index access to prevent IndexError
            _pt_vals = list(PLATFORM_TARGETS.values())
            _pt_idx = self.platform_combo_ai.currentIndex()
            target_ceiling = _pt_vals[_pt_idx]['true_peak'] if 0 <= _pt_idx < len(_pt_vals) else -1.0
            self.post_ceiling_slider.setValue(int(target_ceiling * 10))
            self.post_ceiling_slider.blockSignals(False)

            # Update status
            self.master_progress.setVisible(False)
            if after_lufs is not None:
                self.master_status.setText(f"MASTERED: {after_lufs:.1f} LUFS")
            else:
                self.master_status.setText("MASTERED (LUFS pending)")
            self.master_status.setStyleSheet(f"color:{C_LED_GREEN}; font-size:11px; font-weight:bold;")

            # Re-enable master button
            self.btn_master.setEnabled(True)
            self.btn_master.setText("⚡  M A S T E R")

            # V5.2: Auto-load preview for playback with bypass
            try:
                self._load_preview_for_playback(preview_path)
                # V5.9: Auto-play after Master Assistant completes
                self._on_transport_play()
            except Exception as pe:
                print(f"[MASTER UI] Transport load error (non-fatal): {pe}")

            # Update chain UI
            self._sync_ui_from_chain()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[MASTER UI] _on_master_done error: {e}")
            # Try to re-enable button even if error occurred
            try:
                self.btn_master.setEnabled(True)
                self.btn_master.setText("⚡  M A S T E R")
                self.master_progress.setVisible(False)
                self.master_status.setText(f"ERROR: {e}")
                self.master_status.setStyleSheet(f"color:{C_LED_RED}; font-size:11px; font-weight:bold;")
            except Exception:
                pass

    def _on_master_error(self, msg):
        """Show error state."""
        try:
            print(f"[MASTER UI] ERROR: {msg}")
            if not self.isVisible():
                return
            self.master_progress.setVisible(False)
            self.master_status.setText(f"ERROR: {msg}")
            self.master_status.setStyleSheet(f"color:{C_LED_RED}; font-size:11px; font-weight:bold;")

            self.btn_master.setEnabled(True)
            self.btn_master.setText("⚡  M A S T E R")

            QMessageBox.critical(self, "Mastering Error", f"Failed to complete mastering:\n{msg}")
        except Exception as e:
            print(f"[MASTER UI] _on_master_error display failed: {e}")

    def _on_post_eq_changed(self, value: int):
        """Post-EQ adjustment slider → scales EQ intensity in real-time."""
        self.post_eq_label.setText(f"{value}%")
        # Scale all EQ band gains by this percentage
        scale = value / 100.0
        for band in self.chain.equalizer.bands:
            band._ui_scale = scale  # Store scale for chain to use

    def _on_post_dyn_changed(self, value: int):
        """Post-Dynamics adjustment slider → scales compression intensity."""
        self.post_dyn_label.setText(f"{value}%")
        self.chain.dynamics.post_scale = value / 100.0

    def _on_post_width_changed(self, value: int):
        """Post-Width adjustment slider → updates stereo width directly."""
        self.post_width_label.setText(f"{value}%")
        self.chain.imager.set_width(value)
        # Also update the main width slider visually
        self.img_width_slider.blockSignals(True)
        self.img_width_slider.setValue(value)
        self.img_width_slider.blockSignals(False)

    def _on_post_ceiling_changed(self, value: int):
        """Post-Ceiling adjustment slider → updates limiter ceiling directly."""
        ceiling_db = value / 10.0
        self.post_ceiling_label.setText(f"{ceiling_db:.1f} dB")
        self.chain.maximizer.set_ceiling(ceiling_db)

    def _on_remaster(self):
        """Re-run mastering with adjusted settings from post-mastering sliders."""
        if not hasattr(self, '_last_master_rec') or not self._last_master_rec:
            QMessageBox.warning(self, "No Previous Master", "Please run master first")
            return

        self.btn_master.setEnabled(False)
        self.btn_master.setText("⚡ RE-MASTERING...")
        self.master_progress.setVisible(True)
        self.master_progress.setValue(0)
        self.master_status.setText("RE-APPLYING WITH ADJUSTMENTS...")
        self.master_status.setStyleSheet(f"color:{C_LED_BLUE}; font-size:11px; font-weight:bold;")

        # Get adjustment values
        eq_amount = self.post_eq_slider.value() / 100.0
        dyn_amount = self.post_dyn_slider.value() / 100.0
        width_amount = self.post_width_slider.value() / 100.0
        ceiling = self.post_ceiling_slider.value() / 10.0

        def _do_remaster():
            try:
                # Apply adjustments to recommendation
                rec = self._last_master_rec
                if not rec:
                    self._call_on_main.emit(lambda: self._on_master_error("No previous master recommendation found"))
                    return

                # Modify recommendation based on sliders
                # EQ: scale the gain adjustments
                if hasattr(self.chain, 'equalizer') and hasattr(self.chain.equalizer, 'bands'):
                    for band in self.chain.equalizer.bands:
                        band.gain *= eq_amount

                # Dynamics: scale ratio/threshold — V5.1 FIX: safe nested access
                if (hasattr(self.chain, 'dynamics') and
                    hasattr(self.chain.dynamics, 'single_band') and
                    self.chain.dynamics.single_band is not None):
                    sb = self.chain.dynamics.single_band
                    if hasattr(sb, 'ratio'):
                        sb.ratio *= dyn_amount
                    if hasattr(sb, 'threshold'):
                        sb.threshold *= dyn_amount

                # Imager: scale width
                if hasattr(self.chain, 'imager') and hasattr(self.chain.imager, 'width'):
                    self.chain.imager.width = int(100 + (self.chain.imager.width - 100) * width_amount)

                # Maximizer: set ceiling
                if hasattr(self.chain, 'maximizer') and hasattr(self.chain.maximizer, 'set_ceiling'):
                    self.chain.maximizer.set_ceiling(ceiling)

                self._call_on_main.emit(lambda: self._update_master_progress(50, "RENDERING PREVIEW..."))
                preview_path = self.chain.preview()

                self._call_on_main.emit(lambda r=rec, p=preview_path: self._on_master_done(r, p))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._call_on_main.emit(lambda msg=str(e): self._on_master_error(msg))

        threading.Thread(target=_do_remaster, daemon=True).start()

    def _on_render_full_from_master(self):
        """Trigger full render from master assistant view."""
        if not self.chain.input_path:
            QMessageBox.warning(self, "No Audio", "Please load audio first")
            return

        # Use the standard render flow
        self._on_render()

    def _sync_ui_from_chain(self):
        """Update all UI controls to match current chain settings."""
        try:
            # Intensity
            self.intensity_slider.blockSignals(True)
            self.intensity_slider.setValue(self.chain.intensity)
            self.intensity_slider.blockSignals(False)
            self.intensity_value.setText(f"{self.chain.intensity}%")

            # IRC buttons — V5.0 FIX: safe index with try/except
            irc_keys = list(IRC_MODES.keys())
            try:
                if self.chain.maximizer.irc_mode in irc_keys:
                    idx = irc_keys.index(self.chain.maximizer.irc_mode)
                    for i, btn in enumerate(self.irc_buttons):
                        btn.setChecked(i == idx)
                    self._update_irc_button_styles()
                    self.irc_desc_label.setText(
                        IRC_MODES[self.chain.maximizer.irc_mode].get("description", ""))
            except (ValueError, IndexError):
                pass

            # Tone buttons — V5.0 FIX: safe index
            tone_keys = list(TONE_PRESETS.keys())
            try:
                if self.chain.maximizer.tone in tone_keys:
                    idx = tone_keys.index(self.chain.maximizer.tone)
                    for i, btn in enumerate(self.tone_buttons):
                        btn.setChecked(i == idx)
                    self._update_tone_button_styles()
            except (ValueError, IndexError):
                pass

            # Ceiling
            if hasattr(self, 'max_ceiling_knob'):
                self.max_ceiling_knob.blockSignals(True)
                self.max_ceiling_knob.setValue(self.chain.maximizer.ceiling)
                self.max_ceiling_knob.blockSignals(False)

            # EQ bands
            for i, band in enumerate(self.chain.equalizer.bands):
                if i < len(self.eq_band_sliders):
                    slider, label = self.eq_band_sliders[i]
                    slider.blockSignals(True)
                    slider.setValue(int(band.gain * 10))
                    slider.blockSignals(False)
                    label.setText(f"{band.gain:.1f}")

            # Width
            self.img_width_slider.blockSignals(True)
            self.img_width_slider.setValue(self.chain.imager.width)
            self.img_width_slider.blockSignals(False)
            self.img_width_value.setText(f"{self.chain.imager.width}")

            # Dynamics
            band = self.chain.dynamics.single_band
            for attr, spin in self.dyn_knobs.items():
                spin.blockSignals(True)
                spin.setValue(getattr(band, attr, spin.value()))
                spin.blockSignals(False)
        except Exception as e:
            print(f"[MASTER UI] _sync_ui_from_chain error: {e}")

    # ═══════════════════════════════════════════
    #  V5.4 FIX: Realtime Meter System
    # ═══════════════════════════════════════════

    def _on_meter_data(self, levels: dict):
        """
        Receive real-time meter data from chain processing or playback bridge.

        CRITICAL: Called from MasterWorker thread or main app, NOT necessarily main GUI thread.
        Uses thread-safe buffer — UI updates happen via QTimer on main thread.
        """
        with self._meter_lock:
            self._meter_buffer.append(levels)
            if len(self._meter_buffer) > self._meter_max_buffer_size:
                self._meter_buffer = self._meter_buffer[-self._meter_keep_count:]

        # V5.10: Auto-start meter timer when receiving playback data
        if levels.get("stage") == "playback":
            if hasattr(self, '_meter_update_timer') and not self._meter_update_timer.isActive():
                # Use QTimer.singleShot to start on main thread safely
                QTimer.singleShot(0, lambda: self._start_meter_timer("STAGE: ▶ LIVE"))

    def _update_realtime_meters(self):
        """Update UI with latest meter data (called by QTimer on main thread at 30 Hz)."""
        # V5.10: Poll RT engine meter data when active
        if self._rt_engine is not None and self._rt_playback_active:
            try:
                if self._rt_engine.is_playing():
                    m = self._rt_engine.get_meter_data()
                    l_rms = m.get("rms_l", -70.0)
                    r_rms = m.get("rms_r", -70.0)
                    rt_entry = {
                        "stage": "final",
                        "left_peak_db": m.get("peak_l", -70.0),
                        "right_peak_db": m.get("peak_r", -70.0),
                        "left_rms_db": l_rms,
                        "right_rms_db": r_rms,
                        "gain_reduction_db": m.get("gain_reduction_db", 0.0),
                    }

                    # V5.10 FIX: Estimate LUFS from RMS for RT playback
                    # LUFS ≈ mono_rms_db - 0.691 (approximate, no K-weighting)
                    mono_rms_db = 10.0 * math.log10(
                        (10 ** (l_rms / 10.0) + 10 ** (r_rms / 10.0)) / 2.0
                    ) if l_rms > -60 and r_rms > -60 else -70.0
                    lufs_approx = mono_rms_db - 0.691 if mono_rms_db > -60 else -70.0

                    # Accumulate for sliding windows (momentary=400ms, short-term=3s)
                    if not hasattr(self, '_rt_lufs_buf'):
                        self._rt_lufs_buf = []       # ~30Hz samples → 12 = 400ms, 90 = 3s
                        self._rt_lufs_int_acc = []   # all samples for integrated
                    self._rt_lufs_buf.append(lufs_approx)
                    self._rt_lufs_int_acc.append(lufs_approx)
                    if len(self._rt_lufs_buf) > 90:
                        self._rt_lufs_buf = self._rt_lufs_buf[-90:]
                    if len(self._rt_lufs_int_acc) > 9000:  # ~5 min
                        self._rt_lufs_int_acc = self._rt_lufs_int_acc[-9000:]

                    # Momentary: average of last 12 samples (~400ms)
                    mom_win = self._rt_lufs_buf[-12:] if len(self._rt_lufs_buf) >= 12 else self._rt_lufs_buf
                    mom_valid = [v for v in mom_win if v > -60]
                    lufs_mom = sum(mom_valid) / len(mom_valid) if mom_valid else -70.0

                    # Short-term: average of last 90 samples (~3s)
                    st_win = self._rt_lufs_buf[-90:] if len(self._rt_lufs_buf) >= 90 else self._rt_lufs_buf
                    st_valid = [v for v in st_win if v > -60]
                    lufs_short = sum(st_valid) / len(st_valid) if st_valid else -70.0

                    # Integrated: average of all accumulated samples
                    int_valid = [v for v in self._rt_lufs_int_acc if v > -60]
                    lufs_int = sum(int_valid) / len(int_valid) if int_valid else -70.0

                    # LRA: estimate from short-term block std deviation
                    if len(st_valid) >= 2:
                        import statistics
                        lu_range = max(0.0, max(st_valid) - min(st_valid))
                    else:
                        lu_range = 0.0

                    rt_entry["lufs_momentary"] = lufs_mom
                    rt_entry["lufs_short_term"] = lufs_short
                    rt_entry["lufs_integrated"] = lufs_int
                    rt_entry["lu_range"] = lu_range

                    if hasattr(self, 'meters') and hasattr(self.meters, 'update_live_levels'):
                        self.meters.update_live_levels(rt_entry)
                    if hasattr(self, 'max_gr_history') and self.max_gr_history is not None:
                        self.max_gr_history.set_gr(abs(rt_entry.get("gain_reduction_db", 0.0)))

                    # V5.10 FIX: Also feed WLM meter from RT path
                    if hasattr(self, 'meters') and hasattr(self.meters, 'live_wlm_meter'):
                        self.meters.live_wlm_meter.set_levels(
                            momentary=lufs_mom, short_term=lufs_short,
                            integrated=lufs_int, lra=lu_range,
                            tp_left=rt_entry["left_peak_db"],
                            tp_right=rt_entry["right_peak_db"],
                        )

                    # V5.11.0: Feed Stats for Nerds from RT path
                    if hasattr(self, 'meters') and hasattr(self.meters, 'stats_nerd'):
                        self.meters.stats_nerd.update_stats(
                            lufs_integrated=lufs_int, tp_left=rt_entry["left_peak_db"],
                            tp_right=rt_entry["right_peak_db"], lra=lu_range,
                            lufs_momentary=lufs_mom)

                    # V5.10.6: Feed spectrum analyzers from loaded audio during RT playback
                    self._feed_spectrum_from_rt_playback()

                    # V5.10.5 FIX: Forward RT meter data to popup panels
                    # Without this, popup panels get NO data during RT playback
                    # because _meter_buffer is empty and the method returns early below.
                    if hasattr(self, '_popup_meter_forward') and self._popup_meter_forward:
                        try:
                            fwd = dict(rt_entry)
                            if hasattr(self, 'chain') and hasattr(self.chain, 'stage_meter_data'):
                                fwd['_stage_data'] = self.chain.stage_meter_data.copy()
                            self._popup_meter_forward(fwd)
                        except Exception:
                            pass
            except Exception:
                pass  # Silently skip meter errors

        with self._meter_lock:
            if not self._meter_buffer:
                return
            # V5.8: Keep all buffered entries so we can extract per-stage data
            buf_copy = list(self._meter_buffer)
            self._meter_buffer.clear()

        # V5.8: Find latest entry for each stage we care about
        latest = buf_copy[-1]
        pre_chain_entry = None
        final_entry = None
        for entry in buf_copy:
            s = entry.get("stage", "")
            if s == "pre_chain":
                pre_chain_entry = entry
            elif s in ("final", "post_loudnorm", "post_maximizer"):
                final_entry = entry

        # V5.8: Feed LogicChannelMeter (BEFORE/AFTER) via MetersPanel
        if hasattr(self, 'meters') and hasattr(self.meters, 'live_logic_meter'):
            if pre_chain_entry:
                self.meters.update_live_levels(pre_chain_entry)
            if final_entry:
                self.meters.update_live_levels(final_entry)

        # V5.4: Update LiveMeterPanel with latest (for all other meters)
        if hasattr(self, 'meters') and hasattr(self.meters, 'update_live_levels'):
            self.meters.update_live_levels(latest)

        # V5.5: Update Maximizer LUFS display from offline chain meter data
        stage = latest.get("stage", "")

        # Update LUFS from any stage (not just realtime)
        lufs_int = latest.get("lufs_integrated")
        if lufs_int is not None and lufs_int > -70 and hasattr(self, 'max_lufs_label'):
            self.max_lufs_label.setText(f"{lufs_int:.1f} LUFS")

        # Update GR from chain meter data — feed history widget
        gr_db = abs(latest.get("gain_reduction_db", 0.0))
        if hasattr(self, 'max_gr_history') and self.max_gr_history is not None:
            self.max_gr_history.set_gr(gr_db)

        # V5.6: Update Waves WLM meter in MetersPanel
        if hasattr(self, 'meters') and hasattr(self.meters, 'live_wlm_meter'):
            l_peak = latest.get("left_peak_db", -70.0)
            r_peak = latest.get("right_peak_db", -70.0)
            lufs_mom = latest.get("lufs_momentary", -70.0)
            lufs_short = latest.get("lufs_short_term", -70.0)
            lufs_int = latest.get("lufs_integrated", -70.0)
            lra = latest.get("lu_range", 0.0)
            self.meters.live_wlm_meter.set_levels(
                momentary=lufs_mom, short_term=lufs_short,
                integrated=lufs_int, lra=lra,
                tp_left=l_peak, tp_right=r_peak,
            )

        # V5.11.0: Feed Stats for Nerds
        if hasattr(self, 'meters') and hasattr(self.meters, 'stats_nerd'):
            self.meters.stats_nerd.update_stats(
                lufs_integrated=lufs_int, tp_left=l_peak, tp_right=r_peak,
                lra=lra, lufs_momentary=lufs_mom)

        # True Peak indicator color
        if hasattr(self, 'max_meter_db'):
            max_peak = max(latest.get("left_peak_db", -60),
                           latest.get("right_peak_db", -60))
            tp_over = latest.get("true_peak_over", False)
            if tp_over or max_peak > -0.5:
                col = C_LED_RED
            elif max_peak > -1.0:
                col = C_LED_YELLOW
            else:
                col = C_TEAL_GLOW
            self.max_meter_db.setStyleSheet(
                f"color:{col}; font-family:'Menlo'; font-size:9px; font-weight:bold;")
            self.max_meter_db.setText(f"{max_peak:.1f} dBTP")

        # Update stage indicator (bottom bar)
        stage_names = {
            "pre_chain": "INPUT",
            "post_eq": "EQ",
            "post_dynamics": "DYNAMICS",
            "post_imager": "IMAGER",
            "post_maximizer": "MAXIMIZER",
            "post_loudnorm": "LOUDNESS",
            "final": "✓ READY",
            "playback": "▶ PLAYING",
            "rendering": "⏳ RENDERING...",
        }
        if stage in stage_names and hasattr(self, 'live_stage_label'):
            self.live_stage_label.setText(f"STAGE: {stage_names[stage]}")

        # V5.10.6: Feed spectrum analyzers with audio data from meter buffer
        self._feed_spectrum_from_meter(latest)

        # V5.10.5: Forward meter data to popup panels (gui.py hook)
        # Send latest entry + chain's per-stage data for full coverage
        if hasattr(self, '_popup_meter_forward') and self._popup_meter_forward:
            try:
                fwd = dict(latest)
                # Merge per-stage data from chain if available
                if hasattr(self, 'chain') and hasattr(self.chain, 'stage_meter_data'):
                    fwd['_stage_data'] = self.chain.stage_meter_data.copy()
                self._popup_meter_forward(fwd)
            except Exception:
                pass

    # ═══════════════════════════════════════════
    #  V5.10.6: Spectrum Analyzer Feed System
    # ═══════════════════════════════════════════

    def _feed_all_spectrums(self, samples, sr):
        """Feed audio samples to all spectrum analyzer widgets."""
        import numpy as np
        if samples is None or len(samples) == 0:
            return
        # Feed all module spectrum widgets
        for attr in ('res_spectrum', 'eq_spectrum', 'dyn_spectrum',
                     'img_spectrum', 'max_spectrum'):
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    widget.set_audio_data(samples, sr)
                except Exception:
                    pass

    def _feed_spectrum_from_meter(self, meter_data: dict):
        """Extract audio chunk from meter data and feed to spectrum widgets."""
        chunk = meter_data.get("_spectrum_chunk")
        sr = meter_data.get("_spectrum_sr", 44100)
        if chunk is not None:
            self._feed_all_spectrums(chunk, sr)

    def _feed_spectrum_from_rt_playback(self):
        """Read audio samples at current playback position for spectrum display.

        During RT playback, the Rust engine doesn't send raw samples back.
        Instead we read directly from the loaded audio file at the current position.
        """
        import numpy as np
        try:
            import soundfile as sf
        except ImportError:
            return

        if not hasattr(self, 'chain') or not self.chain.input_path:
            return
        if not hasattr(self, '_rt_engine') or self._rt_engine is None:
            return

        try:
            pos_sec = self._rt_engine.position()
            if pos_sec <= 0:
                return

            info = sf.info(self.chain.input_path)
            sr = info.samplerate
            start_sample = max(0, int(pos_sec * sr) - 4096)
            end_sample = start_sample + 4096

            if end_sample > info.frames:
                end_sample = info.frames
                start_sample = max(0, end_sample - 4096)

            data, sr = sf.read(self.chain.input_path,
                               start=start_sample, stop=end_sample)
            self._feed_all_spectrums(data, sr)
        except Exception:
            pass

    def _start_meter_timer(self, stage_text="STAGE: ⏳ RENDERING..."):
        """Start the meter update timer (for rendering or playback metering)."""
        if hasattr(self, '_meter_update_timer'):
            self._meter_update_timer.start()
        if hasattr(self, 'live_stage_label'):
            self.live_stage_label.setText(stage_text)

    def _stop_meter_timer(self):
        """Stop the meter update timer."""
        if hasattr(self, '_meter_update_timer'):
            self._meter_update_timer.stop()
        if hasattr(self, 'live_stage_label'):
            self.live_stage_label.setText("STAGE: ✓ READY")

    # ═══════════════════════════════════════════
    #  V5.8 E-3: Undo / Redo System
    # ═══════════════════════════════════════════

    def _push_undo(self, module: str, param: str, old_val, new_val, desc: str = ""):
        """Record a parameter change for undo."""
        from modules.master.undo import Command
        if old_val != new_val:
            self._cmd_history.push(Command(module, param, old_val, new_val, desc or f"{module}.{param}"))

    def _undo(self):
        """Undo last parameter change (Ctrl+Z)."""
        cmd = self._cmd_history.undo()
        if cmd:
            self._apply_param(cmd.module, cmd.param, cmd.old_val)
            self.meters.set_status(f"UNDO: {cmd.description}", C_AMBER)
            print(f"[UNDO] {cmd.description}: {cmd.new_val} → {cmd.old_val}")

    def _redo(self):
        """Redo last undone change (Ctrl+Shift+Z)."""
        cmd = self._cmd_history.redo()
        if cmd:
            self._apply_param(cmd.module, cmd.param, cmd.new_val)
            self.meters.set_status(f"REDO: {cmd.description}", C_TEAL_GLOW)
            print(f"[REDO] {cmd.description}: {cmd.old_val} → {cmd.new_val}")

    def _apply_param(self, module: str, param: str, value):
        """Apply a parameter value to the chain module."""
        mod = getattr(self.chain, module, None)
        if mod:
            setattr(mod, param, value)
            self._sync_maximizer_ui()

    # ═══════════════════════════════════════════
    #  V5.8 E-4: AutoSave System
    # ═══════════════════════════════════════════

    def _autosave(self):
        """Auto-save current chain settings every 60 seconds."""
        import os
        import json
        from datetime import datetime

        save_dir = os.path.expanduser("~/.longplay_studio/autosave")
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"autosave_{timestamp}.json")

        try:
            self.chain.save_settings(save_path)

            # Keep only last 10 autosaves
            files = sorted([f for f in os.listdir(save_dir) if f.startswith("autosave_")])
            while len(files) > 10:
                os.remove(os.path.join(save_dir, files.pop(0)))
        except Exception as e:
            print(f"[AUTOSAVE] Error: {e}")

    def _check_autosave_recovery(self):
        """Check for autosave on launch and offer recovery."""
        import os
        import json

        save_dir = os.path.expanduser("~/.longplay_studio/autosave")
        if not os.path.isdir(save_dir):
            return

        files = sorted([f for f in os.listdir(save_dir) if f.startswith("autosave_")])
        if not files:
            return

        latest = os.path.join(save_dir, files[-1])
        try:
            reply = QMessageBox.question(
                self, "Recover Session",
                f"Found autosaved session:\n{files[-1]}\n\nRecover last session?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                self.chain.load_settings(latest)
                self._sync_maximizer_ui()
                self.meters.set_status("SESSION RECOVERED", C_LED_GREEN)
        except Exception as e:
            print(f"[AUTOSAVE] Recovery error: {e}")

    # ═══════════════════════════════════════════
    #  V5.4 FIX: Batch Processing System
    # ═══════════════════════════════════════════

    def _on_batch_render(self):
        """Start batch mastering all tracks in the playlist."""
        # Try to get all tracks from parent GUI
        all_tracks = []
        parent = self.parent()
        if parent and hasattr(parent, 'get_all_track_paths'):
            all_tracks = parent.get_all_track_paths()
        elif parent and hasattr(parent, 'track_list'):
            # Fallback: try to read from track list widget
            try:
                for i in range(parent.track_list.count()):
                    item = parent.track_list.item(i)
                    if hasattr(item, 'data') and item.data(Qt.ItemDataRole.UserRole):
                        all_tracks.append(item.data(Qt.ItemDataRole.UserRole))
            except Exception:
                pass

        # If no tracks from parent, ask user to select files
        if not all_tracks:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select Audio Files for Batch Mastering", "",
                "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a *.ogg);;All Files (*)"
            )
            if files:
                all_tracks = files

        if not all_tracks:
            QMessageBox.warning(self, "Batch Master", "No tracks found. Please load tracks first.")
            return

        if len(all_tracks) < 1:
            QMessageBox.warning(self, "Batch Master", "Need at least 1 track for batch mastering.")
            return

        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Mastered Tracks", "")
        if not output_dir:
            return

        # Confirm batch operation
        reply = QMessageBox.question(
            self, "Batch Master",
            f"Master {len(all_tracks)} tracks?\n\n"
            f"Output: {output_dir}\n"
            f"Target: {self.chain.target_lufs} LUFS / {self.chain.target_tp} dBTP",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Initialize batch state
        self.track_queue = all_tracks
        self.current_track_index = 0
        self.batch_mode = True
        self.batch_output_dir = output_dir
        self._batch_results = []

        # Disable UI during batch
        self._set_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start processing first track
        self._batch_process_next()

    def _batch_process_next(self):
        """Process the next track in the batch queue."""
        if self.current_track_index >= len(self.track_queue):
            self._batch_complete()
            return

        track_path = self.track_queue[self.current_track_index]
        track_name = os.path.basename(track_path)
        total = len(self.track_queue)
        idx = self.current_track_index + 1

        self.progress_text.setText(f"BATCH [{idx}/{total}]: {track_name}")
        self.progress_bar.setValue(int((self.current_track_index / total) * 100))

        # Load the track
        try:
            if not self.chain.load_audio(track_path):
                self._batch_results.append({
                    "path": track_path, "success": False,
                    "lufs": None, "tp": None, "error": "Failed to load audio"
                })
                self.current_track_index += 1
                self._batch_process_next()
                return
        except Exception as e:
            self._batch_results.append({
                "path": track_path, "success": False,
                "lufs": None, "tp": None, "error": str(e)
            })
            self.current_track_index += 1
            self._batch_process_next()
            return

        # Set output path
        name, ext = os.path.splitext(track_name)
        # Always output as WAV for soundfile compatibility
        safe_ext = ext if ext.lower() in ['.wav', '.wave', '.flac'] else '.wav'
        out_name = f"{name}_mastered{safe_ext}"
        self.chain.output_path = os.path.join(self.batch_output_dir, out_name)

        # Start rendering
        self._start_meter_timer()
        self.meters.set_status("⏳ BATCH RENDERING...", C_LED_BLUE)

        self.worker = MasterWorker(self.chain, "render")
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_render_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _batch_complete(self):
        """Show batch completion summary."""
        self.batch_mode = False
        self._stop_meter_timer()
        self.progress_bar.setVisible(False)
        self._set_buttons_enabled(True)

        total = len(self._batch_results)
        success = sum(1 for r in self._batch_results if r["success"])
        failed = total - success

        # Build summary HTML
        rows = ""
        for r in self._batch_results:
            name = os.path.basename(r["path"])
            if r["success"]:
                lufs_str = f"{r['lufs']:.1f}" if r['lufs'] is not None else "—"
                tp_str = f"{r['tp']:.1f}" if r['tp'] is not None else "—"
                rows += f"<tr><td>✅ {name}</td><td>{lufs_str} LUFS</td><td>{tp_str} dBTP</td></tr>"
            else:
                rows += f"<tr><td>❌ {name}</td><td colspan='2'>Error: {r['error']}</td></tr>"

        html = (
            f"<h3>Batch Mastering Complete</h3>"
            f"<p><b>{success}/{total}</b> tracks mastered successfully"
            f"{', <span style=color:red>' + str(failed) + ' failed</span>' if failed else ''}</p>"
            f"<table border='1' cellpadding='4'>"
            f"<tr><th>Track</th><th>LUFS</th><th>True Peak</th></tr>"
            f"{rows}</table>"
            f"<p>Output: {self.batch_output_dir}</p>"
        )

        self.progress_text.setText(f"BATCH DONE: {success}/{total} ✅")
        self.meters.set_status("BATCH COMPLETE", C_LED_GREEN)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Batch Mastering Complete")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(html)
        msg_box.exec()

    # ═══════════════════════════════════════════
    #  V5.9: Tonal Balance Control
    # ═══════════════════════════════════════════

    def _update_tonal_balance(self, audio_path: str):
        """Analyze rendered audio and update tonal balance widget."""
        try:
            from modules.master.tonal_balance import TonalBalanceAnalyzer
            import soundfile as sf

            if not os.path.exists(audio_path):
                return

            # Read audio (max 60s for speed)
            info = sf.info(audio_path)
            max_frames = min(info.frames, info.samplerate * 60)
            data, sr = sf.read(audio_path, frames=max_frames, dtype='float64')

            analyzer = TonalBalanceAnalyzer()

            # Set target based on current genre
            genre = self.genre_combo.currentText() if hasattr(self, 'genre_combo') else ""
            curve_key = analyzer.set_target_for_genre(genre)

            if analyzer.analyze(data, sr):
                # Update widget
                ranges = [analyzer.get_target_range(i) for i in range(4)]
                self.meters.live_tonal_balance.set_data(
                    energies=analyzer.band_energies,
                    target_ranges=ranges,
                    in_range=analyzer.in_range,
                    score=analyzer.score,
                    target_name=curve_key,
                )
        except Exception as e:
            print(f"[TONAL BALANCE] Error: {e}")

    # ─── Helpers ──────────────────────────────
    def _module_title(self, text: str):
        lbl = QLabel(text)
        lbl.setFont(QFont("Menlo", 13, QFont.Weight.Bold))
        lbl.setStyleSheet(
            f"color: {C_AMBER}; letter-spacing: 2px;")
        return lbl

    def _panel_label(self, text: str):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"font-size:9px; color:{C_GOLD}; letter-spacing:1.5px; font-weight:bold;")
        return lbl

    # ------------------------------------------------------------------
    # V5.5: Close event — notify parent so Mini Meter can restore opacity
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        """Stop playback and emit master_closed signal when window closes."""
        try:
            self._preview_player.stop()
            self._bypass_player.stop()
            self._stop_meter_timer()
        except Exception:
            pass
        self.master_closed.emit()
        super().closeEvent(event)
