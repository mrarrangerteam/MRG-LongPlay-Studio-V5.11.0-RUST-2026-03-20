"""
Vintage hardware UI polish — final pass theme system.

Story 5.4 — Epic 5: Polish & Production.

Features:
    - Waves/SSL/Neve-inspired hardware aesthetics
    - 3D beveled controls, brushed metal textures
    - Warm amber VU glow, teal accents
    - Consistent dark theme across all panels
    - High-DPI ready styling
    - Theme variants: Classic Dark, Midnight, Warm Console
"""

from __future__ import annotations

from gui.styles import Colors


# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------
class VintageTheme:
    """Base class for vintage hardware themes."""

    name: str = "Classic Dark"

    # chassis
    CHASSIS = "#111113"
    PANEL = "#1E1E22"
    PANEL_LIGHT = "#2A2A30"
    PANEL_INSET = "#0E0E10"
    FACEPLATE = "#282830"

    # accents
    AMBER = "#E8A832"
    AMBER_GLOW = "#F5C04A"
    AMBER_DIM = "#8B6914"
    GOLD = "#C89B3C"
    TEAL = "#00B4D8"
    TEAL_DIM = "#0077B6"
    TEAL_GLOW = "#48CAE4"

    # text
    TEXT_BRIGHT = "#E8E8E8"
    TEXT_MEDIUM = "#AAAAAA"
    TEXT_DIM = "#666666"
    TEXT_LABEL = "#888890"

    # controls
    KNOB_BODY = "#2A2A30"
    KNOB_RING = "#4A4A52"
    KNOB_POINTER = "#E8A832"
    BUTTON_FACE = "#222228"
    BUTTON_ACTIVE = "#00B4D8"
    BUTTON_HOVER = "#333340"

    # meters
    METER_GREEN = "#43A047"
    METER_YELLOW = "#FDD835"
    METER_RED = "#E53935"
    METER_BG = "#0A0A0C"

    # chrome/bezel
    BEZEL_LIGHT = "#5A5A62"
    BEZEL_DARK = "#1A1A20"
    BEZEL_HIGHLIGHT = "#78787E"
    SCREW = "#3A3A42"
    SCREW_SLOT = "#222228"

    # module identities
    MOD_EQ = "#4FC3F7"
    MOD_DYN = "#FF8A65"
    MOD_IMG = "#CE93D8"
    MOD_MAX = "#FFD54F"
    MOD_AI = "#81C784"

    @classmethod
    def get_global_stylesheet(cls) -> str:
        """Generate a complete application stylesheet."""
        return f"""
        /* === MRG LongPlay Studio V5.5 — {cls.name} Theme === */

        QMainWindow, QWidget {{
            background-color: {cls.CHASSIS};
            color: {cls.TEXT_BRIGHT};
            font-family: "Inter", "SF Pro Display", "Segoe UI", sans-serif;
        }}

        /* Panel sections */
        QFrame {{
            background-color: {cls.PANEL};
            border: 1px solid {cls.BEZEL_DARK};
            border-radius: 3px;
        }}

        QGroupBox {{
            background-color: {cls.FACEPLATE};
            border: 1px solid {cls.BEZEL_DARK};
            border-radius: 4px;
            margin-top: 14px;
            padding-top: 10px;
            font-weight: bold;
            color: {cls.GOLD};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 2px 10px;
            background: {cls.PANEL};
            border: 1px solid {cls.BEZEL_DARK};
            border-radius: 3px;
            color: {cls.AMBER};
            font-size: 10px;
            text-transform: uppercase;
        }}

        /* Buttons — hardware style */
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {cls.PANEL_LIGHT}, stop:1 {cls.BUTTON_FACE});
            border: 1px solid {cls.BEZEL_DARK};
            border-radius: 3px;
            color: {cls.TEXT_BRIGHT};
            padding: 5px 14px;
            font-weight: bold;
            font-size: 11px;
        }}
        QPushButton:hover {{
            background: {cls.BUTTON_HOVER};
            border-color: {cls.TEAL_DIM};
        }}
        QPushButton:pressed {{
            background: {cls.PANEL_INSET};
            border-color: {cls.TEAL};
        }}
        QPushButton:checked {{
            background: {cls.BUTTON_ACTIVE};
            color: {cls.CHASSIS};
            border-color: {cls.TEAL_GLOW};
        }}

        /* Combo boxes */
        QComboBox {{
            background: {cls.PANEL_INSET};
            border: 1px solid {cls.BEZEL_DARK};
            border-radius: 3px;
            color: {cls.AMBER};
            padding: 4px 8px;
            font-size: 11px;
        }}
        QComboBox:hover {{
            border-color: {cls.TEAL_DIM};
        }}
        QComboBox::drop-down {{
            border-left: 1px solid {cls.BEZEL_DARK};
            width: 20px;
        }}

        /* Sliders — Waves-style fader */
        QSlider::groove:horizontal {{
            border: 1px solid {cls.BEZEL_DARK};
            height: 6px;
            background: {cls.PANEL_INSET};
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {cls.BEZEL_HIGHLIGHT}, stop:0.5 {cls.BEZEL_LIGHT}, stop:1 {cls.BEZEL_DARK});
            border: 1px solid {cls.BEZEL_DARK};
            width: 14px;
            margin: -5px 0;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {cls.TEAL_DIM};
        }}

        /* Vertical slider */
        QSlider::groove:vertical {{
            border: 1px solid {cls.BEZEL_DARK};
            width: 6px;
            background: {cls.PANEL_INSET};
            border-radius: 3px;
        }}
        QSlider::handle:vertical {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {cls.BEZEL_HIGHLIGHT}, stop:0.5 {cls.BEZEL_LIGHT}, stop:1 {cls.BEZEL_DARK});
            border: 1px solid {cls.BEZEL_DARK};
            height: 14px;
            margin: 0 -5px;
            border-radius: 3px;
        }}

        /* Spin boxes */
        QSpinBox, QDoubleSpinBox {{
            background: {cls.PANEL_INSET};
            border: 1px solid {cls.BEZEL_DARK};
            border-radius: 3px;
            color: {cls.AMBER};
            padding: 2px 4px;
            font-family: "Courier New", monospace;
        }}

        /* Tabs — channel strip selector */
        QTabWidget::pane {{
            border: 1px solid {cls.BEZEL_DARK};
            background: {cls.PANEL};
        }}
        QTabBar::tab {{
            background: {cls.FACEPLATE};
            border: 1px solid {cls.BEZEL_DARK};
            border-bottom: none;
            padding: 6px 16px;
            color: {cls.TEXT_DIM};
            font-weight: bold;
            font-size: 10px;
        }}
        QTabBar::tab:selected {{
            background: {cls.PANEL};
            color: {cls.TEAL};
            border-bottom: 2px solid {cls.TEAL};
        }}
        QTabBar::tab:hover {{
            color: {cls.TEXT_BRIGHT};
        }}

        /* Progress bar — LED style */
        QProgressBar {{
            border: 1px solid {cls.BEZEL_DARK};
            border-radius: 3px;
            background: {cls.PANEL_INSET};
            text-align: center;
            color: {cls.TEXT_BRIGHT};
            font-size: 10px;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {cls.TEAL_DIM}, stop:1 {cls.TEAL});
            border-radius: 2px;
        }}

        /* Scroll bars */
        QScrollBar:vertical {{
            background: {cls.CHASSIS};
            width: 10px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {cls.BEZEL_DARK};
            border-radius: 4px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {cls.BEZEL_LIGHT};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background: {cls.CHASSIS};
            height: 10px;
        }}
        QScrollBar::handle:horizontal {{
            background: {cls.BEZEL_DARK};
            border-radius: 4px;
            min-width: 20px;
        }}

        /* Labels */
        QLabel {{
            color: {cls.TEXT_MEDIUM};
            border: none;
            background: transparent;
        }}

        /* List/Tree widgets */
        QListWidget, QTreeWidget {{
            background: {cls.PANEL_INSET};
            border: 1px solid {cls.BEZEL_DARK};
            color: {cls.TEXT_BRIGHT};
            alternate-background-color: {cls.FACEPLATE};
        }}
        QListWidget::item:selected {{
            background: {cls.TEAL_DIM};
            color: {cls.TEXT_BRIGHT};
        }}

        /* Tooltips */
        QToolTip {{
            background: {cls.FACEPLATE};
            color: {cls.AMBER};
            border: 1px solid {cls.TEAL_DIM};
            padding: 4px;
            font-size: 11px;
        }}

        /* Menu */
        QMenu {{
            background: {cls.PANEL};
            border: 1px solid {cls.BEZEL_DARK};
            color: {cls.TEXT_BRIGHT};
        }}
        QMenu::item:selected {{
            background: {cls.TEAL_DIM};
        }}
        """


class MidnightTheme(VintageTheme):
    """Deeper blacks with blue accent — modern studio."""
    name = "Midnight"
    CHASSIS = "#0A0A0E"
    PANEL = "#141418"
    PANEL_LIGHT = "#1E1E24"
    FACEPLATE = "#1A1A22"
    TEAL = "#4488FF"
    TEAL_DIM = "#2255AA"
    TEAL_GLOW = "#6699FF"
    BUTTON_ACTIVE = "#4488FF"


class WarmConsoleTheme(VintageTheme):
    """Warmer tones — inspired by Neve/SSL consoles."""
    name = "Warm Console"
    CHASSIS = "#141210"
    PANEL = "#201C18"
    PANEL_LIGHT = "#2C2620"
    FACEPLATE = "#282220"
    AMBER = "#F0A020"
    AMBER_GLOW = "#FFB840"
    TEAL = "#20C8A0"
    TEAL_DIM = "#108860"
    TEAL_GLOW = "#40E8C0"
    BUTTON_ACTIVE = "#20C8A0"
    GOLD = "#D4A030"


# ---------------------------------------------------------------------------
# Theme registry
# ---------------------------------------------------------------------------
THEMES = {
    "classic_dark": VintageTheme,
    "midnight": MidnightTheme,
    "warm_console": WarmConsoleTheme,
}


def get_theme(name: str = "classic_dark") -> type:
    """Get a theme class by name."""
    return THEMES.get(name, VintageTheme)


def get_theme_names() -> list:
    """Return available theme names."""
    return [cls.name for cls in THEMES.values()]


def apply_theme(app, theme_name: str = "classic_dark") -> None:
    """
    Apply a vintage theme to the QApplication.

    Args:
        app: QApplication instance.
        theme_name: Theme key from THEMES dict.
    """
    theme = THEMES.get(theme_name, VintageTheme)
    stylesheet = theme.get_global_stylesheet()
    app.setStyleSheet(stylesheet)
