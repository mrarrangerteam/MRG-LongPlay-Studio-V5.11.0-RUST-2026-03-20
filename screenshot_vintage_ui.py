#!/usr/bin/env python3
"""Generate a screenshot of the vintage mastering UI elements."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                                  QLabel, QGroupBox, QPushButton, QSlider, QFrame)
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont, QPainter, QColor, QImage, QPen, QBrush, QLinearGradient
except ImportError:
    from PySide6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                                    QLabel, QGroupBox, QPushButton, QSlider, QFrame)
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QPainter, QColor, QImage, QPen, QBrush, QLinearGradient

from modules.widgets.rotary_knob import OzoneRotaryKnob
from modules.master.ui_panel import (
    GLOBAL_STYLE, BTN_PRIMARY_STYLE, BTN_SECONDARY_STYLE, BTN_TEAL_STYLE,
    C_CHASSIS, C_PANEL, C_PANEL_LIGHT, C_AMBER, C_AMBER_GLOW, C_GOLD,
    C_TEAL, C_TEAL_GLOW, C_CREAM, C_CREAM_DIM, C_GROOVE, C_RIDGE,
    C_MOD_EQ, C_MOD_DYN, C_MOD_IMG, C_MOD_MAX, C_MOD_AI,
    C_LED_GREEN, C_LED_RED, C_LED_YELLOW,
)


class VintageShowcase(QWidget):
    """Showcase widget demonstrating the vintage hardware UI style."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LongPlay Studio V5.9 — Vintage Hardware UI")
        self.setFixedSize(900, 620)
        self.setStyleSheet(GLOBAL_STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ── Title bar ──
        title = QLabel("LONGPLAY STUDIO  V5.9  —  MASTERING CONSOLE")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Georgia", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C_GOLD}; letter-spacing: 4px; padding: 8px;")
        root.addWidget(title)

        subtitle = QLabel("Inspired by Neve 1073  ·  SSL 4000  ·  Chandler Limited  ·  Pultec EQP-1A")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {C_CREAM_DIM}; font-size: 10px; letter-spacing: 2px;")
        root.addWidget(subtitle)

        # ── Main module strip ──
        strip = QHBoxLayout()
        strip.setSpacing(8)

        # EQ Module
        eq_group = QGroupBox("PARAMETRIC EQ")
        eq_layout = QVBoxLayout()
        eq_knob_row = QHBoxLayout()
        for name, val, unit in [("32 Hz", 0.0, "dB"), ("250 Hz", 1.5, "dB"),
                                 ("1 kHz", -0.5, "dB"), ("8 kHz", 2.0, "dB")]:
            k = OzoneRotaryKnob(name, -12, 12, val, unit, 1)
            eq_knob_row.addWidget(k)
        eq_layout.addLayout(eq_knob_row)
        eq_group.setLayout(eq_layout)
        eq_group.setStyleSheet(eq_group.styleSheet() + f"QGroupBox {{ border-left: 3px solid {C_MOD_EQ}; }}")
        strip.addWidget(eq_group)

        # Dynamics Module
        dyn_group = QGroupBox("DYNAMICS")
        dyn_layout = QVBoxLayout()
        dyn_knob_row = QHBoxLayout()
        for name, val, mn, mx, unit in [("THRESH", -16.0, -40, 0, "dB"),
                                          ("RATIO", 2.5, 1, 20, ":1"),
                                          ("ATTACK", 10.0, 0.1, 100, "ms"),
                                          ("RELEASE", 100.0, 10, 1000, "ms")]:
            k = OzoneRotaryKnob(name, mn, mx, val, unit, 1)
            dyn_knob_row.addWidget(k)
        dyn_layout.addLayout(dyn_knob_row)
        dyn_group.setLayout(dyn_layout)
        dyn_group.setStyleSheet(dyn_group.styleSheet() + f"QGroupBox {{ border-left: 3px solid {C_MOD_DYN}; }}")
        strip.addWidget(dyn_group)

        # Maximizer Module
        max_group = QGroupBox("MAXIMIZER  ·  IRC IV")
        max_layout = QVBoxLayout()
        max_knob_row = QHBoxLayout()
        for name, val, mn, mx, unit in [("GAIN", 8.0, 0, 20, "dB"),
                                          ("CEILING", -1.0, -3, -0.1, "dBTP"),
                                          ("CHARACTER", 5.0, 0, 10, "")]:
            k = OzoneRotaryKnob(name, mn, mx, val, unit, 1, large=True)
            max_knob_row.addWidget(k)
        max_layout.addLayout(max_knob_row)
        max_group.setLayout(max_layout)
        max_group.setStyleSheet(max_group.styleSheet() + f"QGroupBox {{ border-left: 3px solid {C_MOD_MAX}; }}")
        strip.addWidget(max_group)

        root.addLayout(strip)

        # ── Button showcase ──
        btn_group = QGroupBox("CONTROLS")
        btn_layout = QHBoxLayout()

        btn1 = QPushButton("MASTER & EXPORT")
        btn1.setStyleSheet(BTN_PRIMARY_STYLE)
        btn_layout.addWidget(btn1)

        btn2 = QPushButton("PREVIEW 30s")
        btn2.setStyleSheet(BTN_SECONDARY_STYLE)
        btn_layout.addWidget(btn2)

        btn3 = QPushButton("A/B COMPARE")
        btn3.setStyleSheet(BTN_SECONDARY_STYLE)
        btn_layout.addWidget(btn3)

        btn4 = QPushButton("MASTER ASSISTANT")
        btn4.setStyleSheet(BTN_TEAL_STYLE)
        btn_layout.addWidget(btn4)

        btn_group.setLayout(btn_layout)
        root.addWidget(btn_group)

        # ── Imager + Status Row ──
        bottom_row = QHBoxLayout()

        # Imager
        img_group = QGroupBox("STEREO IMAGER")
        img_layout = QVBoxLayout()
        img_knob_row = QHBoxLayout()
        for name, val in [("WIDTH", 120), ("BALANCE", 0), ("MONO BASS", 80)]:
            k = OzoneRotaryKnob(name, 0, 200, val, "%", 0)
            img_knob_row.addWidget(k)
        img_layout.addLayout(img_knob_row)
        img_group.setLayout(img_layout)
        img_group.setStyleSheet(img_group.styleSheet() + f"QGroupBox {{ border-left: 3px solid {C_MOD_IMG}; }}")
        bottom_row.addWidget(img_group)

        # Status / Meter area
        status_group = QGroupBox("METER STATUS")
        status_layout = QVBoxLayout()
        for label_text, color in [("INPUT:  -18.2 LUFS", C_CREAM),
                                    ("OUTPUT: -14.0 LUFS", C_LED_GREEN),
                                    ("TRUE PEAK: -1.0 dBTP", C_LED_YELLOW),
                                    ("GAIN DELTA: +4.2 dB", C_AMBER_GLOW)]:
            lbl = QLabel(label_text)
            lbl.setStyleSheet(f"color: {color}; font-family: 'Menlo', monospace; "
                              f"font-size: 11px; font-weight: bold; letter-spacing: 1px; padding: 3px;")
            status_layout.addWidget(lbl)
        status_group.setLayout(status_layout)
        bottom_row.addWidget(status_group)

        root.addLayout(bottom_row)

        # ── Footer ──
        footer = QLabel("MRG LongPlay Studio  ·  Vintage Hardware Edition  ·  Neve / SSL / Chandler / Pultec")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet(f"color: {C_CREAM_DIM}; font-size: 9px; letter-spacing: 2px; padding: 4px;")
        root.addWidget(footer)


def main():
    app = QApplication(sys.argv)
    window = VintageShowcase()
    window.show()

    # Capture screenshot after rendering
    def capture():
        pixmap = window.grab()
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vintage_ui_screenshot.png")
        pixmap.save(out_path)
        print(f"Screenshot saved: {out_path}")
        app.quit()

    QTimer.singleShot(500, capture)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
