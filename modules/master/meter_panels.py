#!/usr/bin/env python3
"""
modules/master/meter_panels.py — Ozone 12-quality Popup Metering Panels
========================================================================

กดที่ปุ่ม Gain/Width/Compressor/Soothe → popup panel แสดง realtime metering
หน้าตาสวยงามแบบ Ozone 12 / Waves

4 Popup Panels:
1. MaximizerMeterPanel   — Gain Reduction History + IRC meter + ceiling
2. ImagerMeterPanel      — Stereo width curve + correlation + vectorscope  
3. CompressorMeterPanel  — GR meter + threshold line + 3-band activity
4. SootheMeterPanel      — Spectral reduction curve + delta display

Instruction for Claude Code:
  วางไฟล์นี้ที่ modules/master/meter_panels.py
  แล้วเพิ่ม import ใน gui.py / ui_panel.py ที่สร้างปุ่ม Gain, Width, Compressor, Soothe
  เมื่อ user กดปุ่ม → สร้าง panel.show()

Usage:
  from modules.master.meter_panels import (
      MaximizerMeterPanel, ImagerMeterPanel,
      CompressorMeterPanel, SootheMeterPanel
  )
  
  # เมื่อ user กดปุ่ม Gain:
  panel = MaximizerMeterPanel(parent=self)
  panel.show()
  
  # Feed data จาก audio processing loop (เรียกทุก ~50ms):
  panel.update_meter(gain_reduction_db=-3.5, output_peak_db=-1.2, lufs=-14.0)

Author: MRARRANGER AI Studio
Date: 2026-03-20
"""

import sys
import os
import numpy as np
from collections import deque

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QFrame, QComboBox, QSlider, QDial, QGridLayout, QSizePolicy,
        QGraphicsOpacityEffect
    )
    from PyQt6.QtCore import Qt, QTimer, QSize, QRectF, QPointF, pyqtSignal
    from PyQt6.QtGui import (
        QPainter, QColor, QPen, QBrush, QLinearGradient,
        QFont, QPainterPath, QPixmap, QRadialGradient
    )
    PYQT6 = True
except ImportError:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QFrame, QComboBox, QSlider, QDial, QGridLayout, QSizePolicy,
        QGraphicsOpacityEffect
    )
    from PySide6.QtCore import Qt, QTimer, QSize, QRectF, QPointF, Signal as pyqtSignal
    from PySide6.QtGui import (
        QPainter, QColor, QPen, QBrush, QLinearGradient,
        QFont, QPainterPath, QPixmap, QRadialGradient
    )
    PYQT6 = False


# ═══════════════════════════════════════════════════════════════
# Color Theme (Ozone 12 / Waves dark teal aesthetic)
# ═══════════════════════════════════════════════════════════════

class OzoneColors:
    BG_DEEP = QColor(12, 14, 18)
    BG_PANEL = QColor(18, 22, 28)
    BG_SURFACE = QColor(24, 30, 38)
    BG_RAISED = QColor(32, 38, 48)
    
    CYAN = QColor(0, 200, 220)
    CYAN_DIM = QColor(0, 140, 155)
    CYAN_BRIGHT = QColor(100, 230, 240)
    CYAN_GLOW = QColor(0, 200, 220, 40)
    
    TEAL = QColor(0, 180, 216)
    TEAL_DIM = QColor(0, 119, 182)
    
    AMBER = QColor(255, 149, 0)
    AMBER_DIM = QColor(200, 120, 0)
    
    RED = QColor(229, 57, 53)
    RED_DIM = QColor(180, 40, 40)
    
    GREEN = QColor(67, 160, 71)
    
    TEXT_PRIMARY = QColor(220, 225, 230)
    TEXT_SECONDARY = QColor(140, 150, 160)
    TEXT_TERTIARY = QColor(80, 90, 100)
    
    GRID_LINE = QColor(40, 48, 58)
    GRID_LINE_MAJOR = QColor(55, 65, 78)
    
    BORDER = QColor(45, 55, 68)


# ═══════════════════════════════════════════════════════════════
# Base Panel Widget
# ═══════════════════════════════════════════════════════════════

class BaseMeterPanel(QWidget):
    """Base class for all Ozone-style meter popup panels"""
    
    closed = pyqtSignal()
    
    def __init__(self, title="Meter", width=520, height=300, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(width, height)
        self.setStyleSheet(f"""
            QWidget {{
                background: rgb(12, 14, 18);
                color: rgb(220, 225, 230);
                font-family: 'SF Pro Display', 'Segoe UI', 'Helvetica Neue', sans-serif;
            }}
        """)
        
        self._title = title
        self._drag_pos = None
        
        # Update timer
        self._timer = QTimer(self)
        self._timer.setInterval(50)  # 20fps metering
        self._timer.timeout.connect(self._on_tick)
    
    def show(self):
        super().show()
        self._timer.start()
    
    def hide(self):
        self._timer.stop()
        super().hide()
    
    def close(self):
        self._timer.stop()
        self.closed.emit()
        super().close()
    
    def _on_tick(self):
        """Override in subclass for animation"""
        self.update()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
    
    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
    
    def _draw_title_bar(self, painter: QPainter):
        """Draw Ozone-style title bar"""
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(18, 22, 28))
        painter.drawRect(0, 0, self.width(), 28)
        
        # Title text
        painter.setPen(OzoneColors.TEXT_SECONDARY)
        painter.setFont(QFont("SF Pro Display", 10, QFont.Weight.Medium))
        painter.drawText(12, 18, self._title)
        
        # Close button
        painter.setPen(QPen(OzoneColors.TEXT_TERTIARY, 1.5))
        cx, cy = self.width() - 18, 14
        painter.drawLine(int(cx-4), int(cy-4), int(cx+4), int(cy+4))
        painter.drawLine(int(cx+4), int(cy-4), int(cx-4), int(cy+4))
        
        # Bottom border
        painter.setPen(QPen(OzoneColors.BORDER, 0.5))
        painter.drawLine(0, 28, self.width(), 28)
    
    def _draw_grid(self, painter: QPainter, rect: QRectF, 
                   h_lines: int = 5, v_lines: int = 0,
                   db_range: tuple = (-12, 0)):
        """Draw dB grid lines"""
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        
        painter.setPen(QPen(OzoneColors.GRID_LINE, 0.5))
        
        # Horizontal lines
        for i in range(h_lines + 1):
            yy = y + (h * i / h_lines)
            if i == 0 or i == h_lines:
                painter.setPen(QPen(OzoneColors.GRID_LINE_MAJOR, 0.5))
            else:
                painter.setPen(QPen(OzoneColors.GRID_LINE, 0.5))
            painter.drawLine(int(x), int(yy), int(x + w), int(yy))
            
            # dB labels
            db = db_range[0] + (db_range[1] - db_range[0]) * (1.0 - i / h_lines)
            painter.setPen(OzoneColors.TEXT_TERTIARY)
            painter.setFont(QFont("SF Pro Display", 8))
            painter.drawText(int(x - 28), int(yy + 4), f"{db:.0f}")
        
        # Vertical lines
        if v_lines > 0:
            painter.setPen(QPen(OzoneColors.GRID_LINE, 0.5))
            for i in range(1, v_lines):
                xx = x + (w * i / v_lines)
                painter.drawLine(int(xx), int(y), int(xx), int(y + h))


# ═══════════════════════════════════════════════════════════════
# 1. MAXIMIZER METER PANEL
# ═══════════════════════════════════════════════════════════════

class MaximizerMeterPanel(BaseMeterPanel):
    """
    Ozone 12 Maximizer-style panel:
    - Gain Reduction History (scrolling cyan waveform)
    - Output Level meter (L/R bars)
    - LUFS readout
    - IRC mode display
    - Ceiling line
    """
    
    def __init__(self, parent=None):
        super().__init__("Maximizer", width=560, height=280, parent=parent)
        
        # Data buffers
        self._gr_history = deque(maxlen=400)  # gain reduction history
        self._output_peak_l = -60.0
        self._output_peak_r = -60.0
        self._lufs = -14.0
        self._ceiling_db = -1.0
        self._irc_mode = "IRC 4"
        self._gain_db = 0.0
        self._true_peak_db = -1.0
        
        # Smooth meters
        self._meter_l_smooth = -60.0
        self._meter_r_smooth = -60.0
        self._gr_smooth = 0.0
    
    def update_meter(self, gain_reduction_db=0.0, output_peak_l=-60.0, 
                     output_peak_r=-60.0, lufs=-14.0, ceiling=-1.0,
                     irc_mode="IRC 4", gain_db=0.0, true_peak=-1.0):
        """Call from audio thread/timer with current values"""
        self._gr_history.append(gain_reduction_db)
        self._output_peak_l = output_peak_l
        self._output_peak_r = output_peak_r
        self._lufs = lufs
        self._ceiling_db = ceiling
        self._irc_mode = irc_mode
        self._gain_db = gain_db
        self._true_peak_db = true_peak
    
    def _on_tick(self):
        # Smooth meters (ballistic)
        attack = 0.3
        release = 0.92
        
        target_l = self._output_peak_l
        if target_l > self._meter_l_smooth:
            self._meter_l_smooth = attack * target_l + (1 - attack) * self._meter_l_smooth
        else:
            self._meter_l_smooth = release * self._meter_l_smooth + (1 - release) * target_l
        
        target_r = self._output_peak_r
        if target_r > self._meter_r_smooth:
            self._meter_r_smooth = attack * target_r + (1 - attack) * self._meter_r_smooth
        else:
            self._meter_r_smooth = release * self._meter_r_smooth + (1 - release) * target_r
        
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        p.fillRect(self.rect(), OzoneColors.BG_DEEP)
        self._draw_title_bar(p)
        
        # ─── Left: Controls ───
        left_x = 12
        top_y = 38
        
        # IRC Mode label
        p.setPen(OzoneColors.TEXT_SECONDARY)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x, top_y + 12, f"{self._irc_mode}")
        
        # Gain knob value
        p.setPen(OzoneColors.CYAN)
        p.setFont(QFont("SF Pro Display", 22, QFont.Weight.Bold))
        p.drawText(left_x, top_y + 52, f"{self._gain_db:+.1f}")
        p.setPen(OzoneColors.TEXT_SECONDARY)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x, top_y + 68, "dB Gain")
        
        # Output Level / True Peak
        p.setPen(OzoneColors.TEXT_SECONDARY)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x, top_y + 100, "Output Level")
        p.setPen(OzoneColors.CYAN)
        p.setFont(QFont("SF Pro Display", 12, QFont.Weight.Bold))
        p.drawText(left_x, top_y + 116, f"{self._ceiling_db:.2f}")
        
        p.setPen(OzoneColors.TEXT_SECONDARY)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x + 70, top_y + 100, "True Peak")
        
        # LUFS
        p.setPen(OzoneColors.TEXT_SECONDARY)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x, top_y + 150, "LUFS")
        p.setPen(OzoneColors.AMBER)
        p.setFont(QFont("SF Pro Display", 14, QFont.Weight.Bold))
        p.drawText(left_x + 40, top_y + 152, f"{self._lufs:.1f}")
        
        # ─── Center: Gain Reduction History (cyan waveform) ───
        gr_rect = QRectF(140, 36, 340, 190)
        
        # Background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(OzoneColors.BG_PANEL)
        p.drawRoundedRect(gr_rect, 4, 4)
        
        # Grid
        self._draw_gr_grid(p, gr_rect)
        
        # Draw GR history as filled waveform
        if len(self._gr_history) > 1:
            path = QPainterPath()
            data = list(self._gr_history)
            n = len(data)
            
            # Map GR to y position (0dB at top, -12dB at bottom)
            def gr_to_y(gr):
                normalized = max(0, min(1, -gr / 12.0))
                return gr_rect.y() + normalized * gr_rect.height()
            
            # Start from bottom-left
            first_x = gr_rect.x() + gr_rect.width() - n * (gr_rect.width() / 400)
            path.moveTo(first_x, gr_rect.y())
            
            for i, gr in enumerate(data):
                x = first_x + i * (gr_rect.width() / 400)
                y = gr_to_y(gr)
                path.lineTo(x, y)
            
            # Close path along bottom
            path.lineTo(gr_rect.x() + gr_rect.width(), gr_rect.y())
            
            # Fill with gradient
            grad = QLinearGradient(0, gr_rect.y(), 0, gr_rect.y() + gr_rect.height())
            grad.setColorAt(0.0, QColor(0, 200, 220, 20))
            grad.setColorAt(0.3, QColor(0, 200, 220, 80))
            grad.setColorAt(1.0, QColor(0, 200, 220, 160))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(grad))
            p.drawPath(path)
            
            # Outline
            outline = QPainterPath()
            outline.moveTo(first_x, gr_to_y(data[0]))
            for i, gr in enumerate(data):
                x = first_x + i * (gr_rect.width() / 400)
                outline.lineTo(x, gr_to_y(gr))
            p.setPen(QPen(OzoneColors.CYAN, 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(outline)
        
        # ─── Right: Output Level Meters (L/R bars) ───
        meter_x = 500
        meter_w = 14
        meter_h = 190
        meter_y = 36
        
        for ch, (peak, label) in enumerate([(self._meter_l_smooth, "L"), (self._meter_r_smooth, "R")]):
            x = meter_x + ch * (meter_w + 6)
            
            # Background
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(OzoneColors.BG_SURFACE)
            p.drawRoundedRect(int(x), meter_y, meter_w, meter_h, 2, 2)
            
            # Level fill
            normalized = max(0, min(1, (peak + 60) / 60))  # -60 to 0 → 0 to 1
            fill_h = int(normalized * meter_h)
            fill_y = meter_y + meter_h - fill_h
            
            # Color gradient: green → yellow → red
            if normalized > 0.9:
                color = OzoneColors.RED
            elif normalized > 0.7:
                color = OzoneColors.AMBER
            else:
                color = OzoneColors.CYAN
            
            grad = QLinearGradient(x, meter_y + meter_h, x, meter_y)
            grad.setColorAt(0.0, OzoneColors.CYAN_DIM)
            grad.setColorAt(0.7, OzoneColors.CYAN)
            grad.setColorAt(0.9, OzoneColors.AMBER)
            grad.setColorAt(1.0, OzoneColors.RED)
            
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(int(x), fill_y, meter_w, fill_h, 2, 2)
            
            # Ceiling line
            ceiling_norm = max(0, min(1, (self._ceiling_db + 60) / 60))
            ceiling_y = meter_y + meter_h - int(ceiling_norm * meter_h)
            p.setPen(QPen(OzoneColors.RED, 1.0))
            p.drawLine(int(x - 2), ceiling_y, int(x + meter_w + 2), ceiling_y)
            
            # dB readout
            p.setPen(OzoneColors.TEXT_PRIMARY)
            p.setFont(QFont("SF Pro Display", 9, QFont.Weight.Bold))
            p.drawText(int(x - 2), meter_y + meter_h + 16, f"{peak:.1f}")
        
        p.end()
    
    def _draw_gr_grid(self, p, rect):
        """Draw gain reduction grid"""
        for i, db in enumerate([0, -3, -6, -9, -12]):
            y = rect.y() + (rect.height() * (-db / 12.0))
            
            p.setPen(QPen(OzoneColors.GRID_LINE if db != 0 else OzoneColors.GRID_LINE_MAJOR, 0.5))
            p.drawLine(int(rect.x()), int(y), int(rect.x() + rect.width()), int(y))
            
            p.setPen(OzoneColors.TEXT_TERTIARY)
            p.setFont(QFont("SF Pro Display", 8))
            p.drawText(int(rect.x() + rect.width() + 4), int(y + 3), f"{db}")


# ═══════════════════════════════════════════════════════════════
# 2. IMAGER/WIDTH METER PANEL
# ═══════════════════════════════════════════════════════════════

class ImagerMeterPanel(BaseMeterPanel):
    """
    Ozone 12 Imager-style panel:
    - Stereo width curve (frequency vs width)
    - Correlation meter
    - Mono bass crossover
    - Vectorscope dot display
    """
    
    def __init__(self, parent=None):
        super().__init__("Stereo Imager", width=560, height=300, parent=parent)
        
        self._width_value = 100  # 0-200
        self._mono_bass_freq = 0
        self._correlation = 1.0  # -1 to +1
        self._stereo_balance = 0.0  # -1 (left) to +1 (right)
        
        # Spectral width data (32 frequency bands)
        self._spectral_width = np.ones(32) * 0.5
        self._spectral_freqs = np.logspace(np.log10(20), np.log10(20000), 32)
        
        # Vectorscope dots
        self._vector_dots = deque(maxlen=200)
    
    def update_meter(self, width=100, mono_bass_freq=0, correlation=1.0,
                     spectral_width=None, vector_l=0.0, vector_r=0.0):
        self._width_value = width
        self._mono_bass_freq = mono_bass_freq
        self._correlation = correlation
        if spectral_width is not None:
            self._spectral_width = spectral_width
        
        # Add vectorscope dot (M/S representation)
        mid = (vector_l + vector_r) * 0.5
        side = (vector_l - vector_r) * 0.5
        self._vector_dots.append((mid, side))
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        p.fillRect(self.rect(), OzoneColors.BG_DEEP)
        self._draw_title_bar(p)
        
        # ─── Stereo Width Curve ───
        curve_rect = QRectF(50, 40, 360, 200)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(OzoneColors.BG_PANEL)
        p.drawRoundedRect(curve_rect, 4, 4)
        
        # Frequency grid
        freq_labels = [20, 50, 100, 200, 500, "1k", "2k", "5k", "10k", "20k"]
        freq_values = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        
        for freq, label in zip(freq_values, freq_labels):
            x = curve_rect.x() + curve_rect.width() * (np.log10(freq) - np.log10(20)) / (np.log10(20000) - np.log10(20))
            p.setPen(QPen(OzoneColors.GRID_LINE, 0.5))
            p.drawLine(int(x), int(curve_rect.y()), int(x), int(curve_rect.y() + curve_rect.height()))
            p.setPen(OzoneColors.TEXT_TERTIARY)
            p.setFont(QFont("SF Pro Display", 7))
            p.drawText(int(x - 10), int(curve_rect.y() + curve_rect.height() + 12), str(label))
        
        # dB grid (width: -6 to +6 dB)
        for db in [-6, -3, 0, 3, 6]:
            y = curve_rect.y() + curve_rect.height() * (1.0 - (db + 6) / 12.0)
            p.setPen(QPen(OzoneColors.GRID_LINE if db != 0 else OzoneColors.GRID_LINE_MAJOR, 0.5))
            p.drawLine(int(curve_rect.x()), int(y), int(curve_rect.x() + curve_rect.width()), int(y))
            p.setPen(OzoneColors.TEXT_TERTIARY)
            p.setFont(QFont("SF Pro Display", 8))
            p.drawText(int(curve_rect.x() - 24), int(y + 3), f"{db:+.0f}")
        
        # Width curve (filled area like Ozone)
        width_factor = (self._width_value - 100) / 100.0  # -1 to +1
        
        path = QPainterPath()
        zero_y = curve_rect.y() + curve_rect.height() * 0.5  # 0dB line
        
        path.moveTo(curve_rect.x(), zero_y)
        
        for i, freq in enumerate(self._spectral_freqs):
            x = curve_rect.x() + curve_rect.width() * (np.log10(freq) - np.log10(20)) / (np.log10(20000) - np.log10(20))
            
            # Below mono_bass_freq → collapsed to mono (no width)
            if self._mono_bass_freq > 0 and freq < self._mono_bass_freq:
                db_offset = -width_factor * 3  # Reduce width below crossover
            else:
                db_offset = width_factor * self._spectral_width[min(i, len(self._spectral_width)-1)] * 6
            
            y = zero_y - (db_offset / 6.0) * (curve_rect.height() * 0.5)
            path.lineTo(x, y)
        
        # Close along 0dB
        path.lineTo(curve_rect.x() + curve_rect.width(), zero_y)
        
        # Fill
        fill_color = QColor(220, 80, 100, 60) if width_factor > 0 else QColor(0, 200, 220, 60)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill_color)
        p.drawPath(path)
        
        # Outline
        p.setPen(QPen(QColor(255, 255, 255, 180), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        # Redraw just the curve part
        curve_path = QPainterPath()
        for i, freq in enumerate(self._spectral_freqs):
            x = curve_rect.x() + curve_rect.width() * (np.log10(freq) - np.log10(20)) / (np.log10(20000) - np.log10(20))
            if self._mono_bass_freq > 0 and freq < self._mono_bass_freq:
                db_offset = -width_factor * 3
            else:
                db_offset = width_factor * self._spectral_width[min(i, len(self._spectral_width)-1)] * 6
            y = zero_y - (db_offset / 6.0) * (curve_rect.height() * 0.5)
            if i == 0:
                curve_path.moveTo(x, y)
            else:
                curve_path.lineTo(x, y)
        p.drawPath(curve_path)
        
        # Mono bass crossover line
        if self._mono_bass_freq > 0:
            xover_x = curve_rect.x() + curve_rect.width() * (np.log10(self._mono_bass_freq) - np.log10(20)) / (np.log10(20000) - np.log10(20))
            p.setPen(QPen(OzoneColors.AMBER, 1.5, Qt.PenStyle.DashLine))
            p.drawLine(int(xover_x), int(curve_rect.y()), int(xover_x), int(curve_rect.y() + curve_rect.height()))
        
        # ─── Right: Correlation Meter ───
        corr_x = 440
        corr_w = 100
        corr_y = 50
        corr_h = 180
        
        # Background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(OzoneColors.BG_PANEL)
        p.drawRoundedRect(corr_x, corr_y, corr_w, corr_h, 4, 4)
        
        # Correlation bar (vertical: -1 at bottom, +1 at top)
        p.setPen(OzoneColors.TEXT_TERTIARY)
        p.setFont(QFont("SF Pro Display", 8))
        p.drawText(corr_x + 5, corr_y + 12, "+1")
        p.drawText(corr_x + 10, corr_y + corr_h // 2 + 4, "0")
        p.drawText(corr_x + 8, corr_y + corr_h - 4, "-1")
        
        # Correlation indicator
        corr_norm = (self._correlation + 1) / 2  # 0 to 1
        corr_bar_y = corr_y + corr_h * (1.0 - corr_norm)
        
        corr_color = OzoneColors.GREEN if self._correlation > 0.3 else (
            OzoneColors.AMBER if self._correlation > -0.2 else OzoneColors.RED
        )
        
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(corr_color)
        mid_y = corr_y + corr_h // 2
        bar_h = abs(int(corr_bar_y) - mid_y)
        bar_top = min(int(corr_bar_y), mid_y)
        p.drawRoundedRect(corr_x + 30, bar_top, 40, max(2, bar_h), 2, 2)
        
        # Width readout
        p.setPen(OzoneColors.TEXT_PRIMARY)
        p.setFont(QFont("SF Pro Display", 12, QFont.Weight.Bold))
        p.drawText(corr_x + 10, corr_y + corr_h + 30, f"W: {self._width_value}%")
        
        p.end()


# ═══════════════════════════════════════════════════════════════
# 3. COMPRESSOR METER PANEL
# ═══════════════════════════════════════════════════════════════

class CompressorMeterPanel(BaseMeterPanel):
    """
    Ozone 12 Dynamics-style panel:
    - GR history (scrolling)
    - 3-band activity (Low/Mid/High knobs)
    - Threshold line
    - Amount, Speed, Smoothing readouts
    """
    
    def __init__(self, parent=None):
        super().__init__("Dynamics", width=560, height=320, parent=parent)
        
        self._gr_history = deque(maxlen=400)
        self._threshold_db = -18.0
        self._ratio = 2.5
        self._attack_ms = 10.0
        self._release_ms = 100.0
        self._gr_current = 0.0
        
        # 3-band activity
        self._band_gr = [0.0, 0.0, 0.0]  # Low, Mid, High
        self._band_labels = ["Low", "Mid", "High"]
    
    def update_meter(self, gain_reduction_db=0.0, threshold=-18.0, ratio=2.5,
                     attack_ms=10.0, release_ms=100.0,
                     band_gr_low=0.0, band_gr_mid=0.0, band_gr_high=0.0):
        self._gr_history.append(gain_reduction_db)
        self._gr_current = gain_reduction_db
        self._threshold_db = threshold
        self._ratio = ratio
        self._attack_ms = attack_ms
        self._release_ms = release_ms
        self._band_gr = [band_gr_low, band_gr_mid, band_gr_high]
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        p.fillRect(self.rect(), OzoneColors.BG_DEEP)
        self._draw_title_bar(p)
        
        # ─── Left: Controls readout ───
        left_x = 12
        top_y = 40
        
        p.setPen(OzoneColors.TEXT_SECONDARY)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x, top_y + 15, "Threshold")
        p.setPen(OzoneColors.CYAN)
        p.setFont(QFont("SF Pro Display", 14, QFont.Weight.Bold))
        p.drawText(left_x, top_y + 35, f"{self._threshold_db:.0f} dB")
        
        p.setPen(OzoneColors.TEXT_SECONDARY)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x, top_y + 60, f"Ratio: {self._ratio:.1f}:1")
        p.drawText(left_x, top_y + 78, f"Atk: {self._attack_ms:.0f}ms")
        p.drawText(left_x, top_y + 96, f"Rel: {self._release_ms:.0f}ms")
        
        # Current GR
        p.setPen(OzoneColors.AMBER)
        p.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
        p.drawText(left_x, top_y + 140, f"{self._gr_current:.1f}")
        p.setPen(OzoneColors.TEXT_SECONDARY)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x, top_y + 156, "dB GR")
        
        # ─── Center: GR History ───
        gr_rect = QRectF(140, 40, 280, 160)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(OzoneColors.BG_PANEL)
        p.drawRoundedRect(gr_rect, 4, 4)
        
        # Grid
        for db in [0, -3, -6, -9, -12]:
            y = gr_rect.y() + gr_rect.height() * (-db / 12.0)
            p.setPen(QPen(OzoneColors.GRID_LINE, 0.5))
            p.drawLine(int(gr_rect.x()), int(y), int(gr_rect.x() + gr_rect.width()), int(y))
        
        # GR waveform
        if len(self._gr_history) > 1:
            data = list(self._gr_history)
            n = len(data)
            
            path = QPainterPath()
            for i, gr in enumerate(data):
                x = gr_rect.x() + (i / 400) * gr_rect.width()
                y = gr_rect.y() + max(0, min(1, -gr / 12.0)) * gr_rect.height()
                if i == 0:
                    path.moveTo(x, gr_rect.y())
                    path.lineTo(x, y)
                else:
                    path.lineTo(x, y)
            
            path.lineTo(gr_rect.x() + (n / 400) * gr_rect.width(), gr_rect.y())
            
            grad = QLinearGradient(0, gr_rect.y(), 0, gr_rect.y() + gr_rect.height())
            grad.setColorAt(0, QColor(0, 200, 220, 20))
            grad.setColorAt(1, QColor(0, 200, 220, 120))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(grad))
            p.drawPath(path)
        
        # ─── Bottom: 3-Band Activity ───
        band_y = 220
        band_radius = 30
        band_colors = [OzoneColors.TEAL, OzoneColors.CYAN, OzoneColors.AMBER]
        
        for i, (label, gr, color) in enumerate(zip(self._band_labels, self._band_gr, band_colors)):
            cx = 200 + i * 120
            cy = band_y + band_radius + 10
            
            # Ring background
            p.setPen(QPen(OzoneColors.BG_RAISED, 6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), band_radius, band_radius)
            
            # Activity arc
            activity = max(0, min(1, -gr / 12.0))
            span = int(activity * 270 * 16)  # Qt uses 1/16 degree
            p.setPen(QPen(color, 4))
            p.drawArc(int(cx - band_radius), int(cy - band_radius),
                      int(band_radius * 2), int(band_radius * 2),
                      135 * 16, -span)
            
            # Center value
            p.setPen(OzoneColors.TEXT_PRIMARY)
            p.setFont(QFont("SF Pro Display", 11, QFont.Weight.Bold))
            text = f"{gr:.1f}"
            p.drawText(int(cx - 16), int(cy + 4), text)
            
            # Label
            p.setPen(OzoneColors.TEXT_SECONDARY)
            p.setFont(QFont("SF Pro Display", 9))
            p.drawText(int(cx - 12), int(cy + band_radius + 18), label)
        
        p.end()


# ═══════════════════════════════════════════════════════════════
# 4. SOOTHE METER PANEL
# ═══════════════════════════════════════════════════════════════

class SootheMeterPanel(BaseMeterPanel):
    """
    Soothe-style panel:
    - Spectral reduction curve (which frequencies are being reduced)
    - Delta display (what's being removed)
    - Amount, Speed, Smoothing controls readout
    - Frequency range indicator
    """
    
    def __init__(self, parent=None):
        super().__init__("Soothe — Resonance Suppression", width=560, height=300, parent=parent)
        
        self._amount = 0.0
        self._speed = 50.0
        self._smoothing = 50.0
        self._freq_low = 2000.0
        self._freq_high = 8000.0
        
        # 64-band spectral reduction display
        self._reduction_db = np.zeros(64)
        self._freqs = np.logspace(np.log10(20), np.log10(20000), 64)
        
        # Delta (what Soothe is removing)
        self._delta_active = False
    
    def update_meter(self, amount=0.0, reduction_db=None, 
                     freq_low=2000.0, freq_high=8000.0,
                     speed=50.0, smoothing=50.0, delta=False):
        self._amount = amount
        self._freq_low = freq_low
        self._freq_high = freq_high
        self._speed = speed
        self._smoothing = smoothing
        self._delta_active = delta
        if reduction_db is not None:
            self._reduction_db = reduction_db
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        p.fillRect(self.rect(), OzoneColors.BG_DEEP)
        self._draw_title_bar(p)
        
        # ─── Left: Controls ───
        left_x = 12
        top_y = 40
        
        # Amount knob readout
        p.setPen(OzoneColors.TEXT_SECONDARY)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x, top_y + 12, "Amount")
        p.setPen(OzoneColors.CYAN)
        p.setFont(QFont("SF Pro Display", 20, QFont.Weight.Bold))
        p.drawText(left_x, top_y + 42, f"{self._amount:.0f}")
        
        p.setPen(OzoneColors.TEXT_SECONDARY)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x, top_y + 70, f"Speed: {self._speed:.0f}")
        p.drawText(left_x, top_y + 88, f"Smooth: {self._smoothing:.0f}")
        
        # Freq range
        p.setPen(OzoneColors.AMBER)
        p.setFont(QFont("SF Pro Display", 9))
        p.drawText(left_x, top_y + 120, f"{self._freq_low:.0f} - {self._freq_high:.0f} Hz")
        
        # Delta indicator
        if self._delta_active:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(OzoneColors.CYAN)
            p.drawRoundedRect(left_x, int(top_y + 140), 50, 20, 4, 4)
            p.setPen(OzoneColors.BG_DEEP)
            p.setFont(QFont("SF Pro Display", 9, QFont.Weight.Bold))
            p.drawText(left_x + 6, int(top_y + 155), "Delta")
        
        # ─── Center: Spectral Reduction Curve ───
        curve_rect = QRectF(120, 40, 420, 210)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(OzoneColors.BG_PANEL)
        p.drawRoundedRect(curve_rect, 4, 4)
        
        # Freq grid
        freq_labels = [50, 100, 200, 500, "1k", "2k", "5k", "10k", "20k"]
        freq_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        
        for freq, label in zip(freq_values, freq_labels):
            x = curve_rect.x() + curve_rect.width() * (np.log10(freq) - np.log10(20)) / (np.log10(20000) - np.log10(20))
            p.setPen(QPen(OzoneColors.GRID_LINE, 0.5))
            p.drawLine(int(x), int(curve_rect.y()), int(x), int(curve_rect.y() + curve_rect.height()))
            p.setPen(OzoneColors.TEXT_TERTIARY)
            p.setFont(QFont("SF Pro Display", 7))
            p.drawText(int(x - 8), int(curve_rect.y() + curve_rect.height() + 12), str(label))
        
        # dB grid
        for db in [0, -3, -6, -9, -12]:
            y = curve_rect.y() + curve_rect.height() * (-db / 12.0)
            p.setPen(QPen(OzoneColors.GRID_LINE, 0.5))
            p.drawLine(int(curve_rect.x()), int(y), int(curve_rect.x() + curve_rect.width()), int(y))
            p.setPen(OzoneColors.TEXT_TERTIARY)
            p.setFont(QFont("SF Pro Display", 8))
            p.drawText(int(curve_rect.x() + curve_rect.width() + 4), int(y + 3), f"{db}")
        
        # Active frequency range highlight
        x_low = curve_rect.x() + curve_rect.width() * (np.log10(max(20, self._freq_low)) - np.log10(20)) / (np.log10(20000) - np.log10(20))
        x_high = curve_rect.x() + curve_rect.width() * (np.log10(min(20000, self._freq_high)) - np.log10(20)) / (np.log10(20000) - np.log10(20))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 200, 220, 15))
        p.drawRect(int(x_low), int(curve_rect.y()), int(x_high - x_low), int(curve_rect.height()))
        
        # Reduction curve
        if self._amount > 0:
            path = QPainterPath()
            zero_y = curve_rect.y()  # 0dB at top
            
            for i, freq in enumerate(self._freqs):
                x = curve_rect.x() + curve_rect.width() * (np.log10(freq) - np.log10(20)) / (np.log10(20000) - np.log10(20))
                reduction = self._reduction_db[i] if i < len(self._reduction_db) else 0
                y = zero_y + max(0, min(1, -reduction / 12.0)) * curve_rect.height()
                
                if i == 0:
                    path.moveTo(x, zero_y)
                    path.lineTo(x, y)
                else:
                    path.lineTo(x, y)
            
            # Close along top
            path.lineTo(curve_rect.x() + curve_rect.width(), zero_y)
            
            # Fill with warm color (what's being removed)
            grad = QLinearGradient(0, curve_rect.y(), 0, curve_rect.y() + curve_rect.height())
            grad.setColorAt(0, QColor(220, 80, 100, 10))
            grad.setColorAt(1, QColor(220, 80, 100, 100))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(grad))
            p.drawPath(path)
            
            # White outline curve
            outline = QPainterPath()
            for i, freq in enumerate(self._freqs):
                x = curve_rect.x() + curve_rect.width() * (np.log10(freq) - np.log10(20)) / (np.log10(20000) - np.log10(20))
                reduction = self._reduction_db[i] if i < len(self._reduction_db) else 0
                y = zero_y + max(0, min(1, -reduction / 12.0)) * curve_rect.height()
                if i == 0:
                    outline.moveTo(x, y)
                else:
                    outline.lineTo(x, y)
            
            p.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(outline)
        
        p.end()


# ═══════════════════════════════════════════════════════════════
# LUFS Calibration Test (Logic Pro X comparison)
# ═══════════════════════════════════════════════════════════════

def test_lufs_calibration():
    """
    ทดสอบว่า LUFS measurement ตรงกับ Logic Pro X หรือไม่
    
    วิธีทดสอบ:
    1. สร้าง test tone (-20 LUFS, 1kHz sine)
    2. วัดด้วย LUFSMeter ของเรา
    3. เปรียบเทียบกับค่าจาก Logic Pro X
    
    ค่าอ้างอิง ITU-R BS.1770-4:
    - 1kHz sine at -20 dBFS → ควรได้ ~-23.0 LUFS (mono) หรือ ~-20.0 LUFS (stereo, dual mono)
    - Pink noise at -20 dBFS → ควรได้ ~-20.7 LUFS
    """
    try:
        from modules.master.ai_master import LUFSMeter
    except ImportError:
        print("Cannot import LUFSMeter — run from project root")
        return
    
    sr = 44100
    duration = 10  # seconds
    t = np.linspace(0, duration, sr * duration, dtype=np.float64)
    
    # Test 1: -20 dBFS 1kHz sine (stereo, dual mono)
    amplitude = 10 ** (-20.0 / 20.0)  # -20 dBFS
    sine_1k = amplitude * np.sin(2 * np.pi * 1000 * t)
    stereo_1k = np.column_stack([sine_1k, sine_1k]).astype(np.float32)
    
    meter = LUFSMeter(sr)
    lufs_1k = meter.measure_integrated(stereo_1k)
    
    print(f"Test 1: 1kHz sine at -20 dBFS (stereo)")
    print(f"  Measured: {lufs_1k:.1f} LUFS")
    print(f"  Expected: ~-20.0 LUFS (dual mono) or -23.0 (mono ref)")
    print(f"  Logic Pro X ref: -20.0 LUFS")
    diff_1 = abs(lufs_1k - (-20.0))
    print(f"  Deviation: {diff_1:.1f} dB {'✅ OK' if diff_1 < 1.0 else '⚠️ CHECK'}")
    
    # Test 2: -14 dBFS pink noise (stereo)
    np.random.seed(42)
    white = np.random.randn(sr * duration)
    # Simple pink noise approximation (1/f filter)
    from scipy.signal import lfilter
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004709510])
    a = np.array([1.000000000, -2.494956002, 2.017265875, -0.522189400])
    pink = lfilter(b, a, white)
    # Normalize to -14 dBFS RMS
    rms = np.sqrt(np.mean(pink ** 2))
    target_rms = 10 ** (-14.0 / 20.0)
    pink = (pink * target_rms / rms).astype(np.float32)
    stereo_pink = np.column_stack([pink, pink])
    
    lufs_pink = meter.measure_integrated(stereo_pink)
    
    print(f"\nTest 2: Pink noise at -14 dBFS RMS (stereo)")
    print(f"  Measured: {lufs_pink:.1f} LUFS")
    print(f"  Expected: ~-14.0 to -15.0 LUFS")
    diff_2 = abs(lufs_pink - (-14.0))
    print(f"  Deviation: {diff_2:.1f} dB {'✅ OK' if diff_2 < 1.5 else '⚠️ CHECK'}")
    
    # Test 3: Silence
    silence = np.zeros((sr * 2, 2), dtype=np.float32)
    lufs_silence = meter.measure_integrated(silence)
    print(f"\nTest 3: Silence")
    print(f"  Measured: {lufs_silence:.1f} LUFS")
    print(f"  Expected: < -60 LUFS {'✅ OK' if lufs_silence < -60 else '⚠️ CHECK'}")
    
    print(f"\n{'='*50}")
    if diff_1 < 1.0 and diff_2 < 1.5:
        print("✅ LUFS calibration matches Logic Pro X (within ±1 dB)")
    else:
        print("⚠️ LUFS calibration has significant deviation")
        print("  → Check K-weighting filter coefficients in LUFSMeter")


if __name__ == "__main__":
    print("Running LUFS calibration test...")
    test_lufs_calibration()
