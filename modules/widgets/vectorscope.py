"""
VectorscopeWidget — Lissajous stereo field display with correlation meter.
"""

import numpy as np
try:
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtCore import Qt, QPointF
    from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QImage
except ImportError:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QImage


class VectorscopeWidget(QWidget):
    """
    Stereo vectorscope with phosphor-style dots and correlation bar.

    X-axis: L-R (stereo difference)
    Y-axis: L+R (stereo sum)
    Phosphor decay: 85% alpha retention per frame.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 230)  # extra 30px for correlation bar
        self._phosphor = None  # QImage for phosphor buffer
        self._correlation = 0.0
        self._init_phosphor()

    def _init_phosphor(self):
        self._phosphor = QImage(200, 200, QImage.Format.Format_ARGB32_Premultiplied)
        self._phosphor.fill(QColor(0, 0, 0, 0))

    def set_audio_data(self, left: np.ndarray, right: np.ndarray):
        """Feed stereo audio data. Arrays should be float, same length."""
        if left is None or right is None or len(left) == 0:
            return

        # Downsample to ~4096 points for performance
        n = len(left)
        if n > 4096:
            step = n // 4096
            left = left[::step]
            right = right[::step]

        # Compute Lissajous coordinates
        x = left - right  # L-R (difference)
        y = left + right  # L+R (sum)

        # Correlation
        if n > 1:
            l_std = np.std(left)
            r_std = np.std(right)
            if l_std > 1e-10 and r_std > 1e-10:
                self._correlation = float(np.corrcoef(left, right)[0, 1])
            else:
                self._correlation = 1.0
        else:
            self._correlation = 1.0

        # Decay existing phosphor (multiply alpha by 0.85)
        if self._phosphor is not None:
            # Use a fresh painter on the phosphor image to apply decay
            p = QPainter(self._phosphor)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            p.fillRect(0, 0, 200, 200, QColor(0, 0, 0, 217))  # 217/255 ≈ 0.85
            p.end()

            # Draw new dots
            p = QPainter(self._phosphor)
            p.setPen(Qt.PenStyle.NoPen)
            dot_color = QColor(34, 197, 94, 180)  # green phosphor
            p.setBrush(QBrush(dot_color))

            cx, cy = 100.0, 100.0
            scale = 80.0  # pixels per unit

            for i in range(len(x)):
                px = cx + x[i] * scale
                py = cy - y[i] * scale
                if 0 <= px < 200 and 0 <= py < 200:
                    p.drawEllipse(QPointF(px, py), 1.0, 1.0)
            p.end()

        self.update()

    def reset(self):
        self._init_phosphor()
        self._correlation = 0.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        scope_h = 200

        # Background
        p.fillRect(0, 0, w, scope_h, QColor("#0A0A0C"))
        p.setPen(QPen(QColor("#1A1A1E"), 1))
        p.drawRect(0, 0, w - 1, scope_h - 1)

        # Grid lines
        p.setPen(QPen(QColor("#1E1E22"), 1, Qt.PenStyle.DotLine))
        cx, cy = 100, 100
        p.drawLine(0, cy, w, cy)       # horizontal center
        p.drawLine(cx, 0, cx, scope_h) # vertical center
        # Diagonal lines (L and R axes)
        p.setPen(QPen(QColor("#1E1E22"), 1))
        p.drawLine(0, scope_h, w, 0)  # L axis
        p.drawLine(0, 0, w, scope_h)  # R axis

        # L/R labels
        p.setFont(QFont("Menlo", 8))
        p.setPen(QColor("#8E8A82"))
        p.drawText(5, scope_h - 5, "L")
        p.drawText(w - 12, scope_h - 5, "R")
        p.drawText(cx - 3, 12, "M")

        # Draw phosphor buffer
        if self._phosphor is not None:
            p.drawImage(0, 0, self._phosphor)

        # ── Correlation bar (below scope) ──
        bar_y = scope_h + 6
        bar_h = 8
        bar_margin = 10
        bar_w = w - 2 * bar_margin

        # Bar background
        p.fillRect(bar_margin, bar_y, bar_w, bar_h, QColor("#1A1A1E"))
        p.setPen(QPen(QColor("#2A2A30"), 1))
        p.drawRect(bar_margin, bar_y, bar_w, bar_h)

        # Correlation indicator
        corr_x = bar_margin + int((self._correlation + 1.0) / 2.0 * bar_w)
        corr_x = max(bar_margin, min(bar_margin + bar_w, corr_x))

        # Color: red (-1) → yellow (0) → green (+1)
        if self._correlation > 0.5:
            corr_color = QColor("#22C55E")
        elif self._correlation > 0.0:
            corr_color = QColor("#FACC15")
        elif self._correlation > -0.5:
            corr_color = QColor("#F59E0B")
        else:
            corr_color = QColor("#EF4444")

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(corr_color))
        p.drawRect(corr_x - 2, bar_y, 4, bar_h)

        # Labels
        p.setFont(QFont("Menlo", 7))
        p.setPen(QColor("#8E8A82"))
        p.drawText(bar_margin, bar_y + bar_h + 10, "-1")
        p.drawText(bar_margin + bar_w // 2 - 3, bar_y + bar_h + 10, "0")
        p.drawText(bar_margin + bar_w - 8, bar_y + bar_h + 10, "+1")

        p.end()
