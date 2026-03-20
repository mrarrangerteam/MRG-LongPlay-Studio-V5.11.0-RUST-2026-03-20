"""
LoudnessHistoryWidget — Scrolling LUFS timeline graph.
"""

import collections
try:
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtCore import Qt, QPointF
    from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath
except ImportError:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath


class LoudnessHistoryWidget(QWidget):
    """
    Scrolling time-series graph of LUFS over time.

    Three traces: Momentary (thin), Short-term (medium), Integrated (thick dashed).
    Target LUFS line (horizontal dashed teal).
    """

    MAX_SAMPLES = 300  # ~30 seconds at 10 Hz
    DB_MIN = -60.0
    DB_MAX = 0.0

    def __init__(self, target_lufs: float = -14.0, parent=None):
        super().__init__(parent)
        self.setFixedSize(260, 100)
        self._target = target_lufs
        self._momentary = collections.deque(maxlen=self.MAX_SAMPLES)
        self._short_term = collections.deque(maxlen=self.MAX_SAMPLES)
        self._integrated = collections.deque(maxlen=self.MAX_SAMPLES)

    def set_target(self, target_lufs: float):
        self._target = target_lufs
        self.update()

    def append_levels(self, momentary: float = -70.0, short_term: float = -70.0,
                      integrated: float = -70.0):
        self._momentary.append(max(self.DB_MIN, min(self.DB_MAX, momentary)))
        self._short_term.append(max(self.DB_MIN, min(self.DB_MAX, short_term)))
        self._integrated.append(max(self.DB_MIN, min(self.DB_MAX, integrated)))
        self.update()

    def reset(self):
        self._momentary.clear()
        self._short_term.clear()
        self._integrated.clear()
        self.update()

    def _db_to_y(self, db, plot_y, plot_h):
        db = max(self.DB_MIN, min(self.DB_MAX, db))
        ratio = (db - self.DB_MIN) / (self.DB_MAX - self.DB_MIN)
        return plot_y + plot_h - int(ratio * plot_h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        margin = 24
        plot_x = margin
        plot_y = 4
        plot_w = w - margin - 4
        plot_h = h - 8

        # Background
        p.fillRect(0, 0, w, h, QColor("#0A0A0C"))
        p.fillRect(plot_x, plot_y, plot_w, plot_h, QColor("#111114"))
        p.setPen(QPen(QColor("#1E1E22"), 1))
        p.drawRect(plot_x, plot_y, plot_w, plot_h)

        # Grid + dB labels
        p.setFont(QFont("Menlo", 6))
        for db in [-48, -36, -24, -14, -6, 0]:
            y = self._db_to_y(db, plot_y, plot_h)
            p.setPen(QPen(QColor("#1E1E22"), 1, Qt.PenStyle.DotLine))
            p.drawLine(plot_x, y, plot_x + plot_w, y)
            p.setPen(QColor("#8E8A82"))
            p.drawText(2, y + 3, f"{db:>3}")

        # Target line
        target_y = self._db_to_y(self._target, plot_y, plot_h)
        p.setPen(QPen(QColor("#00B4D8"), 1, Qt.PenStyle.DashLine))
        p.drawLine(plot_x, target_y, plot_x + plot_w, target_y)

        # Draw traces
        traces = [
            (self._momentary, QColor(72, 202, 228, 100), 1, Qt.PenStyle.SolidLine),     # thin teal
            (self._short_term, QColor(0, 180, 216, 180), 2, Qt.PenStyle.SolidLine),      # medium teal
            (self._integrated, QColor(232, 168, 50, 200), 2, Qt.PenStyle.DashLine),       # thick amber dashed
        ]

        for data, color, width, style in traces:
            if len(data) < 2:
                continue
            path = QPainterPath()
            n = len(data)
            for i, val in enumerate(data):
                x = plot_x + int(i / max(1, self.MAX_SAMPLES - 1) * plot_w)
                y = self._db_to_y(val, plot_y, plot_h)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.setPen(QPen(color, width, style))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        # Legend
        p.setFont(QFont("Menlo", 6))
        lx = plot_x + 4
        p.setPen(QColor(72, 202, 228))
        p.drawText(lx, plot_y + 8, "M")
        p.setPen(QColor(0, 180, 216))
        p.drawText(lx + 12, plot_y + 8, "S")
        p.setPen(QColor(232, 168, 50))
        p.drawText(lx + 24, plot_y + 8, "I")

        p.end()
