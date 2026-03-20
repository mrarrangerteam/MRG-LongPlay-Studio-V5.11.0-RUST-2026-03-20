"""
TransferCurveWidget — Compressor transfer curve (input vs output dB).
"""

import math
try:
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtCore import Qt, QPointF
    from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath
except ImportError:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath


class TransferCurveWidget(QWidget):
    """
    Transfer curve display: input dB (X) vs output dB (Y).

    Shows 1:1 reference line + compression curve with soft knee.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self._threshold = -20.0   # dB
        self._ratio = 4.0         # :1
        self._knee = 6.0          # dB
        self._makeup = 0.0        # dB

    def set_params(self, threshold: float = -20.0, ratio: float = 4.0,
                   knee: float = 6.0, makeup: float = 0.0):
        self._threshold = threshold
        self._ratio = max(1.0, ratio)
        self._knee = max(0.0, knee)
        self._makeup = makeup
        self.update()

    def _compute_output(self, input_db: float) -> float:
        """Compute output dB from input dB using soft-knee compression."""
        t = self._threshold
        r = self._ratio
        k = self._knee

        if k <= 0:
            # Hard knee
            if input_db < t:
                out = input_db
            else:
                out = t + (input_db - t) / r
        else:
            half_k = k / 2.0
            if input_db < (t - half_k):
                out = input_db
            elif input_db > (t + half_k):
                out = t + (input_db - t) / r
            else:
                # Soft knee region (quadratic interpolation)
                x = input_db - t + half_k
                out = input_db + (1.0 / r - 1.0) * x * x / (2.0 * k)

        return out + self._makeup

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        db_min, db_max = -60.0, 0.0
        db_range = db_max - db_min
        margin = 24
        plot_w = w - margin - 4
        plot_h = h - margin - 4
        plot_x = margin
        plot_y = 4

        # Background
        p.fillRect(0, 0, w, h, QColor("#0A0A0C"))

        # Plot area background
        p.fillRect(plot_x, plot_y, plot_w, plot_h, QColor("#111114"))
        p.setPen(QPen(QColor("#1E1E22"), 1))
        p.drawRect(plot_x, plot_y, plot_w, plot_h)

        def db_to_x(db):
            return plot_x + int((db - db_min) / db_range * plot_w)

        def db_to_y(db):
            return plot_y + plot_h - int((db - db_min) / db_range * plot_h)

        # Grid lines
        p.setPen(QPen(QColor("#1E1E22"), 1, Qt.PenStyle.DotLine))
        for db in [-48, -36, -24, -12, -6, 0]:
            x = db_to_x(db)
            y = db_to_y(db)
            p.drawLine(x, plot_y, x, plot_y + plot_h)
            p.drawLine(plot_x, y, plot_x + plot_w, y)

        # Axis labels
        p.setFont(QFont("Menlo", 6))
        p.setPen(QColor("#8E8A82"))
        for db in [-48, -24, -12, 0]:
            y = db_to_y(db)
            p.drawText(2, y + 3, f"{db:>3}")
            x = db_to_x(db)
            p.drawText(x - 6, h - 2, f"{db}")

        # 1:1 reference line (dashed gray)
        p.setPen(QPen(QColor("#3E3E46"), 1, Qt.PenStyle.DashLine))
        p.drawLine(db_to_x(db_min), db_to_y(db_min),
                   db_to_x(db_max), db_to_y(db_max))

        # Compression curve
        path = QPainterPath()
        steps = 200
        for i in range(steps + 1):
            input_db = db_min + (i / steps) * db_range
            output_db = self._compute_output(input_db)
            output_db = max(db_min, min(db_max, output_db))
            px = db_to_x(input_db)
            py = db_to_y(output_db)
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)

        p.setPen(QPen(QColor("#00B4D8"), 2))
        p.drawPath(path)

        # Threshold marker
        tx = db_to_x(self._threshold)
        p.setPen(QPen(QColor("#E8A832"), 1, Qt.PenStyle.DashLine))
        p.drawLine(tx, plot_y, tx, plot_y + plot_h)

        # Labels
        p.setFont(QFont("Menlo", 7))
        p.setPen(QColor("#8E8A82"))
        p.drawText(plot_x + 4, plot_y + 10, f"T:{self._threshold:.0f} R:{self._ratio:.1f}:1")

        p.end()
