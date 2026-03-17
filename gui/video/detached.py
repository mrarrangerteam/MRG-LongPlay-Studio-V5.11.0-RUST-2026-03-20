"""
Detached (floating) video preview window.

Classes:
    DetachedVideoWindow — Floating video preview that can be moved to any monitor
"""

from gui.utils.compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, Qt, QPixmap, QImage, pyqtSignal,
)
from gui.styles import Colors

# OpenCV availability
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


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
        self.pin_btn = QPushButton("\U0001f4cc")
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
        self.dock_btn = QPushButton("\u2b05 Dock")
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
        self.track_label.setText(f"{track_name} \u00b7 {time_str}")

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
