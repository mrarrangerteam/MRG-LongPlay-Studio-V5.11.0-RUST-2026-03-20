"""
Qt compatibility layer — try PyQt6 first, fall back to PySide6.

Every GUI module in the gui/ package should import Qt symbols from here:

    from gui.utils.compat import (
        QMainWindow, QWidget, Qt, pyqtSignal, QTimer, ...
    )

This avoids duplicating the try/except block in every file.
"""

# ---------- flag -----------------------------------------------------------
PYQT6: bool

# ---------- try PyQt6 first ------------------------------------------------
try:
    from PyQt6.QtWidgets import (  # noqa: F401
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QFrame, QScrollArea, QSlider, QComboBox,
        QTabWidget, QListWidget, QListWidgetItem, QProgressBar,
        QSplitter, QSizePolicy, QSpacerItem, QStackedWidget,
        QApplication, QStyle, QStyleOption, QFileDialog, QDialog,
        QLineEdit, QTextEdit, QCheckBox, QGroupBox, QToolButton,
        QMessageBox, QMenu, QSpinBox, QDoubleSpinBox, QDial,
        QGraphicsOpacityEffect,
        QGraphicsView, QGraphicsScene, QGraphicsRectItem,
        QGraphicsLineItem, QGraphicsTextItem, QGraphicsItem,
        QGraphicsSceneMouseEvent,
    )
    from PyQt6.QtCore import (  # noqa: F401
        Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QUrl,
        QMimeData, QPoint, QPointF, QThread, QRect, QRectF,
    )
    from PyQt6.QtCore import pyqtSignal, pyqtSlot  # noqa: F401
    from PyQt6.QtGui import (  # noqa: F401
        QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QPalette,
        QIcon, QPixmap, QDragEnterEvent, QDropEvent, QPolygon, QAction,
        QShortcut, QKeySequence, QImage, QPainterPath, QCursor,
        QWheelEvent, QMouseEvent,
    )
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput  # noqa: F401
    from PyQt6.QtMultimediaWidgets import QVideoWidget  # noqa: F401
    PYQT6 = True

except ImportError:
    from PySide6.QtWidgets import (  # noqa: F401
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QFrame, QScrollArea, QSlider, QComboBox,
        QTabWidget, QListWidget, QListWidgetItem, QProgressBar,
        QSplitter, QSizePolicy, QSpacerItem, QStackedWidget,
        QApplication, QStyle, QStyleOption, QFileDialog, QDialog,
        QLineEdit, QTextEdit, QCheckBox, QGroupBox, QToolButton,
        QMessageBox, QMenu, QSpinBox, QDoubleSpinBox, QDial,
        QGraphicsOpacityEffect,
        QGraphicsView, QGraphicsScene, QGraphicsRectItem,
        QGraphicsLineItem, QGraphicsTextItem, QGraphicsItem,
        QGraphicsSceneMouseEvent,
    )
    from PySide6.QtCore import (  # noqa: F401
        Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QUrl,
        QMimeData, QPoint, QPointF, QThread, QRect, QRectF,
    )
    from PySide6.QtCore import Signal as pyqtSignal, Slot as pyqtSlot  # noqa: F401
    from PySide6.QtGui import (  # noqa: F401
        QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QPalette,
        QIcon, QPixmap, QDragEnterEvent, QDropEvent, QPolygon, QAction,
        QShortcut, QKeySequence, QImage, QPainterPath, QCursor,
        QWheelEvent, QMouseEvent,
    )
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput  # noqa: F401
    from PySide6.QtMultimediaWidgets import QVideoWidget  # noqa: F401
    PYQT6 = False
