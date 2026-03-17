"""
ProductionPipelineDialog — 7-step mastering pipeline wizard.

Steps: Import → AI DJ → Compile → Master → Hook Extract (Before) →
       Hook Extract (After) → Video + Export
"""

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QProgressBar, QStackedWidget, QWidget, QListWidget, QFileDialog,
        QCheckBox, QComboBox,
    )
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont
except ImportError:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QProgressBar, QStackedWidget, QWidget, QListWidget, QFileDialog,
        QCheckBox, QComboBox,
    )
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont


STEPS = [
    ("IMPORT", "Select audio files to master"),
    ("AI DJ", "Auto-order tracks for optimal flow"),
    ("COMPILE", "Arrange tracks with crossfades"),
    ("MASTER", "Apply mastering chain to all tracks"),
    ("HOOKS (BEFORE)", "Extract hooks from original tracks"),
    ("HOOKS (AFTER)", "Extract hooks from mastered tracks"),
    ("EXPORT", "Generate final output files"),
]


class ProductionPipelineDialog(QDialog):
    """7-step production pipeline wizard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Production Pipeline")
        self.setFixedSize(600, 450)
        self.setStyleSheet("""
            QDialog { background: #111114; color: #E8E4DC; }
            QLabel { color: #E8E4DC; }
            QPushButton {
                background: #1A1A1E; border: 1px solid #0077B6; border-radius: 4px;
                color: #48CAE4; font-weight: bold; font-size: 10px; padding: 6px 16px;
            }
            QPushButton:hover { background: #0077B6; color: #FFF; }
            QPushButton:disabled { color: #4A4844; border-color: #2A2A30; }
            QListWidget { background: #0A0A0C; border: 1px solid #2A2A30; color: #E8E4DC; }
        """)

        self._current_step = 0
        self._files = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Title + step indicator
        title_row = QHBoxLayout()
        self._title = QLabel("PRODUCTION PIPELINE")
        self._title.setFont(QFont("Menlo", 12, QFont.Weight.Bold))
        self._title.setStyleSheet("color: #48CAE4; letter-spacing: 2px;")
        title_row.addWidget(self._title)
        title_row.addStretch()
        self._step_label = QLabel("Step 1 of 7")
        self._step_label.setStyleSheet("color: #8E8A82; font-size: 10px;")
        title_row.addWidget(self._step_label)
        layout.addLayout(title_row)

        # Step progress dots
        dots_row = QHBoxLayout()
        self._dots = []
        for i, (name, _) in enumerate(STEPS):
            dot = QLabel(f"{i+1}")
            dot.setFixedSize(24, 24)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setStyleSheet(
                "background: #2A2A30; border-radius: 12px; color: #8E8A82; font-size: 9px; font-weight: bold;")
            dots_row.addWidget(dot)
            self._dots.append(dot)
        dots_row.addStretch()
        layout.addLayout(dots_row)

        # Content stack
        self._stack = QStackedWidget()
        for i, (name, desc) in enumerate(STEPS):
            page = self._build_step_page(i, name, desc)
            self._stack.addWidget(page)
        layout.addWidget(self._stack, stretch=1)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet("""
            QProgressBar { background: #1A1A1E; border: none; border-radius: 3px; }
            QProgressBar::chunk { background: #00B4D8; border-radius: 3px; }
        """)
        layout.addWidget(self._progress)

        # Navigation buttons
        nav_row = QHBoxLayout()
        self._btn_back = QPushButton("BACK")
        self._btn_back.clicked.connect(self._go_back)
        self._btn_back.setEnabled(False)
        nav_row.addWidget(self._btn_back)

        self._btn_skip = QPushButton("SKIP")
        self._btn_skip.clicked.connect(self._go_next)
        nav_row.addWidget(self._btn_skip)

        nav_row.addStretch()

        self._btn_next = QPushButton("NEXT")
        self._btn_next.clicked.connect(self._go_next)
        self._btn_next.setStyleSheet(
            "QPushButton { background: #00B4D8; color: #FFF; font-weight: bold; "
            "font-size: 11px; padding: 8px 24px; border-radius: 4px; border: none; }")
        nav_row.addWidget(self._btn_next)
        layout.addLayout(nav_row)

        self._update_step_ui()

    def _build_step_page(self, index, name, desc):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 8)

        header = QLabel(name)
        header.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        header.setStyleSheet("color: #E8A832;")
        layout.addWidget(header)

        desc_label = QLabel(desc)
        desc_label.setStyleSheet("color: #8E8A82; font-size: 11px;")
        layout.addWidget(desc_label)

        if index == 0:  # Import
            self._file_list = QListWidget()
            layout.addWidget(self._file_list, stretch=1)
            add_btn = QPushButton("ADD FILES")
            add_btn.clicked.connect(self._add_files)
            layout.addWidget(add_btn)
        else:
            # Placeholder content for other steps
            status = QLabel(f"Ready to {name.lower()}")
            status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status.setFont(QFont("Menlo", 11))
            status.setStyleSheet("color: #48CAE4;")
            layout.addWidget(status, stretch=1)

        return page

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "Audio Files (*.wav *.flac *.mp3 *.aif *.aiff *.ogg *.m4a)")
        for f in files:
            self._files.append(f)
            import os
            self._file_list.addItem(os.path.basename(f))

    def _go_next(self):
        if self._current_step < len(STEPS) - 1:
            self._current_step += 1
            self._update_step_ui()

    def _go_back(self):
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step_ui()

    def _update_step_ui(self):
        self._stack.setCurrentIndex(self._current_step)
        self._step_label.setText(f"Step {self._current_step + 1} of {len(STEPS)}")
        self._btn_back.setEnabled(self._current_step > 0)
        self._btn_next.setText("FINISH" if self._current_step == len(STEPS) - 1 else "NEXT")
        self._progress.setValue(int((self._current_step / (len(STEPS) - 1)) * 100))

        for i, dot in enumerate(self._dots):
            if i < self._current_step:
                dot.setStyleSheet(
                    "background: #00B4D8; border-radius: 12px; color: #FFF; font-size: 9px; font-weight: bold;")
            elif i == self._current_step:
                dot.setStyleSheet(
                    "background: #E8A832; border-radius: 12px; color: #111; font-size: 9px; font-weight: bold;")
            else:
                dot.setStyleSheet(
                    "background: #2A2A30; border-radius: 12px; color: #8E8A82; font-size: 9px; font-weight: bold;")

        if self._current_step == len(STEPS) - 1 and self._btn_next.text() == "FINISH":
            self._btn_next.clicked.disconnect()
            self._btn_next.clicked.connect(self.accept)
