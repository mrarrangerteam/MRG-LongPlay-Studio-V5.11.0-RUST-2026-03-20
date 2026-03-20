"""
History panel widget showing undo/redo action names.

Provides:
    HistoryPanel — QWidget list displaying the command history stack
"""

from __future__ import annotations

from typing import Optional

from gui.utils.compat import (
    QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QHBoxLayout, QShortcut, QKeySequence,
    Qt, pyqtSignal, QFont, QColor,
)
from gui.styles import Colors
from gui.models.commands import CommandHistory


class HistoryPanel(QWidget):
    """Displays the undo/redo history and provides undo/redo buttons."""

    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()

    def __init__(
        self,
        history: Optional[CommandHistory] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._history = history or CommandHistory(on_change=self._refresh)
        # Wire on_change if history was provided without one
        if history is not None and history._on_change is None:
            history._on_change = self._refresh

        self._setup_ui()
        self._refresh()

    @property
    def history(self) -> CommandHistory:
        return self._history

    def set_history(self, history: CommandHistory) -> None:
        self._history = history
        history._on_change = self._refresh
        self._refresh()

    # -- UI ----------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("History")
        title.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-weight: bold; font-size: 12px;"
        )
        layout.addWidget(title)

        self._list = QListWidget()
        self._list.setStyleSheet(f"""
            QListWidget {{
                background: {Colors.BG_PRIMARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                font-size: 11px;
            }}
            QListWidget::item:selected {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.ACCENT};
            }}
        """)
        layout.addWidget(self._list, 1)

        # Buttons
        btn_row = QHBoxLayout()
        self._undo_btn = QPushButton("Undo")
        self._redo_btn = QPushButton("Redo")
        for btn in (self._undo_btn, self._redo_btn):
            btn.setFixedHeight(24)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.BG_TERTIARY};
                    color: {Colors.TEXT_PRIMARY};
                    border: 1px solid {Colors.BORDER};
                    border-radius: 3px;
                    padding: 2px 8px;
                    font-size: 11px;
                }}
                QPushButton:hover {{ background: {Colors.ACCENT_DIM}; }}
                QPushButton:disabled {{ color: {Colors.TEXT_TERTIARY}; }}
            """)
        self._undo_btn.clicked.connect(self._on_undo)
        self._redo_btn.clicked.connect(self._on_redo)
        btn_row.addWidget(self._undo_btn)
        btn_row.addWidget(self._redo_btn)
        layout.addLayout(btn_row)

        # Keyboard shortcuts (Cmd+Z / Cmd+Shift+Z)
        undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_sc.activated.connect(self._on_undo)
        redo_sc = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        redo_sc.activated.connect(self._on_redo)

    # -- slots -------------------------------------------------------------
    def _on_undo(self) -> None:
        if self._history.can_undo:
            self._history.undo()
            self.undo_requested.emit()

    def _on_redo(self) -> None:
        if self._history.can_redo:
            self._history.redo()
            self.redo_requested.emit()

    # -- refresh -----------------------------------------------------------
    def _refresh(self) -> None:
        self._list.clear()

        for cmd in self._history.undo_stack:
            item = QListWidgetItem(cmd.description)
            item.setForeground(QColor(Colors.TEXT_PRIMARY))
            self._list.addItem(item)

        for cmd in reversed(self._history.redo_stack):
            item = QListWidgetItem(f"  {cmd.description}")
            item.setForeground(QColor(Colors.TEXT_TERTIARY))
            self._list.addItem(item)

        self._undo_btn.setEnabled(self._history.can_undo)
        self._redo_btn.setEnabled(self._history.can_redo)

        # Scroll to current position
        undo_count = len(self._history.undo_stack)
        if undo_count > 0:
            self._list.setCurrentRow(undo_count - 1)
