"""
Drag-and-drop list widget for file drops and internal reordering.

Classes:
    DropZoneListWidget — QListWidget with drag/drop support for files AND internal reordering
"""

import os

from gui.utils.compat import (
    QListWidget, Qt, pyqtSignal, QDragEnterEvent, QDropEvent,
)
from gui.styles import Colors


class DropZoneListWidget(QListWidget):
    """QListWidget with drag and drop support for files AND internal reordering"""
    filesDropped = pyqtSignal(list)  # emits list of file paths

    def __init__(self, accepted_extensions: list, placeholder_text: str = "Drop files here", parent=None):
        super().__init__(parent)
        self.accepted_extensions = [ext.lower() for ext in accepted_extensions]
        self.placeholder_text = placeholder_text
        self._is_dragging = False
        self._allow_internal_move = False

        self._update_style_dragging(False)

        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DropOnly)

    def enableInternalMove(self):
        """Enable internal drag & drop reordering"""
        self._allow_internal_move = True
        self.setDragEnabled(True)
        self.setDragDropMode(QListWidget.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event: QDragEnterEvent):
        try:
            mime = event.mimeData()
            if mime and mime.hasUrls():
                for url in mime.urls():
                    file_path = url.toLocalFile()
                    if file_path:
                        ext = os.path.splitext(file_path)[1].lower()
                        if ext in self.accepted_extensions:
                            event.acceptProposedAction()
                            self._is_dragging = True
                            self._update_style_dragging(True)
                            return

            if self._allow_internal_move and event.source() == self:
                event.acceptProposedAction()
                return

            event.ignore()
        except Exception as e:
            print(f"[DROP] dragEnterEvent error: {e}")
            event.ignore()

    def dragMoveEvent(self, event):
        """Required for proper drag and drop"""
        try:
            mime = event.mimeData()
            if mime and mime.hasUrls():
                for url in mime.urls():
                    if url.toLocalFile():
                        event.acceptProposedAction()
                        return

            if self._allow_internal_move and event.source() == self:
                event.acceptProposedAction()
                return

            event.ignore()
        except Exception as e:
            print(f"[DROP] dragMoveEvent error: {e}")
            event.ignore()

    def dragLeaveEvent(self, event):
        self._is_dragging = False
        self._update_style_dragging(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        try:
            self._is_dragging = False
            self._update_style_dragging(False)

            mime = event.mimeData()
            if mime and mime.hasUrls():
                valid_files = []
                for url in mime.urls():
                    file_path = url.toLocalFile()
                    if file_path:
                        ext = os.path.splitext(file_path)[1].lower()
                        if ext in self.accepted_extensions:
                            valid_files.append(file_path)

                if valid_files:
                    self.filesDropped.emit(valid_files)
                    event.acceptProposedAction()
                    return

            if self._allow_internal_move and event.source() == self:
                super().dropEvent(event)
                return

            event.ignore()
        except Exception as e:
            print(f"[DROP] dropEvent error: {e}")
            event.ignore()

    def _update_style_dragging(self, is_dragging: bool):
        if is_dragging:
            self.setStyleSheet(f"""
                QListWidget {{
                    background: {Colors.BG_TERTIARY};
                    border: 2px dashed {Colors.ACCENT};
                    border-radius: 6px;
                }}
                QListWidget::item {{
                    color: {Colors.TEXT_PRIMARY};
                    padding: 8px;
                    border-radius: 4px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QListWidget {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 #0A0A0C, stop:0.02 {Colors.BG_PRIMARY},
                        stop:0.98 {Colors.BG_PRIMARY}, stop:1 #0A0A0C);
                    border: 1px solid {Colors.BORDER};
                    border-top: 1px solid #0A0A0C;
                    border-radius: 4px;
                }}
                QListWidget::item {{
                    color: {Colors.TEXT_PRIMARY};
                    padding: 8px;
                    border-bottom: 1px solid {Colors.BORDER};
                    font-family: 'Menlo', monospace;
                    font-size: 11px;
                }}
                QListWidget::item:hover {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 {Colors.BG_TERTIARY}, stop:1 {Colors.BG_SECONDARY});
                    color: {Colors.ACCENT};
                }}
                QListWidget::item:selected {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 {Colors.ACCENT_DIM}, stop:0.5 {Colors.ACCENT},
                        stop:1 {Colors.ACCENT_DIM});
                    color: #1A1A1E;
                }}
            """)
