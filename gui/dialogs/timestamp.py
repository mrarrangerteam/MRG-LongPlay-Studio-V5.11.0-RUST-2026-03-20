"""
Timestamp dialog — display and copy YouTube timestamps.

Classes:
    TimestampDialog — Dialog to show and copy timestamps
"""

from typing import List

from gui.utils.compat import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit,
    QMessageBox, QApplication, Qt,
)
from gui.styles import Colors


class TimestampDialog(QDialog):
    """Dialog to show and copy timestamps"""
    
    def __init__(self, timestamps: List[str], total_duration: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📋 YouTube Timestamps")
        self.setMinimumSize(500, 500)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel(f"📋 Generated Timestamps ({len(timestamps)} tracks)")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        
        # Info
        info = QLabel(f"Total Duration: {total_duration}")
        info.setStyleSheet(f"color: {Colors.VIDEO_COLOR}; font-size: 12px;")
        layout.addWidget(info)
        
        # Timestamp text
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText("\n".join(timestamps))
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 15px;
                font-family: 'Menlo', 'Courier New';
                font-size: 13px;
            }}
        """)
        layout.addWidget(self.text_edit)
        
        # Copy button
        copy_btn = QPushButton("📋 Copy All")
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #3A80C9;
            }}
        """)
        copy_btn.clicked.connect(self._copy_all)
        layout.addWidget(copy_btn)
        
    def _copy_all(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())
        QMessageBox.information(self, "Copied", "Timestamps copied to clipboard!")
