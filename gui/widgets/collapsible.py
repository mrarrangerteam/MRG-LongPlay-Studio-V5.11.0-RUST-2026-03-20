"""
Collapsible section widget.

Classes:
    CollapsibleSection — Expandable/collapsible panel with header button
"""

from gui.utils.compat import QWidget, QVBoxLayout, QPushButton
from gui.styles import Colors


class CollapsibleSection(QWidget):
    """Collapsible section with header"""

    def __init__(self, title: str, icon: str = "", parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        self.header = QPushButton(f"\u25bc {icon} {title}")
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {Colors.BG_TERTIARY}, stop:0.5 {Colors.BG_SECONDARY},
                    stop:1 {Colors.BG_PRIMARY});
                color: {Colors.GOLD};
                border: none;
                border-bottom: 1px solid {Colors.BORDER};
                border-top: 1px solid {Colors.BORDER_LIGHT};
                padding: 10px;
                text-align: left;
                font-size: 11px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                font-family: 'Menlo', 'Menlo', 'Courier New', monospace;
            }}
            QPushButton:hover {{
                color: {Colors.ACCENT};
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #2E2E36, stop:0.5 {Colors.BG_TERTIARY},
                    stop:1 {Colors.BG_SECONDARY});
            }}
        """)
        self.header.clicked.connect(self._toggle)
        layout.addWidget(self.header)

        # Content
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.content)

        self.title_text = title
        self.icon = icon

    def _toggle(self):
        visible = self.header.isChecked()
        self.content.setVisible(visible)
        arrow = "\u25bc" if visible else "\u25b6"
        self.header.setText(f"{arrow} {self.icon} {self.title_text}")

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)
