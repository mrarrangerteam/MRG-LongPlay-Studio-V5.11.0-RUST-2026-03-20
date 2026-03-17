"""Shared test fixtures."""

import os
import pytest
import sys


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Create a QApplication instance for all tests that need Qt widgets.

    Uses offscreen platform in headless environments (CI, SSH).
    """
    # Force offscreen rendering if no display available
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
