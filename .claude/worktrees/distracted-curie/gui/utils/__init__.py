"""GUI utility modules."""
from gui.utils.compat import *  # noqa: F401,F403 — re-export Qt compat symbols


def ffmpeg_escape_path(p: str) -> str:
    """Escape single quotes in file paths for FFmpeg concat list files."""
    return p.replace("'", "'\\''")

