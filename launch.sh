#!/bin/bash
# LongPlay Studio V5 Launcher
# Uses Rust audio engine → C++ fallback → Python fallback

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/../Mrarranger_LongPlay_Studio_V5/venv"

# Use venv Python if available
if [ -f "$VENV/bin/python3" ]; then
    PYTHON="$VENV/bin/python3"
else
    PYTHON="python3"
fi

# Ensure FFmpeg is in PATH
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export PYTHONPATH="$DIR"

cd "$DIR"
exec "$PYTHON" gui.py "$@"
