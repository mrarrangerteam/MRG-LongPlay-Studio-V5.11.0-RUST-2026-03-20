#!/usr/bin/env bash
# LongPlay Studio V5 - Double-click to run!
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
python3 gui.py
