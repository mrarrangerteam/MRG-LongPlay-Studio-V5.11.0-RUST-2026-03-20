#!/bin/bash
# verify_longplay.sh — Integration verification script for LongPlay Studio
# Usage: bash verify_longplay.sh

set -e
PASS=0
FAIL=0
TOTAL=0

check() {
    TOTAL=$((TOTAL + 1))
    echo -n "  [$TOTAL] $1... "
    if eval "$2" > /dev/null 2>&1; then
        echo "✅ PASS"
        PASS=$((PASS + 1))
    else
        echo "❌ FAIL"
        FAIL=$((FAIL + 1))
    fi
}

echo "═══════════════════════════════════════════"
echo "  LongPlay Studio v5.9.0 — Verification"
echo "═══════════════════════════════════════════"
echo ""

# 1. Python version
echo "── Environment ──"
check "Python >= 3.10" "python3 -c 'import sys; assert sys.version_info >= (3, 10)'"

# 2. Dependencies
echo ""
echo "── Dependencies ──"
check "PySide6/PyQt6" "python3 -c 'try:\n import PySide6\nexcept ImportError:\n import PyQt6'"
check "NumPy" "python3 -c 'import numpy'"
check "soundfile" "python3 -c 'import soundfile'"

# Optional (graceful fallback if missing)
echo -n "  [*] SciPy (optional)... "
python3 -c 'import scipy' 2>/dev/null && echo "✅ INSTALLED" || echo "⚠️  NOT INSTALLED (optional)"
echo -n "  [*] pedalboard (optional)... "
python3 -c 'import pedalboard' 2>/dev/null && echo "✅ INSTALLED" || echo "⚠️  NOT INSTALLED (optional)"
echo -n "  [*] pyloudnorm (optional)... "
python3 -c 'import pyloudnorm' 2>/dev/null && echo "✅ INSTALLED" || echo "⚠️  NOT INSTALLED (optional)"

# 3. Import test
echo ""
echo "── Import Test ──"
check "LongPlayStudioV4 import" "python3 -c 'from gui import LongPlayStudioV4; print(\"OK\")'"

# 4. Module imports
echo ""
echo "── Module Imports ──"
check "MasterChain" "python3 -c 'from modules.master.chain import MasterChain'"
check "Maximizer" "python3 -c 'from modules.master.maximizer import Maximizer'"
check "Equalizer" "python3 -c 'from modules.master.equalizer import Equalizer'"
check "Dynamics" "python3 -c 'from modules.master.dynamics import Dynamics'"
check "Imager" "python3 -c 'from modules.master.imager import Imager'"
check "LoudnessMeter" "python3 -c 'from modules.master.loudness import LoudnessMeter'"
check "OzoneRotaryKnob" "python3 -c 'from modules.widgets.rotary_knob import OzoneRotaryKnob'"
check "VectorscopeWidget" "python3 -c 'from modules.widgets.vectorscope import VectorscopeWidget'"
check "TransferCurveWidget" "python3 -c 'from modules.widgets.transfer_curve import TransferCurveWidget'"
check "SpectrumAnalyzerWidget" "python3 -c 'from modules.widgets.spectrum_analyzer import SpectrumAnalyzerWidget'"
check "CommandHistory" "python3 -c 'from modules.master.undo import CommandHistory'"
check "ProductionPipeline" "python3 -c 'from modules.master.pipeline import ProductionPipelineDialog'"
check "ContentFactory" "python3 -c 'from gui.content_factory import ContentPlanner, BatchMasterLite, LongVideoBuilder, ShortVideoBuilder, MetadataGenerator'"
check "ContentFactoryDialog" "python3 -c 'from gui.dialogs.content_factory import ContentFactoryDialog'"

# 5. Pytest
echo ""
echo "── Test Suite ──"
check "pytest (all tests)" "python3 -m pytest tests/ -q --tb=no"

# 6. Rust .so
echo ""
echo "── Rust Engine ──"
check "Compiled .so files exist" "ls *.so 2>/dev/null | head -1"

# 7. Genre & Platform counts
echo ""
echo "── Data Validation ──"
check "50+ genre presets" "python3 -c 'from modules.master.genre_profiles import GENRE_PROFILES; assert len(GENRE_PROFILES) >= 50'"
check "11+ platform targets" "python3 -c 'from modules.master.genre_profiles import PLATFORM_TARGETS; assert len(PLATFORM_TARGETS) >= 11'"

echo ""
echo "═══════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed (out of $TOTAL)"
echo "═══════════════════════════════════════════"

if [ $FAIL -gt 0 ]; then
    exit 1
else
    echo "  ✅ ALL CHECKS PASSED"
    exit 0
fi
