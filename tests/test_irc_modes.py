#!/usr/bin/env python3
"""
tests/test_irc_modes.py — Verify IRC modes produce different output
Validates that each IRC mode in the mastering chain creates
audibly different results, not identical passthrough.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def generate_test_signal(sr=44100, duration=5):
    """Generate a dynamic test signal with bass, mid, high + dynamics."""
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)

    signal = (
        0.3 * np.sin(2 * np.pi * 100 * t) +    # bass
        0.2 * np.sin(2 * np.pi * 1000 * t) +    # mid
        0.1 * np.sin(2 * np.pi * 5000 * t) +    # high
        0.15 * np.random.randn(len(t)).astype(np.float32)  # noise
    )

    # Add dynamics (quiet + loud sections)
    envelope = np.ones_like(signal)
    envelope[sr:2*sr] = 0.1      # quiet section
    envelope[3*sr:4*sr] = 1.5    # loud section
    signal = signal * envelope
    stereo = np.column_stack([signal, signal * 0.95])
    return stereo, sr


def test_irc_modes_differ():
    """Test that different IRC modes produce different output."""
    try:
        from modules.master.maximizer import Maximizer
        print("  Maximizer imported")
    except ImportError as e:
        print(f"  Cannot import Maximizer: {e}")
        return False

    stereo, sr = generate_test_signal()
    print(f"  Test signal: {stereo.shape}, peak={np.max(np.abs(stereo)):.4f}")

    modes = ["IRC 1", "IRC 2", "IRC 3", "IRC 4", "IRC 5", "IRC LL"]
    results = {}

    for mode in modes:
        try:
            m = Maximizer()
            m.set_irc_mode(mode)
            m.set_gain(6.0)
            m.set_ceiling(-1.0)

            # Get FFmpeg filters (the chain uses these)
            filters = m.get_ffmpeg_filters()

            results[mode] = {
                "n_filters": len(filters),
                "filter_str": ",".join(filters)[:100],
            }

            print(f"  {mode}: {len(filters)} filters")

        except Exception as e:
            print(f"  {mode}: Error: {e}")
            return False

    # Check that modes generate different filter chains
    filter_strs = set(r["filter_str"] for r in results.values())
    if len(filter_strs) <= 1:
        print("  ALL MODES PRODUCE IDENTICAL FILTER CHAINS!")
        return False
    else:
        print(f"  {len(filter_strs)} unique filter chains out of {len(modes)} modes")
        return True


def test_soothe_processor():
    """Test that SootheProcessor works."""
    try:
        from modules.master.soothe import SootheProcessor
        print("  SootheProcessor imported")
    except ImportError as e:
        print(f"  Cannot import SootheProcessor: {e}")
        return False

    stereo, sr = generate_test_signal(duration=1)

    proc = SootheProcessor(sr)
    proc.enabled = True
    proc.set_params(amount=50.0)

    output = proc.process(stereo)

    diff = np.max(np.abs(output - stereo))
    print(f"  Input peak: {np.max(np.abs(stereo)):.4f}")
    print(f"  Output peak: {np.max(np.abs(output)):.4f}")
    print(f"  Difference: {diff:.6f}")

    if diff < 0.0001:
        print("  Soothe produced identical output!")
        return False
    else:
        print(f"  Soothe working — diff={diff:.4f}")
        return True


def test_ai_master_import():
    """Test that AIMasterEngine imports and initializes."""
    try:
        from modules.master.ai_master import AIMasterEngine
        engine = AIMasterEngine()
        print("  AIMasterEngine imported and initialized")
        return True
    except ImportError as e:
        print(f"  Cannot import AIMasterEngine: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("IRC Modes & Mastering Chain Verification")
    print("=" * 60)

    tests = [
        ("IRC Modes Differ", test_irc_modes_differ),
        ("Soothe Processor", test_soothe_processor),
        ("AI Master Import", test_ai_master_import),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            result = test_fn()
            if result:
                print(f"  PASSED")
                passed += 1
            else:
                print(f"  FAILED")
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
