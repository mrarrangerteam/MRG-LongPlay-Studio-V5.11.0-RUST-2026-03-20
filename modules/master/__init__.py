# LongPlay Studio V5.5 — AI Master Module
# 3-tier backend: Rust (PyO3) → C++ (pybind11) → Python (original)

# Try Rust/C++ backend first, fallback to Python
try:
    from .rust_chain import MasterChain, _RUST_BACKEND_AVAILABLE
    if not _RUST_BACKEND_AVAILABLE:
        raise ImportError("No native backend, using Python fallback")
    NATIVE_BACKEND = True
except ImportError as e:
    print(f"[LongPlay] Native backend unavailable, using Python fallback: {e}")
    from .chain import MasterChain
    NATIVE_BACKEND = False

from .ai_assist import AIAssist
from .analyzer import AudioAnalyzer
from .limiter import LookAheadLimiter

__all__ = ['MasterChain', 'AIAssist', 'AudioAnalyzer', 'LookAheadLimiter', 'NATIVE_BACKEND']
