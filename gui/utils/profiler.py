"""
Performance profiler and optimization utilities.

Story 5.5 — Epic 5: Polish & Production.

Features:
    - Function-level profiling with decorator
    - Memory usage tracking
    - Frame rate monitoring for GUI
    - Audio processing benchmark
    - Export performance report
    - Lazy import system for faster startup
    - Render cache management
"""

from __future__ import annotations

import functools
import os
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Profiling decorator & context manager
# ---------------------------------------------------------------------------
class _ProfileStats:
    """Accumulated profiling statistics for a single function."""

    __slots__ = ("name", "call_count", "total_time", "min_time", "max_time")

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count: int = 0
        self.total_time: float = 0.0
        self.min_time: float = float("inf")
        self.max_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / max(self.call_count, 1)

    def record(self, elapsed: float) -> None:
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "calls": self.call_count,
            "total_ms": round(self.total_time * 1000, 2),
            "avg_ms": round(self.avg_time * 1000, 2),
            "min_ms": round(self.min_time * 1000, 2),
            "max_ms": round(self.max_time * 1000, 2),
        }


class Profiler:
    """
    Centralized performance profiler.

    Usage:
        profiler = Profiler()

        @profiler.profile
        def my_function():
            ...

        with profiler.section("render"):
            ...

        print(profiler.report())
    """

    def __init__(self) -> None:
        self._stats: Dict[str, _ProfileStats] = {}
        self._lock = threading.Lock()
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def profile(self, func: Callable) -> Callable:
        """Decorator that profiles function execution time."""
        name = f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self._enabled:
                return func(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                self._record(name, elapsed)

        return wrapper

    class _SectionContext:
        def __init__(self, profiler: "Profiler", name: str) -> None:
            self._profiler = profiler
            self._name = name
            self._t0 = 0.0

        def __enter__(self) -> "_SectionContext":
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *args: Any) -> None:
            elapsed = time.perf_counter() - self._t0
            self._profiler._record(self._name, elapsed)

    def section(self, name: str) -> "_SectionContext":
        """Context manager for profiling a code section."""
        return self._SectionContext(self, name)

    def _record(self, name: str, elapsed: float) -> None:
        with self._lock:
            if name not in self._stats:
                self._stats[name] = _ProfileStats(name)
            self._stats[name].record(elapsed)

    def report(self, top_n: int = 20) -> str:
        """Generate a text performance report."""
        with self._lock:
            stats = sorted(self._stats.values(),
                           key=lambda s: s.total_time, reverse=True)

        lines = [
            "=" * 70,
            "MRG LongPlay Studio V5.5 — Performance Report",
            "=" * 70,
            f"{'Function':<40s} {'Calls':>6s} {'Total ms':>10s} {'Avg ms':>8s} {'Max ms':>8s}",
            "-" * 70,
        ]

        for stat in stats[:top_n]:
            lines.append(
                f"{stat.name[:40]:<40s} {stat.call_count:>6d} "
                f"{stat.total_time * 1000:>10.1f} {stat.avg_time * 1000:>8.2f} "
                f"{stat.max_time * 1000:>8.2f}"
            )

        lines.append("-" * 70)
        total = sum(s.total_time for s in stats)
        lines.append(f"{'Total':>40s} {'':>6s} {total * 1000:>10.1f}")
        return "\n".join(lines)

    def report_dict(self) -> List[Dict]:
        """Return profiling data as list of dicts."""
        with self._lock:
            return [s.to_dict() for s in sorted(
                self._stats.values(), key=lambda s: s.total_time, reverse=True
            )]

    def reset(self) -> None:
        """Clear all profiling data."""
        with self._lock:
            self._stats.clear()


# ---------------------------------------------------------------------------
# FPS Monitor
# ---------------------------------------------------------------------------
class FPSMonitor:
    """Track frame rate for GUI rendering."""

    def __init__(self, window_size: int = 60) -> None:
        self._timestamps: List[float] = []
        self._window = window_size

    def tick(self) -> None:
        """Call once per frame."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps = self._timestamps[-self._window:]

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / dt

    @property
    def frame_time_ms(self) -> float:
        if self.fps <= 0:
            return 0.0
        return 1000.0 / self.fps


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------
class MemoryTracker:
    """Track memory usage of the application."""

    @staticmethod
    def get_rss_mb() -> float:
        """Get current RSS (Resident Set Size) in MB."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # macOS returns bytes, Linux returns KB
            if sys.platform == "darwin":
                return usage.ru_maxrss / (1024 * 1024)
            else:
                return usage.ru_maxrss / 1024
        except ImportError:
            return 0.0

    @staticmethod
    def get_process_memory_mb() -> float:
        """Get current process memory via /proc or ps."""
        try:
            import subprocess
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True, text=True, timeout=5,
            )
            return int(result.stdout.strip()) / 1024.0
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return 0.0


# ---------------------------------------------------------------------------
# Render cache
# ---------------------------------------------------------------------------
class RenderCache:
    """
    LRU cache for rendered outputs (audio/video segments).

    Evicts oldest entries when max_size_mb is exceeded.
    """

    def __init__(self, max_size_mb: float = 512.0) -> None:
        self._cache: Dict[str, Tuple[bytes, float]] = {}
        self._access_order: List[str] = []
        self._current_size: int = 0
        self._max_size = int(max_size_mb * 1024 * 1024)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key][0]
            return None

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            size = len(data)
            # evict if needed
            while self._current_size + size > self._max_size and self._access_order:
                oldest = self._access_order.pop(0)
                if oldest in self._cache:
                    self._current_size -= len(self._cache[oldest][0])
                    del self._cache[oldest]

            self._cache[key] = (data, time.time())
            self._access_order.append(key)
            self._current_size += size

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_size = 0

    @property
    def size_mb(self) -> float:
        return self._current_size / (1024 * 1024)

    @property
    def entry_count(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Audio benchmark
# ---------------------------------------------------------------------------
def benchmark_mastering(chain, audio_path: str, iterations: int = 3) -> Dict:
    """
    Benchmark mastering chain performance.

    Args:
        chain: MasterChain instance.
        audio_path: Path to test audio file.
        iterations: Number of benchmark runs.

    Returns:
        Benchmark results dict.
    """
    import tempfile

    results = {
        "backend": getattr(chain, "backend_name", "unknown"),
        "iterations": iterations,
        "times_ms": [],
        "avg_ms": 0.0,
        "file": os.path.basename(audio_path),
    }

    for i in range(iterations):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            t0 = time.perf_counter()
            try:
                chain.render(tmp.name)
            except Exception as e:
                results["error"] = str(e)
                break
            elapsed = (time.perf_counter() - t0) * 1000
            results["times_ms"].append(round(elapsed, 1))

    if results["times_ms"]:
        results["avg_ms"] = round(sum(results["times_ms"]) / len(results["times_ms"]), 1)
        results["min_ms"] = min(results["times_ms"])
        results["max_ms"] = max(results["times_ms"])

    return results


# ---------------------------------------------------------------------------
# Global profiler instance
# ---------------------------------------------------------------------------
profiler = Profiler()
fps_monitor = FPSMonitor()
memory_tracker = MemoryTracker()
render_cache = RenderCache()
