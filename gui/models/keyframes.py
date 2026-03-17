"""
Keyframe animation system for MRG LongPlay Studio.

Provides keyframe types, interpolation (linear, ease, bezier),
keyframe tracks, and value-at-time evaluation.

Story 3.1 — Epic 3: CapCut Features.
"""

from __future__ import annotations

import bisect
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class KeyframeType(Enum):
    """Interpolation modes between keyframes."""
    LINEAR = auto()
    EASE_IN = auto()
    EASE_OUT = auto()
    EASE_IN_OUT = auto()
    BEZIER = auto()


@dataclass
class Keyframe:
    """A single keyframe on a parameter track.

    Attributes:
        time:           Position on the timeline in seconds.
        value:          Parameter value at this keyframe.
        interpolation:  How to interpolate *from* this keyframe to the next.
        bezier_cp1:     First control point for BEZIER mode (t, v) relative.
        bezier_cp2:     Second control point for BEZIER mode (t, v) relative.
        id:             Unique keyframe identifier.
    """
    time: float = 0.0
    value: float = 0.0
    interpolation: KeyframeType = KeyframeType.LINEAR
    bezier_cp1: Tuple[float, float] = (0.33, 0.0)
    bezier_cp2: Tuple[float, float] = (0.67, 1.0)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# ---------------------------------------------------------------------------
# Bezier helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _cubic_bezier_y(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """Evaluate the cubic bezier at parameter *t* (0..1)."""
    u = 1.0 - t
    return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3


def _cubic_bezier_x(t: float, x0: float, x1: float, x2: float, x3: float) -> float:
    """X-coordinate of the cubic bezier at parameter *t*."""
    u = 1.0 - t
    return u * u * u * x0 + 3.0 * u * u * t * x1 + 3.0 * u * t * t * x2 + t * t * t * x3


def bezier_interpolate(
    progress: float,
    cp1: Tuple[float, float],
    cp2: Tuple[float, float],
) -> float:
    """Map a linear progress [0..1] through a cubic-bezier timing curve.

    The curve's endpoints are implicitly (0, 0) and (1, 1).
    *cp1* and *cp2* are the two interior control points (x, y).

    Returns the eased value in [0..1].
    """
    progress = _clamp(progress)
    # Newton-Raphson to find t for given x (progress)
    x1, y1 = cp1
    x2, y2 = cp2
    t = progress  # initial guess
    for _ in range(8):
        x_at_t = _cubic_bezier_x(t, 0.0, x1, x2, 1.0)
        dx = x_at_t - progress
        if abs(dx) < 1e-7:
            break
        # derivative of x w.r.t. t
        dxdt = 3.0 * (1 - t) * (1 - t) * x1 + 6.0 * (1 - t) * t * (x2 - x1) + 3.0 * t * t * (1.0 - x2)
        if abs(dxdt) < 1e-7:
            break
        t -= dx / dxdt
        t = _clamp(t)
    return _cubic_bezier_y(t, 0.0, y1, y2, 1.0)


# Pre-baked ease curves expressed as bezier control points
_EASE_CURVES: Dict[KeyframeType, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    KeyframeType.EASE_IN: ((0.42, 0.0), (1.0, 1.0)),
    KeyframeType.EASE_OUT: ((0.0, 0.0), (0.58, 1.0)),
    KeyframeType.EASE_IN_OUT: ((0.42, 0.0), (0.58, 1.0)),
}


def interpolate_value(
    kf_a: Keyframe,
    kf_b: Keyframe,
    time: float,
) -> float:
    """Interpolate between two keyframes at *time*."""
    span = kf_b.time - kf_a.time
    if span <= 0:
        return kf_a.value
    progress = _clamp((time - kf_a.time) / span)

    interp = kf_a.interpolation
    if interp == KeyframeType.LINEAR:
        t = progress
    elif interp == KeyframeType.BEZIER:
        t = bezier_interpolate(progress, kf_a.bezier_cp1, kf_a.bezier_cp2)
    elif interp in _EASE_CURVES:
        cp1, cp2 = _EASE_CURVES[interp]
        t = bezier_interpolate(progress, cp1, cp2)
    else:
        t = progress  # fallback linear

    return kf_a.value + (kf_b.value - kf_a.value) * t


class KeyframeTrack:
    """Ordered collection of keyframes for a single parameter.

    Attributes:
        parameter_name:  Name of the parameter being animated.
        keyframes:       Sorted list of Keyframe objects.
    """

    def __init__(self, parameter_name: str = "", keyframes: Optional[List[Keyframe]] = None) -> None:
        self.parameter_name = parameter_name
        self.keyframes: List[Keyframe] = keyframes if keyframes is not None else []
        self._sort()

    # -- public API ---------------------------------------------------------

    def add_keyframe(self, kf: Keyframe) -> None:
        """Insert a keyframe and maintain sort order."""
        self.keyframes.append(kf)
        self._sort()

    def remove_keyframe(self, kf_id: str) -> Optional[Keyframe]:
        """Remove and return a keyframe by its id."""
        for i, kf in enumerate(self.keyframes):
            if kf.id == kf_id:
                return self.keyframes.pop(i)
        return None

    def get_value_at(self, time: float) -> Optional[float]:
        """Evaluate the parameter value at *time*.

        Returns ``None`` if the track has no keyframes.
        """
        if not self.keyframes:
            return None
        if len(self.keyframes) == 1:
            return self.keyframes[0].value

        # Before first / after last — hold
        if time <= self.keyframes[0].time:
            return self.keyframes[0].value
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value

        # Binary search for the surrounding pair
        times = [kf.time for kf in self.keyframes]
        idx = bisect.bisect_right(times, time) - 1
        idx = max(0, min(idx, len(self.keyframes) - 2))
        return interpolate_value(self.keyframes[idx], self.keyframes[idx + 1], time)

    def sort(self) -> None:
        """Public alias for re-sorting the keyframe list."""
        self._sort()

    # -- internal -----------------------------------------------------------

    def _sort(self) -> None:
        self.keyframes.sort(key=lambda kf: kf.time)
