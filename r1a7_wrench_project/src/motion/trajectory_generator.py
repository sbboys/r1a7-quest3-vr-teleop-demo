from __future__ import annotations

from typing import List, Sequence


def quintic_interpolate(start: Sequence[float], goal: Sequence[float], duration_s: float, dt_s: float) -> List[List[float]]:
    if len(start) != len(goal):
        raise ValueError("start and goal must have the same length")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    steps = max(2, int(round(duration_s / dt_s)) + 1)
    path: List[List[float]] = []
    for i in range(steps):
        t = min(1.0, i / (steps - 1))
        blend = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        path.append([float(a) + (float(b) - float(a)) * blend for a, b in zip(start, goal)])
    return path
