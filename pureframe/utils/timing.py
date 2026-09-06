"""Lightweight per-phase wall-clock timers for pipeline instrumentation.

Used by `--verbose` output and the `pureframe bench` command to attribute
processing time to phases (probe, scene detection, frame sampling, NudeNet,
context models, fusion, densify, render). Overhead is a dict update per
phase exit - negligible next to the work being measured.
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager


class PhaseTimers:
    def __init__(self):
        self._lock = threading.Lock()
        self._seconds: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)

    @contextmanager
    def phase(self, name: str):
        start = time.perf_counter()
        try:
            yield self
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._seconds[name] += elapsed
                self._calls[name] += 1

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                name: {"seconds": round(sec, 3), "calls": self._calls[name]}
                for name, sec in self._seconds.items()
            }

    def summary(self) -> str:
        rows = sorted(self.as_dict().items(), key=lambda kv: -kv[1]["seconds"])
        lines = ["Phase timings (wall clock):", "  phase            seconds  calls"]
        for name, d in rows:
            lines.append(f"  {name:<16} {d['seconds']:>7.3f}  {d['calls']:>5}")
        return "\n".join(lines)
