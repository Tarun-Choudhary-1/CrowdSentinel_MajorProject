
import time
from collections import deque


class FPSCounter:
    def __init__(self, window: int = 30):
        self._ts = deque(maxlen=window)

    def tick(self):
        self._ts.append(time.perf_counter())

    def get(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        elapsed = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / elapsed if elapsed > 0 else 0.0
