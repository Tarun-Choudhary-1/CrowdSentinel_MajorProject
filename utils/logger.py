
import os, csv, time
from datetime import datetime


class CrowdLogger:
    def __init__(self, log_dir: str = "static/outputs",
                 interval_sec: float = 2.0):
        self.log_dir      = log_dir
        self.interval_sec = interval_sec
        self._last_write  = 0.0
        self._csv_path    = None
        os.makedirs(log_dir, exist_ok=True)
        fname = f"crowd_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._csv_path = os.path.join(log_dir, fname)
        with open(self._csv_path, "w", newline="") as f:
            csv.DictWriter(
                f, fieldnames=["timestamp", "count", "risk_score",
                               "risk_level", "density", "avg_speed"]
            ).writeheader()

    def log(self, count, risk_score, risk_level, density, avg_speed):
        now = time.time()
        if (now - self._last_write) < self.interval_sec:
            return
        self._last_write = now
        record = {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "count":      count,
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "density":    round(density, 4),
            "avg_speed":  round(avg_speed, 4),
        }
        try:
            with open(self._csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=list(record.keys())).writerow(record)
        except Exception:
            pass
