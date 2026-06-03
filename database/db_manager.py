

import os
import sqlite3
import time
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path: str = "database/crowd_logs.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def init(self):
        with self._con() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS crowd_events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT    NOT NULL,
                    unix_ts    REAL    NOT NULL,
                    count      INTEGER DEFAULT 0,
                    risk_score REAL    DEFAULT 0.0,
                    risk_level TEXT    DEFAULT 'LOW',
                    density    REAL    DEFAULT 0.0,
                    avg_speed  REAL    DEFAULT 0.0
                )
            """)
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts ON crowd_events(unix_ts)")
        print(f"[DB] Ready: {self.db_path}")

    def insert(self, count, risk_score, risk_level, density, avg_speed):
        now = time.time()
        ts  = datetime.fromtimestamp(now).isoformat(timespec="seconds")
        try:
            with self._con() as con:
                con.execute(
                    "INSERT INTO crowd_events "
                    "(timestamp,unix_ts,count,risk_score,risk_level,density,avg_speed) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (ts, now, count, round(risk_score, 4),
                     risk_level, round(density, 4), round(avg_speed, 4))
                )
        except Exception as e:
            print(f"[DB] Insert error: {e}")

    def fetch_recent(self, n: int = 50) -> list[dict]:
        try:
            with self._con() as con:
                cur  = con.execute(
                    "SELECT timestamp,count,risk_score,risk_level,density,avg_speed "
                    "FROM crowd_events ORDER BY unix_ts DESC LIMIT ?", (n,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:
            return []

    def clear(self):
        with self._con() as con:
            con.execute("DELETE FROM crowd_events")

    def _con(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
