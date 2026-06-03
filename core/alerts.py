
import os
import time
import threading


class AlertSystem:
    ALERT_LEVELS  = {"HIGH", "CRITICAL"}
    COOLDOWN_SECS = 10

    def __init__(self, sound_path: str = "static/alerts/siren.wav"):
        self.sound_path     = sound_path
        self._last_alert_ts = 0.0
        self._pygame_ok     = False
        self._lock          = threading.Lock()
        self._init_pygame()

    def _init_pygame(self):
        try:
            import pygame
            pygame.mixer.init()
            self._pygame_ok = True
        except Exception:
            pass

    def check_and_trigger(self, risk_level: str, _score: float) -> bool:
        now      = time.time()
        is_alert = risk_level in self.ALERT_LEVELS
        if is_alert and (now - self._last_alert_ts) > self.COOLDOWN_SECS:
            self._last_alert_ts = now
            threading.Thread(target=self._play, daemon=True).start()
        return is_alert and (now - self._last_alert_ts) < (self.COOLDOWN_SECS / 2 + 0.1)

    def _play(self):
        if not self._pygame_ok or not os.path.exists(self.sound_path):
            return
        try:
            import pygame
            with self._lock:
                pygame.mixer.music.load(self.sound_path)
                pygame.mixer.music.play()
        except Exception:
            pass

    def stop(self):
        if not self._pygame_ok:
            return
        try:
            import pygame
            pygame.mixer.music.stop()
        except Exception:
            pass
