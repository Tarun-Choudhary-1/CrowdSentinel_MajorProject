

class RiskEngine:
    

    ALPHA = 0.45   # density weight
    BETA  = 0.30   # speed weight
    GAMMA = 0.25   # direction-variance weight

    MAX_SPEED_REF = 6.0

    COUNT_GATES = [
        (26, 1.00, "CRITICAL"),
        (11, 0.74, "HIGH"),
        (4,  0.54, "MEDIUM"),
        (0,  0.29, "LOW"),
    ]

    BANDS = [
        (0.75, "CRITICAL"),
        (0.55, "HIGH"),
        (0.30, "MEDIUM"),
        (0.00, "LOW"),
    ]

    def __init__(self,
                 alpha: float = 0.45,
                 beta:  float = 0.30,
                 gamma: float = 0.25):
        total      = alpha + beta + gamma
        self.alpha = alpha / total
        self.beta  = beta  / total
        self.gamma = gamma / total
        self._ema        = 0.0
        self._k_normal   = 0.35   # normal EMA smoothing factor
        self._k_fast     = 0.70   # fast decay when people leave
        self._prev_count = 0

    def compute(self, count: int, density: float,
                speed: float, direction_variance: float) -> tuple[float, str]:
        
        norm_speed = min(speed / self.MAX_SPEED_REF, 1.0)

        behaviour = (self.alpha * density
                     + self.beta  * norm_speed
                     + self.gamma * direction_variance)
        behaviour  = float(max(0.0, min(behaviour, 1.0)))

        max_score = self._count_cap(count)
        capped     = min(behaviour, max_score)

        if count == 0:
            
            k = self._k_fast
            capped = 0.0  
        elif count < self._prev_count * 0.5:
            
            k = self._k_fast
        else:
            k = self._k_normal

        self._ema  = k * capped + (1 - k) * self._ema
        score      = round(self._ema, 4)

        if count == 0 and score < 0.05:
            score = 0.0
            self._ema = 0.0

        self._prev_count = count
        level = self._classify(score, count)
        return score, level

    def _count_cap(self, count: int) -> float:
        for min_count, cap, _ in self.COUNT_GATES:
            if count >= min_count:
                return cap
        return 0.29

    def _classify(self, score: float, count: int) -> str:
        ceiling = "LOW"
        for min_count, _, lvl in self.COUNT_GATES:
            if count >= min_count:
                ceiling = lvl
                break

        score_level = "LOW"
        for threshold, label in self.BANDS:
            if score >= threshold:
                score_level = label
                break

        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        return order[min(order.index(score_level), order.index(ceiling))]

    def reset(self):
        self._ema = 0.0
        self._prev_count = 0

    @property
    def weights(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}
