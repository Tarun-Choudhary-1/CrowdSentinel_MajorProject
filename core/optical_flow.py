
import cv2
import numpy as np


class OpticalFlowAnalyzer:
    
    FLOW_WIDTH = 320   

    def __init__(self,
                 winsize: int = 13,
                 levels: int  = 3,
                 mag_threshold: float = 0.8):
        self.params = dict(
            pyr_scale=0.5, levels=levels, winsize=winsize,
            iterations=2,  
            poly_n=5, poly_sigma=1.1, flags=0,
        )
        self.mag_threshold = mag_threshold
        self._prev_small = None   

    def analyze(self, prev_gray: np.ndarray,
                curr_gray: np.ndarray) -> tuple[float, float]:
       
        h, w = curr_gray.shape[:2]
        scale = self.FLOW_WIDTH / w if w > self.FLOW_WIDTH else 1.0

        if scale < 1.0:
            new_w = self.FLOW_WIDTH
            new_h = int(h * scale)
            curr_small = cv2.resize(curr_gray, (new_w, new_h),
                                    interpolation=cv2.INTER_AREA)
            if self._prev_small is None or self._prev_small.shape != curr_small.shape:
                self._prev_small = curr_small
                return 0.0, 0.0
            prev_small = self._prev_small
            self._prev_small = curr_small
        else:
            prev_small = prev_gray
            curr_small = curr_gray

        flow     = cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None, **self.params)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mask = mag > self.mag_threshold
        if mask.sum() < 50:     
            return 0.0, 0.0

        avg_speed    = float(mag[mask].mean())
        angles       = ang[mask]
        sin_m, cos_m = np.sin(angles).mean(), np.cos(angles).mean()
        r            = float(np.sqrt(sin_m ** 2 + cos_m ** 2))
        dir_variance = float(1.0 - r)   

        return avg_speed, dir_variance

    def reset(self):
        """Reset cached previous frame."""
        self._prev_small = None
