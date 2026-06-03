
import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray, target_width: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)),
                      interpolation=cv2.INTER_LINEAR)
