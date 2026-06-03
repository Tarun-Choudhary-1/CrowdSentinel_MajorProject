
import cv2
import numpy as np

RISK_BGR = {
    "LOW":      (34,  197, 94),
    "MEDIUM":   (37,  183, 245),
    "HIGH":     (20,  120, 245),
    "CRITICAL": (30,   30, 220),
}


def draw_detections(frame: np.ndarray, tracks: list[dict],
                    risk_level: str) -> np.ndarray:
    color = RISK_BGR.get(risk_level, (200, 200, 200))
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid = t["id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid}"
        (lw, lh), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1),
                      color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, count: int, risk_level: str,
             risk_score: float, fps: float,
             alert_active: bool) -> np.ndarray:
    
    color = RISK_BGR.get(risk_level, (200, 200, 200))
    h, w  = frame.shape[:2]

    
    cv2.rectangle(frame, (0, 0), (w, 44), (15, 15, 20), -1)

    items = [
        f" People: {count}",
        f"Risk: {risk_level}",
        f"Score: {risk_score:.2f}",
        f"FPS: {fps:.1f}",
    ]
    cv2.putText(frame, "   |   ".join(items),
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, color, 1, cv2.LINE_AA)

    
    if alert_active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        banner = "  ⚠  STAMPEDE RISK — EVACUATE IMMEDIATELY  ⚠  "
        bw, _  = cv2.getTextSize(
            banner, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
        cv2.putText(frame, banner,
                    ((w - bw) // 2, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (30, 30, 255), 2, cv2.LINE_AA)
    return frame
