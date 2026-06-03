
import cv2
import numpy as np


class PersonDetector:
    
    PERSON_CLASS = 0

    def __init__(self,
                 model_path: str   = "models/yolov8n.pt",
                 confidence: float = 0.45,
                 iou:        float = 0.45,
                 device:     str   = "cpu"):
        self.confidence = confidence
        self.iou        = iou
        self.device     = device
        self.model      = None
        self._load(model_path)

    def _load(self, path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            print(f"[Detector] YOLOv8 loaded: {path}")
        except Exception as e:
            print(f"[Detector] Model unavailable ({e}). Demo mode active.")

    def detect(self, frame: np.ndarray, tiled: bool = False) -> list[dict]:
        
        if self.model is None:
            return []

        raw_dets = self._run_inference(frame, self.confidence, imgsz=640)

        if tiled:
            raw_dets = self._tiled_detect(frame, raw_dets)

        return self._nms(raw_dets, iou_thresh=0.40)

    
    def _tiled_detect(self, frame: np.ndarray,
                      base_dets: list[dict]) -> list[dict]:
        
        h, w = frame.shape[:2]
        all_dets = list(base_dets)

        strips = [
            (0.0,  0.50),   
            (0.25, 0.75),   
        ]

        tile_conf = max(self.confidence - 0.05, 0.35)

        for y_start_r, y_end_r in strips:
            y1 = int(h * y_start_r)
            y2 = int(h * y_end_r)
            strip = frame[y1:y2, :]

            if strip.shape[0] < 60:
                continue

            
            strip_dets = self._run_inference(strip, tile_conf, imgsz=640)

            for d in strip_dets:
                d["bbox"][1] += y1
                d["bbox"][3] += y1
                cx, cy = d["centroid"]
                d["centroid"] = (cx, cy + y1)

            all_dets.extend(strip_dets)

        return all_dets

    def _run_inference(self, img: np.ndarray, conf: float,
                       imgsz: int = 640) -> list[dict]:
        results = self.model.predict(
            source       = img,
            conf         = conf,
            iou          = self.iou,
            classes      = [self.PERSON_CLASS],
            imgsz        = imgsz,
            agnostic_nms = True,
            verbose      = False,
            device       = self.device,
        )
        dets = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != self.PERSON_CLASS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1

            
                if w < 8 or h < 15:
                    continue
                if h / max(w, 1) < 0.3:
                    continue

                if w * h < 200:
                    continue

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dets.append({
                    "bbox":       [x1, y1, x2, y2],
                    "centroid":   (cx, cy),
                    "confidence": float(box.conf[0]),
                    "area":       w * h,
                })
        return dets

    def _nms(self, dets: list[dict], iou_thresh: float) -> list[dict]:
        """
        Two-pass NMS to eliminate duplicates from tiled merge:
        1. Standard IoU-based NMS (tight threshold)
        2. Centroid-distance based dedup (removes near-identical detections)
        """
        if not dets:
            return []

        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)

       
        kept, suppressed = [], set()
        for i, d in enumerate(dets):
            if i in suppressed:
                continue
            kept.append(d)
            ax1, ay1, ax2, ay2 = d["bbox"]
            a_area = (ax2 - ax1) * (ay2 - ay1)

            for j in range(i + 1, len(dets)):
                if j in suppressed:
                    continue
                bx1, by1, bx2, by2 = dets[j]["bbox"]
                b_area = (bx2 - bx1) * (by2 - by1)

                ix = max(0, min(ax2, bx2) - max(ax1, bx1))
                iy = max(0, min(ay2, by2) - max(ay1, by1))
                inter = ix * iy
                union = a_area + b_area - inter

                if union > 0 and inter / union > iou_thresh:
                    suppressed.add(j)


        final = []
        for d in kept:
            cx, cy = d["centroid"]
            duplicate = False
            for f in final:
                fx, fy = f["centroid"]
                dist = ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5
                area_ratio = d["area"] / max(f["area"], 1)
                if dist < 35 and 0.4 < area_ratio < 2.5:
                    duplicate = True
                    break
            if not duplicate:
                final.append(d)

        return final

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
