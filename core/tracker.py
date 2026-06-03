

import numpy as np


class PersonTracker:
    
    def __init__(self, max_age: int = 10, n_init: int = 3):
        self._tracker   = None
        self._next_id   = 1
        self._centroids = {}   
        self._init_deepsort(max_age, n_init)

    def _init_deepsort(self, max_age, n_init):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_cosine_distance=0.3,
                nn_budget=60,            # capped: prevents memory leak
                embedder="mobilenet",
                half=False,
                bgr=True,
                embedder_gpu=False,
            )
            print("[Tracker] DeepSORT ready.")
        except Exception as e:
            print(f"[Tracker] DeepSORT unavailable ({e}). Using centroid tracker.")

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """Update tracker. Returns empty list immediately when no detections."""
        if self._tracker:
            return self._deepsort_update(detections, frame)
        return self._centroid_update(detections)

    def _deepsort_update(self, detections, frame) -> list[dict]:
        # Build DeepSORT input
        if not detections:
            # Still call update with empty list to age out stale tracks
            self._tracker.update_tracks([], frame=frame)
            return []

        ds_in = [
            ([d["bbox"][0], d["bbox"][1],
              d["bbox"][2] - d["bbox"][0],
              d["bbox"][3] - d["bbox"][1]],
             d["confidence"], "person")
            for d in detections
        ]
        tracks  = self._tracker.update_tracks(ds_in, frame=frame)
        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()
            # Guard against invalid bboxes
            if any(np.isnan(ltrb)) or any(np.isinf(ltrb)):
                continue
            x1, y1, x2, y2 = map(int, ltrb)
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            results.append({"id": t.track_id, "bbox": [x1, y1, x2, y2],
                             "centroid": (cx, cy)})
        return results

    def _centroid_update(self, detections) -> list[dict]:
        if not detections:
            self._centroids = {}
            return []

        new_data  = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            new_data.append(((cx, cy), d["bbox"]))

        updated  = {}
        used_ids = set()

        for (cx, cy), bbox in new_data:
            best_id, best_dist = None, 60
            for eid, ec in self._centroids.items():
                if eid in used_ids:
                    continue
                dist = ((cx - ec[0]) ** 2 + (cy - ec[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, eid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            updated[best_id] = (cx, cy)
            used_ids.add(best_id)

        self._centroids = updated
        results = []
        for eid, (cx, cy) in updated.items():
            for (ocx, ocy), bbox in new_data:
                if abs(ocx - cx) < 5 and abs(ocy - cy) < 5:
                    results.append({"id": eid, "bbox": bbox,
                                    "centroid": (cx, cy)})
                    break
        return results
