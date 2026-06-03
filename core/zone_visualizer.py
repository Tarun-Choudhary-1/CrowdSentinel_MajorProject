

import cv2
import numpy as np



ZONE_COLORS = {
    "safe":     (34,  197, 94),    
    "caution":  (22,  163, 245),   
    "high":     (22,  120, 245),   
    "critical": (30,  30,  220),   
}
ZONE_ALPHA = 0.28    


class ZoneVisualizer:
    """
    Renders a grid-based risk-zone overlay on each frame.
    """

    def __init__(self, grid_rows: int = 4, grid_cols: int = 4,
                 critical_per_cell: int = 8):
        self.rows              = grid_rows
        self.cols              = grid_cols
        self.critical_per_cell = critical_per_cell

    
    def render(self, frame: np.ndarray,
               centroids: list[tuple],
               grid_counts: np.ndarray | None = None) -> np.ndarray:
        
        h, w   = frame.shape[:2]
        out    = frame.copy()
        overlay = frame.copy()

        if grid_counts is None:
            grid_counts = self._build_grid(centroids, w, h)

        cell_w = w // self.cols
        cell_h = h // self.rows

        for r in range(self.rows):
            for c in range(self.cols):
                count = int(grid_counts[r, c])
                color = self._zone_color(count)

                x1 = c * cell_w
                y1 = r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                if count == 0:
                    continue

                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

                
                label = str(count)
                lw, lh = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                cv2.putText(out, label,
                            (x1 + (cell_w - lw) // 2,
                             y1 + (cell_h + lh) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)

        
        cv2.addWeighted(overlay, ZONE_ALPHA, out, 1 - ZONE_ALPHA, 0, out)

        
        self._draw_clusters(out, centroids, w, h)

        return out

    def _zone_color(self, count: int) -> tuple[int, int, int]:
        if count == 0:
            return (80, 80, 80)
        ratio = min(count / self.critical_per_cell, 1.0)
        if ratio < 0.25:
            return ZONE_COLORS["safe"]
        elif ratio < 0.50:
            return ZONE_COLORS["caution"]
        elif ratio < 0.80:
            return ZONE_COLORS["high"]
        else:
            return ZONE_COLORS["critical"]

    
    def _build_grid(self, centroids, frame_w, frame_h) -> np.ndarray:
        grid   = np.zeros((self.rows, self.cols), dtype=np.int32)
        cell_w = max(frame_w // self.cols, 1)
        cell_h = max(frame_h // self.rows, 1)
        for cx, cy in centroids:
            col = min(int(cx // cell_w), self.cols - 1)
            row = min(int(cy // cell_h), self.rows - 1)
            grid[row, col] += 1
        return grid

    
    def _draw_clusters(self, frame, centroids, fw, fh):
        """Draw danger-radius circle around clusters of >= 3 people."""
        if len(centroids) < 3:
            return

        pts = np.array(centroids, dtype=np.float32)
        cell_r = min(fw // self.cols, fh // self.rows) * 0.6

        for i, (cx, cy) in enumerate(centroids):
            neighbours = [
                (ox, oy) for j, (ox, oy) in enumerate(centroids)
                if i != j and
                ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 < cell_r
            ]
            if len(neighbours) >= 2:
                
                all_pts = [(cx, cy)] + neighbours
                avg_x   = int(np.mean([p[0] for p in all_pts]))
                avg_y   = int(np.mean([p[1] for p in all_pts]))
                radius  = int(max(
                    ((cx - avg_x) ** 2 + (cy - avg_y) ** 2) ** 0.5 + 30,
                    40
                ))
                cv2.circle(frame, (avg_x, avg_y), radius,
                           (30, 30, 220), 1, cv2.LINE_AA)
