

import numpy as np
from scipy.spatial.distance import pdist


class DensityEstimator:
    

    
    COUNT_GATE = {
        "LOW":      0,
        "MEDIUM":   8,
        "HIGH":     18,
        "CRITICAL": 35,
    }

    def __init__(self, grid_rows: int = 4, grid_cols: int = 4,
                 critical_per_cell: int = 8,
                 min_safe_distance: float = 60.0):
        
        self.grid_rows        = grid_rows
        self.grid_cols        = grid_cols
        self.critical_per_cell = critical_per_cell
        self.min_safe_dist    = min_safe_distance

    
    def compute(self, centroids: list[tuple],
                frame_w: int, frame_h: int) -> tuple[float, np.ndarray]:
        
        count = len(centroids)
        grid  = self._build_grid(centroids, frame_w, frame_h)

        if count == 0:
            return 0.0, grid

        
        total_capacity = self.grid_rows * self.grid_cols * self.critical_per_cell
        fill_ratio = min(count / total_capacity, 1.0)

        
        hotspot = min(int(grid.max()) / self.critical_per_cell, 1.0)

        
        if count >= 2:
            pts = np.array(centroids, dtype=float)
            dists = pdist(pts)
            
            close_fraction = float((dists < self.min_safe_dist).mean())
        else:
            close_fraction = 0.0

        
        raw = (0.35 * fill_ratio
               + 0.40 * hotspot
               + 0.25 * close_fraction)

        raw = float(np.clip(raw, 0.0, 1.0))
        return raw, grid

    
    def _build_grid(self, centroids, frame_w, frame_h) -> np.ndarray:
        grid   = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        cell_w = max(frame_w // self.grid_cols, 1)
        cell_h = max(frame_h // self.grid_rows, 1)
        for cx, cy in centroids:
            col = min(int(cx // cell_w), self.grid_cols - 1)
            row = min(int(cy // cell_h), self.grid_rows - 1)
            grid[row, col] += 1
        return grid
