"""HSV-threshold ROI obstacle and pit detector.

Scans a horizontal strip ahead of the player cube each frame to produce:
  - dist_obstacle: normalised distance [0,1] to nearest obstacle above ground
  - obstacle_height: normalised height [0,1] of that obstacle
  - dist_pit: normalised distance [0,1] to nearest gap in the ground

All values use sentinel 1.0 (dist) / 0.0 (height) when nothing is detected
within SCAN_DEPTH — agreed in grill-me Q5.

Coordinate convention: image origin top-left, y increases downward.
"""

import numpy as np
import cv2
from numpy.typing import NDArray

from configs.config import PlayerCalibration


# HSV range for ground/obstacle tiles in Stereo Madness (yellow-orange).
# Calibrate empirically in Phase 2 debug mode if misdetections appear.
_TILE_HSV_LOWER = np.array([15, 80, 80],  dtype=np.uint8)
_TILE_HSV_UPPER = np.array([45, 255, 255], dtype=np.uint8)

# Minimum consecutive orange pixels in a column to count as obstacle presence.
_MIN_OBSTACLE_PX = 4
# Minimum gap columns to count as a pit (filters 1-2px noise).
_MIN_PIT_COLS = 6


class ObstacleDetector:
    """Detects obstacles and pits from a BGR frame.

    Args:
        cal: Calibrated player pixel constants.
    """

    def __init__(self, cal: PlayerCalibration) -> None:
        self._cal = cal
        # ROI x-start: skip the player cube body + particle guard
        self._roi_x0 = cal.player_right_x + cal.guard_px
        # scan_depth measured from roi_x0 (after the guard), not from player_right_x
        self._roi_x1 = cal.player_right_x + cal.guard_px + cal.scan_depth
        # Vertical ROI: from (ground_y - 4*cube_height) down to ground_y+4
        # Wide enough to catch tall multi-tile obstacles.
        self._roi_y0 = max(0, cal.ground_y - 4 * cal.cube_height_px)
        self._roi_y1 = cal.ground_y + 4  # slight margin below ground

    def detect(
        self, frame: NDArray[np.uint8]
    ) -> tuple[float, float, float]:
        """Return (dist_obstacle, obstacle_height, dist_pit).

        Args:
            frame: BGR uint8 array, full capture region (H×W×3).

        Returns:
            Tuple of three normalised floats in [0, 1].
            Sentinels: dist=1.0, height=0.0 when nothing detected.
        """
        h, w = frame.shape[:2]
        x0 = max(0, self._roi_x0)
        x1 = min(w, self._roi_x1)
        y0 = max(0, self._roi_y0)
        y1 = min(h, self._roi_y1)

        roi = frame[y0:y1, x0:x1]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        tile_mask = cv2.inRange(hsv_roi, _TILE_HSV_LOWER, _TILE_HSV_UPPER)

        n_cols = x1 - x0
        ground_row_in_roi = self._cal.ground_y - y0  # ground_y relative to ROI top

        dist_obstacle, obstacle_height = self._find_obstacle(
            tile_mask, ground_row_in_roi, n_cols
        )
        dist_pit = self._find_pit(tile_mask, ground_row_in_roi, n_cols)

        return dist_obstacle, obstacle_height, dist_pit

    def _find_obstacle(
        self,
        mask: NDArray[np.uint8],
        ground_row: int,
        n_cols: int,
    ) -> tuple[float, float]:
        """Find nearest column with tile pixels strictly above ground_row."""
        scan_top = max(0, ground_row - 4 * self._cal.cube_height_px)

        for col in range(n_cols):
            col_slice = mask[scan_top:ground_row, col]
            if int(col_slice.sum()) >= _MIN_OBSTACLE_PX * 255:
                # Height: count tile pixels above ground
                px_above = int(col_slice.sum()) // 255
                norm_dist = col / max(n_cols - 1, 1)
                norm_height = min(px_above / (4 * self._cal.cube_height_px), 1.0)
                return norm_dist, norm_height

        return 1.0, 0.0

    def _find_pit(
        self,
        mask: NDArray[np.uint8],
        ground_row: int,
        n_cols: int,
    ) -> float:
        """Find nearest column where ground_row has no tile pixels (= pit)."""
        # Scan a 3-pixel band around ground_y for robustness
        band_top = max(0, ground_row - 1)
        band_bot = min(mask.shape[0], ground_row + 2)
        ground_band = mask[band_top:band_bot, :]  # shape (3, n_cols)

        consecutive = 0
        for col in range(n_cols):
            if ground_band[:, col].max() == 0:
                consecutive += 1
                if consecutive >= _MIN_PIT_COLS:
                    first_col = col - consecutive + 1
                    return first_col / max(n_cols - 1, 1)
            else:
                consecutive = 0

        return 1.0
