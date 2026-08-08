"""Death and level-completion detector based on pixel heuristics.

Two signals:
  - is_dead(): True when GD shows the death/restart screen.
  - is_grounded(): True when the cube sprite is on the ground (pixel-based,
    no timer — agreed in grill-me Q4).

Death detection strategy (no process memory access):
  Primary:  detect the "practice mode restart" or "level failed" overlay by
            checking for a near-black screen flash that GD plays on death,
            followed by the bright retry button appearing.
  Fallback: timeout — if is_dead() has not fired within MAX_EPISODE_STEPS the
            env loop treats it as a non-death terminal.

The death flash heuristic: GD dims the screen to near-black for ~3 frames
on cube death. We detect this by measuring mean luminance of the center
region of the frame.
"""

import numpy as np
import cv2
from numpy.typing import NDArray

from configs.config import PlayerCalibration


# Mean luminance below this threshold in the centre ROI = death flash.
# GD background luminance ~38 (from calibration frame); death flash drops to near-black.
# Threshold 15 leaves a clear margin without false-positives from dark parallax layers.
_DEATH_LUMINANCE_THRESHOLD = 15
# Centre ROI: middle 40% of the frame
_CENTRE_FRACTION = 0.4


class DeathDetector:
    """Detects cube death and grounded state from BGR frames.

    Args:
        cal: Calibrated player pixel constants.
        frame_shape: (H, W) of the capture region.
    """

    def __init__(self, cal: PlayerCalibration, frame_shape: tuple[int, int]) -> None:
        self._cal = cal
        h, w = frame_shape
        cx, cy = w // 2, h // 2
        dw = int(w * _CENTRE_FRACTION / 2)
        dh = int(h * _CENTRE_FRACTION / 2)
        self._cx0 = cx - dw
        self._cx1 = cx + dw
        self._cy0 = cy - dh
        self._cy1 = cy + dh

        # Grounded check: single pixel at top of cube when landed.
        # ground_y - cube_height_px is the cube top when sitting on ground.
        self._grounded_x = cal.player_right_x - cal.cube_height_px // 2
        self._grounded_y = cal.ground_y - cal.cube_height_px

        # HSV range for the player cube colour (bright red, set by user).
        # These are defaults; update after seeing the actual cube colour.
        self._cube_hsv_lower = np.array([0,   150, 150], dtype=np.uint8)
        self._cube_hsv_upper = np.array([10,  255, 255], dtype=np.uint8)
        # Red wraps around H=180 in OpenCV HSV, so we also check the upper range
        self._cube_hsv_lower2 = np.array([170, 150, 150], dtype=np.uint8)
        self._cube_hsv_upper2 = np.array([180, 255, 255], dtype=np.uint8)

    def is_dead(self, frame: NDArray[np.uint8]) -> bool:
        """Return True if the frame shows the death flash (near-black screen).

        Args:
            frame: BGR uint8 array.
        """
        centre = frame[self._cy0:self._cy1, self._cx0:self._cx1]
        gray = cv2.cvtColor(centre, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < _DEATH_LUMINANCE_THRESHOLD

    def is_grounded(self, frame: NDArray[np.uint8]) -> bool:
        """Return True if the cube sprite is at ground level.

        Checks a single pixel at the expected position of the cube top
        when the cube is sitting on the ground. If the pixel matches the
        cube HSV colour, the cube is grounded.

        Args:
            frame: BGR uint8 array.
        """
        h, w = frame.shape[:2]
        x = max(0, min(self._grounded_x, w - 1))
        y = max(0, min(self._grounded_y, h - 1))

        px_bgr = frame[y, x].reshape(1, 1, 3).astype(np.uint8)
        px_hsv = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2HSV)[0, 0]

        in_range1 = (
            (self._cube_hsv_lower[0] <= px_hsv[0] <= self._cube_hsv_upper[0])
            and (self._cube_hsv_lower[1] <= px_hsv[1] <= self._cube_hsv_upper[1])
            and (self._cube_hsv_lower[2] <= px_hsv[2] <= self._cube_hsv_upper[2])
        )
        in_range2 = (
            (self._cube_hsv_lower2[0] <= px_hsv[0] <= self._cube_hsv_upper2[0])
            and (self._cube_hsv_lower2[1] <= px_hsv[1] <= self._cube_hsv_upper2[1])
            and (self._cube_hsv_lower2[2] <= px_hsv[2] <= self._cube_hsv_upper2[2])
        )
        return in_range1 or in_range2
