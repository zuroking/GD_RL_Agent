"""Assembles the 4-float feature vector from a raw BGR frame.

Feature vector (agreed in grill-me Q5):
  [dist_obstacle, dist_pit, obstacle_height, is_grounded]

All floats in [0, 1]. Sentinel for "nothing detected": dist=1.0, height=0.0.
"""

import numpy as np
from numpy.typing import NDArray

from configs.config import AgentConfig
from vision.obstacle_detector import ObstacleDetector
from vision.death_detector import DeathDetector


class StateExtractor:
    """Extracts the RL feature vector from a captured frame.

    Args:
        config: Full agent config (uses player calibration).
    """

    def __init__(self, config: AgentConfig) -> None:
        cal = config.player
        self._obstacle = ObstacleDetector(cal)
        frame_h = config.capture_region.height
        frame_w = config.capture_region.width
        self._death = DeathDetector(cal, (frame_h, frame_w))
        self._use_lookahead = config.hp.use_obstacle_lookahead

    def extract(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Return feature vector as a float32 array of shape (4,) or (6,).

        Args:
            frame: BGR uint8 array of the full capture region.

        Returns:
            [dist_obstacle, dist_pit, obstacle_height, is_grounded]
            If use_obstacle_lookahead is True, appends [dist_obstacle_2, height_2].
        """
        dist_obs, obs_height, dist_pit = self._obstacle.detect(frame)
        grounded = float(self._death.is_grounded(frame))

        vec = [dist_obs, dist_pit, obs_height, grounded]

        if self._use_lookahead:
            # Second obstacle: scan from just past the first obstacle column
            # Not yet implemented — returns sentinels until Phase 5
            vec += [1.0, 0.0]

        return np.array(vec, dtype=np.float32)

    def is_dead(self, frame: NDArray[np.uint8]) -> bool:
        return self._death.is_dead(frame)
