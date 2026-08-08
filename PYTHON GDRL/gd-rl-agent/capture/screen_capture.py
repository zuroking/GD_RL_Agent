"""mss-based screen grabber with configurable capture region."""

import time
from typing import Generator

import mss
import mss.tools
import numpy as np
from numpy.typing import NDArray

from configs.config import CaptureRegion


class ScreenCapture:
    """Grabs BGR frames from a fixed screen region using mss.

    Args:
        region: Absolute screen coordinates to capture.
    """

    def __init__(self, region: CaptureRegion) -> None:
        self._region = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        self._sct = mss.mss()

    def grab(self) -> NDArray[np.uint8]:
        """Capture one frame as an (H, W, 3) BGR uint8 array."""
        raw = self._sct.grab(self._region)
        # mss returns BGRA; drop alpha channel
        frame = np.array(raw, dtype=np.uint8)[:, :, :3]
        return frame

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def measure_fps(self, n_frames: int = 60) -> float:
        """Measure the achievable capture rate for the configured region.

        Args:
            n_frames: Number of frames to average over.

        Returns:
            Frames per second.
        """
        t0 = time.perf_counter()
        for _ in range(n_frames):
            self.grab()
        elapsed = time.perf_counter() - t0
        return n_frames / elapsed
