"""Jump input injection for Geometry Dash on Windows.

Uses pydirectinput (DirectInput scan codes) so that the game receives
the keypress even when it is in the foreground but not the active focus
of a different input method. Requires the GD window to be in the
foreground — call bring_to_foreground() before starting the loop.

The default key binding in GD for jump is Space.
"""

import time

import numpy as np
import pydirectinput

from capture.window_utils import WindowInfo

# GD default jump key
_JUMP_KEY = "space"

# Minimum hold duration in seconds so GD registers a tap (one physics frame)
_TAP_DURATION = 0.05


class Controller:
    """Sends jump commands to Geometry Dash and measures round-trip latency.

    Args:
        window: WindowInfo for the GD window (used only for focus checks).
    """

    def __init__(self, window: WindowInfo) -> None:
        self._window = window
        # pydirectinput pause between calls (default 0.1s is too slow)
        pydirectinput.PAUSE = 0.0

    def jump(self) -> float:
        """Send a single jump keypress and return the time taken in seconds.

        Returns:
            Duration from keydown to keyup in seconds.
        """
        t0 = time.perf_counter()
        pydirectinput.keyDown(_JUMP_KEY)
        time.sleep(_TAP_DURATION)
        pydirectinput.keyUp(_JUMP_KEY)
        return time.perf_counter() - t0

    def no_op(self) -> None:
        """Do nothing (explicit no-op for clarity in the env loop)."""

    def measure_injection_latency(self, n: int = 20) -> dict[str, float]:
        """Measure keyDown+keyUp round-trip latency.

        This measures the software-side injection cost only (time from
        pydirectinput.keyDown call to pydirectinput.keyUp return). It does
        NOT measure the full pipeline latency to visual confirmation in-game —
        that requires a separate capture-loop measurement in the calibrate CLI.

        Args:
            n: Number of samples.

        Returns:
            Dict with keys: mean_ms, min_ms, max_ms, std_ms.
        """
        samples: list[float] = []
        for _ in range(n):
            duration_s = self.jump()
            samples.append(duration_s * 1000)
            time.sleep(0.2)  # avoid rapid-fire during measurement

        mean = sum(samples) / len(samples)
        return {
            "mean_ms": mean,
            "min_ms": min(samples),
            "max_ms": max(samples),
            "std_ms": float(np.std(samples)),
        }
