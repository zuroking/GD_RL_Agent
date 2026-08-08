"""Utilities for locating the Geometry Dash window on Windows."""

import ctypes
import ctypes.wintypes
from dataclasses import dataclass

import win32api
import win32con
import win32gui
import win32process


@dataclass
class WindowInfo:
    hwnd: int
    left: int
    top: int
    width: int
    height: int
    # Client area (excludes title bar / borders) — use for capture region
    client_left: int
    client_top: int
    client_width: int
    client_height: int


_GD_TITLES = ("Geometry Dash", "GeometryDash")


def _match_title(title: str) -> bool:
    return any(t.lower() in title.lower() for t in _GD_TITLES)


def find_gd_window() -> WindowInfo:
    """Find the Geometry Dash window and return its screen geometry.

    Raises:
        RuntimeError: If no matching window is found or client coords cannot
            be resolved.
    """
    found: list[int] = []

    def _cb(hwnd: int, _: None) -> bool:
        if win32gui.IsWindowVisible(hwnd) and _match_title(win32gui.GetWindowText(hwnd)):
            found.append(hwnd)
        return True

    win32gui.EnumWindows(_cb, None)

    if not found:
        raise RuntimeError(
            "Geometry Dash window not found. Make sure the game is running."
        )

    hwnd = found[0]
    rect = win32gui.GetWindowRect(hwnd)
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top

    # Client rect gives us the drawable area without the OS chrome
    client_rect = win32gui.GetClientRect(hwnd)
    cl_width = client_rect[2]
    cl_height = client_rect[3]

    # Convert client origin to screen coords
    pt = ctypes.wintypes.POINT(0, 0)
    result = ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    if not result:
        raise RuntimeError(
            f"ClientToScreen failed for HWND={hwnd}. "
            "The window may have been closed or minimised to tray."
        )

    return WindowInfo(
        hwnd=hwnd,
        left=left,
        top=top,
        width=width,
        height=height,
        client_left=pt.x,
        client_top=pt.y,
        client_width=cl_width,
        client_height=cl_height,
    )


def bring_to_foreground(hwnd: int) -> bool:
    """Bring the given window to the foreground (required for pydirectinput).

    Tries two approaches in order:
    1. AttachThreadInput to borrow focus from the current foreground process.
    2. Alt-key trick: briefly simulate an Alt keypress so Windows permits the
       foreground switch (documented Windows workaround for UIPI focus rules).

    Returns:
        True if the window is now in the foreground, False if both attempts
        failed (caller should warn the user to click GD manually).
    """
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    # Attempt 1: AttachThreadInput
    fg_hwnd = win32gui.GetForegroundWindow()
    fg_thread_id = win32process.GetWindowThreadProcessId(fg_hwnd)[0]
    my_thread_id = win32api.GetCurrentThreadId()

    if fg_thread_id != my_thread_id:
        attached = bool(
            ctypes.windll.user32.AttachThreadInput(fg_thread_id, my_thread_id, True)
        )
        try:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                ctypes.windll.user32.AttachThreadInput(fg_thread_id, my_thread_id, False)
    else:
        ctypes.windll.user32.SetForegroundWindow(hwnd)

    if win32gui.GetForegroundWindow() == hwnd:
        return True

    # Attempt 2: Alt-key trick — a simulated Alt keypress temporarily unlocks
    # the foreground-window restriction in Windows.
    VK_MENU = 0x12
    KEYEVENTF_KEYUP = 0x0002
    ctypes.windll.user32.keybd_event(VK_MENU, 0, 0, 0)
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    ctypes.windll.user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

    return win32gui.GetForegroundWindow() == hwnd
