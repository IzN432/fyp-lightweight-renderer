"""ImGui panels from Python.

Call these only from a ``Viewer.on_gui`` callback; anywhere else they raise ``RuntimeError`` (ImGui would
otherwise assert and abort the process). Value widgets follow the usual Python-ImGui convention of
returning ``(changed, value)``::

    def gui():
        with lr.gui.window("Controls"):
            changed, settings.exposure = lr.gui.slider_float("Exposure", settings.exposure, 0.1, 4.0)
            if lr.gui.button("Reset"):
                settings.exposure = 1.0

    viewer.on_gui(gui)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

from ._lr import _gui


@contextmanager
def window(
    title: str, size: tuple[float, float] | None = None, position: tuple[float, float] | None = None
) -> Iterator[bool]:
    """A window; yields whether its contents are visible (False when collapsed). Always closed on exit,
    even if the body raises. `size` / `position` (pixels) apply the first time the window appears;
    after that the user's resizing and moving win (ImGui remembers them in imgui.ini)."""
    visible, _ = _gui.begin(title, False, size, position)
    try:
        yield visible
    finally:
        _gui.end()


def begin(
    title: str,
    closable: bool = False,
    size: tuple[float, float] | None = None,
    position: tuple[float, float] | None = None,
) -> tuple[bool, bool]:
    """Low-level window start: returns (visible, open); `open` turns False when a closable window's close
    button is pressed. Must be paired with end() — prefer `with window(...)`."""
    return _gui.begin(title, closable, size, position)


def end() -> None:
    _gui.end()


def text(text: str) -> None:
    _gui.text(text)


def button(label: str) -> bool:
    """True on the frame the button is clicked."""
    return _gui.button(label)


def checkbox(label: str, value: bool) -> tuple[bool, bool]:
    return _gui.checkbox(label, value)


def slider_float(
    label: str, value: float, min: float, max: float, format: str = "%.3f", logarithmic: bool = False
) -> tuple[bool, float]:
    return _gui.slider_float(label, value, min, max, format, logarithmic)


def slider_int(label: str, value: int, min: int, max: int) -> tuple[bool, int]:
    return _gui.slider_int(label, value, min, max)


def drag_float(label: str, value: float, speed: float = 0.01, min: float = 0.0, max: float = 0.0) -> tuple[bool, float]:
    """Drag to change; min == max means unbounded."""
    return _gui.drag_float(label, value, speed, min, max)


def color_edit3(label: str, color: Sequence[float]) -> tuple[bool, tuple[float, float, float]]:
    return _gui.color_edit3(label, tuple(color))


def color_edit4(label: str, color: Sequence[float]) -> tuple[bool, tuple[float, float, float, float]]:
    return _gui.color_edit4(label, tuple(color))


def combo(label: str, current: int, items: Sequence[str]) -> tuple[bool, int]:
    """A drop-down; `current` and the returned value are indices into `items`."""
    return _gui.combo(label, current, list(items))


def collapsing_header(label: str, default_open: bool = True) -> bool:
    """A collapsible section header; True when expanded."""
    return _gui.collapsing_header(label, default_open)


def separator() -> None:
    _gui.separator()


def same_line() -> None:
    """Put the next widget on the same line as the previous one."""
    _gui.same_line()


def spacing() -> None:
    _gui.spacing()


def want_capture_mouse() -> bool:
    """True while the mouse is over (or dragging) ImGui UI — leave mouse input to the UI then. Safe anywhere."""
    return _gui.want_capture_mouse()


def want_capture_keyboard() -> bool:
    """True while a UI text field has keyboard focus. Safe anywhere."""
    return _gui.want_capture_keyboard()


def framerate() -> float:
    """ImGui's smoothed frames-per-second estimate. Safe anywhere."""
    return _gui.framerate()
