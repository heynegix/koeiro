"""Startup splash: app-color overlay inside the main window, icon above the wordmark.

The overlay covers exactly the main window (not the screen), so only the splash
is seen while the window behind it stays hidden in plain sight; afterwards the
window is revealed at once. Input stays locked for the whole sequence.

Timeline (all non-blocking; timers are children of the overlay, so an early
close cancels the rest):
    content fade in   900 ms
    hold              900 ms
    content fade out 1200 ms  (-> 3000 ms splash, then the window appears)
"""
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QVBoxLayout, QWidget

BACKGROUND = "#0B1020"

FADE_IN_MS = 900
HOLD_MS = 900
FADE_OUT_MS = 1200
SPLASH_MS = FADE_IN_MS + HOLD_MS + FADE_OUT_MS


def asset_path(name):
    """Brand file under assets/, or None when it is not there.

    Missing assets degrade to no icon/splash rather than a crash, so a source
    checkout without the images still runs.
    """
    candidate = Path(__file__).resolve().parents[2] / "assets" / name
    return candidate if candidate.is_file() else None


def window_icon():
    """The app icon for the window corner and the taskbar (may be null)."""
    path = asset_path("icon.png")
    return QIcon(str(path)) if path is not None else QIcon()


def _fade(target, prop, start, value, duration, on_finished=None):
    animation = QPropertyAnimation(target, prop, target)
    animation.setDuration(duration)
    animation.setStartValue(start)
    animation.setEndValue(value)
    animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
    if on_finished is not None:
        animation.finished.connect(on_finished)
    animation.start()
    return animation


def _after(ms, parent, slot):
    timer = QTimer(parent)
    timer.setSingleShot(True)
    timer.setInterval(ms)
    timer.timeout.connect(slot)
    timer.start()
    return timer


def _finish(overlay, window):
    """End state shared by the timer path and the tests."""
    try:
        overlay.releaseKeyboard()
    except RuntimeError:
        pass


class _FollowHost(QObject):
    """Keep the overlay glued to the central widget through resizes."""

    def __init__(self, overlay, host):
        super().__init__(overlay)
        self._overlay = overlay
        self._host = host

    def eventFilter(self, watched, event):
        if watched is self._host and event.type() == QEvent.Type.Resize:
            self._overlay.setGeometry(self._host.rect())
        return False


def show_splash(window):
    """Cover the screen, play the fade, then reveal the main window at once.

    Returns the overlay widget, or None when the brand images are missing.
    The main window is shown maximized (disabled) by this call.
    """
    icon_path, logo_path = asset_path("icon.png"), asset_path("logo.png")
    if icon_path is None or logo_path is None:
        if not window.isVisible():
            window.showMaximized()
        return None
    if not window.isVisible():
        # Maximized is the standard launch state: a full-monitor window that
        # still leaves the taskbar visible and stays behind/in front of others.
        window.showMaximized()

    central = getattr(window, "centralWidget", None)
    host = central() if callable(central) else window
    if host is None:
        host = window
    overlay = QWidget(host)
    overlay.setObjectName("splash")
    overlay.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    overlay.setStyleSheet(f"background-color: {BACKGROUND};")
    # Input lock without graying the window: the overlay covers everything
    # (mouse) and holds the keyboard grab (keys). Disabling the window instead
    # would render every pixmap underneath in grayscale.
    overlay.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    # A child, not a window: it travels with the application and can never
    # cover anything outside it.
    overlay.setGeometry(host.rect())
    host.installEventFilter(_FollowHost(overlay, host))
    frame = QVBoxLayout(overlay)
    frame.setContentsMargins(0, 0, 0, 0)
    frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
    content = QWidget()
    content.setObjectName("splashContent")
    content.setStyleSheet("background-color: transparent;")
    layout = QVBoxLayout(content)
    layout.setContentsMargins(28, 28, 28, 28)
    layout.setSpacing(12)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    icon_label = QLabel()
    icon_label.setObjectName("splashIcon")
    icon_label.setPixmap(QPixmap(str(icon_path)).scaled(
        180, 180, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation))
    icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    logo_label = QLabel()
    logo_label.setObjectName("splashLogo")
    logo_label.setPixmap(QPixmap(str(logo_path)).scaled(
        340, 340, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation))
    logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(icon_label)
    layout.addWidget(logo_label)
    frame.addWidget(content)
    overlay.show()
    # Above every sibling (the floating guide button re-raises itself).
    overlay.raise_()
    overlay.grabKeyboard()

    effect = QGraphicsOpacityEffect(content)
    effect.setOpacity(0.0)
    content.setGraphicsEffect(effect)

    def begin_fade_out():
        overlay._fade_out = _fade(
            effect, b"opacity", 1.0, 0.0, FADE_OUT_MS, on_finished=begin_reveal)

    def begin_reveal():
        _finish(overlay, window)
        overlay.close()

    overlay._fade_in = _fade(effect, b"opacity", 0.0, 1.0, FADE_IN_MS)
    _after(FADE_IN_MS + HOLD_MS, overlay, begin_fade_out)
    overlay._finish = lambda: _finish(overlay, window)
    return overlay
