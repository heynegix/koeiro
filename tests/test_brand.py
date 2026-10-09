"""Brand assets and the startup splash.

The icon and wordmark are committed sources; the splash shows them with one
3-second fade. Headless-safe throughout (offscreen platform included).
"""
from pathlib import Path

from PIL import Image

from src.gui import splash
from src.gui.splash import (FADE_IN_MS, FADE_OUT_MS, HOLD_MS, SPLASH_MS,
                             asset_path, show_splash, window_icon)
from tests.test_ai_gui import ai_window, pump  # noqa: F401  (ai_window is used as a fixture)


def test_brand_source_and_derived_assets_exist():
    root = Path("assets")
    for name in ("icon.png", "logo.png", "brand.png", "social.png",
                 "icon.ico"):
        assert (root / name).is_file(), name
    assert asset_path("icon.png") is not None
    assert asset_path("missing.png") is None


def test_window_icon_uses_the_raw_badge():
    """The live icon ships as drawn: no backing is added anywhere."""
    with Image.open("assets/icon.png").convert("RGBA") as raw:
        assert raw.getpixel((0, 0))[3] == 0
    assert not Path("assets/icon_small.png").exists()


def test_brand_banner_stacks_icon_above_logo():
    """Top half holds the badge art, bottom half the wordmark, on black."""
    with Image.open("assets/brand.png") as banner:
        assert banner.size == (1200, 1140)
        dark = (0, 0, 0)
        # Margins stay black.
        assert banner.getpixel((600, 10))[:3] == dark
        assert banner.getpixel((600, 1130))[:3] == dark

        def lit(box):
            left, top, right, bottom = box
            return any(banner.getpixel((x, y))[:3] != dark
                       for y in range(top, bottom, 7) for x in range(left, right, 7))

        assert lit((360, 60, 840, 540))  # icon zone
        assert lit((100, 640, 1100, 1080))  # logo zone


def test_splash_timing_totals_three_seconds():
    assert SPLASH_MS == 3000
    assert FADE_IN_MS + HOLD_MS + FADE_OUT_MS == SPLASH_MS


def test_window_carries_the_app_icon(ai_window):
    _app, window, _backend, _bridge = ai_window
    assert not window.windowIcon().isNull()
    assert not window_icon().isNull()


def test_sidebar_brand_uses_the_icon(ai_window):
    from PySide6.QtWidgets import QLabel
    _app, window, _backend, _bridge = ai_window
    mark = window.sidebar.findChild(QLabel, "brandIcon")
    assert mark is not None and not mark.pixmap().isNull()


def test_splash_locks_and_reveals_the_window(ai_window):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QWidget
    app, window, _backend, _bridge = ai_window
    overlay = show_splash(window)
    assert overlay is not None and overlay.isVisible()
    # Locked without graying the window: the overlay covers the mouse and
    # holds the keyboard grab instead of disabling anything.
    assert window.isEnabled()
    assert window.isVisible()
    assert overlay.focusPolicy() == Qt.FocusPolicy.StrongFocus
    # A child of the central widget: it travels with the app window and
    # can never cover anything outside it.
    assert overlay.parentWidget() is window.centralWidget()
    assert overlay.geometry() == window.centralWidget().rect()
    content = overlay.findChild(QWidget, "splashContent")
    assert content is not None
    labels = [child for child in content.findChildren(QLabel)
              if child.objectName() in ("splashIcon", "splashLogo")]
    assert len(labels) == 2
    positions = {child.objectName(): child.geometry().center().y() for child in labels}
    assert all(child.pixmap() is not None and not child.pixmap().isNull()
               for child in labels)
    assert positions["splashIcon"] < positions["splashLogo"]
    overlay._finish()
    overlay.close()
    # close() hides synchronously; C++ deletion timing is Qt-internal.
    assert overlay.isHidden()


def test_splash_renders_in_full_color(ai_window, tmp_path):
    """Disabled ancestors gray every pixmap: lock input without disabling.

    Renders the overlay and requires real chroma in the icon zone, so a
    gray-out regression like setEnabled(False) fails here, not in production.
    """
    import colorsys
    from PySide6.QtWidgets import QLabel, QWidget
    app, window, _backend, _bridge = ai_window
    overlay = show_splash(window)
    assert overlay is not None
    overlay._fade_in.stop()
    content = overlay.findChild(QWidget, "splashContent")
    content.graphicsEffect().setOpacity(1.0)
    app.processEvents()
    target = tmp_path / "splash.png"
    overlay.grab().save(str(target))
    icon_label = content.findChild(QLabel, "splashIcon")
    top_left = icon_label.mapTo(overlay, icon_label.rect().topLeft())
    box = (top_left.x() + 20, top_left.y() + 20,
           top_left.x() + icon_label.width() - 20,
           top_left.y() + icon_label.height() - 20)
    with Image.open(target).convert("RGB") as shot:
        pixels = [shot.getpixel((x, y))
                  for y in range(box[1], box[3], 4) for x in range(box[0], box[2], 4)]
    saturation = sum(colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[1]
                     for r, g, b in pixels) / len(pixels)
    assert saturation > 0.2, saturation
    overlay.close()


def test_no_splash_widget_without_explicit_show(ai_window):
    """Automation never gets a splash unless it asks for one."""
    _app, window, _backend, _bridge = ai_window
    found = [child for child in window.children()
             if child.objectName() == "splash" and child.isVisible()]
    assert found == []
