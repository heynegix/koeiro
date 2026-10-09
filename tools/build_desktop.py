"""Regenerate the Linux desktop entry and the icons a taskbar needs (Pillow).

Sources (committed, do not edit by hand):
    assets/icon.png                  app icon (transparent badge)

Outputs (committed):
    assets/icon-<n>.png              standard sizes: 16…512, plus 1024
    Koeiro.desktop                   application entry naming the icon

Why Linux needs these and Windows does not: Windows takes the window icon from Qt
and the taskbar from the icon embedded in the executable, so assets/icon.ico is
enough there. A Linux desktop draws the taskbar/dock entry from the application
entry's ``Icon=``, matched against the icon theme by name -- with no entry and no
sized icons, only the in-window icon appears and the dock falls back to a generic
mark. The in-window icon is unaffected by this: it comes from QApplication's window
icon in app.py, the same on both systems.

The sizes, not a single large image, are what let a dock pick a crisp icon instead
of scaling one down. Discord's activity image is unrelated: Discord fetches that
from the URL in src/presence/activity.py, so it is identical on Windows and Linux.
"""
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("error: Pillow が必要です（pip install pillow）")
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "icon.png"
SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)
DESKTOP = ROOT / "Koeiro.desktop"
# The icon is named, not pathed: a desktop resolves it through the icon theme,
# which is what makes it appear in the dock and the application menu.
ICON_NAME = "Koeiro"

TEMPLATE = """[Desktop Entry]
Type=Application
Name=Koeiro（声彩）
Name[en]=Koeiro
Comment=登録した声にリアルタイム変換するボイスチェンジャー
Comment[en]=Real-time voice changer into a registered voice
Exec=koeiro
Icon=Koeiro
Terminal=false
Categories=AudioVideo;Audio;
Keywords=voice;changer;koeiro;声彩;
StartupWMClass=Koeiro
"""


def main():
    icon = Image.open(SOURCE).convert("RGBA")
    # One icon name for the running app and the desktop entry: the window icon is
    # loaded from assets/icon.png, and the dock matches this entry by that name.
    largest = max(SIZES)
    resized = icon.resize((largest, largest), Image.LANCZOS)
    resized.save(ROOT / "assets" / "Koeiro.png", "PNG", optimize=True)
    print(f"wrote assets/Koeiro.png ({largest}px, apps/ and one file both work)")
    for size in SIZES:
        resized = icon.resize((size, size), Image.LANCZOS)
        target = ROOT / "assets" / f"Koeiro-{size}.png"
        resized.save(target, "PNG", optimize=True)
        print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size} bytes)")
    DESKTOP.write_text(TEMPLATE, "utf-8")
    print(f"wrote {DESKTOP.relative_to(ROOT)} (Exec=koeiro, Icon={ICON_NAME})")


if __name__ == "__main__":
    main()
