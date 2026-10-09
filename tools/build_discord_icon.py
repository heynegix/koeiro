"""Regenerate the image Discord shows, from the brand icon (needs Pillow).

Source (committed, do not edit by hand):
    assets/icon.png                  app icon (transparent badge)

Output (committed: Discord fetches it by URL, so it must exist on the published branch):
    assets/discord/icon.png          512x512 card image

Why a separate file and not assets/icon.png directly: the Discord activity points at a
URL on the published branch, so the bytes it fetches should be small, fixed-size and
PNG -- the same reason the portal route uploads a dedicated image. 512 covers the
"playing" card and its corner icon with room to spare; Discord renders uploaded assets
at 512 and documents PNG/JPEG/WebP only, never ICO.
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
TARGET = ROOT / "assets" / "discord" / "icon.png"
SIDE = 512


def main():
    icon = Image.open(SOURCE).convert("RGBA")
    if icon.size != (SIDE, SIDE):
        icon = icon.resize((SIDE, SIDE), Image.LANCZOS)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    icon.save(TARGET, "PNG", optimize=True)
    print(f"wrote {TARGET.relative_to(ROOT)} {icon.size} ({TARGET.stat().st_size} bytes)")
    print("assets/discord/icon.png は公開ブランチに push されて初めて Discord から取得できます")


if __name__ == "__main__":
    main()
