"""Regenerate derived brand assets from the two sources (needs Pillow).

Sources (committed, do not edit by hand):
    assets/icon.png   app icon (transparent badge)
    assets/logo.png   koeiro wordmark (transparent, white text)

Outputs:
    assets/icon.ico   multi-size Windows icon (taskbar / exe embedding)
    assets/brand.png  icon-above-logo banner for the README top (1200px)
    assets/social.png wide banner for the GitHub social preview (1280x640)

The icon ships as drawn: no backing is added anywhere, so the badge looks
the same on the taskbar, the window corner and in Explorer.
"""
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("error: Pillow が必要です（pip install pillow）")
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
BACKGROUND = (0, 0, 0, 255)


def main():
    icon = Image.open(ASSETS / "icon.png").convert("RGBA")
    logo = Image.open(ASSETS / "logo.png").convert("RGBA")

    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon.save(ASSETS / "icon.ico", sizes=sizes)
    print("wrote assets/icon.ico")

    # README banner: icon on top, logo below, on black (the white wordmark
    # needs a dark backing everywhere it appears).
    width = 1200
    icon_side = 480
    logo_width = 1000
    logo_height = round(logo.height * logo_width / logo.width)
    margin, gap = 60, 40
    banner = Image.new("RGBA", (width, margin + icon_side + gap + logo_height + margin),
                       BACKGROUND)
    small_icon = icon.resize((icon_side, icon_side), Image.LANCZOS)
    banner.alpha_composite(small_icon, ((width - icon_side) // 2, margin))
    small_logo = logo.resize((logo_width, logo_height), Image.LANCZOS)
    banner.alpha_composite(small_logo, ((width - logo_width) // 2, margin + icon_side + gap))
    banner.save(ASSETS / "brand.png")
    print(f"wrote assets/brand.png {banner.size}")

    # Social preview: icon left, logo right, 1280x640.
    social = Image.new("RGBA", (1280, 640), BACKGROUND)
    mark = icon.resize((440, 440), Image.LANCZOS)
    social.alpha_composite(mark, (90, 100))
    word = logo.resize((640, round(logo.height * 640 / logo.width)), Image.LANCZOS)
    social.alpha_composite(word, (580, (640 - word.height) // 2))
    social.save(ASSETS / "social.png")
    print(f"wrote assets/social.png {social.size}")


if __name__ == "__main__":
    main()
