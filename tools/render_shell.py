"""Headless render of the new shell so the layout can be inspected as an image.

Qt paints widgets without a visible window when `grab()` is called on an unshown widget,
so this produces a real render of the current stylesheet and geometry rather than a
redrawing of the design by hand.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--width', type=int, default=1520)
    parser.add_argument('--height', type=int, default=980)
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication
    application = QApplication.instance() or QApplication([])

    from src.gui import shell

    # Standalone shell render: the sidebar, orb and rail exactly as the window uses them.
    window = shell.MicOrb()
    window.resize(760, 420)
    window.set_state('マイク入力中', '声が変換されて出力されます', active=True)
    window.set_level(0.62)
    window.grab().save(args.out)
    print('orb:', args.out)

    rail = shell.PresetRail(MainWindowPresets())
    rail.resize(322, 620)
    rail.select_silently('default')
    rail_path = str(Path(args.out).with_name(Path(args.out).stem+'_rail.png'))
    rail.grab().save(rail_path)
    print('rail:', rail_path)

    nav = shell.Sidebar()
    nav.resize(196, 620)
    nav_path = str(Path(args.out).with_name(Path(args.out).stem+'_sidebar.png'))
    nav.grab().save(nav_path)
    print('sidebar:', nav_path)


def MainWindowPresets():
    return (('default', 'デフォルト', '自然な声', '♫'),
            ('male_low', '男性（低音）', '落ち着いた低音', '♂'),
            ('female_high', '女性（高音）', 'やさしく高い声', '♀'),
            ('robot', 'ロボット', '機械的な声', '⚙'),
            ('anime', 'アニメ声', '可愛らしい声', '★'),
            ('echo', 'エコー', '残響のある声', '◔'))


if __name__ == '__main__':
    main()