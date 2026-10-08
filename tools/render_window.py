"""Headless screenshot of the whole window, for visual inspection.

Uses the real MainWindow with mock audio devices, exactly as the GUI tests do, so the
render reflects the shipped layout rather than a separate redrawing.
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
    parser.add_argument('--page', default='home')
    parser.add_argument('--running', action='store_true',
                        help='Render as if the engine were running')
    args = parser.parse_args()

    import tempfile
    import time
    from PySide6.QtWidgets import QApplication
    application = QApplication.instance() or QApplication([])
    sys.path.insert(0, str(ROOT))
    from tests.fakes import FakeBackend
    from tests.test_ai_voice import FakeClient
    from src.audio.controller import AudioController
    from src.audio.engine import AudioEngine
    from src.processors.chain import ProcessorChain
    from src.processors.voice_router import VoiceRouter
    from src.processors.ai_voice import AIVoiceProcessor
    from src.settings.manager import SettingsManager
    from src.vc.bridge import AIBridge
    from src.gui.main_window import MainWindow

    bridge = AIBridge(client_factory=FakeClient)
    router = VoiceRouter(ai=AIVoiceProcessor(bridge=bridge))
    controller = AudioController(AudioEngine(ProcessorChain(router), FakeBackend()))
    window = MainWindow(SettingsManager(Path(tempfile.mkdtemp())/'settings.json'), controller)
    window.show()
    until = time.monotonic()+5
    while time.monotonic() < until and not window.start_button.isEnabled():
        application.processEvents()
        time.sleep(0.005)
    window.resize(args.width, args.height)
    window._show_page(args.page)
    if args.running:
        # Feed the orb and meters a level so the running state is visible.
        window._display_input = 0.42
        window._display_output = 0.55
        window.orb.set_level(0.55)
    window.grab().save(args.out)
    print('saved', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())