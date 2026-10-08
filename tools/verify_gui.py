"""Opt-in native Windows GUI / real CABLE regression, with labelled WAV replay."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.settings.manager import SettingsManager
from src.presets.female_presets import PRESETS
from src.processors.chain import ProcessorChain
from tools.verify_audio import VerificationSignal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=int, required=True)
    parser.add_argument('--output', type=int, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=Path('test-results/v021-gui.json'))
    args = parser.parse_args()
    sys.setswitchinterval(.001)
    app = QApplication([])
    window = MainWindow(SettingsManager(Path('test-results/gui-settings.json')))
    window.show()

    def pump(predicate=lambda: False, seconds=5, required=True):
        until = time.monotonic()+seconds
        while time.monotonic() < until:
            app.processEvents()
            if predicate():
                return
            time.sleep(.002)
        if required:
            raise RuntimeError('GUI operation timed out')

    report = dict(source=str(args.wav), input_kind='Human WAV replay at Main, not live microphone', cycles=[])
    try:
        pump(lambda: window.start_button.isEnabled())
        for combo, index in ((window.input_device, args.input), (window.output_device, args.output)):
            row = next(i for i in range(combo.count()) if combo.itemData(i).index == index)
            combo.setCurrentIndex(row)
        if 'cable input' not in window.output_device.currentData().name.lower():
            raise ValueError('Only CABLE Input is allowed')
        window.controller.engine.chain = ProcessorChain(main_processor=VerificationSignal(window.dsp, args.wav))
        for cycle in range(10):
            window.preset.setCurrentText('Anime Test')
            window.start_button.click()
            pump(lambda: window.stop_button.isEnabled())
            for change in range(10):
                window.preset.setCurrentText(list(PRESETS)[(cycle*10+change) % 4])
                window.pitch.setValue((change*13)%120)
                window.formant.setValue((change*5)%60)
                window.brightness.setValue((change*9)%100)
                pump(seconds=.10, required=False)
            engine = window.controller.engine
            report['cycles'].append(dict(callback=asdict(engine.performance.snapshot()),
                dsp=asdict(engine.chain.performance.snapshot()), native=window.dsp._native.timing_snapshot(),
                underflow=engine.underflows, overflow=engine.overflows, diagnostics=engine.diagnostics.snapshot()))
            window.stop_button.click()
            pump(lambda: window.start_button.isEnabled())
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        window.close()
        pump(lambda: not window.controller.alive, seconds=10)
        app.processEvents()
        report['control_thread_stopped'] = not window.controller.alive
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
