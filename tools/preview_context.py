"""Three-minute user listening window; starts no audio automatically."""
from pathlib import Path
import sys
import shutil
import json
import time
import argparse
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.settings.manager import SettingsManager
from src.utils.logging import configure_logging


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-only',choices=('full','gain','pitch'),default='full')
    parser.add_argument('--wait-ms',type=int,choices=range(20,157))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    path=root/'test-results/v08-listening/settings.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if (root/'settings.json').exists(): shutil.copy2(root/'settings.json',path)
    # Use exactly the scheduling policy of app.py / verify_ai_gui.py.
    sys.setswitchinterval(.001)
    configure_logging(path.parent/'logs')
    app=QApplication([])
    window=MainWindow(SettingsManager(path))
    window.prosody_on.setChecked(False)
    if args.wait_ms is not None:
        window.ai_wait.setValue(args.wait_ms)
    if args.control_only!='full':
        original=window.router.ai.bridge.prosody.control
        def isolated_control(now=None):
            pitch,gain=original(now)
            return (0.,gain) if args.control_only=='gain' else (pitch,0.)
        window.router.ai.bridge.prosody.control=isolated_control
    window.setWindowTitle(f'v0.8 安定性比較 {args.control_only} · AI待ち{args.wait_ms or window.ai_wait.value()}ms · 初期OFF · 180秒終了')
    window.show()
    samples=[]; started=time.monotonic()
    def sample():
        ai=window.router.ai.bridge.snapshot()
        samples.append(dict(seconds=time.monotonic()-started,ai=ai,
            callback=asdict(window.controller.engine.performance.snapshot()),
            underflow=window.controller.engine.underflows,overflow=window.controller.engine.overflows,
            gate_db=window.gate.value()/10,gain_db=window.gain.value()/10,
            input_peak=window.controller.engine.input_peak,output_peak=window.controller.engine.output_peak,
            gate_open=window.controller.engine.chain.gate._opened,
            requested_mode=window.mode.currentData(),prosody_enabled=window.prosody_on.isChecked()))
    timer=QTimer(window); timer.setInterval(1000); timer.timeout.connect(sample); timer.start()
    QTimer.singleShot(180000,window.close)
    result=app.exec()
    (path.parent/f'diagnostics-{args.control_only}.json').write_text(json.dumps(dict(control_only=args.control_only,
        samples=samples,elapsed=time.monotonic()-started),
        ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__': sys.exit(main())
