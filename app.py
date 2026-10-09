import argparse
import json
import logging
from pathlib import Path
import sys

from src.utils.logging import configure_logging


def main():
    from src.runtime_paths import data_dir
    from src.version import APP_VERSION
    parser = argparse.ArgumentParser(description=f"Koeiro v{APP_VERSION} human-approved fixed voices (Windows and Linux)")
    parser.add_argument("--data-dir", type=Path, default=data_dir(),
                        help="Directory for settings.json and logs/app.log")
    parser.add_argument("--list-devices", action="store_true", help="List devices without opening audio")
    parser.add_argument("--smoke-test", action="store_true", help="Open GUI and close safely after 1 second; no audio")
    parser.add_argument("--apply-update", metavar="PLAN", default=None,
                        help="Apply a staged update from a plan file, then exit (used by the updater)")
    args = parser.parse_args()
    if args.apply_update:
        from src.update_swap import main as apply_update
        return apply_update(args.apply_update)
    if sys.platform not in ("win32", "linux"):
        parser.error("This build supports Windows 10/11 and Linux only")
    # Reduce pure-Python GIL scheduling stalls relative to a 5.33 ms deadline.
    # This changes only this app process, not Windows or other applications.
    sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
    configure_logging(args.data_dir / "logs")
    logging.getLogger("app").info("App startup; Python=%s; platform=%s", sys.version, sys.platform)
    if args.list_devices:
        from src.audio.devices import enumerate_devices
        for device in enumerate_devices():
            print(json.dumps({"index": device.index, "name": device.name, "host_api": device.host_api,
                              "inputs": device.max_input_channels, "outputs": device.max_output_channels},
                             ensure_ascii=False))
        return 0
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow
    from src.settings.manager import SettingsManager
    qt_app = QApplication(sys.argv[:1])
    qt_app.setApplicationName("Koeiro")
    window = MainWindow(SettingsManager(args.data_dir / "settings.json"))
    window.show()
    if args.smoke_test:
        QTimer.singleShot(1000, window.close)
    result = qt_app.exec()
    logging.getLogger("app").info("App exit; code=%d", result)
    return result


if __name__ == "__main__":
    sys.exit(main())
