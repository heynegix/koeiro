"""Update check, download staging and install swap.

Network is never touched: the GitHub API is stubbed, and the install swap runs
against temporary directories. The one thing automation cannot prove — that a
helper process replaces a locked executable after exit — stays out of scope.
"""
import io
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from src import app_update
from src.app_update import UpdateError
from src import update_swap
from src.update_swap import SwapError
from tests.test_ai_gui import ai_window, pump  # noqa: F401  (ai_window is used as a fixture)


def test_parse_version_orders_releases_and_previews():
    assert app_update.parse_version("v1.2.3") == (1, 2, 3, 1, 0)
    assert app_update.parse_version("0.11.0-preview.3") == (0, 11, 0, 0, 3)
    assert app_update.parse_version("v0.11.0-preview.3") < app_update.parse_version("v0.11.0")
    assert app_update.parse_version("v0.11.0-preview.3") < app_update.parse_version("v0.11.0-preview.4")
    assert app_update.parse_version("v0.9.2-preview.1") < app_update.parse_version("v0.11.0-preview.3")
    for bad in ("", "v1.2", "1.2.3.4", "latest", None):
        with pytest.raises(UpdateError):
            app_update.parse_version(bad)


def test_is_newer_ignores_unparsable_tags():
    assert app_update.is_newer("v9.9.9", "v0.11.0-preview.3") is True
    assert app_update.is_newer("v0.11.0-preview.3", "v0.11.0-preview.3") is False
    assert app_update.is_newer("v0.11.0", "v0.11.0-preview.3") is True
    assert app_update.is_newer("garbage", "v0.11.0-preview.3") is False


def test_pick_asset_selects_this_os_package():
    release = {"tag_name": "v1.0.0", "assets": [
        {"name": "koeiro-v1.0.0-windows.zip", "browser_download_url": "http://w", "size": 10},
        {"name": "koeiro-v1.0.0-linux.tar.gz", "browser_download_url": "http://l", "size": 20},
        {"name": "notes.txt", "browser_download_url": "http://n", "size": 1},
    ]}
    windows = app_update.pick_asset(release, platform="windows")
    assert windows == {"name": "koeiro-v1.0.0-windows.zip", "url": "http://w", "size": 10}
    linux = app_update.pick_asset(release, platform="linux")
    assert linux["url"] == "http://l"
    assert app_update.pick_asset(release, platform="darwin") is None
    assert app_update.pick_asset({"tag_name": "v1.0.0", "assets": []}, platform="windows") is None


def test_check_now_reports_found_uptodate_and_errors(monkeypatch):
    monkeypatch.setattr(app_update, "fetch_latest_release",
                        lambda *args, **kwargs: {"tag_name": "v9.9.9", "body": "notes",
                                          "assets": [{"name": "koeiro-v9.9.9-windows.zip",
                                                      "browser_download_url": "http://w",
                                                      "size": 5}]})
    monkeypatch.setattr(app_update, "pick_asset",
                        lambda release, platform=None: {"name": "x", "url": "http://w", "size": 5})
    found = app_update.check_now(current="v0.11.0-preview.3")
    assert found["status"] == "found" and found["tag"] == "v9.9.9"
    assert found["notes"] == "notes" and found["asset"]["url"] == "http://w"
    monkeypatch.setattr(app_update, "fetch_latest_release",
                        lambda *args, **kwargs: {"tag_name": "v0.11.0-preview.3"})
    assert app_update.check_now(current="v0.11.0-preview.3")["status"] == "uptodate"
    monkeypatch.setattr(app_update, "fetch_latest_release", lambda *args, **kwargs: None)
    assert app_update.check_now()["status"] == "error"
    monkeypatch.setattr(app_update, "fetch_latest_release",
                        lambda *args, **kwargs: {"tag_name": "v9.9.9", "assets": []})
    monkeypatch.setattr(app_update, "pick_asset", lambda release, platform=None: None)
    assert app_update.check_now(current="v0.1.0")["status"] == "error"


def _package_tree(root, version="v9.9.9"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "release.json").write_text(json.dumps({"app": "Koeiro", "version": version,
                                                   "platform": "windows"}), "utf-8")
    (root / "app.py").write_text("new", "utf-8")
    voices = root / "models" / "user_voices" / "user_default"
    voices.mkdir(parents=True)
    (voices / "reference.wav").write_bytes(b"new-default")
    return root


def test_extract_and_validate_package(tmp_path):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("koeiro/release.json", json.dumps({"app": "Koeiro", "version": "v9.9.9"}))
        bundle.writestr("koeiro/app.py", "new")
    package = app_update.extract_package(archive, tmp_path / "staging")
    assert app_update.read_package_release(package)["version"] == "v9.9.9"
    (tmp_path / "broken" / "app.py").parent.mkdir(parents=True)
    (tmp_path / "broken" / "app.py").write_text("x", "utf-8")
    with pytest.raises(UpdateError):
        app_update.read_package_release(tmp_path / "broken")
    with pytest.raises(UpdateError):
        app_update.extract_package(tmp_path / "missing.zip", tmp_path / "out")


def test_download_asset_streams_and_cleans_up(monkeypatch, tmp_path):
    payload = b"x" * 100

    class FakeResponse:
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            if not hasattr(self, "_sent"):
                self._sent = True
                return payload
            return b""

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())
    seen = []
    target = app_update.download_asset("http://example/x.zip", tmp_path / "x.zip",
                                       progress=lambda done, total: seen.append((done, total)))
    assert target.read_bytes() == payload
    assert seen == [(100, 100)]

    def failing(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", failing)
    with pytest.raises(UpdateError):
        app_update.download_asset("http://example/x.zip", tmp_path / "y.zip")
    assert not (tmp_path / "y.zip").exists()


def _install_tree(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "release.json").write_text(json.dumps({"app": "Koeiro", "version": "v0.11.0-preview.3"}),
                                       "utf-8")
    (root / "app.py").write_text("old", "utf-8")
    (root / "settings.json").write_text(json.dumps({"gain": 1}), "utf-8")
    default = root / "models" / "user_voices" / "user_default"
    default.mkdir(parents=True)
    (default / "reference.wav").write_bytes(b"old-default")
    custom = root / "models" / "user_voices" / "user_mine"
    custom.mkdir(parents=True)
    (custom / "reference.wav").write_bytes(b"mine")
    (root / "recordings" / "note.txt").parent.mkdir(parents=True)
    (root / "recordings" / "note.txt").write_text("keep", "utf-8")
    (root / "logs" / "old.log").parent.mkdir(parents=True)
    (root / "logs" / "old.log").write_text("drop", "utf-8")
    return root


def test_swap_install_replaces_app_but_keeps_user_content(tmp_path):
    install = _install_tree(tmp_path / "install")
    staging = _package_tree(tmp_path / "staging")
    report = update_swap.swap_install(staging, install)
    assert Path(report["install"]) == install
    assert (install / "app.py").read_text("utf-8") == "new"
    assert json.loads((install / "settings.json").read_text("utf-8")) == {"gain": 1}
    assert (install / "models" / "user_voices" / "user_default" / "reference.wav").read_bytes() == b"new-default"
    assert (install / "models" / "user_voices" / "user_mine" / "reference.wav").read_bytes() == b"mine"
    assert (install / "recordings" / "note.txt").read_text("utf-8") == "keep"
    assert not (install / "logs" / "old.log").exists()
    backup = Path(report["backup"])
    assert (backup / "app.py").read_text("utf-8") == "old"
    assert not staging.exists()


def test_swap_install_refuses_broken_input(tmp_path):
    install = _install_tree(tmp_path / "install")
    with pytest.raises(SwapError):
        update_swap.swap_install(install, install)
    with pytest.raises(SwapError):
        update_swap.swap_install(tmp_path / "missing", install)
    with pytest.raises(SwapError):
        update_swap.swap_install(_package_tree(tmp_path / "staging"), tmp_path / "missing")


def test_swap_install_restores_backup_on_failure(tmp_path, monkeypatch):
    install = _install_tree(tmp_path / "install")
    staging = _package_tree(tmp_path / "staging")
    real_move = update_swap.shutil.move

    def failing_move(source, destination):
        if Path(destination).name == "install":
            raise OSError("disk full")
        return real_move(source, destination)

    monkeypatch.setattr(update_swap.shutil, "move", failing_move)
    with pytest.raises(SwapError):
        update_swap.swap_install(staging, install)
    assert (install / "app.py").read_text("utf-8") == "old"


def test_wait_for_exit_returns_for_dead_processes():
    update_swap.wait_for_exit(os.getpid() + 300000, timeout=5)
    with pytest.raises(SwapError):
        update_swap.wait_for_exit(os.getpid(), timeout=0)


def test_launch_command_points_at_this_copy():
    command = app_update.launch_command()
    assert command and Path(command[-1]).name == ("Koeiro.exe" if getattr(sys, "frozen", False) else "app.py")


def test_begin_update_refuses_source_checkouts(monkeypatch, tmp_path):
    monkeypatch.setattr(app_update, "install_dir", lambda: None)
    with pytest.raises(UpdateError):
        app_update.begin_update_and_restart(tmp_path, "v9.9.9")


def test_update_settings_roundtrip():
    from src.settings.manager import AppSettings
    assert AppSettings().app_update_check is True
    assert AppSettings().app_update_ignored == ""
    values = AppSettings.from_dict({"app_update_check": False, "app_update_ignored": "v9.9.9"})
    assert values.app_update_check is False
    assert values.app_update_ignored == "v9.9.9"
    broken = AppSettings.from_dict({"app_update_check": "yes", "app_update_ignored": 123})
    assert broken.app_update_check is True
    assert broken.app_update_ignored == ""


def test_update_section_shows_version_and_drains_results(ai_window, monkeypatch):
    import json
    from src.version import APP_VERSION
    app, window, backend, bridge = ai_window
    assert APP_VERSION in window.update_version.text()
    assert window.update_auto.isChecked()
    window._offer_update = lambda: None  # stub the modal dialog
    window.files.results["update_check"] = json.dumps(
        {"status": "found", "tag": "v9.9.9", "notes": "n",
         "asset": {"name": "a.zip", "url": "http://x", "size": 1}})
    window._drain_update_results()
    assert window._update_pending["tag"] == "v9.9.9"
    assert window.update_check_btn.text() == "ダウンロードして更新"
    assert "v9.9.9" in window.update_status.text()
    window.files.results["update_check"] = json.dumps({"status": "uptodate", "tag": "v9.9.9"})
    window._drain_update_results()
    assert window._update_pending is None
    assert window.update_check_btn.text() == "更新を確認"
    window.files.results["update_check"] = json.dumps({"status": "error", "message": "offline"})
    window._drain_update_results()
    assert "確認できませんでした" in window.update_status.text()


def test_update_download_in_source_checkout_reports_gracefully(ai_window):
    import json
    app, window, backend, bridge = ai_window
    window._update_pending = {"tag": "v9.9.9", "asset": {"name": "a.zip", "url": "http://x"}}
    window.files.results["update_download"] = json.dumps({"staging": "C:/tmp/staging"})
    window._drain_update_results()
    assert "更新を開始できませんでした" in window.update_status.text()


def test_startup_auto_check_is_silent_under_pytest(ai_window):
    app, window, backend, bridge = ai_window
    window._auto_update_check()
    assert "update_check" not in window.files._pending


def test_route_hint_describes_virtual_routing(ai_window):
    from src.audio.devices import AudioDevice
    app, window, backend, bridge = ai_window
    mic = AudioDevice(8, "USB Mic", "ALSA", 2, 0)
    loop = AudioDevice(9, "Loopback: PCM (hw:1,0)", "ALSA", 2, 2)
    window.input_device.addItem(mic.label, mic)
    window.output_device.addItem(loop.label, loop)
    window.input_device.setCurrentIndex(window.input_device.findData(mic))
    window.output_device.setCurrentIndex(window.output_device.findData(loop))
    window._route_changed()
    assert "Loopback" in window.route_hint.text()


def test_route_hint_points_linux_users_at_snd_aloop(ai_window, monkeypatch):
    import sys
    from src.audio.devices import AudioDevice
    app, window, backend, bridge = ai_window
    monkeypatch.setattr(sys, "platform", "linux")
    mic = AudioDevice(8, "USB Mic", "ALSA", 2, 0)
    speaker = AudioDevice(9, "Speakers", "ALSA", 0, 2)
    window.input_device.addItem(mic.label, mic)
    window.output_device.addItem(speaker.label, speaker)
    window.input_device.setCurrentIndex(window.input_device.findData(mic))
    window.output_device.setCurrentIndex(window.output_device.findData(speaker))
    window._route_changed()
    assert "snd-aloop" in window.route_hint.text()
