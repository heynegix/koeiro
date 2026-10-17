"""Update check, download staging and install swap.

Network is never touched: the GitHub API is stubbed. The install swap runs against
temporary directories, and the frozen path is checked up to the point of launching
the setup program — that Inno Setup really replaces a running ``Koeiro.exe`` is a
manual check (PUBLISHING.md), not something a test can prove here.
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
    # '²'.isdigit() is True while int('²') raises ValueError, and the callers here
    # only turn UpdateError into something the user can read: a background update
    # check that hit it simply stopped answering. isdecimal is the honest predicate.
    for bad in ("", "v1.2", "1.2.3.4", "latest", None,
                "v².0.0", "v1.².0", "v1.2.³"):
        with pytest.raises(UpdateError):
            app_update.parse_version(bad)
    # A digit the parser can read is still a number even when it is not ASCII.
    assert app_update.parse_version("v٢.0.0") == (2, 0, 0, 1, 0)
    # A suffix that is not "preview.<number>" degrades to the base's first pre-release.
    assert app_update.parse_version("v1.2.3-preview.²") == (1, 2, 3, 0, 0)


def test_is_newer_ignores_unparsable_tags():
    assert app_update.is_newer("v9.9.9", "v0.11.0-preview.3") is True
    assert app_update.is_newer("v0.11.0-preview.3", "v0.11.0-preview.3") is False
    assert app_update.is_newer("v0.11.0", "v0.11.0-preview.3") is True
    for garbage in ("garbage", "v1.2.³"):
        assert app_update.is_newer(garbage, "v0.11.0-preview.3") is False
        assert app_update._version_matches(garbage, "v0.11.0") is False


def test_pick_asset_ignores_asset_entries_that_are_not_objects(monkeypatch):
    """A non-dict asset raised AttributeError inside the background update check,
    where nothing shows it, so the check just stopped answering. Upstream junk has
    to read as "no distribution for this OS", the same as an empty asset list.
    """
    for assets in ([None, 5, "x", [], {"name": "other"}], [1, 2],
                   ["koeiro-v9.9.9-setup.exe"]):
        release = {"tag_name": "v9.9.9", "assets": assets}
        for kind in ("frozen", "package"):
            assert app_update.pick_asset(release, platform="windows", kind=kind) is None
    for broken in (5, "text", {"name": "koeiro-v9.9.9-setup.exe"}, None):
        assert app_update.pick_asset({"tag_name": "v9.9.9", "assets": broken},
                                     platform="windows", kind="frozen") is None
    with monkeypatch.context() as patch:
        patch.setattr(app_update, "_api_get",
                      lambda *args, **kwargs: {"tag_name": "v9.9.9", "assets": [None, 5]})
        assert app_update.check_now(timeout=1)["status"] == "error"


def test_pick_asset_selects_this_os_package():
    release = {"tag_name": "v1.0.0", "assets": [
        {"name": "koeiro-v1.0.0-windows.zip", "browser_download_url": "http://w", "size": 10},
        {"name": "koeiro-v1.0.0-linux.tar.gz", "browser_download_url": "http://l", "size": 20},
        {"name": "notes.txt", "browser_download_url": "http://n", "size": 1},
    ]}
    windows = app_update.pick_asset(release, platform="windows")
    assert windows == {"name": "koeiro-v1.0.0-windows.zip", "url": "http://w", "size": 10,
                       "kind": "package"}
    linux = app_update.pick_asset(release, platform="linux")
    assert linux["url"] == "http://l"
    assert app_update.pick_asset(release, platform="darwin") is None
    assert app_update.pick_asset({"tag_name": "v1.0.0", "assets": []}, platform="windows") is None


def _installer_release():
    return {"tag_name": "v1.0.0", "assets": [
        {"name": "koeiro-v1.0.0-windows.zip", "browser_download_url": "http://w", "size": 10},
        {"name": "koeiro-v1.0.0-windows-setup.exe", "browser_download_url": "http://s", "size": 99},
        {"name": "koeiro-v1.0.0-windows-checksums.txt", "browser_download_url": "http://c", "size": 1},
        {"name": "koeiro-v1.0.0-linux.tar.gz", "browser_download_url": "http://l", "size": 20},
    ]}


def test_frozen_install_picks_the_installer_never_the_source_tree():
    """The zip holds app.py and src/, no Koeiro.exe: swapping it into an install kills it."""
    asset = app_update.pick_asset(_installer_release(), kind="frozen")
    assert asset == {"name": "koeiro-v1.0.0-windows-setup.exe", "url": "http://s", "size": 99,
                     "kind": "setup", "checksums_url": "http://c",
                     "checksums_name": "koeiro-v1.0.0-windows-checksums.txt"}
    assert app_update.pick_asset(_installer_release(), kind="package")["kind"] == "package"
    source_only = {"tag_name": "v1.0.0", "assets": [
        {"name": "koeiro-v1.0.0-windows.zip", "browser_download_url": "http://w", "size": 10}]}
    assert app_update.pick_asset(source_only, kind="frozen") is None
    assert app_update.pick_asset(_installer_release(), platform="linux", kind="frozen") is None


def test_check_now_says_so_when_a_release_has_no_executable_asset(monkeypatch):
    monkeypatch.setattr(app_update, "current_kind", lambda: "frozen")
    monkeypatch.setattr(app_update, "fetch_latest_release",
                        lambda *args, **kwargs: {"tag_name": "v9.9.9", "assets": [
                            {"name": "koeiro-v9.9.9-windows.zip",
                             "browser_download_url": "http://w", "size": 5}]})
    result = app_update.check_now(current="v0.13.1")
    assert result["status"] == "error" and "実行ファイル用の配布" in result["message"]


def test_parse_checksums_reads_the_sha256sum_format():
    text = "\n".join([
        "a" * 64 + "  koeiro-v1.0.0-windows-setup.exe",
        "b" * 64 + " *koeiro-v1.0.0-windows.zip",
        "garbage line",
        "",
    ]) + "\n"
    table = app_update.parse_checksums(text)
    assert table["koeiro-v1.0.0-windows-setup.exe"] == "a" * 64
    assert table["koeiro-v1.0.0-windows.zip"] == "b" * 64
    assert len(table) == 2


def test_verify_downloaded_refuses_mismatch_and_missing(tmp_path):
    payload = b"setup-bytes"
    target = tmp_path / "koeiro-v1.0.0-windows-setup.exe"
    target.write_bytes(payload)
    digest = app_update.sha256_of(target)
    table = {target.name: digest, "other.exe": "0" * 64}
    assert app_update.verify_downloaded(target, target.name, table) == digest
    with pytest.raises(UpdateError):
        app_update.verify_downloaded(target, target.name, {target.name: "0" * 64})
    with pytest.raises(UpdateError):
        app_update.verify_downloaded(target, "missing.exe", table)


def test_prepare_installer_downloads_then_verifies(monkeypatch, tmp_path):
    import hashlib
    monkeypatch.setattr(app_update, "updates_dir", lambda: tmp_path)
    monkeypatch.setattr(app_update, "log_update", lambda message: None)
    payload = b"setup-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    asset = app_update.pick_asset(_installer_release(), kind="frozen")

    def serve(url, destination, progress=None, timeout=120):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name.endswith("checksums.txt"):
            destination.write_text(f"{digest}  koeiro-v1.0.0-windows-setup.exe\n", "utf-8")
        else:
            destination.write_bytes(payload)
        return destination

    monkeypatch.setattr(app_update, "download_asset", serve)
    assert app_update.prepare_installer(asset, "v1.0.0").read_bytes() == payload

    def corrupted(url, destination, progress=None, timeout=120):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name.endswith("checksums.txt"):
            destination.write_text(f"{'0' * 64}  koeiro-v1.0.0-windows-setup.exe\n", "utf-8")
        else:
            destination.write_bytes(payload)
        return destination

    monkeypatch.setattr(app_update, "download_asset", corrupted)
    with pytest.raises(UpdateError):
        app_update.prepare_installer(asset, "v1.0.0")
    with pytest.raises(UpdateError):
        app_update.prepare_installer(dict(asset, checksums_url=""), "v1.0.0")
    with pytest.raises(UpdateError):
        app_update.prepare_installer({"url": "http://w", "kind": "package"}, "v1.0.0")


def test_installer_command_is_silent_and_unattended():
    command = app_update.installer_command(Path("C:/tmp/koeiro-v1.0.0-windows-setup.exe"))
    assert command[0].lower().endswith(".exe")
    for flag in ("/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"):
        assert flag in command


def test_run_installer_refuses_a_source_checkout(monkeypatch, tmp_path):
    installer = tmp_path / "koeiro-v1.0.0-windows-setup.exe"
    installer.write_bytes(b"x")
    with pytest.raises(UpdateError):
        app_update.run_installer_and_restart(installer)
    with pytest.raises(UpdateError):
        app_update.run_installer_and_restart(tmp_path / "absent.exe")


def test_install_dir_is_none_for_a_frozen_windows_build(monkeypatch):
    from src import runtime_paths
    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "platform", "win32")
    assert app_update.install_dir() is None


def _linux_release():
    return {"tag_name": "v1.0.0", "assets": [
        {"name": "koeiro-v1.0.0-linux.tar.gz", "browser_download_url": "http://l", "size": 20},
        {"name": "koeiro-v1.0.0-linux-app.tar.gz", "browser_download_url": "http://b", "size": 88},
        {"name": "koeiro-v1.0.0-linux-checksums.txt", "browser_download_url": "http://c", "size": 1},
    ]}


def test_frozen_linux_picks_the_bundle_not_the_source_tree():
    asset = app_update.pick_asset(_linux_release(), platform="linux", kind="frozen")
    assert asset["kind"] == "bundle"
    assert asset["name"] == "koeiro-v1.0.0-linux-app.tar.gz"
    assert asset["checksums_name"] == "koeiro-v1.0.0-linux-checksums.txt"
    # A release-directory install still takes the source tree, and the exact-name match
    # keeps the two assets from shadowing each other.
    assert app_update.pick_asset(_linux_release(), platform="linux")["name"] == \
        "koeiro-v1.0.0-linux.tar.gz"
    assert app_update.pick_asset(_linux_release(), platform="windows") is None


def test_install_dir_points_at_the_bundle_folder_on_linux(monkeypatch, tmp_path):
    from src import runtime_paths
    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Koeiro" / "Koeiro"))
    assert app_update.install_dir() == tmp_path / "Koeiro"


def _bundle_tar(directory, name="bundle.tar.gz"):
    """A staged frozen Linux bundle: a launcher plus its _internal tree."""
    directory = Path(directory)
    root = directory / "stage-src"
    (root / "_internal").mkdir(parents=True)
    (root / "Koeiro").write_text("#!/bin/sh\n", "utf-8")
    (root / "_internal" / "base_library.zip").write_bytes(b"zip")
    archive = directory / name
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(str(root), arcname="Koeiro")
    return archive


def test_check_bundle_needs_the_launcher_and_the_internal_tree(tmp_path):
    good = tmp_path / "Koeiro"
    (good / "_internal").mkdir(parents=True)
    (good / "Koeiro").write_text("x", "utf-8")
    assert app_update.check_bundle(good) == good
    (good / "Koeiro").unlink()
    with pytest.raises(UpdateError):
        app_update.check_bundle(good)


def test_prepare_bundle_downloads_verifies_and_unpacks(monkeypatch, tmp_path):
    import hashlib
    monkeypatch.setattr(app_update, "updates_dir", lambda: tmp_path)
    monkeypatch.setattr(app_update, "log_update", lambda message: None)
    archive = _bundle_tar(tmp_path)
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    def serve(url, destination, progress=None, timeout=120):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name.endswith("checksums.txt"):
            destination.write_text(f"{digest}  koeiro-v1.0.0-linux-app.tar.gz\n", "utf-8")
        else:
            destination.write_bytes(payload)
        return destination

    monkeypatch.setattr(app_update, "download_asset", serve)
    asset = app_update.pick_asset(_linux_release(), platform="linux", kind="frozen")
    staging = app_update.prepare_bundle(asset, "v1.0.0")
    assert (staging / "Koeiro").is_file()
    assert (staging / "_internal").is_dir()
    with pytest.raises(UpdateError):
        app_update.prepare_bundle({"kind": "setup", "url": "http://s"}, "v1.0.0")


def test_frozen_linux_plan_carries_the_pack_preservation(monkeypatch, tmp_path, capfd):
    """An installed AI pack must survive the swap: preserve_dirs goes into the plan."""
    from src import runtime_paths
    install = tmp_path / "Koeiro"
    (install / "_internal").mkdir(parents=True)
    (install / "Koeiro").write_text("x", "utf-8")
    staging = tmp_path / "staging" / "Koeiro"
    (staging / "_internal").mkdir(parents=True)
    (staging / "Koeiro").write_text("new", "utf-8")
    launched = []
    monkeypatch.setattr(runtime_paths, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "executable", str(install / "Koeiro"))
    monkeypatch.setattr(app_update.subprocess, "Popen",
                        lambda command, **kwargs: launched.append(command))
    plan_path = app_update.begin_update_and_restart(staging, "v1.0.0")
    plan = json.loads(Path(plan_path).read_text("utf-8"))
    assert plan["install"] == str(install.resolve())
    assert "vc_models" in plan["preserve_dirs"]
    assert launched[0][0] == str(install.resolve() / "Koeiro")
    assert launched[0][1:] == ["--apply-update", str(plan_path)]


@pytest.mark.parametrize("replacement_worker", [False, True])
def test_swap_install_preserves_the_ai_pack_tree(monkeypatch, tmp_path, replacement_worker):
    from src import runtime_paths

    install = _install_tree(tmp_path / "install")
    pack = install / "vc_models" / "meanvc2" / "repo"
    pack.mkdir(parents=True)
    (pack / "ckpts.pt").write_bytes(b"weights")
    worker = install / "KoeiroWorker"
    worker.write_bytes(b"old-worker")
    worker.chmod(0o755)
    dependency = install / "_internal_worker" / "libpython.so"
    dependency.parent.mkdir()
    dependency.write_bytes(b"old-runtime")
    staging = _package_tree(tmp_path / "staging")
    if replacement_worker:
        (staging / "KoeiroWorker").write_bytes(b"new-worker")
        (staging / "KoeiroWorker").chmod(0o755)
        (staging / "_internal_worker").mkdir()
        (staging / "_internal_worker" / "libpython.so").write_bytes(b"new-runtime")
    monkeypatch.setenv(runtime_paths.ROOT_ENV, str(install))
    report = update_swap.swap_install(staging, install,
                                      preserve_dirs=update_swap.BUNDLE_PRESERVE_DIRS)
    assert runtime_paths.worker_executable() == worker
    assert worker.read_bytes() == (b"new-worker" if replacement_worker else b"old-worker")
    if os.name != "nt":
        assert worker.stat().st_mode & 0o777 == 0o755
        assert os.access(worker, os.X_OK)
    assert dependency.read_bytes() == (b"new-runtime" if replacement_worker else b"old-runtime")
    assert (install / "vc_models" / "meanvc2" / "repo" / "ckpts.pt").read_bytes() == b"weights"
    assert (install / "app.py").read_text("utf-8") == "new"
    assert Path(report["install"]) == install


def test_check_now_reports_found_uptodate_and_errors(monkeypatch):
    monkeypatch.setattr(app_update, "fetch_latest_release",
                        lambda *args, **kwargs: {"tag_name": "v9.9.9", "body": "notes",
                                          "assets": [{"name": "koeiro-v9.9.9-windows.zip",
                                                      "browser_download_url": "http://w",
                                                      "size": 5}]})
    monkeypatch.setattr(app_update, "pick_asset",
                        lambda release, platform=None, kind=None: {
                            "name": "x", "url": "http://w", "size": 5, "kind": "package"})
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
    monkeypatch.setattr(app_update, "pick_asset",
                        lambda release, platform=None, kind=None: None)
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
    # A release.json nested past the parser's depth limit is broken in exactly the
    # same way; RecursionError is not a ValueError, so it must be named explicitly.
    deep = tmp_path / "deep"
    deep.mkdir()
    (deep / "release.json").write_text("[" * 30000 + "]" * 30000, "utf-8")
    with pytest.raises(UpdateError):
        app_update.read_package_release(deep)
    with pytest.raises(UpdateError):
        app_update.extract_package(tmp_path / "missing.zip", tmp_path / "out")


def test_version_match_ignores_the_leading_v():
    """release.json carries 0.11.0 while the tag reads v0.11.0: same release."""
    assert app_update._version_matches("0.11.0", "v0.11.0")
    assert app_update._version_matches("v0.11.0", "v0.11.0")
    assert not app_update._version_matches("0.11.0", "v0.12.0")
    assert not app_update._version_matches("nope", "v0.11.0")


def test_extract_rejects_members_escaping_the_destination(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("koeiro/release.json", json.dumps({"app": "Koeiro", "version": "v9.9.9"}))
        bundle.writestr("koeiro/app.py", "new")
        bundle.writestr("../escape.txt", "outside")
    with pytest.raises(UpdateError):
        app_update.extract_package(archive, tmp_path / "staging")
    assert not (tmp_path / "escape.txt").exists()


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


def test_update_installer_result_launches_or_reports(ai_window):
    """The setup branch must not fall through to the swap branch."""
    import json
    app, window, backend, bridge = ai_window
    window._update_pending = {"tag": "v9.9.9",
                              "asset": {"name": "k.exe", "url": "http://x", "kind": "setup"}}
    window.files.results["update_download"] = json.dumps({"installer": "C:/tmp/k.exe"})
    window._drain_update_results()
    status = window.update_status.text()
    assert "更新を開始できませんでした" in status
    assert "実行ファイル版以外では使えません" in status


def test_fetch_update_package_routes_a_setup_asset(monkeypatch, tmp_path):
    from src.gui.main_window import MainWindow
    monkeypatch.setattr(app_update, "prepare_installer",
                        lambda asset, tag, progress=None: tmp_path / "k.exe")
    result = MainWindow._fetch_update_package(
        {"name": "k.exe", "url": "http://s", "kind": "setup"}, "v1.0.0")
    assert result["installer"].endswith("k.exe")
    monkeypatch.setattr(app_update, "prepare_installer",
                        lambda asset, tag, progress=None: (_ for _ in ()).throw(
                            UpdateError("検証に失敗")))
    result = MainWindow._fetch_update_package(
        {"name": "k.exe", "url": "http://s", "kind": "setup"}, "v1.0.0")
    assert "検証に失敗" in result["error"]


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
