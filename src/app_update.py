"""Check, download and stage application updates from GitHub Releases.

Standard library only, so the frozen worker-less updater path and the unit tests
never need Qt, audio or model packages. Network access happens only when the user
asks (or consented to the startup check): nothing here phones home by itself.

Release layout this understands (see PUBLISHING.md):

* tag like ``v1.2.3`` (optionally ``-preview.N``)
* asset ``koeiro-<tag>-windows.zip`` / ``koeiro-<tag>-linux.tar.gz``
* package root holds ``release.json`` (``{"app": "Koeiro", "version": ..., ...}``)
"""
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from .version import APP_VERSION, UPDATE_REPO

API_LATEST = "https://api.github.com/repos/{repo}/releases/latest"
API_ALL = "https://api.github.com/repos/{repo}/releases?per_page=20"
USER_AGENT = "Koeiro-update-check"

PLATFORM_ASSETS = {
    "windows": ("windows", ".zip"),
    "linux": ("linux", ".tar.gz"),
}


class UpdateError(Exception):
    """Anything that stops an update: network, format or install problem."""


def parse_version(tag):
    """'v1.2.3' / '0.11.0-preview.3' -> comparable tuple.

    A final release compares higher than previews of the same base, so an
    installed preview always offers the final release as an update.
    """
    text = str(tag).strip()
    if text[:1].lower() == "v":
        text = text[1:]
    base, dash, pre = text.partition("-")
    parts = base.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise UpdateError(f"Unparsable version: {tag!r}")
    major, minor, patch = (int(part) for part in parts)
    if not dash:
        return (major, minor, patch, 1, 0)
    kind, num = pre, 0
    if pre[:8] == "preview." and pre[8:].isdigit():
        kind, num = "preview", int(pre[8:])
    return (major, minor, patch, 0, 0 if kind != "preview" else num)


def is_newer(latest_tag, current_tag=APP_VERSION):
    try:
        return parse_version(latest_tag) > parse_version(current_tag)
    except UpdateError:
        return False


def _api_get(url, timeout):
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise UpdateError(f"更新情報を取得できませんでした: {error}") from error


def fetch_latest_release(repo=UPDATE_REPO, timeout=15, allow_prerelease=False):
    """Newest published release dict, or None when the repo has no releases."""
    if allow_prerelease:
        try:
            releases = _api_get(API_ALL.format(repo=repo), timeout)
        except UpdateError:
            return None
        for item in releases if isinstance(releases, list) else []:
            if isinstance(item, dict) and not item.get("draft") and item.get("tag_name"):
                return item
        return None
    try:
        release = _api_get(API_LATEST.format(repo=repo), timeout)
    except UpdateError:
        return None
    if not isinstance(release, dict) or not release.get("tag_name"):
        return None
    return release


def pick_asset(release, platform=None):
    """Download candidate for this OS, or None when the release has none."""
    if platform is None:
        platform = "windows" if sys.platform == "win32" else (
            "linux" if sys.platform.startswith("linux") else None)
    wanted = PLATFORM_ASSETS.get(platform)
    if wanted is None or not isinstance(release, dict):
        return None
    token, extension = wanted
    tag = str(release.get("tag_name", ""))
    prefix = f"koeiro-{tag}-{token}"
    for asset in release.get("assets") or []:
        name = str(asset.get("name", ""))
        url = asset.get("browser_download_url")
        if name.startswith(prefix) and name.endswith(extension) and url:
            return {"name": name, "url": url, "size": asset.get("size") or 0}
    return None


def download_asset(url, destination, progress=None, timeout=120):
    """Stream an asset to disk. progress(done_bytes, total_bytes|0) is optional."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(destination, "wb") as stream:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    stream.write(chunk)
                    done += len(chunk)
                    if callable(progress):
                        progress(done, total)
    except Exception as error:
        try:
            destination.unlink()
        except OSError:
            pass
        raise UpdateError(f"ダウンロードに失敗しました: {error}") from error
    return destination


def extract_package(archive, destination):
    """Unpack a release asset; zip for Windows, tar.gz for Linux."""
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(destination)
        else:
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(destination)
    except Exception as error:
        raise UpdateError(f"展開に失敗しました: {error}") from error
    # A package may wrap everything in one top folder; unwrap it.
    children = [path for path in destination.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return destination


def read_package_release(package_dir):
    """release.json of an extracted package, validated."""
    try:
        info = json.loads((Path(package_dir) / "release.json").read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise UpdateError(f"更新パッケージが壊れています: {error}") from error
    if not isinstance(info, dict) or info.get("app") != "Koeiro":
        raise UpdateError("更新パッケージが壊れています")
    try:
        parse_version(info.get("version", ""))
    except UpdateError:
        raise UpdateError("更新パッケージが壊れています") from None
    if not (Path(package_dir) / "app.py").is_file():
        raise UpdateError("更新パッケージが壊れています")
    return info


def current_kind():
    """'frozen' (installed exe), 'release-dir' (marked copy) or 'source'."""
    from .runtime_paths import asset_root, is_frozen
    if is_frozen():
        return "frozen"
    if (asset_root() / "release.json").is_file():
        return "release-dir"
    return "source"


def install_dir():
    """Directory an update would replace, or None for a source checkout."""
    from .runtime_paths import asset_root, is_frozen
    if is_frozen():
        return Path(sys.executable).resolve().parent
    if (asset_root() / "release.json").is_file():
        return asset_root()
    return None


def updates_dir():
    """Writable staging area for downloads (per-user, never the install dir)."""
    from .runtime_paths import data_dir
    target = data_dir() / "updates"
    target.mkdir(parents=True, exist_ok=True)
    return target


def launch_command():
    """How this running copy was started, for the post-update relaunch."""
    if getattr(sys, "frozen", False):
        return [str(Path(sys.executable).resolve())]
    root = Path(__file__).resolve().parents[1]
    return [sys.executable, str(root / "app.py")]


def begin_update_and_restart(staging_dir, version):
    """Hand a validated staging dir to the apply helper and return its plan path.

    The helper waits for this process to exit, swaps the install, then launches
    the new copy. The caller must quit the app right after this returns.
    """
    from .runtime_paths import is_frozen
    target = install_dir()
    if target is None:
        raise UpdateError("この実行形式では自動更新できません")
    info = read_package_release(staging_dir)
    if str(info.get("version")) != str(version):
        raise UpdateError("更新パッケージが壊れています")
    plan = {
        "staging": str(Path(staging_dir).resolve()),
        "install": str(Path(target).resolve()),
        "launch": launch_command(),
        "parent_pid": os.getpid(),
    }
    plan_path = Path(staging_dir) / "update-plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), "utf-8")
    if is_frozen():
        command = [str(Path(sys.executable).resolve()), "--apply-update", str(plan_path)]
    else:
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(root / "app.py"), "--apply-update", str(plan_path)]
    try:
        if sys.platform == "win32":
            flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        else:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    except Exception as error:
        raise UpdateError(f"更新ヘルパーを起動できませんでした: {error}") from error
    return plan_path


def check_now(current=APP_VERSION, repo=UPDATE_REPO, timeout=15):
    """One update check: 'uptodate' | {'tag', 'notes', 'asset'} | {'error'}.

    Returns plain data (JSON-safe) so a background thread can hand it to the GUI.
    """
    release = fetch_latest_release(repo, timeout)
    if release is None:
        return {"status": "error", "message": "更新情報を取得できませんでした"}
    tag = str(release.get("tag_name", ""))
    try:
        newer = parse_version(tag) > parse_version(current)
    except UpdateError:
        return {"status": "error", "message": "更新情報を取得できませんでした"}
    if not newer:
        return {"status": "uptodate", "tag": tag}
    asset = pick_asset(release)
    if asset is None:
        return {"status": "error", "message": "このOS用の配布がありません"}
    notes = str(release.get("body") or "")
    return {"status": "found", "tag": tag, "notes": notes[:2000], "asset": asset}
