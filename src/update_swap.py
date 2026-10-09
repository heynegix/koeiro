"""Apply a staged update: swap the install dir with backup and rollback.

Standard library only and no Qt/audio/model imports, so `app.py --apply-update`
can run it in a bare helper process after the GUI has exited (on Windows the
running executable itself is locked, which is why the swap never happens
in-process). Every function here is unit-tested against temporary directories.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


class SwapError(Exception):
    """Raised when the install cannot be replaced safely."""


# User content that survives an update. Everything else comes from the package.
# `models/user_voices` merges by folder: shipped voices refresh from the package,
# user-added voices are carried over.
PRESERVE_FILES = ("settings.json",)
PRESERVE_DIRS = ("recordings",)


def wait_for_exit(pid, timeout=60):
    """Return once the process is gone; raise SwapError on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        except OSError:
            return
        time.sleep(0.2)
    raise SwapError("アプリの終了を待機できませんでした")


def _copy_tree(source, destination):
    shutil.copytree(source, destination, dirs_exist_ok=True)


def swap_install(staging, install):
    """Replace install with staging; backup + user content preserved.

    Returns a small report dict. On failure the previous install is restored
    whenever possible and SwapError is raised.
    """
    staging, install = Path(staging), Path(install)
    if not staging.is_dir() or staging == install:
        raise SwapError("更新パッケージが壊れています")
    if not install.is_dir():
        raise SwapError("インストール先が見つかりません")
    backup = install.parent / (install.name + ".bak")
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    try:
        os.rename(install, backup)
    except OSError as error:
        raise SwapError(f"バックアップに失敗しました: {error}") from error
    try:
        shutil.move(str(staging), str(install))
        # User voices: shipped folders refresh from the package, the rest carry over.
        staged_voices = install / "models" / "user_voices"
        backup_voices = backup / "models" / "user_voices"
        if backup_voices.is_dir():
            staged_voices.mkdir(parents=True, exist_ok=True)
            staged_names = {path.name for path in staged_voices.iterdir()}
            for path in backup_voices.iterdir():
                if path.name not in staged_names:
                    target = staged_voices / path.name
                    if path.is_dir():
                        _copy_tree(path, target)
                    else:
                        shutil.copy2(path, target)
        for name in PRESERVE_FILES:
            source = backup / name
            if source.is_file() and not (install / name).exists():
                shutil.copy2(source, install / name)
        for name in PRESERVE_DIRS:
            source = backup / name
            if source.is_dir() and not (install / name).exists():
                _copy_tree(source, install / name)
    except Exception as error:
        try:
            shutil.rmtree(install, ignore_errors=True)
            if not install.exists():
                os.rename(backup, install)
        except OSError:
            pass
        raise SwapError(f"更新の適用に失敗しました: {error}") from error
    return {"install": str(install), "backup": str(backup)}


def launch_new(argv):
    """Start the updated copy without any tie to this helper process."""
    try:
        if sys.platform == "win32":
            flags = 0x00000008 | 0x00000200
            subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        else:
            subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    except Exception as error:
        raise SwapError(f"再起動に失敗しました: {error}") from error


def main(plan_path):
    """Helper entry: wait, swap, launch. Returns a process exit code."""
    try:
        plan = json.loads(Path(plan_path).read_text("utf-8"))
        wait_for_exit(int(plan["parent_pid"]))
        report = swap_install(plan["staging"], plan["install"])
        launch_new(plan["launch"])
    except SwapError as error:
        print(f"update failed: {error}")
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"update failed: {error}")
        return 1
    print(f"updated: {report['install']}")
    return 0
