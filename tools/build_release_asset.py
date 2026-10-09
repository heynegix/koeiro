"""Build a GitHub Release asset from a tag: tracked tree + release.json.

Standard library only. The asset is a source tree, not a frozen binary: the
first launch runs tools/setup_release.py, which creates the per-OS virtualenvs
and fetches the models. This keeps every asset far below the 2GB hosting limit
(a torch-bundling build cannot fit) and identical for both OSes except for the
archive format and the release.json platform field.

Layout contract (see PUBLISHING.md and src/app_update.py):

* ``koeiro-<tag>-windows.zip`` / ``koeiro-<tag>-linux.tar.gz``
* package root holds ``release.json``: {"app": "Koeiro", "version": "<tag w/o v>",
  "platform": "windows" | "linux"} plus app.py, src/, models default voice, …
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


class BuildError(Exception):
    pass


def _git(root, *args):
    try:
        completed = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                   text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        raise BuildError(f"git が実行できません: {error}") from error
    if completed.returncode != 0:
        raise BuildError(f"git {' '.join(args)} に失敗: {completed.stderr.strip()}")
    return completed.stdout.strip()


def check_tree(root, tag):
    """Refuse to package anything but the exact tagged commit, clean."""
    root = Path(root)
    try:
        tagged = _git(root, "rev-parse", f"{tag}^{{commit}}")
        head = _git(root, "rev-parse", "HEAD")
    except BuildError as error:
        raise BuildError(f"タグ {tag} を解決できません: {error}") from error
    if tagged != head:
        raise BuildError(f"HEAD がタグ {tag} と一致しません（{head[:8]} != {tagged[:8]}）")
    if _git(root, "status", "--porcelain"):
        raise BuildError("作業ツリーが汚れています。コミットしてから実行してください")


def write_release_json(directory, tag, platform):
    if platform not in ("windows", "linux"):
        raise BuildError(f"platform は windows/linux のみ: {platform}")
    version = tag[1:] if tag[:1] == "v" else tag
    info = {"app": "Koeiro", "version": version, "platform": platform, "tag": tag}
    (Path(directory) / "release.json").write_text(json.dumps(info, indent=2), "utf-8")
    return info


def required_paths():
    return ["app.py", "src/version.py", "src/app_update.py", "tools/setup_release.py",
            "requirements.txt", "vc_models/meanvc2/requirements.lock.txt",
            "models/user_voices/user_3b11387911244c6e9a013b42fa88a458/reference.wav"]


def verify_stage(stage):
    missing = [name for name in required_paths() if not (Path(stage) / name).is_file()]
    if missing:
        raise BuildError(f"必須ファイルがありません: {', '.join(missing)}")
    try:
        info = json.loads((Path(stage) / "release.json").read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise BuildError(f"release.json が壊れています: {error}") from error
    if info.get("app") != "Koeiro" or not info.get("version"):
        raise BuildError("release.json が壊れています")
    return info


def asset_name(tag, platform):
    if platform == "windows":
        return f"koeiro-{tag}-windows.zip"
    if platform == "linux":
        return f"koeiro-{tag}-linux.tar.gz"
    raise BuildError(f"platform は windows/linux のみ: {platform}")


def build_asset(root, tag, platform, out_dir, allow_dirty=False):
    """Package the tag into out_dir. Returns (asset_path, sha256, release_info)."""
    root, out_dir = Path(root), Path(out_dir)
    if not allow_dirty:
        check_tree(root, tag)
    work = Path(tempfile.mkdtemp(prefix="koeiro-asset-"))
    try:
        tree_archive = work / "tree.zip"
        _git(root, "archive", "--format=zip", tag, "-o", str(tree_archive))
        stage = work / "stage"
        stage.mkdir()
        with zipfile.ZipFile(tree_archive) as bundle:
            bundle.extractall(stage)
        info = write_release_json(stage, tag, platform)
        verify_stage(stage)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = asset_name(tag, platform)
        target = out_dir / name
        if target.exists():
            target.unlink()
        if platform == "windows":
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
                for path in sorted(stage.rglob("*")):
                    if path.is_file():
                        bundle.write(path, path.relative_to(stage).as_posix())
        else:
            with tarfile.open(target, "w:gz", compresslevel=9) as bundle:
                for path in sorted(stage.rglob("*")):
                    if path.is_file():
                        bundle.add(path, path.relative_to(stage).as_posix())
        digest = hashlib.sha256()
        with open(target, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return target, digest.hexdigest(), info
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="例: v0.11.0")
    parser.add_argument("--platform", required=True, choices=("windows", "linux"))
    parser.add_argument("--out", required=True, help="アセット出力先ディレクトリ")
    parser.add_argument("--root", default=".", help="リポジトリの場所")
    parser.add_argument("--allow-dirty", action="store_true", help="タグ一致・clean検査を省略（手元確認用）")
    args = parser.parse_args(argv)
    try:
        target, digest, info = build_asset(args.root, args.tag, args.platform,
                                           args.out, allow_dirty=args.allow_dirty)
    except BuildError as error:
        print(f"error: {error}")
        return 1
    size_mb = target.stat().st_size / 1024 / 1024
    print(f"asset: {target} ({size_mb:.1f} MB)")
    print(f"sha256: {digest}")
    print(f"release: {json.dumps(info, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
