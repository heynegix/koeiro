"""Release tooling: asset builder and first-run setup script.

Only logic that runs without network, git-optional paths aside: the asset test
builds a throwaway git repository (skipped when git is missing), and the setup
script is exercised through --dry-run plus its pure helpers.
"""
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tools import build_release_asset
from tools import setup_release


def has_git():
    return shutil.which("git") is not None


def test_asset_name_convention():
    assert build_release_asset.asset_name("v1.2.3", "windows") == "koeiro-v1.2.3-windows.zip"
    assert build_release_asset.asset_name("v1.2.3", "linux") == "koeiro-v1.2.3-linux.tar.gz"
    with pytest.raises(build_release_asset.BuildError):
        build_release_asset.asset_name("v1.2.3", "darwin")


def test_write_release_json_strips_the_v():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        written = build_release_asset.write_release_json(tmp, "v1.2.3", "windows")
        assert written == {"app": "Koeiro", "version": "1.2.3", "platform": "windows",
                           "tag": "v1.2.3"}
        assert json.loads((Path(tmp) / "release.json").read_text("utf-8")) == written
        with pytest.raises(build_release_asset.BuildError):
            build_release_asset.write_release_json(tmp, "v1.2.3", "darwin")


@pytest.mark.skipif(not has_git(), reason="git が必要")
def test_build_asset_from_a_tag(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *args: subprocess.run(["git", *args], cwd=repo, capture_output=True,
                                       text=True, timeout=60, check=True)
    run("init")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "test")
    (repo / "app.py").write_text("app", "utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "version.py").write_text("x", "utf-8")
    (repo / "src" / "app_update.py").write_text("x", "utf-8")
    (repo / "tools").mkdir()
    (repo / "tools" / "setup_release.py").write_text("x", "utf-8")
    (repo / "requirements.txt").write_text("x", "utf-8")
    lock = repo / "vc_models" / "meanvc2"
    lock.mkdir(parents=True)
    (lock / "requirements.lock.txt").write_text("x", "utf-8")
    voice = repo / "models" / "user_voices" / "user_3b11387911244c6e9a013b42fa88a458"
    voice.mkdir(parents=True)
    (voice / "reference.wav").write_bytes(b"RIFF....")
    run("add", "-A")
    run("commit", "-m", "test")
    run("tag", "v9.9.9")
    out = tmp_path / "out"
    for platform in ("windows", "linux"):
        target, digest, info = build_release_asset.build_asset(
            repo, "v9.9.9", platform, out, allow_dirty=False)
        assert target.is_file() and len(digest) == 64
        assert info["version"] == "9.9.9" and info["platform"] == platform
    assert (out / "koeiro-v9.9.9-windows.zip").is_file()
    assert (out / "koeiro-v9.9.9-linux.tar.gz").is_file()
    with zipfile.ZipFile(out / "koeiro-v9.9.9-windows.zip") as bundle:
        names = bundle.namelist()
        assert "release.json" in names and "app.py" in names
    with pytest.raises(build_release_asset.BuildError):
        build_release_asset.build_asset(repo, "v0.0.0", "windows", out)


@pytest.mark.skipif(not has_git(), reason="git が必要")
def test_build_asset_refuses_dirty_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *args: subprocess.run(["git", *args], cwd=repo, capture_output=True,
                                       text=True, timeout=60, check=True)
    run("init")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "test")
    (repo / "app.py").write_text("app", "utf-8")
    run("add", "-A")
    run("commit", "-m", "test")
    run("tag", "v9.9.9")
    (repo / "app.py").write_text("dirty", "utf-8")
    with pytest.raises(build_release_asset.BuildError):
        build_release_asset.build_asset(repo, "v9.9.9", "windows", tmp_path / "out")


def test_setup_lock_filter_drops_windows_only_pins():
    text = "torch==2.5.1+cpu\nwin32-setctime==1.2.0\nnumpy==1.26.4\n"
    linux = setup_release.filter_lock_lines(text, False)
    assert "win32-setctime" not in linux and "torch==2.5.1+cpu" in linux
    windows = setup_release.filter_lock_lines(text, True)
    assert "win32-setctime" in windows


def test_setup_plan_lists_the_required_steps(tmp_path):
    steps = setup_release.plan_steps(tmp_path, with_models=True, with_lavasr=True)
    titles = [title for title, _cmd in steps]
    assert titles == ["GUI 環境 (.venv)", "GUI 依存関係", "AI 環境 (vc_models/meanvc2/.venv)",
                      "AI: torch 取得", "AI: 固定依存関係", "LavaSR 取得", "MeanVC2 リポジトリ"]
    commands = " ".join(str(part) for _title, cmd in steps for part in cmd)
    assert "requirements.txt" in commands and "install_lavasr.py" in commands
    plain = setup_release.plan_steps(tmp_path, with_models=False, with_lavasr=False)
    assert [title for title, _cmd in plain] == [title for title, _cmd in steps[:5]]


def test_setup_dry_run_executes_nothing(tmp_path, capsys):
    assert setup_release.main(["--root", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "GUI 依存関係" in out and "torch" in out
    assert not (tmp_path / ".venv").exists()


def test_setup_requires_python_312_for_real_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 0))
    assert setup_release.main(["--root", str(tmp_path), "--no-models", "--no-lavasr"]) == 1


def test_swap_install_carries_the_virtualenvs_over(tmp_path):
    from src import update_swap
    install = tmp_path / "install"
    install.mkdir()
    (install / "app.py").write_text("old", "utf-8")
    venv = install / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("interpret", "utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "app.py").write_text("new", "utf-8")
    report = update_swap.swap_install(staging, install)
    assert (install / "app.py").read_text("utf-8") == "new"
    assert (install / ".venv" / "bin" / "python").read_text("utf-8") == "interpret"
    assert Path(report["backup"]).is_dir()
