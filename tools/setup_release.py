"""First-run setup for release copies (and fresh source checkouts).

Creates the GUI and AI-worker virtualenvs, installs the pinned dependencies,
fetches LavaSR and the model checkpoints, then verifies the interpreters.
Standard library only, so it runs on a bare Python 3.12.

Usage (run from the repository/package root):

    py -3.12 tools/setup_release.py                  # Windows, everything
    python3.12 tools/setup_release.py --dry-run      # print the plan only
    python3.12 tools/setup_release.py --no-models    # envs + LavaSR, models later
    python3.12 tools/setup_release.py --no-lavasr --recreate
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
MEANVC2_REPO = "https://github.com/ASLP-lab/MeanVC2"
MEANVC2_HF = "https://huggingface.co/ASLP-lab/MeanVC2/resolve/main"
KNN_WAVLM_URL = "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt"
ECAPA_DRIVE_ID = "1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"

# (package-relative destination, download URL). None means "clone the repo first".
MODEL_FILES = [
    ("vc_models/meanvc2/repo/preprocess/ckpts/fastu2pp_80ms.pt",
     f"{MEANVC2_HF}/preprocess/ckpts/fastu2pp_80ms.pt"),
    ("vc_models/meanvc2/repo/preprocess/ckpts/fastu2pp_160ms.pt",
     f"{MEANVC2_HF}/preprocess/ckpts/fastu2pp_160ms.pt"),
    ("vc_models/meanvc2/repo/ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors",
     f"{MEANVC2_HF}/ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors"),
    ("vc_models/meanvc2/repo/ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors",
     f"{MEANVC2_HF}/ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors"),
    ("vc_models/meanvc2/repo/ckpts/vocos/vocos.pt",
     f"{MEANVC2_HF}/ckpts/vocos/vocos.pt"),
    ("vc_models/meanvc2/repo/preprocess/ckpts/wavlm_large.pt", KNN_WAVLM_URL),
]

VERIFY_GUI = ["PySide6", "sounddevice", "numpy", "psutil"]
VERIFY_WORKER = ["torch", "torchaudio", "scipy", "soundfile", "s3prl", "librosa"]


class SetupError(Exception):
    pass


def is_windows():
    return os.name == "nt"


def venv_python(root, name):
    root = Path(root)
    if is_windows():
        return root / name / "Scripts" / "python.exe"
    return root / name / "bin" / "python"


def gui_python(root):
    return venv_python(root, ".venv")


def worker_python(root):
    return venv_python(root, "vc_models/meanvc2/.venv")


def filter_lock_lines(text, windows):
    """Drop Windows-only pins on other platforms (pip would fail on them)."""
    kept = []
    for line in text.splitlines():
        name = line.split("=")[0].strip().lower()
        if not windows and (name.startswith("win32-") or name.startswith("pywin32")):
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def run(cmd, cwd):
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    try:
        completed = subprocess.run([str(part) for part in cmd], cwd=str(cwd))
    except OSError as error:
        raise SetupError(f"実行できません: {cmd[0]} ({error})") from error
    if completed.returncode != 0:
        raise SetupError(f"失敗 (exit={completed.returncode}): {' '.join(str(p) for p in cmd)}")


def download(url, destination, minimum_bytes=1024 * 1024):
    """Fetch a URL with the standard library; skips when already present."""
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size >= minimum_bytes:
        print(f"  あり: {destination} ({destination.stat().st_size // 1024 // 1024} MB)")
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  取得: {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Koeiro-setup"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            content_type = response.headers.get("Content-Type", "")
            with open(destination, "wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 256)
    except Exception as error:
        raise SetupError(f"ダウンロードに失敗: {url} ({error})") from error
    if destination.stat().st_size < minimum_bytes or "text/html" in content_type:
        destination.unlink(missing_ok=True)
        return False
    print(f"  完了: {destination} ({destination.stat().st_size // 1024 // 1024} MB)")
    return True


def plan_steps(root, with_models=True, with_lavasr=True, recreate=False):
    """Ordered (title, argv) executable plan. Informational sections print separately."""
    root = Path(root)
    gui, worker = gui_python(root), worker_python(root)
    steps = []
    if recreate or not gui.is_file():
        steps.append(("GUI 環境 (.venv)", [sys.executable, "-m", "venv", ".venv"]))
    steps.append(("GUI 依存関係", [str(gui), "-m", "pip", "install", "-r", "requirements.txt"]))
    if recreate or not worker.is_file():
        steps.append(("AI 環境 (vc_models/meanvc2/.venv)",
                      [sys.executable, "-m", "venv", "vc_models/meanvc2/.venv"]))
    steps.append(("AI: torch 取得", [str(worker), "-m", "pip", "install", "torch", "torchaudio",
                                     "--index-url", TORCH_INDEX]))
    steps.append(("AI: 固定依存関係", [str(worker), "-m", "pip", "install", "-r", "<filtered-lock>"]))
    if with_lavasr:
        steps.append(("LavaSR 取得", [str(gui), "tools/install_lavasr.py"]))
    if with_models and not (Path(root) / "vc_models" / "meanvc2" / "repo").is_dir():
        steps.append(("MeanVC2 リポジトリ", ["git", "clone", MEANVC2_REPO, "vc_models/meanvc2/repo"]))
    return steps


def install_filtered_lock(worker, root, dry_run):
    lock = Path(root) / "vc_models" / "meanvc2" / "requirements.lock.txt"
    text = lock.read_text(encoding="utf-8")
    filtered = filter_lock_lines(text, is_windows())
    if dry_run:
        dropped = len(text.splitlines()) - len(filtered.splitlines())
        print(f"  lock: {dropped} 行を除外（Windows 専用）" if dropped else "  lock: 除外なし")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as stream:
        stream.write(filtered)
        temporary = stream.name
    try:
        run([str(worker), "-m", "pip", "install", "-r", temporary], root)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fetch_models(root, dry_run):
    repo = Path(root) / "vc_models" / "meanvc2" / "repo"
    if not repo.is_dir():
        if shutil.which("git") is None:
            raise SetupError("git がありません。git を入れるか、README の手順で repo を手で置いてください")
        if not dry_run:
            run(["git", "clone", MEANVC2_REPO, "vc_models/meanvc2/repo"], root)
        else:
            print(f"  予定: git clone {MEANVC2_REPO} vc_models/meanvc2/repo")
    missing = []
    for relative, url in MODEL_FILES:
        if dry_run:
            print(f"  予定: {relative}")
            continue
        try:
            if not download(url, Path(root) / relative):
                missing.append(relative)
        except SetupError as error:
            print(f"  {error}")
            missing.append(relative)
    if not dry_run:
        # ECAPA finetune lives on Google Drive; the direct link needs a confirm
        # token for large files, so a failed fetch falls back to manual steps.
        ecapa = Path(root) / "vc_models" / "meanvc2" / "repo" / "preprocess" / "ckpts" / "wavlm_large_finetune.pth"
        if not (ecapa.is_file() and ecapa.stat().st_size >= 1024 * 1024):
            print("  ECAPA finetune を取得します（失敗時は手動手順を表示）")
            try:
                direct = f"https://drive.google.com/uc?export=download&id={ECAPA_DRIVE_ID}"
                if not download(direct, ecapa):
                    raise SetupError("confirm が必要です")
            except SetupError:
                print("  gdown で取得してください: "
                      f"gdown {ECAPA_DRIVE_ID} -O {ecapa.as_posix()}")
                missing.append(ecapa.relative_to(root).as_posix())
    if missing and not dry_run:
        raise SetupError("未取得のモデルがあります:\n  - " + "\n  - ".join(missing) +
                         "\nREADME の「モデル配置」に従って手で置いてください")


def verify_interpreters(root):
    gui = [str(gui_python(root)), "-c",
           "import " + ", ".join(VERIFY_GUI) + "; print('gui ok')"]
    run(gui, root)
    worker = [str(worker_python(root)), "-c",
              "import " + ", ".join(VERIFY_WORKER) + "; print('worker ok')"]
    run(worker, root)


def check_host_tools(with_lavasr):
    if with_lavasr and (shutil.which("uv") is None or shutil.which("git") is None):
        raise SetupError("LavaSR 取得に uv と git が必要です（--no-lavasr で後回し可）")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="リポジトリ/パッケージの場所")
    parser.add_argument("--dry-run", action="store_true", help="手順だけ表示")
    parser.add_argument("--no-models", action="store_true", help="モデル取得を省略")
    parser.add_argument("--no-lavasr", action="store_true", help="LavaSR 取得を省略")
    parser.add_argument("--recreate", action="store_true", help="venv を作り直す")
    args = parser.parse_args(argv)
    root = Path(args.root)
    dry_run = args.dry_run
    try:
        if not dry_run and sys.version_info[:2] != (3, 12):
            raise SetupError(f"Python 3.12 が必要です（現在 {sys.version.split()[0]}）")
        if dry_run:
            print(f"想定 Python: 3.12（現在 {sys.version.split()[0]}）")
        check_host_tools(with_lavasr=not args.no_lavasr and not dry_run)
        gui, worker = gui_python(root), worker_python(root)
        for title, cmd in plan_steps(root, with_models=not args.no_models,
                                     with_lavasr=not args.no_lavasr, recreate=args.recreate):
            print(f"== {title}")
            if dry_run:
                print("  $ " + " ".join(str(part) for part in cmd))
                continue
            if "<filtered-lock>" in cmd:
                install_filtered_lock(worker, root, dry_run=False)
            elif cmd[0] == "git":
                if not (Path(root) / "vc_models" / "meanvc2" / "repo").is_dir():
                    run(cmd, root)
                else:
                    print("  あり: vc_models/meanvc2/repo")
            else:
                run(cmd, root)
        if not args.no_lavasr:
            print("== LavaSR 取得")
            if not dry_run:
                run([str(gui), "tools/install_lavasr.py"], root)
        if not args.no_models:
            print("== MeanVC2 モデル取得")
            fetch_models(root, dry_run)
        if not dry_run:
            print("== 動作確認")
            verify_interpreters(root)
        print("完了。起動: " + (str(gui) + " app.py"))
    except SetupError as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
