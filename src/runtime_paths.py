"""Read-only asset root and writable data directory, with a frozen-bundle branch.

Unfrozen, the repository root holds both. In a packaged build the assets are read
from the bundle's ``_internal`` tree while settings, logs and caches are written to
a per-user directory so the bundle itself can live under Program Files.
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)

APP_NAME = 'Koeiro'
WORKER_EXE = 'KoeiroWorker.exe'
# Linux builds the same frozen worker without an extension, so both names are
# recognised wherever a build looks for one beside the GUI.
WORKER_NAMES = (WORKER_EXE, 'KoeiroWorker')
DATA_ENV = 'KOEIRO_DATA'
# A packaged GUI can be launched from anywhere -- a shortcut, Explorer, another
# folder -- so the install folder can be named outright instead of inferred. Users
# who move the executable keep a working AI by pointing this at their Koeiro folder.
ROOT_ENV = 'KOEIRO_ROOT'
# A packaged build records the folder it was built from, so the executable finds the
# environment tools/setup_release.py installed even though it sits in dist/Koeiro/
# with no vc_models beside it. Written by tools/write_build_root.py and bundled.
ROOT_FILE = 'koeiro-root.txt'
# A packaged build that ran under the pre-rename name keeps its settings.
LEGACY_APP_NAME = 'AnimeVoiceChanger'
LEGACY_DATA_ENV = 'ANIME_VOICE_CHANGER_DATA'
MARKER = 'bundle.json'
# Same ceiling the settings loader uses, so a hand-edited file is never adopted.
SETTINGS_SIZE_LIMIT = 64 * 1024


def is_frozen():
    return bool(getattr(sys, 'frozen', False))


def asset_root():
    """Directory holding models/, vc_models/ and the bundled package tree."""
    if is_frozen():
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def data_dir():
    """Writable directory for settings.json, logs/ and inference caches."""
    override = os.environ.get(DATA_ENV) or os.environ.get(LEGACY_DATA_ENV)
    if override:
        return Path(override)
    if is_frozen():
        if sys.platform == "win32":
            base = os.environ.get('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
        else:
            base = os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local' / 'share')
        target = Path(base) / APP_NAME
        _adopt_previous_settings(Path(base) / LEGACY_APP_NAME, target)
        return target
    return Path(__file__).resolve().parents[1]


def _adopt_previous_settings(previous, target):
    """Carry settings.json over from the pre-rename data directory, once.

    Bounded startup work: a single small JSON file, copied only while the new
    directory has none, and only while it fits the loader's own size ceiling.
    The old directory is left untouched, so nothing a user had is destroyed.
    """
    source = previous / 'settings.json'
    destination = target / 'settings.json'
    if destination.exists() or not source.is_file():
        return
    try:
        if source.stat().st_size > SETTINGS_SIZE_LIMIT:
            return
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        log.info('Adopted settings from %s', previous)
    except OSError:
        log.warning('Could not adopt settings from %s', previous, exc_info=True)


def cache_dir():
    """Writable cache root for torch/huggingface/number caches."""
    return data_dir() / 'cache'


def build_root():
    """The folder this build recorded as its source, or None when it recorded none."""
    try:
        recorded = (asset_root() / ROOT_FILE).read_text('utf-8').strip()
    except OSError:
        return None
    return Path(recorded) if recorded else None


def install_root():
    """Directory the packaged build takes its AI environment from.

    A packaged build can be started from any working directory, so this is never the
    process's cwd. The order is: an explicit ``KOEIRO_ROOT``, then the folder the
    build recorded (the executable sits in ``dist/Koeiro/``, where no ``vc_models``
    exists, while the installed environment is beside the source tree), then the
    executable's own folder -- the parent of a one-dir bundle's ``_internal``, since
    the running interpreter lives *inside* that tree.
    """
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override)
    if is_frozen():
        recorded = build_root()
        if recorded is not None:
            return recorded
        bundle = asset_root()
        if bundle.is_dir() and bundle.name == '_internal':
            return bundle.parent
    return Path(sys.executable).resolve().parent


def worker_executable():
    """The frozen worker beside the GUI, or None when this build ships none."""
    root = install_root()
    for name in WORKER_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def is_bundled_worker(executable):
    """Whether ``executable`` is the frozen worker rather than a Python interpreter.

    A frozen worker is launched with the service arguments alone -- there is no
    interpreter to hand ``-u -m src.vc.service`` to.
    """
    return Path(executable).name in WORKER_NAMES


def worker_environment(python=sys.executable):
    """The installed AI worker environment to launch, or None when there is none.

    Two layouts reach a frozen GUI. A full freeze could ship ``KoeiroWorker.exe``
    beside the GUI, which needs no Python at all; a bundle that only freezes the GUI
    runs the worker from the environment ``tools/setup_release.py`` installed for it.
    Neither is assumed: the frozen executable's own folder is checked first (where
    setup_release puts ``vc_models/meanvc2/.venv``), then the unpacked bundle root.

    ``python`` exists so a test can drive the platform branches without a real
    build, and defaults to the running interpreter everywhere else.
    """
    roots = [install_root(), asset_root()] if is_frozen() else [asset_root()]
    candidates = []
    for root in roots:
        candidates.extend((root / 'vc_models' / 'meanvc2' / '.venv', root / '.venv-ai'))
    candidates.append(Path(python).parent.parent / '.venv-ai')
    for candidate in candidates:
        launcher = candidate / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if launcher.is_file():
            return candidate
    return None


def _marker():
    try:
        return json.loads((asset_root() / 'models' / MARKER).read_text('utf-8'))
    except (OSError, ValueError):
        return {}


def bundle_profiles():
    """Allowed voice profile keys in this bundle, or None when unrestricted."""
    profiles = _marker().get('profiles')
    if isinstance(profiles, list) and profiles:
        return frozenset(str(key) for key in profiles)
    return None


def bundle_deliveries():
    """Allowed delivery-mode combo keys in this bundle, or None when unrestricted."""
    deliveries = _marker().get('deliveries')
    if isinstance(deliveries, list) and deliveries:
        return frozenset(str(key) for key in deliveries)
    return None
