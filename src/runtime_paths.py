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
DATA_ENV = 'KOEIRO_DATA'
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


def worker_executable():
    """Packaged AI worker; it is a separate process so the bridge can own it."""
    return Path(sys.executable).resolve().parent / WORKER_EXE


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
