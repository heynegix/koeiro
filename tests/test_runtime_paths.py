"""The renamed app keeps one identity, and no existing settings are stranded."""
import json
import sys
from pathlib import Path

from src import runtime_paths


def test_application_identity_is_koeiro():
    assert runtime_paths.APP_NAME == 'Koeiro'
    assert runtime_paths.WORKER_EXE == 'KoeiroWorker.exe'
    assert runtime_paths.DATA_ENV == 'KOEIRO_DATA'
    assert runtime_paths.LEGACY_APP_NAME == 'AnimeVoiceChanger'
    assert runtime_paths.LEGACY_DATA_ENV == 'ANIME_VOICE_CHANGER_DATA'


def test_data_dir_honours_the_new_and_the_previous_override(monkeypatch, tmp_path):
    monkeypatch.setenv(runtime_paths.DATA_ENV, str(tmp_path / 'new'))
    assert runtime_paths.data_dir() == tmp_path / 'new'
    monkeypatch.delenv(runtime_paths.DATA_ENV)
    monkeypatch.setenv(runtime_paths.LEGACY_DATA_ENV, str(tmp_path / 'old'))
    assert runtime_paths.data_dir() == tmp_path / 'old'


def test_settings_are_adopted_once_from_the_previous_directory(tmp_path):
    previous, target = tmp_path / 'AnimeVoiceChanger', tmp_path / 'Koeiro'
    previous.mkdir()
    (previous / 'settings.json').write_text('{"monitor": true}', 'utf-8')
    runtime_paths._adopt_previous_settings(previous, target)
    assert json.loads((target / 'settings.json').read_text('utf-8')) == {'monitor': True}
    # A reader who then changes a setting keeps it: adoption never overwrites.
    (target / 'settings.json').write_text('{"monitor": false}', 'utf-8')
    runtime_paths._adopt_previous_settings(previous, target)
    assert json.loads((target / 'settings.json').read_text('utf-8')) == {'monitor': False}


def test_adoption_is_skipped_without_a_previous_file_or_when_it_is_too_large(tmp_path):
    previous, target = tmp_path / 'AnimeVoiceChanger', tmp_path / 'Koeiro'
    runtime_paths._adopt_previous_settings(previous, target)
    assert not target.exists()
    previous.mkdir()
    (previous / 'settings.json').write_bytes(b'x' * (runtime_paths.SETTINGS_SIZE_LIMIT + 1))
    runtime_paths._adopt_previous_settings(previous, target)
    assert not (target / 'settings.json').exists()


# ------------------------------------------------------- packaged worker lookup


def test_the_frozen_worker_name_covers_windows_and_linux():
    assert runtime_paths.WORKER_NAMES == ('KoeiroWorker.exe', 'KoeiroWorker')
    assert runtime_paths.is_bundled_worker(Path('C:/x/KoeiroWorker.exe'))
    assert runtime_paths.is_bundled_worker(Path('/opt/koeiro/KoeiroWorker'))
    # A real Python interpreter is handed the module invocation instead.
    assert not runtime_paths.is_bundled_worker(Path('/opt/koeiro/Koeiro'))


def _install_root(tmp_path):
    return tmp_path / 'install' / 'Koeiro'


def _fake_environment(tmp_path):
    """A worker virtualenv shaped for the running OS."""
    root = _install_root(tmp_path) / 'vc_models' / 'meanvc2' / '.venv'
    launcher = root / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b'')
    return root


def test_a_packaged_gui_finds_the_worker_environment_installed_beside_it(tmp_path, monkeypatch):
    """The regression: a bundled GUI must not demand a frozen worker it never ships.

    PyInstaller unpacks to ``_MEIPASS``, where no ``vc_models`` exists; the worker
    environment lives next to the executable, installed by tools/setup_release.py.
    """
    environment = _fake_environment(tmp_path)
    bundle = tmp_path / 'bundle' / '_internal'
    bundle.mkdir(parents=True)
    monkeypatch.setattr(runtime_paths, 'install_root', lambda: _install_root(tmp_path))
    monkeypatch.setattr(runtime_paths, 'asset_root', lambda: bundle)
    monkeypatch.setattr(runtime_paths, 'is_frozen', lambda: True)
    assert runtime_paths.worker_environment() == environment
    # Nothing is shipped beside the GUI, so no frozen worker is invented.
    assert runtime_paths.worker_executable() is None


def test_a_packaged_gui_falls_back_to_the_bundle_root(tmp_path, monkeypatch):
    """The legacy ``.venv-ai`` layout is still honoured, but only when usable."""
    root = tmp_path / 'install' / 'Koeiro'
    environment = root / '.venv-ai'
    launcher = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b'')
    bundle = tmp_path / 'bundle' / '_internal'
    bundle.mkdir(parents=True)
    monkeypatch.setattr(runtime_paths, 'install_root', lambda: root)
    monkeypatch.setattr(runtime_paths, 'asset_root', lambda: bundle)
    monkeypatch.setattr(runtime_paths, 'is_frozen', lambda: True)
    assert runtime_paths.worker_environment() == environment


def test_a_packaged_gui_without_any_environment_says_so(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle' / '_internal'
    bundle.mkdir(parents=True)
    monkeypatch.setattr(runtime_paths, 'install_root', lambda: _install_root(tmp_path))
    monkeypatch.setattr(runtime_paths, 'asset_root', lambda: bundle)
    monkeypatch.setattr(runtime_paths, 'is_frozen', lambda: True)
    assert runtime_paths.worker_environment() is None


def test_a_bundled_worker_is_found_by_either_name(tmp_path, monkeypatch):
    root = _install_root(tmp_path)
    root.mkdir(parents=True)
    (root / 'KoeiroWorker').write_bytes(b'')
    monkeypatch.setattr(runtime_paths, 'install_root', lambda: root)
    assert runtime_paths.worker_executable() == root / 'KoeiroWorker'


def test_a_packaged_build_uses_the_folder_it_was_built_in(tmp_path, monkeypatch):
    """dist/Koeiro/ holds no vc_models, so the build records where its environment is.

    Keeping the repository out of the bundle is deliberate (torch cannot ship), which
    makes this record the difference between a working packaged build and one that
    reports a missing worker with no environment variable set.
    """
    environment = _fake_environment(tmp_path)
    source = tmp_path / 'install' / 'Koeiro'
    bundle = source / 'dist' / 'Koeiro' / '_internal'
    bundle.mkdir(parents=True)
    (bundle / runtime_paths.ROOT_FILE).write_text(str(source), 'utf-8')
    monkeypatch.delenv(runtime_paths.ROOT_ENV, raising=False)
    monkeypatch.setattr(runtime_paths, 'asset_root', lambda: bundle)
    monkeypatch.setattr(runtime_paths, 'is_frozen', lambda: True)
    assert runtime_paths.install_root() == source
    assert runtime_paths.worker_environment() == environment


def test_the_root_override_still_wins_over_the_recorded_folder(tmp_path, monkeypatch):
    """A moved executable is repointed with KOEIRO_ROOT, record or not."""
    moved = tmp_path / 'elsewhere' / 'Koeiro'
    moved.mkdir(parents=True)
    bundle = tmp_path / 'install' / 'Koeiro' / '_internal'
    bundle.mkdir(parents=True)
    (bundle / runtime_paths.ROOT_FILE).write_text(str(tmp_path / 'install' / 'Koeiro'), 'utf-8')
    monkeypatch.setenv(runtime_paths.ROOT_ENV, str(moved))
    monkeypatch.setattr(runtime_paths, 'asset_root', lambda: bundle)
    monkeypatch.setattr(runtime_paths, 'is_frozen', lambda: True)
    assert runtime_paths.install_root() == moved


def test_a_packaged_worker_runs_from_the_install_folder(tmp_path, monkeypatch):
    """The worker imports src.vc.service, which only exists in the source tree.

    PyInstaller's bundle holds src/gui and src/processors for the GUI but not the
    worker's packages, so the worker's working directory must be the install folder --
    where its own code, its models and its environment all are. Running it in the
    unpacked tree fails with "No module named 'src.vc'" before any model loads, and
    naming the tree on PYTHONPATH instead lets directories like tools/psutil/ shadow
    the environment's own packages.
    """
    import subprocess

    from src.vc import client as vc_client
    from src.vc.config import AIParameters

    environment_root = _fake_environment(tmp_path)
    (environment_root / 'pyvenv.cfg').write_text(f'home = {Path(sys.executable).parent}\n',
                                                 encoding='utf-8')
    install = tmp_path / 'install' / 'Koeiro'  # created with the fake environment
    captured = {}

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs
            self.stdin = self.stdout = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(vc_client, 'is_frozen', lambda: True)
    monkeypatch.setattr(vc_client, 'worker_environment', lambda: environment_root)
    monkeypatch.setattr(vc_client, 'worker_executable', lambda: None)
    monkeypatch.setattr(vc_client, 'install_root', lambda: install)
    monkeypatch.setattr(vc_client, 'data_dir', lambda: tmp_path / 'userdata')
    monkeypatch.setattr(vc_client, 'asset_root', lambda: tmp_path / 'bundle' / '_internal')
    (tmp_path / 'userdata').mkdir()
    client = vc_client.ServiceClient(AIParameters(device='cpu'))
    try:
        client.start()
        assert captured['kwargs']['cwd'] == install
        # The interpreter is handed the module itself, since it is not frozen.
        assert captured['args'][0][1:3] == ['-u', '-m']
    finally:
        client.close()
