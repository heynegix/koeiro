"""The renamed app keeps one identity, and no existing settings are stranded."""
import json

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
