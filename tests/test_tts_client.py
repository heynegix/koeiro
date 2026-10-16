"""The TTS bridge: one request to the environment that owns kokoro-onnx.

The real model is not installed in this environment, so the tests answer the parts
that can be answered honestly: the request/response contract, the WAV it must write,
the fallback in ``build_provider``, and the worker's own failure reporting through a
real interpreter. Nothing here listens to audio.
"""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import venv
import wave

import numpy as np
import pytest

from src.tts import client as tts_client
from src.tts import provider as tts

ROOT = Path(__file__).resolve().parents[1]


def make_model_files(root):
    base = tts.model_dir(root)
    base.mkdir(parents=True, exist_ok=True)
    (base / tts.MODEL_FILE).write_bytes(b"x")
    (base / tts.VOICES_FILE).write_bytes(b"x")
    return base


def make_fake_venv(root):
    """A launcher file is all ``launcher()`` looks for; nothing runs it in these tests."""
    launcher = tts_client.venv_dir(root) / ('Scripts/python.exe' if sys.platform == 'win32'
                                            else 'bin/python')
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_bytes(b"stub")
    return launcher


def write_wav(path, samples, rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(samples, dtype='<i2').tobytes())
    return path


class FakeRunner:
    """Answers a request the way the worker does, from inside the same temp folder."""

    def __init__(self, status='ok', samples=480, rate=24000, stderr=''):
        self.status, self.samples, self.rate, self.stderr = status, samples, rate, stderr
        self.requests = []

    def __call__(self, request_path):
        request = json.loads(Path(request_path).read_text(encoding='utf-8'))
        self.requests.append(request)
        if self.status == 'ok':
            write_wav(request['out'], np.full(self.samples, 8192, dtype='<i2'), self.rate)
            stdout = json.dumps(dict(status='ok', engine='kokoro-onnx', load_seconds=1.5,
                                     synthesis_seconds=0.25))
        elif self.status == 'silent':
            stdout = ''  # a crashed interpreter: nothing useful on stdout at all
        else:
            stdout = json.dumps(dict(status='error', kind='TtsUnavailable', error='boom'))
        return subprocess.CompletedProcess([], 0, stdout, self.stderr)


def test_available_needs_both_the_models_and_an_environment(tmp_path):
    assert tts_client.worker_available(tmp_path) is False
    with pytest.raises(tts.TtsUnavailable):
        tts_client.KokoroClient(tmp_path)
    make_model_files(tmp_path)
    assert tts_client.worker_available(tmp_path) is False
    make_fake_venv(tmp_path)
    assert tts_client.worker_available(tmp_path) is True


def test_build_provider_falls_back_to_the_tts_environment(tmp_path):
    make_model_files(tmp_path)
    make_fake_venv(tmp_path)
    # kokoro-onnx is not importable here, which is exactly the shipped case: the
    # provider must reach the dedicated environment instead of returning nothing.
    built = tts.build_provider(tmp_path, voice='am_michael')
    assert isinstance(built, tts_client.KokoroClient)
    assert built.voice == 'am_michael'


def test_a_request_comes_back_as_resampled_audio(tmp_path):
    make_model_files(tmp_path)
    make_fake_venv(tmp_path)
    runner = FakeRunner(samples=480, rate=24000)
    client = tts_client.KokoroClient(tmp_path, speed=1.25)
    client._run = runner
    audio = client.synthesize('hello there', sample_rate=48000)
    assert audio.dtype == np.float32 and audio.size == 960
    assert np.allclose(audio, 8192 / 32768.0, atol=1e-4)
    assert client.last_timing == {'load_seconds': 1.5, 'synthesis_seconds': 0.25}
    request = runner.requests[0]
    assert request['text'] == 'hello there' and request['speed'] == 1.25
    assert request['sample_rate'] == 48000 and request['voice'] == tts.DEFAULT_VOICE
    assert request['root'] == str(tmp_path) and Path(request['out']).name == 'speech.wav'


def test_the_worker_runs_in_the_tts_environment(tmp_path, monkeypatch):
    """The subprocess is started from the installed tree, without the GUI's paths."""
    make_model_files(tmp_path)
    make_fake_venv(tmp_path)
    seen = {}

    def fake_run(command, **kwargs):
        seen['command'], seen['kwargs'] = command, kwargs
        payload = json.dumps(dict(status='error', kind='X', error='stop here'))
        return subprocess.CompletedProcess(command, 1, payload, '')

    monkeypatch.setattr(tts_client.subprocess, 'run', fake_run)
    with pytest.raises(RuntimeError, match='stop here'):
        tts_client.KokoroClient(tmp_path).synthesize('hello')
    assert seen['command'][1:4] == ['-u', '-m', 'src.tts.service']
    assert seen['kwargs']['cwd'] == str(tmp_path)
    assert 'PYTHONPATH' not in seen['kwargs']['env']


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX venv launcher')
def test_posix_tts_interpreter_preserves_real_venv_prefix(tmp_path):
    """Resolving pyvenv.cfg's home must not escape the dedicated environment."""
    environment_root = tts_client.venv_dir(tmp_path)
    venv.EnvBuilder(with_pip=False).create(environment_root)
    base_home = tmp_path / 'base'
    base_home.mkdir()
    (base_home / 'python').symlink_to(sys._base_executable)
    config = environment_root / 'pyvenv.cfg'
    config.write_text('\n'.join(
        'home = ' + str(base_home) if line.startswith('home = ') else line
        for line in config.read_text().splitlines()) + '\n')
    launcher_path = tts_client.launcher(tmp_path)
    prefixes = []
    for executable in (launcher_path, tts_client._interpreter(launcher_path)):
        completed = subprocess.run(
            [str(executable), '-c', 'import sys; print(sys.prefix)'],
            env=tts_client._environment(launcher_path), capture_output=True,
            text=True, check=True, timeout=10)
        prefixes.append(Path(completed.stdout.strip()).resolve())
    assert prefixes == [environment_root.resolve(), environment_root.resolve()]


@pytest.mark.parametrize('base_exists', [True, False])
def test_windows_tts_interpreter_retains_redirector_setup(tmp_path, monkeypatch, base_exists):
    launcher_path = tmp_path / 'tts' / 'Scripts' / 'python.exe'
    launcher_path.parent.mkdir(parents=True)
    launcher_path.touch()
    base_home = tmp_path / 'base'
    base_home.mkdir()
    base_python = base_home / 'python.exe'
    if base_exists:
        base_python.touch()
    (launcher_path.parent.parent / 'pyvenv.cfg').write_text('home = ' + str(base_home))
    monkeypatch.setattr(tts_client, 'os', SimpleNamespace(name='nt', environ={
        'PYTHONHOME': 'inherited', 'PYTHONPATH': 'inherited', '__PYVENV_LAUNCHER__': 'inherited',
    }))
    assert tts_client._interpreter(launcher_path) == (base_python if base_exists else launcher_path)
    assert tts_client._environment(launcher_path) == {'__PYVENV_LAUNCHER__': str(launcher_path)}


def test_empty_text_never_starts_a_worker(tmp_path):
    make_model_files(tmp_path)
    make_fake_venv(tmp_path)
    runner = FakeRunner()
    client = tts_client.KokoroClient(tmp_path)
    client._run = runner
    assert client.synthesize('   ').size == 0
    assert runner.requests == []


def test_a_failing_worker_is_reported_with_its_own_message(tmp_path):
    make_model_files(tmp_path)
    make_fake_venv(tmp_path)
    client = tts_client.KokoroClient(tmp_path)
    client._run = FakeRunner(status='error')
    with pytest.raises(RuntimeError, match='boom'):
        client.synthesize('hello')


def test_a_worker_that_says_nothing_usable_reports_stderr(tmp_path):
    make_model_files(tmp_path)
    make_fake_venv(tmp_path)
    client = tts_client.KokoroClient(tmp_path)
    client._run = FakeRunner(status='silent', stderr='Traceback: no module named numpy')
    with pytest.raises(RuntimeError, match='no module named numpy'):
        client.synthesize('hello')


def test_read_wav_downmixes_and_scales(tmp_path):
    stereo = np.array([[32767, 32767], [-32768, -32768]], dtype='<i2')
    with wave.open(str(tmp_path / 'stereo.wav'), 'wb') as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(stereo.tobytes())
    rate, audio = tts_client.read_wav(tmp_path / 'stereo.wav')
    assert rate == 16000 and audio.size == 2
    assert audio[0] == pytest.approx(1.0, abs=1e-4) and audio[1] == pytest.approx(-1.0, abs=1e-4)


def test_the_worker_reports_a_missing_runtime_as_json(tmp_path):
    """A real interpreter, a real import failure, one JSON line the client can read."""
    make_model_files(tmp_path)
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'root': str(tmp_path), 'text': 'hello',
                                   'out': str(tmp_path / 'out.wav')}), encoding='utf-8')
    completed = subprocess.run([sys.executable, '-u', '-m', 'src.tts.service',
                                '--request', str(request)],
                               cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert completed.returncode == 1
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload['status'] == 'error' and payload['error']
    assert not (tmp_path / 'out.wav').exists()
