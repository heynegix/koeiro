"""Run the optional Kokoro TTS through the environment that owns it.

:func:`src.tts.provider.build_provider` speaks the model directly, which is what a user
who installed ``kokoro-onnx`` into the app's own environment gets. The shipped setup
deliberately does not: onnxruntime and espeak are large, so ``tools/install_tts.py``
puts the runtime in ``vc_models/tts/.venv`` and the weights in ``models/tts/kokoro``.
:class:`KokoroClient` is the bridge between the two -- one request, one subprocess in
that environment, one WAV back.

It satisfies the same ``SpeechSynth`` protocol as the in-process provider, so
``PreservingSynth`` and the translation pipeline take it unchanged. One process per
request is intentional: the lab synthesizes a single line, and paying the model load is
cheaper than keeping a second interpreter alive in the background.
"""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import wave

import numpy as np

from src.tts.provider import (DEFAULT_LANG, DEFAULT_VOICE, NATIVE_SAMPLE_RATE,
                              TtsUnavailable, _resample, available)

VENV_DIR = Path('vc_models') / 'tts' / '.venv'
DEFAULT_TIMEOUT = 600.0


def venv_dir(root):
    return Path(root) / VENV_DIR


def launcher(root):
    """The TTS environment's interpreter, or None when it was never created."""
    base = venv_dir(root)
    for relative in ('Scripts/python.exe', 'bin/python'):
        candidate = base / relative
        if candidate.is_file():
            return candidate
    return None


def worker_available(root):
    """True when both the model files and the TTS environment are installed."""
    return available(root) and launcher(root) is not None


def _interpreter(launcher_path):
    """The CPython behind a venv launcher, or the launcher itself.

    A venv's ``python.exe`` is a stub that re-executes the base interpreter named in
    ``pyvenv.cfg``; the ASR worker resolves the same way, and using the base binary
    directly keeps a moved or renamed venv folder working. On POSIX the launcher is
    kept: pyvenv.cfg's ``home`` usually names the system directory that holds the
    base ``python``, and resolving to it escapes the venv (``sys.prefix`` turns into
    the system one) even though the launcher re-execs correctly.
    """
    if os.name != 'nt':
        return launcher_path
    home = None
    try:
        config = (launcher_path.parent.parent / 'pyvenv.cfg').read_text(encoding='utf-8')
    except OSError:
        config = ''
    for line in config.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            if key.strip() == 'home':
                home = value.strip()
                break
    if home:
        candidate = Path(home) / ('python.exe' if os.name == 'nt' else 'python')
        if candidate.is_file():
            return candidate
    return launcher_path


def _environment(launcher_path):
    """A clean environment: the GUI's own interpreter must not leak into the worker."""
    environment = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH', '__PYVENV_LAUNCHER__'):
        environment.pop(key, None)
    if os.name == 'nt':
        environment['__PYVENV_LAUNCHER__'] = str(launcher_path)
    return environment


def read_wav(path):
    """Mono float32 and its rate from a 16-bit PCM WAV."""
    with wave.open(str(path), 'rb') as handle:
        if handle.getsampwidth() != 2:
            raise RuntimeError('The TTS worker wrote a WAV that is not 16-bit PCM')
        rate = handle.getframerate()
        channels = handle.getnchannels()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype='<i2')
    audio = samples.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return rate, audio


class KokoroClient:
    """Kokoro living in the TTS environment, driven from the app process."""

    def __init__(self, root, voice=DEFAULT_VOICE, lang=DEFAULT_LANG, speed=1.0,
                 timeout=DEFAULT_TIMEOUT):
        self.launcher = launcher(root)
        if self.launcher is None:
            raise TtsUnavailable(f'No TTS environment under {venv_dir(root)}'
                                 ' (run tools/install_tts.py)')
        self.root = str(Path(root))
        self.voice = voice or DEFAULT_VOICE
        self.lang = lang or DEFAULT_LANG
        self.speed = float(speed)
        self.timeout = float(timeout)
        # What the last request cost, split into model load and synthesis, so a report
        # can say which of the two is slow instead of only the total.
        self.last_timing = {}

    def synthesize(self, text, voice='', sample_rate=NATIVE_SAMPLE_RATE):
        text = (text or '').strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        sample_rate = int(sample_rate or NATIVE_SAMPLE_RATE)
        with tempfile.TemporaryDirectory(prefix='koeiro-tts-') as folder:
            request = Path(folder) / 'request.json'
            output = Path(folder) / 'speech.wav'
            request.write_text(json.dumps(
                {'root': self.root, 'text': text, 'voice': voice or self.voice,
                 'lang': self.lang, 'speed': self.speed, 'sample_rate': sample_rate,
                 'out': str(output)}, ensure_ascii=False), encoding='utf-8')
            result = self._result(self._run(request))
            if not output.is_file():
                raise RuntimeError('The TTS worker reported success but wrote no WAV')
            rate, audio = read_wav(output)
        self.last_timing = {'load_seconds': float(result.get('load_seconds') or 0.0),
                            'synthesis_seconds': float(result.get('synthesis_seconds') or 0.0)}
        return _resample(audio, rate, sample_rate)

    def close(self):
        """Nothing stays open between requests; present so callers can close uniformly."""

    def _run(self, request):
        command = [str(_interpreter(self.launcher)), '-u', '-m', 'src.tts.service',
                   '--request', str(request)]
        try:
            return subprocess.run(command, cwd=self.root, env=_environment(self.launcher),
                                  stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                  timeout=self.timeout,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except OSError as error:
            raise TtsUnavailable(f'Could not start the TTS environment: {error}') from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f'The TTS worker did not finish in {self.timeout:.0f}s') from error

    @staticmethod
    def _result(completed):
        """The worker's last stdout line, or a failure naming what actually happened."""
        for line in reversed((completed.stdout or '').splitlines()):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                break
            if isinstance(payload, dict):
                if payload.get('status') == 'ok':
                    return payload
                raise RuntimeError(payload.get('error') or 'The TTS worker failed')
            break
        detail = [line for line in (completed.stderr or '').strip().splitlines() if line.strip()]
        raise RuntimeError('The TTS worker said nothing usable'
                           + (f': {detail[-1]}' if detail else f' (exit {completed.returncode})'))
