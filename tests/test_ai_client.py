"""AI process ownership and environment isolation, without external models."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from src.vc.client import worker_python


def test_crashed_worker_closes_every_handle_and_is_idempotent():
    from types import SimpleNamespace
    from src.vc.client import ServiceClient

    class Stream:
        def __init__(self, broken=False):
            self.closed = False
            self.broken = broken

        def close(self):
            self.closed = True
            if self.broken:
                raise OSError(22, 'dead worker pipe')

    stdin, stdout, log = Stream(True), Stream(), Stream()
    client = ServiceClient(None)
    client.process = SimpleNamespace(stdin=stdin, stdout=stdout,
                                     poll=lambda: 1, wait=lambda timeout: 1)
    client.log_file = log
    client.close()
    client.close()
    assert stdin.closed and stdout.closed and log.closed
    assert client.process is None and client.log_file is None


def test_missing_ai_environment_reports_setup(tmp_path):
    with pytest.raises(RuntimeError, match='AI環境'):
        worker_python(tmp_path)


@pytest.mark.skipif(os.name != 'nt', reason='Windows venv redirector')
def test_direct_worker_keeps_venv_and_is_terminable(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    expected = root/'vc_models/meanvc2/.venv/Scripts/python.exe'
    if not expected.exists():
        pytest.skip('Optional isolated MeanVC2 Python not installed')
    monkeypatch.setenv('PYTHONHOME', 'invalid inherited home')
    monkeypatch.setenv('PYTHONPATH', 'invalid inherited modules')
    executable, environment = worker_python(root)
    # The isolated MeanVC2 venv is retained through the launcher variable, while the
    # interpreter CPython actually runs is the one pyvenv.cfg points at (no stub).
    assert executable.is_file() and Path(executable) != expected
    assert environment['__PYVENV_LAUNCHER__'] == str(expected)
    assert 'PYTHONHOME' not in environment and 'PYTHONPATH' not in environment
    script = ('import sys,os,json,time; '
              'print(json.dumps([os.getpid(),sys.executable,sys.prefix]),flush=True); time.sleep(60)')
    process = subprocess.Popen([str(executable), '-u', '-c', script], env=environment,
                               stdout=subprocess.PIPE, text=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        pid, invoked, prefix = json.loads(process.stdout.readline())
        assert pid == process.pid  # actual interpreter, not a wrapper parent
        assert Path(invoked) == expected
        assert Path(prefix) == expected.parents[1]
    finally:
        process.terminate()
        process.wait(timeout=3)
        process.stdout.close()
    assert process.returncode is not None


@pytest.mark.skipif(os.name != 'nt', reason='Windows venv configuration')
def test_bad_home_rejected_before_process_launch(tmp_path):
    scripts = tmp_path/'vc_models/meanvc2/.venv/Scripts'
    scripts.mkdir(parents=True)
    (scripts/'python.exe').touch()
    (tmp_path/'vc_models/meanvc2/.venv/pyvenv.cfg').write_text('home = Z:/missing-python\n')
    with pytest.raises(RuntimeError, match='Python本体'):
        worker_python(tmp_path)
