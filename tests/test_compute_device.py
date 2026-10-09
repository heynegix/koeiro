"""Compute-device plumbing: config, settings, worker resolution, client env, GUI."""
import sys
import types

import pytest

from src.vc.config import AIParameters, DEVICES
from src.settings.manager import AppSettings
from tests.test_ai_gui import ai_window, pump  # noqa: F401 (ai_window is used as a fixture)


def test_device_defaults_to_auto_and_validates():
    assert AIParameters().device == 'auto'
    assert DEVICES == ('auto', 'cpu', 'cuda')
    for valid in DEVICES:
        assert AIParameters(device=valid).device == valid
    for bad in ('CUDA', 'gpu', '', None, 0, True):
        with pytest.raises(ValueError):
            AIParameters(device=bad)


def test_device_survives_settings_roundtrip():
    assert AppSettings().ai_device == 'auto'
    assert AppSettings.from_dict({'ai_device': 'cuda'}).ai_device == 'cuda'
    assert AppSettings.from_dict({'ai_device': 'cuda'}).ai_parameters().device == 'cuda'
    assert AppSettings.from_dict({'ai_device': 'gpu'}).ai_device == 'auto'
    assert AppSettings.from_dict({'ai_device': None}).ai_device == 'auto'


def _fake_torch(cuda_available):
    fake_cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    return types.SimpleNamespace(cuda=fake_cuda)


def test_resolve_worker_device_matrix(monkeypatch):
    from src.vc import service
    monkeypatch.setitem(sys.modules, 'torch', _fake_torch(True))
    assert service.resolve_worker_device('cuda') == 'cuda'
    assert service.resolve_worker_device('auto') == 'cuda'
    assert service.resolve_worker_device('cpu') == 'cpu'
    monkeypatch.setitem(sys.modules, 'torch', _fake_torch(False))
    assert service.resolve_worker_device('auto') == 'cpu'
    assert service.resolve_worker_device('cpu') == 'cpu'
    with pytest.raises(ValueError):
        service.resolve_worker_device('cuda')
    with pytest.raises(ValueError):
        service.resolve_worker_device('gpu')
    # No torch at all (plain GUI env) always resolves to CPU.
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    monkeypatch.setitem(sys.modules, 'torch', None)
    import builtins
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == 'torch':
            raise ImportError('No module named torch')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_torch)
    assert service.resolve_worker_device('auto') == 'cpu'
    assert service.resolve_worker_device('cpu') == 'cpu'
    with pytest.raises(ValueError):
        service.resolve_worker_device('cuda')


def test_client_hides_gpus_only_for_forced_cpu(monkeypatch, tmp_path):
    import pathlib
    import subprocess
    from src.vc.client import ServiceClient
    captured = {}

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr('src.vc.client.worker_python',
                        lambda root, model=None: (tmp_path / 'python', {'PATH': 'x'}))
    monkeypatch.setattr('src.vc.client.asset_root', lambda: tmp_path)
    # The worker's log goes to the writable data directory, never the bundle.
    monkeypatch.setattr('src.vc.client.data_dir', lambda: tmp_path / 'userdata')
    client = ServiceClient(AIParameters(device='cpu'))
    client.log_file = None
    client.start()
    command = captured['args'][0]
    assert '--device' in command and command[command.index('--device') + 1] == 'cpu'
    assert captured['kwargs']['env'].get('CUDA_VISIBLE_DEVICES') == ''
    client2 = ServiceClient(AIParameters(device='auto'))
    client2.log_file = None
    client2.start()
    command2 = captured['args'][0]
    assert command2[command2.index('--device') + 1] == 'auto'
    assert 'CUDA_VISIBLE_DEVICES' not in captured['kwargs']['env']


def test_backend_and_lavasr_reject_bad_devices():
    from src.vc.meanvc2 import MeanVC2Backend
    with pytest.raises(ValueError):
        MeanVC2Backend(threads=2, device='tpu')
    from src.vc.lavasr import LavaSR
    with pytest.raises(ValueError):
        LavaSR(device='tpu')


def test_backend_cuda_without_gpu_fails_cleanly(monkeypatch):
    import types
    from pathlib import Path
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        set_num_threads=lambda *args: None,
        set_num_interop_threads=lambda *args: None,
    )
    fake_kaldi = types.SimpleNamespace()
    fake_torchaudio = types.SimpleNamespace(
        compliance=types.SimpleNamespace(kaldi=fake_kaldi))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'torchaudio', fake_torchaudio)
    monkeypatch.setitem(sys.modules, 'torchaudio.compliance', fake_torchaudio.compliance)
    monkeypatch.setitem(sys.modules, 'torchaudio.compliance.kaldi', fake_kaldi)
    from src.vc.meanvc2 import MeanVC2Backend
    from src.vc.models import default_voice_id, profile
    root = Path(__file__).resolve().parents[1]
    folder = root / 'models' / profile(default_voice_id())['folder']
    backend = MeanVC2Backend(threads=1, device='cuda')
    with pytest.raises(ValueError, match='CUDA requested'):
        backend.load(folder)


def test_gui_device_combo_drives_parameters(ai_window):
    app, window, backend, bridge = ai_window
    assert window.ai_device.currentData() == 'auto'
    window.ai_device.setCurrentIndex(window.ai_device.findData('cuda'))
    assert window._ai_parameters().device == 'cuda'
    window._capture_settings()
    assert window.settings.ai_device == 'cuda'
    window.ai_device.setCurrentIndex(window.ai_device.findData('cpu'))
    assert window._ai_parameters().device == 'cpu'
