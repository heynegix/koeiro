"""Opt-in isolated model tests; invalid copies never touch installed models."""
from pathlib import Path
import pytest
import numpy as np
pytest.importorskip('sherpa_onnx')
from src.asr.backend import ReazonBackend


def test_invalid_model_error(tmp_path):
    for name in ('encoder-epoch-99-avg-1.int8.onnx','decoder-epoch-99-avg-1.onnx','joiner-epoch-99-avg-1.onnx','tokens.txt'):
        (tmp_path/name).write_bytes(b'invalid model copy')
    with pytest.raises(Exception): ReazonBackend(tmp_path).load()


def test_missing_model_error(tmp_path):
    with pytest.raises(ValueError,match='missing'): ReazonBackend(tmp_path).load()


def test_reazon_load_finite_length_reset_unload():
    root=Path(__file__).resolve().parents[1]/'models/asr-reazon'
    if not (root/'encoder-epoch-99-avg-1.int8.onnx').is_file(): pytest.skip('Optional installed model missing')
    backend=ReazonBackend(root); backend.load()
    assert isinstance(backend.transcribe(np.zeros(16000,dtype=np.float32)),str)
    backend.reset(); backend.unload()
    with pytest.raises(RuntimeError): backend.transcribe(np.zeros(16000,dtype=np.float32))


@pytest.mark.parametrize('bad',[np.array([np.nan],dtype=np.float32),np.array([np.inf],dtype=np.float32),
                              np.zeros(0,dtype=np.float32),np.zeros((2,2),dtype=np.float32)])
def test_backend_input_protection(bad):
    backend=ReazonBackend('unused'); backend.recognizer=object()
    with pytest.raises(ValueError): backend.transcribe(bad)
