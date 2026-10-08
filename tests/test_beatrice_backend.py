import hashlib
import json
import struct
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from src.vc.beatrice_vst import BeatriceVSTBackend
from src.vc.base import VoiceConversionBackend
from src.vc.vst_state import set_model_preset


def preset():
    state = struct.pack('<hii',1,2,0)+struct.pack('<hii',2,0,0)+struct.pack('<hid',4,1,0)
    comp = struct.pack('<I',len(state))+state
    header = b'VST3'+struct.pack('<i',1)+b'A'*32+struct.pack('<q',48+len(comp))
    return header+comp+b'List'+struct.pack('<i',1)+struct.pack('<4sqq',b'Comp',48,len(comp))


def test_vst_state_preserves_wrapper_and_sets_utf8_path_voice_and_pitch(tmp_path):
    model = tmp_path/'日本語.toml'
    result = set_model_preset(preset(),model,1,4)
    offset = struct.unpack_from('<q',result,40)[0]
    assert result[offset:offset+4] == b'List'
    assert str(model.resolve()).encode('utf-8') in result
    assert struct.pack('<hii',2,0,1) in result
    assert struct.pack('<hid',4,1,4) in result


@pytest.mark.parametrize('data',[b'',b'VST3',b'wrong'*100,preset()[:50]])
def test_truncated_or_invalid_vst_state_rejected_before_native_plugin(data,tmp_path):
    with pytest.raises(ValueError):
        set_model_preset(data,tmp_path/'model.toml',1,4)


def test_missing_model_parameters_do_not_silently_use_plugin_defaults(tmp_path):
    comp=struct.pack('<I',0)
    data=(b'VST3'+struct.pack('<i',1)+b'A'*32+struct.pack('<q',52)+comp+
          b'List'+struct.pack('<i',1)+struct.pack('<4sqq',b'Comp',48,4))
    with pytest.raises(ValueError,match='parameters missing'):
        set_model_preset(data,tmp_path/'model.toml',1,4)


def assets(tmp_path):
    folder = tmp_path/'beatrice'
    folder.mkdir()
    (folder/'plugin.vst3').write_bytes(b'fake plugin')
    (folder/'model.toml').write_text('[model]')
    manifest = {p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in folder.iterdir()}
    (tmp_path/'beatrice-manifest.json').write_text(json.dumps({'files':manifest,'license':'test'}))
    (tmp_path/'runtime.json').write_text(json.dumps({'plugin':'beatrice/plugin.vst3','model':'beatrice/model.toml',
        'voice':1,'pitch':4,'name':'Mock Girl'}))


def test_beatrice_contract_load_warmup_reset_unload_and_finite(tmp_path,monkeypatch):
    assets(tmp_path)
    class Plugin:
        preset_data=preset()
        reported_latency_samples=0
        def __init__(self):self.calls=0;self.resets=0
        def process(self,audio,rate,buffer_size,reset):
            assert rate==48000 and not reset
            self.calls+=1
            return audio*.4
        def reset(self):self.resets+=1
    plugin = Plugin()
    monkeypatch.setitem(sys.modules,'pedalboard',SimpleNamespace(load_plugin=lambda path:plugin))
    backend = BeatriceVSTBackend()
    assert isinstance(backend,VoiceConversionBackend)
    backend.load(tmp_path);backend.warmup()
    assert plugin.calls==12 and plugin.resets==1
    result = backend.process_chunk(np.ones(624,dtype=np.float32))
    assert result.dtype==np.float32 and np.all(result==np.float32(.4))
    assert backend.get_stats()['model_alignment_delay_ms'] is None
    with pytest.raises(ValueError):backend.process_chunk(np.full(624,np.nan,dtype=np.float32))
    backend.reset();assert plugin.resets==2
    backend.unload()
    with pytest.raises(RuntimeError):backend.process_chunk(np.ones(624,dtype=np.float32))


def test_beatrice_checksum_failure_prevents_loading_plugin(tmp_path,monkeypatch):
    assets(tmp_path)
    (tmp_path/'beatrice/plugin.vst3').write_bytes(b'corrupt')
    monkeypatch.setitem(sys.modules,'pedalboard',None)
    with pytest.raises(ValueError,match='checksum'):
        BeatriceVSTBackend().load(tmp_path)


def test_beatrice_nonfinite_output_rejected_for_original_fallback():
    backend=BeatriceVSTBackend()
    backend.plugin=SimpleNamespace(process=lambda *args,**kwargs:np.full(624,np.inf,dtype=np.float32))
    with pytest.raises(RuntimeError):backend.process_chunk(np.ones(624,dtype=np.float32))
