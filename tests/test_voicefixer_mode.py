"""Fifth delivery mode: finished-utterance VoiceFixer2 restoration (Q012 pick)."""
import io
import numpy as np
import pytest
from src.vc.config import AIParameters
from src.vc.bridge import AIBridge
from src.settings.manager import AppSettings
from .test_ai_voice import FakeClient, wait


def test_utterance_bridge_allocates_large_rings_and_scaled_timeout():
    class Client(FakeClient):
        def request(self,header,audio=None):
            return ({'status':'Ready','stats':{}},np.zeros(7680,dtype=np.float32))
    bridge=AIBridge(AIParameters(enhancer='lavasr'),Client)
    try:
        assert bridge.input.capacity>=60*48000
        assert bridge.output.capacity>=60*48000  # one whole-utterance write must fit
        bridge.load()
        wait(lambda:bridge.status=='Ready')
        bridge.set_active(True)
        wait(lambda:bridge.ack_generation==bridge.generation)
        audio=np.ones(8*48000,dtype=np.float32)*.1
        bridge._request({'op':'utterance'},audio)
        # Measured LavaSR RTF 0.043-0.064, budgeted at 8x on top of a 30 s allowance.
        assert bridge._request_timeout>=30.+8*8-1e-6
        bridge._request({'op':'utterance'},np.ones(60*48000,dtype=np.float32))
        # The 60 s ceiling is capped at 270 s so a full utterance cannot hang the worker.
        assert bridge._request_timeout==270.
    finally:bridge.stop()


def test_removed_enhancer_names_are_rejected():
    for name in ('flashsr','mossformer','voicefixer'):
        with pytest.raises(ValueError):AIParameters(enhancer=name)


def test_voicefixer_wrapper_manifest_is_pinned_and_wrapper_validates_input():
    import json
    from pathlib import Path
    info=json.loads(Path('src/vc/voicefixer_runtime.json').read_text('utf-8'))
    assert info['sha256']['vf.ckpt']=='748411b70089cadf34a6c11054f95f3a454e614af562c23b13a82f6cb413109f'
    assert info['sha256']['vocoder_44100.pt']=='9410d0b528c10a251ae947bd299d1939b0b3247df680c81c4164e94f5d87dc45'
    from src.vc import voicefixer_sr
    wrapper=voicefixer_sr.VoiceFixerSR.__new__(voicefixer_sr.VoiceFixerSR)  # skip model load; test guards only
    silent=np.zeros(48000,dtype=np.float32)
    np.testing.assert_array_equal(wrapper.process(silent),silent)
    with pytest.raises(RuntimeError):wrapper.process(np.zeros((2,100),dtype=np.float32))
    with pytest.raises(RuntimeError):wrapper.process(np.full(48000,np.nan,dtype=np.float32))
