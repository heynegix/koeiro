"""Fourth delivery mode: finished-utterance MossFormer2_SR_48K band restoration."""
import io
import threading
import numpy as np
import pytest
from src.vc.config import AIParameters
from src.vc.protocol import send, receive, MAX_PAYLOAD
from src.vc.bridge import AIBridge
from src.vc.utterance import UtteranceCollector, convert_utterance
from src.settings.manager import AppSettings
from .test_ai_voice import FakeClient, wait


def test_removed_enhancer_names_are_rejected():
    for name in ('flashsr','mossformer','voicefixer'):
        with pytest.raises(ValueError):AIParameters(enhancer=name)


def test_utterance_limit_is_60_seconds_and_payload_admits_it():
    # Validation runs before the backend is touched, so None stubs suffice.
    with pytest.raises(ValueError):
        convert_utterance(None,None,None,np.zeros(60*48000+1,dtype=np.float32),max_seconds=60)
    with pytest.raises(ValueError):
        convert_utterance(None,None,None,np.zeros(31*48000,dtype=np.float32))  # default 30 s
    audio=np.ones(60*48000,dtype=np.float32)
    stream=io.BytesIO();send(stream,{'op':'utterance','status':'Ready'},audio);stream.seek(0)
    header,result=receive(stream)
    assert header['op']=='utterance' and len(result)==len(audio)
    with pytest.raises(ValueError):send(io.BytesIO(),{'op':'utterance'},np.zeros(MAX_PAYLOAD//4+1,dtype=np.float32))


def test_quality_collector_splits_at_60_seconds_and_other_modes_stay_at_30():
    from src.vc.utterance import MAX_SECONDS, QUALITY_MAX_SECONDS
    assert (MAX_SECONDS,QUALITY_MAX_SECONDS)==(30,60)
    collector=UtteranceCollector(max_seconds=60)
    results=collector.feed(np.ones(48000*61,dtype=np.float32)*.1)
    assert len(results)==1 and len(results[0])<=48000*60
    assert collector.limit_splits==1
    with pytest.raises(ValueError):UtteranceCollector(max_seconds=61)


def test_enhancer_failure_is_reported_not_silently_skipped():
    class Backend:
        chunk_samples=2560;sample_rate=16000
        def reset(self):self.delay=np.zeros(15360,dtype=np.float32)
        def get_stats(self):return {'algorithmic_buffer_ms':960}
        def process_chunk(self,block):
            self.delay=np.concatenate((self.delay,block));output=self.delay[:2560];self.delay=self.delay[2560:];return output
    class Down:
        def reset(self):pass
        def process(self,block):return block[::3]
    class Up:
        def reset(self):pass
        def process(self,block):return np.repeat(block,3)
    class Bad:
        def __call__(self,audio):raise RuntimeError('model exploded')
    audio=np.repeat(np.arange(16003,dtype=np.float32)/16003,3)
    with pytest.raises(RuntimeError):convert_utterance(Backend(),Down(),Up(),audio,enhance=Bad())
    good=lambda x:x*2
    result=convert_utterance(Backend(),Down(),Up(),audio,enhance=good)
    np.testing.assert_allclose(result,audio*2,rtol=1e-6)

