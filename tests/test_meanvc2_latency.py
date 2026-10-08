import numpy as np
import pytest
from tools.measure_meanvc2_latency import observed_delay,ReplayGate


def test_delay_uses_common_clock_offset_and_not_capture_start_as_input_start():
    rate=48000
    source=np.zeros(rate*5,dtype=np.float32)
    rng=np.random.default_rng(4)
    source[rate:rate*2]=rng.normal(0,.03,rate)
    source[rate*3:rate*4]=rng.normal(0,.06,rate)
    capture=np.concatenate([np.zeros(round(2.35*rate)),source,np.zeros(rate)])
    result=observed_delay(source,capture,input_start=1,capture_start=0)
    assert result['valid'] and result['observed_input_to_cable_ms']==1350
    assert result['correlation']>.999


def test_silent_audio_cannot_be_claimed_as_latency_measurement():
    with pytest.raises(ValueError):observed_delay(np.zeros(48000),np.zeros(96000),0,0)


def test_replay_overwrites_microphone_and_preserves_tail_and_reset():
    class Gate:
        def reset(self):pass
        def process(self,audio,rate):return audio
    p=ReplayGate(Gate(),np.array([.1,.2,.3],dtype=np.float32))
    block=np.full((4,1),.8,dtype=np.float32)
    np.testing.assert_allclose(p.process(block,48000)[:,0],[.1,.2,.3,0])
    assert not p.process(block,48000).any()
    p.reset();assert p.offset==0 and p.started is None
