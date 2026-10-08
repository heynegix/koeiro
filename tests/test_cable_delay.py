import numpy as np
import pytest
from tools.measure_cable_delay import delay_estimate,CallbackCapture


def test_energy_delay_measurement_with_different_stream_start_and_gain():
    rng=np.random.default_rng(14)
    t=np.arange(100.,110.,.005)
    energy=np.convolve(rng.random(len(t)),np.ones(20)/20,mode='same')+.01
    output_t=t+.175
    value=delay_estimate(t,energy,output_t,energy*1.8)
    assert value['reliable'] and value['delay_ms']==pytest.approx(175)
    assert value['correlation']>.99


def test_silence_cannot_claim_verified_delay():
    t=np.arange(0,10,.005)
    with pytest.raises(ValueError,match='Silent'):
        delay_estimate(t,np.zeros(len(t)),t,np.zeros(len(t)))


def test_unrelated_output_cannot_claim_verified_delay():
    t=np.arange(0,10,.005);rng=np.random.default_rng(32)
    with pytest.raises(ValueError,match='reliable'):
        delay_estimate(t,rng.random(len(t)),t,rng.random(len(t)))


def test_callback_capture_is_bounded_and_envelope_uses_only_published_audio():
    capture=CallbackCapture(.01,block=256)
    capture.append(np.full(256,.2,dtype=np.float32))
    capture.append(np.ones(256,dtype=np.float32))
    assert capture.overruns==1 and capture.samples==256
    stamps,rms=capture.envelope()
    assert len(stamps)==1 and rms[0]==pytest.approx(.2)


@pytest.mark.parametrize('times,energy',[(np.array([]),np.array([])),
    (np.array([0.,0.]),np.array([.1,.2])),
    (np.array([0.,.1]),np.array([.1,np.nan]))])
def test_invalid_captures_cannot_produce_latency_claim(times,energy):
    with pytest.raises(ValueError,match='Invalid envelope'):
        delay_estimate(times,energy,times,energy)


def test_missing_speech_is_flagged_after_delay_alignment():
    from tools.measure_cable_delay import capture_safety
    t=np.arange(0,3,.005)
    source=np.full(len(t),.1);output=source.copy()
    output[200:220]=0
    value=capture_safety(t,source,t+.2,output,200)
    assert len(value['suspicious_silence_over_40ms'])==1
    assert value['suspicious_silence_over_40ms'][0]['duration_ms']==pytest.approx(100)
