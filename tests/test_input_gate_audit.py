import numpy as np
from tools.audit_input_gate import audit


def test_offline_audit_exposes_weak_signal_rejected_by_high_threshold():
    # Constant RMS -49 dBFS: passes -50 dB, cannot open -43.8 dB gate.
    signal=np.full(48000,10**(-49/20),dtype=np.float32)
    _,strict=audit(signal,-43.8)
    _,soft=audit(signal,-50)
    assert strict['active_retained_fraction']==0
    assert soft['active_retained_fraction']>.98
    assert soft['energy_retained_ratio']>.98


def test_silence_is_not_a_gate_retention_success_claim():
    result,report=audit(np.zeros(48000,dtype=np.float32),-50)
    assert not result.any()
    assert report['active_blocks']==0
    assert report['active_retained_fraction'] is None
