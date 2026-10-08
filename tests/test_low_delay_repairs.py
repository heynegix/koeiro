import numpy as np
from types import SimpleNamespace

from src.processors.noise_gate import NoiseGateProcessor
from src.processors.chain import ProcessorChain
from src.processors.passthrough import PassthroughProcessor
from src.vc.phrase_prosody import HISTORY, HOP, LOOKAHEAD, repair_window
from src.vc.models import DEFAULT_VOICE_ID
from tests.test_meanvc2_startup_voice import ready_processor


def test_quiet_consonant_is_retained_without_sample_delay():
    outputs = []
    for protected in (False, True):
        gate = NoiseGateProcessor(-45)
        gate.prepare(48000, 256)
        gate.protect_quiet = protected
        x = np.full((256, 1), 10**(-51/20), np.float32)
        outputs.append(gate.process(x, 48000).copy())
    assert not outputs[0].any()
    assert outputs[1][0, 0] > 0
    assert outputs[1][-1, 0] > 10**(-51/20) * .65


def test_protected_gate_still_closes_on_deep_noise():
    gate = NoiseGateProcessor(-45)
    gate.prepare(48000, 256)
    gate.protect_quiet = True
    x = np.full((256, 1), 1e-5, np.float32)
    assert not gate.process(x, 48000).any()


def test_gate_protection_follows_selected_route_without_changing_other_slots():
    router = PassthroughProcessor()
    router.mode = 'ai_voice'
    router.ai = SimpleNamespace(bridge=SimpleNamespace(parameters=SimpleNamespace(model=DEFAULT_VOICE_ID)))
    chain = ProcessorChain(router)
    chain.prepare(48000, 256)
    chain.process(np.zeros((256, 1), np.float32), 48000)
    assert chain.gate.protect_quiet
    router.mode = 'original'
    chain.process(np.zeros((256, 1), np.float32), 48000)
    assert not chain.gate.protect_quiet


def test_recovery_needs_two_hops_and_never_plays_dry_voice():
    p, b = ready_processor(DEFAULT_VOICE_ID)
    p._epoch = b.generation
    p._primed = True
    p._recovering = False
    p._last = .1
    p.process(np.full((256, 1), .8, np.float32), 48000)
    assert not p._primed
    waiting = p.process(np.full((256, 1), .8, np.float32), 48000)
    assert np.max(abs(waiting)) < .001
    b.output.write(np.full(2*b.chunk_frames, .1, np.float32))
    out = p.process(np.full((256, 1), .8, np.float32), 48000)
    assert p._primed and np.max(out) <= .101
    assert out[0, 0] < .001 and out[-1, 0] > .099


def test_uncertain_feature_boundary_is_exact_bypass():
    size = HISTORY + LOOKAHEAD
    voice = np.full(size, .1, np.float32)
    positions = np.arange(1280, size+1, 640)
    a = np.tile([110., .99, -20.], (len(positions), 1))
    b = np.tile([220., .99, -20.], (len(positions), 1))
    # Supply a strong relative energy change adjacent to an unvoiced window.
    times = positions - 640
    a[times >= HISTORY, 2] = -14
    b[np.argmin(abs(times - (HISTORY+640))), 1] = 0
    out, _ = repair_window(voice, voice, HISTORY, pitch=False, features=(a,b))
    np.testing.assert_array_equal(out[:1280], voice[HISTORY:HISTORY+1280])
    assert out.shape == (HOP,)
