import numpy as np
from src.vc.bridge import AIBridge
from src.vc.config import AIParameters
from src.vc.models import DEFAULT_VOICE_ID
from src.processors.ai_voice import AIVoiceProcessor
from src.processors.voice_router import VoiceRouter
from src.processors.female_dsp import FemaleDSPProcessor
from src.presets.female_presets import load_preset
from tests.test_ai_voice import FakeClient
from tests.test_ai_voice import wait


def ready_processor(model=None):
    # One voice ships, so the helper names it through the model layer.
    b=AIBridge(AIParameters(model=model or DEFAULT_VOICE_ID),FakeClient)
    p=AIVoiceProcessor(bridge=b);p.prepare(48000,256)
    b.status='Ready';b.ack_generation=b.generation
    return p,b


def test_approved_voice_buffers_first_short_phrase_without_dry_male_voice():
    p,b=ready_processor()
    audio=np.full((256,1),.3,dtype=np.float32)
    assert not p.process(audio,48000).any()
    assert b.input.available==256 and not p._primed


def test_startup_fade_uses_silence_instead_of_undelayed_dry_audio():
    p,b=ready_processor();p._epoch=b.generation
    b.output.write(np.full(b.parameters.startup_frames,.1,dtype=np.float32))
    out=p.process(np.full((256,1),.3,dtype=np.float32),48000)
    assert out[0,0]<.01 and out[-1,0]<=.101


def test_router_does_not_restore_original_during_approved_voice_preroll():
    p,b=ready_processor();r=VoiceRouter(FemaleDSPProcessor(load_preset('Anime Test')),p)
    r.prepare(48000,256);r.mode='ai_voice'
    b.status='Ready';b.active=True;b.ack_generation=b.generation
    assert not r.process(np.full((256,1),.3,dtype=np.float32),48000).any()
    assert b.input.available==256


def test_worker_error_still_falls_back_to_original():
    p,b=ready_processor();b.status='Error'
    audio=np.full((256,1),.3,dtype=np.float32)
    np.testing.assert_array_equal(p.process(audio.copy(),48000),audio)


def test_worker_status_error_falls_back_before_buffering():
    p,b=ready_processor()
    b.status='Error'
    audio=np.full((256,1),.3,dtype=np.float32)
    np.testing.assert_array_equal(p.process(audio.copy(),48000),audio)


def test_start_epoch_is_acknowledged_before_first_input_is_offered():
    p,b=ready_processor()
    try:
        b.load();wait(lambda:b.status=='Ready')
        b.activate_prepared()
        generation=b.generation
        p.process(np.full((256,1),.3,dtype=np.float32),48000)
        assert b.generation==generation==b.ack_generation
        assert b.input.available==256
    finally:b.stop()


def test_router_prepares_active_epoch_off_callback_and_keeps_first_block():
    p,b=ready_processor()
    r=VoiceRouter(FemaleDSPProcessor(load_preset('Anime Test')),p);r.mode='ai_voice'
    try:
        b.load();wait(lambda:b.status=='Ready')
        r.prepare(48000,256)
        generation=b.generation
        assert b.active and b.ack_generation==generation
        r.process(np.full((256,1),.3,dtype=np.float32),48000)
        assert b.generation==generation and b.input.available==256
    finally:b.stop()
