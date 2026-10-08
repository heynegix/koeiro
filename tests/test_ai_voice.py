import hashlib
import io
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.vc.base import VoiceConversionBackend
from src.vc.bridge import AIBridge
from src.vc.config import QUALITY_FACTORS, AIParameters
from src.vc.models import DEFAULT_VOICE_ID
from src.vc.llvc_onnx import LLVCOnnxBackend
from src.vc.protocol import send,receive
from src.vc.ring import AudioRing
from src.processors.ai_voice import AIVoiceProcessor
from src.processors.voice_router import VoiceRouter
from src.processors.female_dsp import FemaleDSPProcessor
from src.presets.female_presets import load_preset
from src.settings.manager import AppSettings,SettingsManager


class FakeClient:
    def __init__(self,parameters):
        self.parameters=parameters
        self.closed=False
        self.calls=0
        self.bad=False
    def start(self): pass
    def read(self): return {'status':'Ready','model':{'fake':True}},np.empty(0,dtype=np.float32)
    def request(self,header,audio=None):
        if self.closed:
            raise EOFError('Stopped')
        if header['op']=='reset':
            return {'status':'Ready'},np.empty(0,dtype=np.float32)
        self.calls+=1
        result=audio*.5
        if self.bad:
            result.fill(np.nan)
        return {'status':'Ready','stats':{'rtf':.1}},result
    def interrupt(self): self.closed=True
    def close(self): self.closed=True


def wait(predicate,seconds=3):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if predicate(): return
        time.sleep(.002)
    raise AssertionError('Worker did not respond')


def test_full_output_waits_for_callback_notification_and_resumes():
    bridge=AIBridge(AIParameters(quality='low_latency'),FakeClient)
    original_wait=bridge.input_ready.wait
    full_wait=__import__('threading').Event()
    def observed_wait(stop):
        if bridge.output.available==bridge.output.capacity:
            full_wait.set()
        original_wait(stop)
    bridge.input_ready.wait=observed_wait
    try:
        bridge.load();bridge.set_active(True)
        wait(lambda:bridge.status=='Ready' and bridge.ack_generation==bridge.generation)
        block=np.ones(bridge.chunk_frames,dtype=np.float32)
        for _ in range(bridge.parameters.queue_chunks):
            expected=bridge.client.calls+1
            assert bridge.input.write(block)
            bridge.input_ready.signal()
            wait(lambda:bridge.client.calls>=expected)
        assert full_wait.wait(1)
        assert bridge.input.write(block)
        consumed=np.empty_like(block)
        assert bridge.output.read_into(consumed)
        bridge.input_ready.signal()
        wait(lambda:bridge.output.available==bridge.output.capacity)
        assert bridge.overruns==bridge.output_overruns==0
    finally:
        bridge.stop()


@pytest.mark.parametrize('capacity', [1,17,256,1248])
def test_ring_wrap_preserves_order_and_bounds(capacity):
    ring=AudioRing(capacity)
    for i in range(10):
        samples=np.arange(capacity,dtype=np.float32)+i
        assert ring.write(samples)
        assert not ring.write(np.ones(1,dtype=np.float32))
        out=np.empty_like(samples)
        assert ring.read_into(out)
        np.testing.assert_array_equal(out,samples)
        assert ring.available==0
    assert ring.maximum_depth==capacity


def test_ring_partial_wrap_underrun_and_old_drop():
    ring=AudioRing(10)
    assert ring.write(np.arange(7,dtype=np.float32))
    out=np.empty(5,dtype=np.float32)
    assert ring.read_into(out)
    assert ring.write(np.arange(7,13,dtype=np.float32))
    assert ring.discard(3)==3
    assert ring.read_into(out)
    np.testing.assert_array_equal(out,np.arange(8,13))
    out.fill(99)
    assert not ring.read_into(out)
    assert np.all(out==99)


@pytest.mark.parametrize('changes',[{'quality':'bad'},{'threads':True},{'threads':8},{'model':'other'},
 {'brightness':float('nan')},{'brightness':-1},{'limiter':'ON'},{'post_fx':1},{'delivery':'broken'},
 {'enhancer':'not_a_model'}])
def test_ai_settings_validation(changes):
    with pytest.raises(ValueError): AIParameters(**changes)


def test_missing_voice_stays_constructible_and_neutral():
    params = AIParameters(model=None)
    assert params.model is None
    assert (params.delivery, params.enhancer, params.experiment) == ('streaming', 'none', 'none')
    assert params.lavasr_denoise is False
    assert params.chunk_frames == 624*QUALITY_FACTORS[params.quality]
    assert params.queue_chunks >= 4 and params.startup_chunks >= 2
    assert params.mute_during_startup is False


def test_empty_library_falls_back_to_no_voice(monkeypatch):
    import src.vc.models as models
    import src.settings.manager as manager
    monkeypatch.setattr(models, 'VOICE_PROFILES', {})
    monkeypatch.setattr(models, 'DEFAULT_VOICE_ID', 'user_missing')
    assert models.default_voice_id() is None
    values = manager.AppSettings.from_dict({'ai_model': 'user_missing'})
    assert values.ai_model is None
    assert values.ai_parameters().model is None


def test_quality_does_not_resize_the_shipped_route():
    # The MeanVC2 route has a fixed hop, so quality cannot change its buffering.
    from src.vc.models import profile
    expected=profile(DEFAULT_VOICE_ID)
    sizes={(AIParameters(quality=q).chunk_frames,AIParameters(quality=q).queue_chunks)
           for q in QUALITY_FACTORS}
    assert len(sizes)==1
    assert AIParameters().chunk_frames==7680
    assert AIParameters().queue_chunks==expected['queue_chunks']


def test_protocol_exact_binary_roundtrip_and_bounds():
    stream=io.BytesIO()
    samples=np.array([.1,-.2,.3],dtype=np.float32)
    send(stream,{'op':'process'},samples)
    stream.seek(0)
    header,audio=receive(stream)
    assert header['op']=='process'
    np.testing.assert_array_equal(samples,audio)
    with pytest.raises(ValueError): send(io.BytesIO(),{},np.zeros(20000,dtype=np.float32))
    with pytest.raises(EOFError): receive(io.BytesIO(b'abc'))


def test_worker_lazy_loading_and_ten_start_stop():
    bridge=AIBridge(client_factory=FakeClient)
    assert bridge.status=='Not Loaded' and not bridge.alive
    for _ in range(10):
        bridge.load(); bridge.load()
        wait(lambda:bridge.status=='Ready')
        bridge.prepare(); bridge.set_active(True)
        wait(lambda:bridge.ack_generation==bridge.generation)
        assert bridge.input.write(np.ones(bridge.chunk_frames,dtype=np.float32))
        wait(lambda:bridge.output.available>=bridge.chunk_frames)
        assert bridge.client.calls==1
        bridge.stop()
        assert not bridge.alive and bridge.status=='Not Loaded'


def test_worker_invalid_output_falls_back_without_abort():
    bridge=AIBridge(client_factory=FakeClient)
    processor=AIVoiceProcessor(bridge=bridge)
    processor.prepare(48000,256)
    bridge.load(); bridge.set_active(True)
    wait(lambda:bridge.status=='Ready' and bridge.ack_generation==bridge.generation)
    bridge.client.bad=True
    # A full chunk of input is what triggers a worker request on this route.
    for _ in range(bridge.chunk_frames//256 + 2):
        processor.process(np.full((256,1),.1,dtype=np.float32),48000)
    wait(lambda:bridge.status=='Error',seconds=10)
    original=np.full((256,1),.3,dtype=np.float32)
    np.testing.assert_array_equal(processor.process(original.copy(),48000),original)
    processor.stop()


def test_ai_input_overflow_does_not_block_or_grow_queue():
    bridge=AIBridge(AIParameters(quality='low_latency'),FakeClient)
    processor=AIVoiceProcessor(bridge=bridge)
    processor.prepare(48000,256)
    bridge.status='Ready'; bridge.ack_generation=bridge.generation
    for _ in range(50*bridge.chunk_frames//256):
        processor.process(np.full((256,1),.2,dtype=np.float32),48000)
    assert bridge.input.available<=bridge.input.capacity
    assert bridge.overruns>0 and bridge.dropped_frames>0
    assert bridge.underruns==0  # no output ever primed; startup is separate


def test_underrun_output_remains_finite_and_counted():
    bridge=AIBridge(AIParameters(quality='low_latency'),FakeClient)
    processor=AIVoiceProcessor(bridge=bridge)
    processor.prepare(48000,256)
    bridge.status='Ready'; bridge.ack_generation=bridge.generation
    processor._epoch=bridge.generation
    processor._primed=True
    processor._last=.5
    block=processor.process(np.zeros((256,1),dtype=np.float32),48000)
    assert bridge.underruns==1
    assert np.isfinite(block).all() and block[0,0]>.4 and abs(block[-1,0])<.003


def test_three_mode_switches_reject_old_epoch_and_keep_finite():
    bridge=AIBridge(client_factory=FakeClient)
    router=VoiceRouter(FemaleDSPProcessor(load_preset('Anime Test')),AIVoiceProcessor(bridge=bridge))
    router.prepare(48000,256)
    try:
        for i in range(100):
            router.select(('original','female_dsp','ai_voice')[i%3])
            block=router.process(np.full((256,1),.05,dtype=np.float32),48000)
            assert np.isfinite(block).all() and np.max(abs(block))<=1
            time.sleep(.002)
        assert router.dsp._native.timing_snapshot()['count']<100
    finally:
        router.stop()
    assert not bridge.alive


def test_ai_route_never_invokes_signalsmith(monkeypatch):
    router=VoiceRouter(ai=AIVoiceProcessor(bridge=AIBridge(client_factory=FakeClient)))
    router.prepare(48000,256)
    def forbidden(*args): raise AssertionError('AI ran through Female DSP')
    monkeypatch.setattr(router.dsp,'process',forbidden)
    try:
        router.select('ai_voice')
        router.process(np.full((256,1),.1,dtype=np.float32),48000)
    finally: router.stop()


def test_old_settings_migrate_and_ai_persist(tmp_path):
    old=AppSettings.from_dict({'dsp_mode':'female_dsp','pitch_semitones':4})
    assert old.voice_mode=='female_dsp' and old.pitch_semitones==4
    new=AppSettings.from_dict({'voice_mode':'ai_voice','ai_quality':'stable','ai_threads':3,
        'ai_brightness':40,'ai_low_cut':False,'ai_post_fx':False})
    manager=SettingsManager(tmp_path/'settings.json')
    assert manager.save(new)
    assert manager.load().ai_parameters()==new.ai_parameters()
    assert manager.load().voice_mode=='ai_voice'


def test_invalid_ai_settings_migrate_safely():
    settings=AppSettings.from_dict({'voice_mode':[],'ai_quality':{},'ai_threads':True,'ai_brightness':float('inf')})
    assert settings.voice_mode=='ai_voice' and settings.ai_parameters()==AppSettings().ai_parameters()


def test_onnx_backend_contract_mock_load_reset_warmup_unload(tmp_path,monkeypatch):
    path=tmp_path/'graph.onnx'; path.write_bytes(b'mock')
    (tmp_path/'onnx.json').write_text(json.dumps({'graphs':{'1':{'filename':path.name,'sha256':hashlib.sha256(b'mock').hexdigest()}}}))
    class Session:
        def __init__(self,*args,**kwargs): self.calls=0
        def get_inputs(self): return [SimpleNamespace(name='audio',shape=[1,1,240]),SimpleNamespace(name='state',shape=[1,2])]
        def run(self,_,inputs): self.calls+=1; return [inputs['audio'][:,:,32:]*.5,inputs['state']+1]
    class Options:
        def add_session_config_entry(self,*args): pass
    monkeypatch.setitem(sys.modules,'onnxruntime',SimpleNamespace(SessionOptions=Options,InferenceSession=Session,__version__='mock'))
    backend=LLVCOnnxBackend()
    assert isinstance(backend,VoiceConversionBackend)
    backend.load(tmp_path); backend.warmup()
    assert backend.session.calls==12 and np.all(backend.state[0]==0)
    output=backend.process_chunk(np.ones(208,dtype=np.float32))
    assert np.all(output==.5) and output.dtype==np.float32
    backend.reset(); assert np.all(backend.state[0]==0)
    with pytest.raises(ValueError): backend.process_chunk(np.full(208,np.nan,dtype=np.float32))
    backend.unload()
    with pytest.raises(RuntimeError): backend.process_chunk(np.ones(208,dtype=np.float32))


def test_invalid_model_rejected_before_runtime_import(tmp_path):
    (tmp_path/'graph.onnx').write_bytes(b'wrong')
    (tmp_path/'onnx.json').write_text(json.dumps({'graphs':{'1':{'filename':'graph.onnx','sha256':'invalid'}}}))
    with pytest.raises(ValueError,match='checksum'): LLVCOnnxBackend().load(tmp_path)


def test_client_creation_error_is_reported_and_worker_exits():
    def broken(parameters):
        raise RuntimeError('Cannot create AI process')
    bridge = AIBridge(client_factory=broken)
    bridge.load()
    wait(lambda: bridge.status == 'Error')
    wait(lambda: not bridge.alive)
    assert 'Cannot create' in bridge.error
    bridge.stop()


@pytest.mark.parametrize('fail_body', [False, True])
def test_audio_scheduling_balances_cleanup(fail_body):
    from src.utils.windows import audio_scheduling
    calls = []
    avrt = SimpleNamespace(AvSetMmThreadCharacteristicsW=lambda name, index: calls.append(name) or 123,
                           AvRevertMmThreadCharacteristics=lambda handle: calls.append(('revert', handle)))
    winmm = SimpleNamespace(timeBeginPeriod=lambda period: calls.append(('begin', period)) or 0,
                            timeEndPeriod=lambda period: calls.append(('end', period)))
    try:
        with audio_scheduling(avrt, winmm) as result:
            assert result == {'mmcss': True, 'timer_1ms': True}
            if fail_body:
                raise ValueError('test cleanup')
    except ValueError:
        assert fail_body
    assert calls == ['Pro Audio', ('begin', 1), ('end', 1), ('revert', 123)]


def test_audio_scheduling_unavailable_is_not_fatal():
    from src.utils.windows import audio_scheduling
    avrt = SimpleNamespace(AvSetMmThreadCharacteristicsW=lambda *args: None)
    winmm = SimpleNamespace(timeBeginPeriod=lambda period: 1)
    with audio_scheduling(avrt, winmm) as result:
        assert result == {'mmcss': False, 'timer_1ms': False}


def test_ai_startup_keeps_original_until_output_is_ready():
    bridge = AIBridge(AIParameters(quality='low_latency'), FakeClient)
    processor = AIVoiceProcessor(bridge=bridge)
    processor.prepare(48000,256)
    bridge.status='Ready'; bridge.ack_generation=bridge.generation
    source = np.full((256,1), .3, dtype=np.float32)
    # Startup mutes, so the route does not pass the unconverted voice through.
    assert not processor.process(source.copy(),48000).any()
    assert bridge.underruns == 0
    assert bridge.output.write(np.full(bridge.parameters.startup_frames-1, .1, dtype=np.float32))
    assert not processor.process(source.copy(),48000).any()
    assert not processor._primed and bridge.underruns == 0
    assert bridge.output.write(np.full(1, .1, dtype=np.float32))
    output = processor.process(source.copy(),48000)
    assert processor._primed
    assert output[0,0] < output[-1,0] <= .101  # ramps up into the buffered level


def test_ring_concurrent_producer_consumer_preserve_order():
    import threading
    ring = AudioRing(1024)
    result = []
    def producer():
        for i in range(200):
            block = np.arange(i*127,(i+1)*127,dtype=np.float32)
            until=time.monotonic()+3
            while not ring.write(block):
                if time.monotonic()>until:
                    return
                time.sleep(.0001)
    thread = threading.Thread(target=producer)
    thread.start()
    target = np.empty(127,dtype=np.float32)
    until=time.monotonic()+5
    while len(result)<200 and time.monotonic()<until:
        if ring.read_into(target):
            result.append(target.copy())
        else:
            time.sleep(.0001)
    thread.join(timeout=3)
    assert not thread.is_alive() and len(result)==200
    np.testing.assert_array_equal(np.concatenate(result), np.arange(25400,dtype=np.float32))
    assert ring.maximum_depth <= ring.capacity


def test_ai_incidents_are_bounded_and_snapshot_in_chronological_order():
    bridge=AIBridge(client_factory=FakeClient)
    for i in range(140):
        bridge.generation=i
        bridge.record_incident(1)
    snapshot=bridge.snapshot()
    assert snapshot['incident_count']==140 and len(snapshot['incidents'])==128
    assert snapshot['incidents'][0][5]==12 and snapshot['incidents'][-1][5]==139
    assert np.isfinite(snapshot['incidents']).all()


def test_client_cleanup_error_is_reported_and_reference_released():
    class BadClose(FakeClient):
        def close(self):raise RuntimeError('mock close failed')
    bridge=AIBridge(client_factory=BadClose)
    bridge.load()
    wait(lambda: bridge.status=='Ready')
    bridge.stop()
    assert not bridge.alive and bridge.client is None
    assert 'cleanup failed' in bridge.error


def test_hung_native_request_times_out_without_stopping_audio_engine():
    import threading
    class HungClient(FakeClient):
        def __init__(self,parameters):
            super().__init__(parameters)
            self.wake=threading.Event()
        def request(self,header,audio=None):
            self.wake.wait(3)
            raise EOFError('interrupted hung native request')
        def interrupt(self):
            self.wake.set()
            super().interrupt()
    bridge=AIBridge(client_factory=HungClient)
    bridge.rpc_timeout_seconds=.06
    processor=AIVoiceProcessor(bridge=bridge)
    processor.prepare(48000,256)
    bridge.set_active(True)
    bridge.load()
    wait(lambda: bridge.status=='Error')
    wait(lambda: not bridge.alive)
    source=np.full((256,1),.2,dtype=np.float32)
    np.testing.assert_array_equal(processor.process(source.copy(),48000),source)
    assert 'timed out' in bridge.error and bridge.client is None
    bridge.stop()
    assert not bridge.watchdog.is_alive()


def test_protocol_writes_complete_packet_once():
    class CountingStream(io.BytesIO):
        writes=0
        def write(self,data):
            self.writes+=1
            return super().write(data)
    stream=CountingStream()
    send(stream,{'status':'Ready'},np.ones(624,dtype=np.float32))
    assert stream.writes==1
    stream.seek(0)
    header,audio=receive(stream)
    assert header=={'status':'Ready'} and np.all(audio==1)


def test_partial_ai_output_is_played_before_missing_tail_ramp():
    bridge=AIBridge(client_factory=FakeClient)
    processor=AIVoiceProcessor(bridge=bridge)
    processor.prepare(48000,256)
    bridge.status='Ready';bridge.ack_generation=bridge.generation
    processor._epoch=bridge.generation;processor._primed=True
    bridge.output.write(np.full(128,.1,dtype=np.float32))
    result=processor.process(np.full((256,1),.3,dtype=np.float32),48000)
    np.testing.assert_allclose(result[:128],.1)
    assert 0<result[-1,0]<.01
    assert bridge.underruns==1 and bridge.missing_frames==128 and bridge.output.available==0


def test_initial_crossfade_keeps_dry_mix_across_two_128_frame_callbacks():
    bridge=AIBridge(client_factory=FakeClient)
    processor=AIVoiceProcessor(bridge=bridge)
    processor.prepare(48000,128)
    bridge.status='Ready';bridge.ack_generation=bridge.generation
    processor._epoch=bridge.generation
    bridge.output.write(np.full(bridge.parameters.startup_chunks*bridge.chunk_frames,.1,dtype=np.float32))
    first=processor.process(np.full((128,1),.3,dtype=np.float32),48000)
    second=processor.process(np.full((128,1),.3,dtype=np.float32),48000)
    assert abs(first[-1,0]-second[0,0])<.002
    assert second[-1,0]==pytest.approx(.1)


def test_public_processor_reset_advances_backend_epoch_and_rejects_old_output():
    bridge=AIBridge(client_factory=FakeClient)
    processor=AIVoiceProcessor(bridge=bridge)
    processor.prepare(48000,256)
    epoch=bridge.generation
    bridge.status='Ready';bridge.ack_generation=epoch
    processor.reset()
    assert bridge.generation==epoch+1 and not processor._primed
    source=np.full((256,1),.2,dtype=np.float32)
    # The new epoch has buffered nothing, so the route stays silent instead of
    # emitting output that belongs to the epoch that was just reset.
    assert not processor.process(source.copy(),48000).any()


def test_entering_dsp_clears_suspended_analysis_window_once(monkeypatch):
    router=VoiceRouter(FemaleDSPProcessor(load_preset('Anime Test')),
        AIVoiceProcessor(bridge=AIBridge(client_factory=FakeClient)))
    router.prepare(48000,256)
    resets=[]
    reset=router.dsp.reset
    def observed():
        resets.append(True)
        reset()
    monkeypatch.setattr(router.dsp,'reset',observed)
    try:
        router.select('female_dsp')
        for _ in range(2):router.process(np.full((256,1),.05,dtype=np.float32),48000)
        assert len(resets)==1
        router.select('original')
        router.process(np.zeros((256,1),dtype=np.float32),48000)
        router.select('female_dsp')
        router.process(np.zeros((256,1),dtype=np.float32),48000)
        assert len(resets)==2
    finally:router.stop()
