from dataclasses import asdict
import json
from threading import Event
import time
import numpy as np
import pytest

from src.audio.performance import TimingStats
from src.processors.ai_voice import AIVoiceProcessor
from src.processors.voice_router import VoiceRouter
from src.settings.manager import AppSettings,SettingsManager
from src.vc.bridge import AIBridge
from src.vc.config import AIParameters
from src.vc.diagnostics import startup_diagnostics
from src.vc.models import DEFAULT_VOICE_ID, profile
from src.vc.health import realtime_health
from src.vc.ring import AudioRing
from src.utils.background_files import BackgroundFiles
from .test_ai_voice import FakeClient,wait


def test_percentiles_include_tail_and_histogram_cap():
    stats=TimingStats()
    for n in range(1,101): stats.record(n*100000,256,48000)
    result=stats.snapshot()
    assert result.p50_ms==pytest.approx(5.01)
    assert result.p95_ms==pytest.approx(9.51)
    assert result.p99_ms==pytest.approx(9.91)
    assert result.average_ms==pytest.approx(5.05)
    stats.record(200000000,256,48000)
    assert stats.snapshot().maximum_ms==200
    empty=TimingStats().snapshot()
    assert empty.p50_ms==empty.p95_ms==empty.p99_ms==0


@pytest.mark.parametrize('wait_ms',[25,30,35,39,45])
def test_exact_preroll_target(wait_ms):
    # The shipped route buffers a fixed number of startup chunks, so the requested
    # pre-roll only ever raises that floor, never replaces it.
    p=AIParameters(output_wait_ms=wait_ms)
    assert p.startup_frames==max(wait_ms*48,p.startup_chunks*p.chunk_frames)
    assert p.startup_frames==p.startup_chunks*p.chunk_frames
    bridge=AIBridge(p,FakeClient)
    ai=AIVoiceProcessor(bridge=bridge)
    ai.prepare(48000,256)
    bridge.status='Ready';bridge.ack_generation=bridge.generation
    ai._epoch=bridge.generation
    assert bridge.output.write(np.full(p.startup_frames-1,.1,dtype=np.float32))
    source=np.full((256,1),.3,dtype=np.float32)
    # The shipped route mutes instead of passing the raw voice through, so the
    # speaker never hears the unconverted signal while the pre-roll fills.
    assert not ai.process(source.copy(),48000).any()
    assert not ai._primed
    assert bridge.output.write(np.full(1,.1,dtype=np.float32))
    ai.process(source.copy(),48000)
    assert ai._primed and bridge.underruns==0


@pytest.mark.parametrize('changes',[{'output_wait_ms':19},{'output_wait_ms':157},
    {'output_wait_ms':float('nan')},{'crossfade_ms':19},{'crossfade_ms':101},{'crossfade_ms':True}])
def test_core_parameters_reject_unbounded_values(changes):
    with pytest.raises(ValueError):AIParameters(**changes)


@pytest.mark.parametrize('wait_ms',[39,78,100])
def test_extended_preroll_has_fixed_bounded_capacity(wait_ms):
    # The queue is sized by the route's startup chunks, not by the requested pre-roll,
    # so a larger pre-roll setting cannot grow the backlog.
    p=AIParameters(quality='low_latency',output_wait_ms=wait_ms)
    reference=AIParameters(quality='low_latency',output_wait_ms=20)
    assert p.queue_chunks==reference.queue_chunks
    assert p.startup_frames+p.chunk_frames<=p.queue_chunks*p.chunk_frames
    restored=AppSettings.from_dict(dict(ai_output_wait_ms=wait_ms))
    assert restored.ai_output_wait_ms==wait_ms


def test_queue_wait_depth_highwater_and_reset():
    ring=AudioRing(100)
    ring.write(np.ones(80,dtype=np.float32))
    ring.write(np.ones(10,dtype=np.float32))
    assert ring.high_water_events==1
    ring.read_into(np.empty(50,dtype=np.float32))
    ring.write(np.ones(40,dtype=np.float32))
    snapshot=ring.snapshot(25)
    assert snapshot['high_water_events']==2
    assert snapshot['maximum']==3.6 and snapshot['current']==3.2
    assert 0<=snapshot['wait']['average_ms']<1000
    assert snapshot['wait']['count']==1
    assert snapshot['average']<=snapshot['maximum']
    ring.reset()
    assert ring.snapshot(25)['wait']['count']==0
    assert ring.high_water_events==0


def test_worker_backpressure_retains_playable_output_and_single_worker():
    bridge=AIBridge(client_factory=FakeClient)
    bridge.load();wait(lambda:bridge.status=='Ready')
    bridge.set_active(True);wait(lambda:bridge.ack_generation==bridge.generation)
    original_thread=bridge.thread
    chunk=bridge.chunk_frames
    full=bridge.output.capacity
    try:
        # Fill the output ring through the worker, then confirm one more input chunk
        # is retained rather than discarded while the consumer is stalled.
        while bridge.output.available<full:
            bridge.input.write(np.ones(chunk,dtype=np.float32))
            bridge.input_ready.signal()
            wait(lambda:bridge.input.available==0)
        wait(lambda:bridge.output.available==full)
        bridge.input.write(np.ones(chunk,dtype=np.float32))
        bridge.input_ready.signal()
        time.sleep(.035)  # longer than former one-chunk discard timeout
        assert bridge.input.available==chunk and bridge.output_overruns==0
        for _ in range(20):bridge.load()
        assert bridge.thread is original_thread
        bridge.output.read_into(np.empty(chunk,dtype=np.float32))
        wait(lambda:bridge.input.available==0)
        assert bridge.dropped_output_frames==0
        assert bridge.worker_state=='Running'
    finally:bridge.stop()
    assert bridge.worker_state=='Stopped' and not bridge.alive


def test_router_preroll_and_full_two_route_crossfade_then_clear():
    bridge=AIBridge(client_factory=FakeClient)
    router=VoiceRouter(ai=AIVoiceProcessor(bridge=bridge))
    router.prepare(48000,256)
    # Use callback-ready manual worker state for deterministic signal checks.
    router.mode='ai_voice';bridge.set_active(True)
    bridge.status='Ready';bridge.ack_generation=bridge.generation
    source=np.full((256,1),.3,dtype=np.float32)
    assert not router.process(source.copy(),48000).any()
    assert router._current=='original'
    # The pre-roll target decides when the route may start emitting, so fill exactly that.
    bridge.output.write(np.full(bridge.parameters.startup_frames,.1,dtype=np.float32))
    # Startup mutes and ramps in over 240 frames, so the first block is near silence
    # rather than the dry input, and reaches the buffered level by the end of it.
    blocks=[router.process(source.copy(),48000) for _ in range(8)]
    first=blocks[0]
    assert abs(first[0,0])<=1e-6
    # The startup ramp and the router crossfade both run in blocks, so the level rises
    # monotonically and then settles on the buffered signal.
    assert all(np.all(b>=a) for a,b in zip(blocks,blocks[1:]))
    assert .09<blocks[-1][-1,0]<=.1
    assert router._current=='ai_voice'
    router.select('original')
    assert bridge.active  # old AI stays available through fade
    for _ in range(4):result=router.process(source.copy(),48000)
    assert router._current=='original' and not bridge.active
    assert bridge.output.available==0
    assert result[-1,0]==pytest.approx(.3)
    assert bridge.crossfade_count==2


def test_crash_fallback_crossfades_and_can_reload_without_app_restart():
    bridge=AIBridge(client_factory=FakeClient)
    router=VoiceRouter(ai=AIVoiceProcessor(bridge=bridge))
    router.prepare(48000,256)
    router.mode='ai_voice';router._current=router._target='ai_voice';router._last=.1
    bridge.status='Error';bridge.worker_state='Error'
    source=np.full((256,1),.3,dtype=np.float32)
    first=router.process(source.copy(),48000)
    assert .1<first[0,0]<.101
    for _ in range(3):router.process(source.copy(),48000)
    assert router._current=='original'
    router.select('ai_voice');wait(lambda:bridge.status=='Ready')
    bridge.stop()
    assert not bridge.alive


def test_watchdog_detects_alive_but_no_queue_progress():
    bridge=AIBridge(client_factory=FakeClient)
    bridge.rpc_timeout_seconds=.06
    bridge.load();wait(lambda:bridge.status=='Ready')
    bridge.set_active(True);wait(lambda:bridge.ack_generation==bridge.generation)
    # Full output plus pending input models a stalled consumer/queue.
    bridge.output.write(np.ones(bridge.output.capacity,dtype=np.float32))
    bridge.input.write(np.ones(bridge.chunk_frames,dtype=np.float32))
    bridge.input_ready.signal()
    wait(lambda:bridge.status=='Error',seconds=10)
    wait(lambda:not bridge.alive)
    assert bridge.error=='AI stalled' and bridge.worker_state=='Error'
    bridge.stop()


@pytest.mark.parametrize('rtf,drop,queue,expected',[(.3,0,1,'Excellent'),(.6,0,1,'Good'),
    (.8,0,1,'Warning'),(.3,0,3.5,'Warning'),(1.1,0,0,'Critical'),(.3,1,0,'Critical')])
def test_realtime_health(rtf,drop,queue,expected):
    assert realtime_health(dict(status='Ready',rtf=rtf,dropped_chunks=drop,
        inference={'count':100},queue_current=queue,queue_capacity=4))==expected


def test_v03_settings_migration_new_fields_and_corruption(tmp_path):
    old=AppSettings.from_dict({'voice_mode':'ai_voice','ai_brightness':47})
    assert old.ai_output_wait_ms==39 and old.ai_crossfade_ms==20
    manager=SettingsManager(tmp_path/'settings.json')
    settings=AppSettings.from_dict(dict(ai_output_wait_ms=30,ai_crossfade_ms=40))
    manager.save(settings)
    assert manager.load().ai_parameters().output_wait_ms==30
    manager.path.write_text('{broken')
    assert manager.load()==AppSettings()
    invalid=AppSettings.from_dict(dict(ai_output_wait_ms=1000,ai_crossfade_ms=float('nan')))
    assert invalid.ai_output_wait_ms==39 and invalid.ai_crossfade_ms==20


@pytest.mark.parametrize('content',[None,'{broken','[]','null',json.dumps({'backend':'invalid'})])
def test_missing_invalid_model_diagnostics_are_safe(tmp_path,content):
    # Point the check at a voice whose assets are absent, so it reports unavailable
    # instead of raising on a folder the test controls.
    folder=tmp_path/'models'/profile(DEFAULT_VOICE_ID)['folder'];folder.mkdir(parents=True)
    if content is not None:(folder/'runtime.json').write_text(content)
    result=startup_diagnostics(tmp_path)
    assert not result['ai_available'] and result['issues']


def test_background_file_jobs_coalesce_and_drain_without_gui_wait():
    files=BackgroundFiles();entered=Event();release=Event();results=[]
    def blocked():entered.set();release.wait(2)
    try:
        files.submit('settings',blocked);assert entered.wait(1)
        for n in range(20):assert files.submit('settings',lambda n=n:results.append(n))
        assert files.submit('snapshot',lambda:results.append('snapshot'))
        assert not files.submit('third',lambda:None)
        files.close();release.set()
        wait(lambda:not files.alive)
        assert results==[19,'snapshot']
    finally:release.set();files.close()


@pytest.mark.parametrize('target',[3,5,-12,12])
def test_native_pitch_changes_are_worker_side_steps_without_model_reload(target):
    from types import SimpleNamespace
    from src.vc.beatrice_vst import BeatriceVSTBackend
    backend=BeatriceVSTBackend()
    backend.plugin=SimpleNamespace(pitch_shift_st=4)
    backend.stats={'pitch':4}
    backend.set_pitch(target)
    assert abs(backend.stats['pitch']-4)==.125
    for _ in range(200):backend.set_pitch(target)
    assert backend.stats['pitch']==target and backend.plugin.pitch_shift_st==target


@pytest.mark.parametrize('target',[float('nan'),float('inf'),True,13])
def test_native_pitch_invalid_parameter_rejected(target):
    from src.vc.beatrice_vst import BeatriceVSTBackend
    with pytest.raises(ValueError):BeatriceVSTBackend().set_pitch(target)


def test_ai_selection_during_old_fade_cleanup_reconciles_callback_activity():
    bridge=AIBridge(client_factory=FakeClient)
    router=VoiceRouter(ai=AIVoiceProcessor(bridge=bridge))
    router.prepare(48000,256)
    # GUI selected AI while the previous Original departure still completed.
    router.mode='ai_voice';bridge.status='Ready';bridge.active=False
    bridge.ack_generation=bridge.generation
    source=np.full((256,1),.3,dtype=np.float32)
    # The route has not buffered yet, so it stays silent rather than passing the source.
    assert not router.process(source.copy(),48000).any()
    assert bridge.active and bridge.ack_generation!=bridge.generation
    assert bridge.input.available==0  # never feed an inactive/old epoch


def test_cancelled_ai_preroll_releases_activity_without_a_chunk_drop():
    bridge=AIBridge(client_factory=FakeClient)
    router=VoiceRouter(ai=AIVoiceProcessor(bridge=bridge))
    router.prepare(48000,256)
    bridge.status='Ready';bridge.set_active(True);bridge.ack_generation=bridge.generation
    router.mode='ai_voice'
    router.process(np.full((256,1),.3,dtype=np.float32),48000)
    assert router._current=='original' and bridge.active
    router.mode='original'
    router.process(np.full((256,1),.3,dtype=np.float32),48000)
    assert not bridge.active and bridge.output.available==0
    assert bridge.overruns==bridge.output_overruns==0


def test_dsp_preroll_keeps_original_until_analysis_window_is_ready():
    from src.processors.base import AudioProcessor
    class DelayedDSP(AudioProcessor):
        algorithmic_latency_samples=1024
        def prepare(self,*args):self.frames=0
        def reset(self):self.frames=0
        def process(self,audio,rate):
            self.frames+=len(audio)
            audio.fill(.1 if self.frames>1024 else 0)
            return audio
    router=VoiceRouter(dsp=DelayedDSP(),ai=AIVoiceProcessor(bridge=AIBridge(client_factory=FakeClient)))
    router.prepare(48000,256)
    router.select('female_dsp')
    source=np.full((256,1),.3,dtype=np.float32)
    for _ in range(4):np.testing.assert_array_equal(router.process(source.copy(),48000),source)
    assert router._current=='original'
    first=router.process(source.copy(),48000)
    assert .299<first[0,0]<.3
    for _ in range(3):last=router.process(source.copy(),48000)
    assert router._current=='female_dsp' and last[-1,0]==pytest.approx(.1)


def test_error_retry_without_stop_advances_epoch_and_discards_old_voice():
    bridge=AIBridge(client_factory=FakeClient)
    ai=AIVoiceProcessor(bridge=bridge);ai.prepare(48000,256)
    bridge.load();bridge.set_active(True)
    wait(lambda:bridge.status=='Ready' and bridge.ack_generation==bridge.generation)
    bridge.client.bad=True
    bridge.input.write(np.ones(bridge.chunk_frames,dtype=np.float32));bridge.input_ready.signal()
    wait(lambda:bridge.status=='Error' and not bridge.alive,seconds=10)
    old=bridge.generation;ai._epoch=old;ai._primed=True
    bridge.output.write(np.full(bridge.chunk_frames,.75,dtype=np.float32))
    try:
        bridge.load()
        wait(lambda:bridge.status=='Ready' and bridge.ack_generation==bridge.generation,seconds=10)
        assert bridge.generation>old
        source=np.full((256,1),.2,dtype=np.float32)
        # The restarted epoch has buffered nothing, so the old voice is discarded.
        assert not ai.process(source.copy(),48000).any()
        assert bridge.output.available==0 and not ai._primed
    finally:bridge.stop()


@pytest.mark.parametrize('enabled',[True,False])
def test_disposable_worker_gc_policy_restores_state_after_error(enabled):
    import gc
    from src.vc.realtime_gc import RealtimeGC
    original=gc.isenabled()
    try:
        gc.enable() if enabled else gc.disable()
        with pytest.raises(RuntimeError):
            with RealtimeGC():
                assert not gc.isenabled()
                raise RuntimeError('worker failed')
        assert gc.isenabled()==enabled
    finally:
        gc.enable() if original else gc.disable()
