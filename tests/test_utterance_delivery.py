import io
import threading
import time
import numpy as np
import pytest
from src.vc.utterance import UtteranceCollector, convert_utterance
from src.vc.config import AIParameters
from src.vc.protocol import send,receive,MAX_PAYLOAD
from src.vc.bridge import AIBridge
from src.processors.ai_voice import AIVoiceProcessor
from src.processors.voice_router import VoiceRouter
from src.settings.manager import AppSettings
from src.vc.models import DEFAULT_VOICE_ID
from .test_ai_voice import FakeClient,wait

# This build offers one voice, so the tests name it through the model layer instead of
# hard-coding a profile id that no longer exists.
VOICE = DEFAULT_VOICE_ID


def test_endpoint_retains_onset_and_tail_and_never_emits_during_speech():
    collector=UtteranceCollector()
    assert collector.feed(np.zeros(48000,dtype=np.float32))==[]
    speech=np.linspace(.1,.2,48000,dtype=np.float32)
    assert collector.feed(speech)==[]
    assert collector.feed(np.zeros(24000,dtype=np.float32))==[]
    outputs=collector.feed(np.zeros(14400,dtype=np.float32))
    assert len(outputs)==1
    np.testing.assert_array_equal(outputs[0][9600:57600],speech)
    assert len(outputs[0])==67200  # 200ms pre-roll + speech + 200ms final pause


def test_silence_and_clicks_do_not_produce_utterances_and_limit_is_bounded():
    collector=UtteranceCollector()
    assert collector.feed(np.ones(960,dtype=np.float32)*.1)==[]
    assert collector.feed(np.zeros(48000,dtype=np.float32))==[]
    results=collector.feed(np.ones(48000*31,dtype=np.float32)*.1)
    assert len(results)==1 and len(results[0])<=48000*30
    assert collector.limit_splits==1
    assert collector.samples<=48000*2


def test_manual_finish_keeps_subframe_tail():
    collector=UtteranceCollector()
    audio=np.ones(48000+113,dtype=np.float32)*.1
    assert collector.feed(audio)==[]
    np.testing.assert_array_equal(collector.flush(),audio)


def test_continuation_flags_follow_cap_not_endpoint():
    collector=UtteranceCollector(max_seconds=1)
    assert collector.take_result_flags() == []
    speech=np.ones(24000,dtype=np.float32)*.1
    assert collector.feed(speech)==[]
    assert collector.take_result_flags() == []
    # Crossing the 1 s cap completes a segment that continues.
    capped=collector.feed(np.ones(28800,dtype=np.float32)*.1)
    assert len(capped)==1 and len(capped[0])==48000
    assert collector.take_result_flags()==[True]
    assert collector.limit_splits==1
    # Draining twice never duplicates or leaks flags.
    assert collector.take_result_flags()==[]
    # A short tail below the cap ends on silence: not a continuation.
    done=collector.feed(np.zeros(48000,dtype=np.float32))
    assert len(done)==1
    assert collector.take_result_flags()==[False]
    # Manual finish is never a continuation either.
    assert collector.feed(np.ones(4800,dtype=np.float32)*.1)==[]
    assert collector.flush() is not None
    assert collector.take_result_flags()==[False]
    # Reset clears undrained flags so nothing leaks across epochs.
    collector=UtteranceCollector(max_seconds=1)
    collector.feed(np.ones(48000*2,dtype=np.float32)*.1)
    collector.reset()
    assert collector.take_result_flags()==[]


def test_utterance_drain_removes_fixed_delay_and_flushes_complete_tail():
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
    audio=np.repeat(np.arange(16003,dtype=np.float32)/16003,3)
    result=convert_utterance(Backend(),Down(),Up(),audio)
    np.testing.assert_array_equal(result,audio)


def test_large_audio_is_only_allowed_for_bounded_utterance_packets():
    audio=np.ones(48000*25,dtype=np.float32)
    stream=io.BytesIO();send(stream,{'op':'utterance','status':'Ready'},audio);stream.seek(0)
    header,result=receive(stream);np.testing.assert_array_equal(result,audio)
    assert header['op']=='utterance'
    with pytest.raises(ValueError):send(io.BytesIO(),{'op':'process'},audio)
    with pytest.raises(ValueError):send(io.BytesIO(),{'op':'utterance'},np.zeros(MAX_PAYLOAD//4+1,dtype=np.float32))


def test_delivery_validation_and_settings_roundtrip():
    params=AIParameters(enhancer='lavasr')
    assert params.delivery=='utterance' and params.mute_during_startup
    assert AIParameters(enhancer='none').delivery=='streaming'
    values=AppSettings.from_dict({'ai_model':VOICE,'ai_delivery':'utterance','ai_enhancer':'lavasr'})
    assert values.ai_parameters().enhancer=='lavasr'
    assert values.ai_parameters().lavasr_denoise is True
    assert AppSettings().ai_lavasr_denoise is True
    assert AppSettings.from_dict({'ai_lavasr_denoise':'yes'}).ai_lavasr_denoise is True
    # A model this build removed falls back to the one voice it ships.
    stale=AppSettings.from_dict({'ai_model':'meanvc2_ref20'})
    assert stale.ai_model==VOICE
    with pytest.raises(ValueError):AIParameters(delivery='broken')


def test_lavasr_is_a_valid_utterance_enhancer_with_a_60s_limit():
    from src.vc.utterance import MAX_SECONDS, QUALITY_MAX_SECONDS, utterance_limit
    params=AIParameters(enhancer='lavasr')
    assert params.delivery=='utterance'
    assert AIParameters(enhancer='none').delivery=='streaming'
    values=AppSettings.from_dict({'ai_model':VOICE,'ai_delivery':'streaming','ai_enhancer':'lavasr'})
    assert values.ai_delivery=='utterance' and values.ai_parameters().enhancer=='lavasr'
    assert values.ai_parameters().lavasr_denoise is True
    # LavaSR is measured at RTF 0.043-0.064 and 1.15 GiB peak, so it may take the
    # long utterance. The short limit applies to the route with no restoration.
    assert utterance_limit('lavasr')==QUALITY_MAX_SECONDS==60
    assert utterance_limit('none')==MAX_SECONDS
    with pytest.raises(ValueError):AIParameters(enhancer='not_a_model')


def test_lavasr_needs_the_long_form_ring_capacity():
    """A 60 s utterance must fit one input ring and one output ring write."""
    from src.vc.bridge import AIBridge
    bridge=AIBridge(AIParameters(enhancer='lavasr'))
    try:
        assert bridge.input.capacity>=48000*60
        assert bridge.output.capacity>=48000*60
    finally:
        bridge.close() if hasattr(bridge,'close') else None


def test_bridge_captures_during_inference_and_rejects_full_queue_without_blocking():
    gate=threading.Event();entered=threading.Event()
    class Slow(FakeClient):
        def request(self,header,audio=None):
            if header['op']=='utterance':entered.set();gate.wait(3)
            return super().request(header,audio)
    bridge=AIBridge(AIParameters(delivery='utterance',enhancer='lavasr'),Slow)
    def utter():
        assert bridge.input.write(np.ones(48000,dtype=np.float32)*.1)
        assert bridge.input.write(np.zeros(48000,dtype=np.float32))
        bridge.input_ready.signal()
        wait(lambda:bridge.input.available<bridge.chunk_frames)
    try:
        bridge.load();bridge.set_active(True)
        wait(lambda:bridge.ack_generation==bridge.generation)
        utter();assert entered.wait(1)
        assert bridge.output.available==0
        utter();utter()
        wait(lambda:bridge.utterance_stats['rejected']==1)
        assert bridge.utterance_stats['queued']==1
        gate.set();wait(lambda:bridge.utterance_stats['completed']==2)
        assert bridge.output.available>48000
    finally:gate.set();bridge.stop()
    assert not bridge.alive and not bridge.utterance_thread.is_alive()


def test_reset_discards_late_sentence_and_manual_finish_without_pause():
    gate=threading.Event();entered=threading.Event()
    class Slow(FakeClient):
        def request(self,header,audio=None):
            if header['op']=='utterance':entered.set();gate.wait(3)
            return super().request(header,audio)
    bridge=AIBridge(AIParameters(delivery='utterance',enhancer='lavasr'),Slow)
    try:
        bridge.load();bridge.set_active(True);wait(lambda:bridge.ack_generation==bridge.generation)
        bridge.input.write(np.ones(48000,dtype=np.float32)*.1);bridge.input_ready.signal()
        wait(lambda:bridge.input.available<bridge.chunk_frames)
        bridge.finish_utterance.set();bridge.input_ready.signal();assert entered.wait(1)
        bridge.set_active(False);gate.set();time.sleep(.05)
        assert bridge.output.available==0
    finally:gate.set();bridge.stop()


def test_callback_wait_is_intentionally_silent_and_not_an_underrun():
    bridge=AIBridge(AIParameters(delivery='utterance',enhancer='lavasr'),FakeClient)
    processor=AIVoiceProcessor(bridge=bridge);processor.prepare(48000,256)
    bridge.status='Ready';bridge.ack_generation=bridge.generation
    audio=np.ones((256,1),dtype=np.float32)*.4
    processor.process(audio,48000)
    assert np.max(abs(audio))==0 and bridge.underruns==0
    bridge.status='Error';processor.process(audio.fill(.4) or audio,48000)
    assert np.max(abs(audio))==0


def test_router_enters_sentence_route_before_output_and_never_leaks_raw_on_error():
    class DSP:
        def prepare(self,*args):pass
        def reset(self):pass
        def process(self,audio,*args):return audio
    bridge=AIBridge(AIParameters(delivery='utterance',enhancer='lavasr'),FakeClient)
    processor=AIVoiceProcessor(bridge=bridge)
    router=VoiceRouter(dsp=DSP(),ai=processor,mode='ai_voice')
    try:
        bridge.load();wait(lambda:bridge.status=='Ready')
        router.prepare(48000,256)
        for _ in range(8):
            block=np.ones((256,1),dtype=np.float32)*.2;router.process(block,48000)
            assert np.max(abs(block))==0
        assert router._current=='ai_voice'
        bridge.utterance_output_generation=bridge.generation
        assert bridge.output.write(np.ones(1024,dtype=np.float32)*.3)
        block=np.ones((256,1),dtype=np.float32)*.2;router.process(block,48000)
        assert np.max(block)>.2
        bridge.status='Error'
        block.fill(.2);router.process(block,48000)
        assert np.max(abs(block))==0
    finally:bridge.stop()


def _feed_stream(bridge, samples, chunk=None):
    """Paced writes that never overflow the input ring."""
    size = chunk or bridge.chunk_frames
    for start in range(0, len(samples), size):
        while bridge.input.available > bridge.input.capacity - 2*size:
            time.sleep(.002)
        assert bridge.input.write(samples[start:start+size])
        bridge.input_ready.signal()
        time.sleep(.001)


def test_finished_utterance_emits_whole_without_waiting_for_next_speech():
    from src.vc.config import AIParameters as P
    bridge=AIBridge(P(delivery='utterance',enhancer='lavasr',experiment='natural'),FakeClient)
    try:
        bridge.load();bridge.set_active(True)
        wait(lambda:bridge.ack_generation==bridge.generation)
        voiced = np.ones(12*bridge.chunk_frames,dtype=np.float32)*.3
        _feed_stream(bridge, np.concatenate((voiced,np.zeros(7*bridge.chunk_frames,dtype=np.float32))))
        wait(lambda:bridge.utterance_stats.get('completed',0)>=1,seconds=10)
        # 1.92 s voiced + 0.5 s silence to endpoint, minus 0.4 s kept tail.
        expected = len(voiced)+25*960-5*960
        assert bridge.output.available==expected
        out=np.empty(expected,dtype=np.float32)
        assert bridge.output.read_into(out)
        np.testing.assert_array_equal(out, np.concatenate((voiced,np.zeros(20*960,dtype=np.float32)))*.5)
        assert bridge.utterance_stats.get('rejected',0)==0
    finally:bridge.stop()
    assert not bridge.alive


def test_capped_utterance_still_holds_tail_for_its_continuation():
    from src.vc.config import AIParameters as P
    from src.vc.utterance import crossfade_blend
    bridge=AIBridge(P(delivery='utterance',enhancer='lavasr',experiment='natural'),FakeClient)
    try:
        bridge.load();bridge.set_active(True)
        wait(lambda:bridge.ack_generation==bridge.generation)
        overlap = int(0.3*48000)
        cap = 60*48000
        _feed_stream(bridge, np.ones(cap+48000,dtype=np.float32)*.3)
        _feed_stream(bridge, np.zeros(7*bridge.chunk_frames,dtype=np.float32))
        wait(lambda:bridge.utterance_stats.get('completed',0)>=1,seconds=15)
        # The capped segment holds its tail for the continuation.
        assert bridge.output.available==cap-overlap
        first=np.empty(cap-overlap,dtype=np.float32)
        assert bridge.output.read_into(first)
        wait(lambda:bridge.utterance_stats.get('completed',0)>=2,seconds=15)
        # Remainder voiced + endpoint silence, minus the kept tail.
        seg2len = 48000+25*960-5*960
        assert bridge.output.available==seg2len
        second=np.empty(seg2len,dtype=np.float32)
        assert bridge.output.read_into(second)
        result1=np.ones(cap,dtype=np.float32)*.15
        result2=np.concatenate((np.ones(48000,dtype=np.float32)*.15,
                                np.zeros(seg2len-48000,dtype=np.float32)))
        np.testing.assert_array_equal(first, result1[:-overlap])
        np.testing.assert_array_equal(second, crossfade_blend(result1[-overlap:],result2,overlap))
        assert bridge.utterance_stats.get('rejected',0)==0
    finally:bridge.stop()
    assert not bridge.alive
