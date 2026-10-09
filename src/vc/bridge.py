"""Callback-safe rings -> AI bridge thread -> isolated inference subprocess."""
from dataclasses import asdict
import logging
import math
import threading
import time

from .models import is_meanvc2
import numpy as np
from src.audio.performance import TimingStats
from src.utils.windows import audio_scheduling
from .client import ServiceClient
from .config import AIParameters
from .ring import AudioRing
from .wakeup import InputWakeup

log=logging.getLogger(__name__)


class AIBridge:
    rpc_timeout_seconds=1.0
    load_timeout_seconds=30.0

    def __init__(self, parameters=None, client_factory=ServiceClient):
        self.parameters = parameters or AIParameters()
        self.client_factory = client_factory
        self.client = None
        self.thread = None
        self.watchdog = None
        self._request_started = self._load_started = 0
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self.input_ready=InputWakeup()
        self._idle = threading.Event()
        self._idle.set()
        self.active = False
        self.generation = 0
        self.ack_generation = -1
        self.status = 'Not Loaded'
        self.error = ''
        self.worker_state = 'Not Started'
        self.last_response = self.last_output = 0.0
        self._pending_since = 0.0
        self.crossfade_count = 0
        self.stats = {}
        self.model_stats = {}
        self.finish_utterance = threading.Event()
        self.utterance_stats = {}
        self.utterance_thread = None
        self._request_timeout = 0.
        self.utterance_output_generation = -1
        from src.prosody.bridge import ProsodyBridge
        self.prosody = ProsodyBridge()
        self._allocate()

    def _allocate(self):
        self.load_timeout_seconds=120.0 if is_meanvc2(self.parameters.model) else 30.0
        self.rpc_timeout_seconds=2.0 if is_meanvc2(self.parameters.model) else 1.0
        self.chunk_frames = self.parameters.chunk_frames
        capacity = self.chunk_frames*self.parameters.queue_chunks
        large = self.parameters.delivery=='utterance' and self.parameters.enhancer in ('mossformer','voicefixer','lavasr')
        self.input = AudioRing(48000*60 if large else 48000*3 if self.parameters.delivery=='utterance' else capacity)
        # A finished utterance must fit the output ring in ONE write; the ring
        # rejects partial writes. 60 s quality-mode results need 61 s of room.
        self.output = AudioRing(48000*61 if large else 48000*32 if self.parameters.delivery=='utterance' else capacity)
        self._chunk = np.empty(self.chunk_frames, dtype=np.float32)
        self.reset_counters()

    def reset_counters(self):
        self.underruns = self.overruns = self.output_overruns = 0
        self.dropped_frames = self.dropped_output_frames = 0
        self.missing_frames=0
        self.preroll_observed_frames=0
        self.rpc_time = TimingStats()
        self.reset_time = TimingStats()
        self.incidents = np.zeros((128, 7), dtype=np.float64)
        self.incident_count = 0

    def record_incident(self, kind):
        """Bounded callback diagnostics; format/write them only off callback."""
        row = self.incidents[self.incident_count % len(self.incidents)]
        row[0] = time.monotonic()
        row[1] = kind  # 1=output underrun, 2=input overrun
        row[2] = self.input.available/self.chunk_frames
        row[3] = self.output.available/self.chunk_frames
        row[4] = self.rpc_time.last_ns/1e6
        row[5] = self.generation
        row[6] = self.stats.get('inference', {}).get('last_ms', 0)
        self.incident_count += 1

    @property
    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def configure(self, parameters):
        if not isinstance(parameters, AIParameters):
            raise TypeError('Expected validated AI parameters')
        structural = (parameters.quality, parameters.threads, parameters.model, parameters.queue_chunks,parameters.delivery,parameters.enhancer,parameters.lavasr_denoise,parameters.experiment) != (
                      self.parameters.quality, self.parameters.threads, self.parameters.model, self.parameters.queue_chunks,self.parameters.delivery,self.parameters.enhancer,self.parameters.lavasr_denoise,self.parameters.experiment)
        if structural:
            self.stop()
            self.parameters = parameters
            self._allocate()
        else:
            self.parameters = parameters  # immutable publish; no callback lock

    def load(self):
        if self.parameters.model is None:
            self.status, self.error = 'Error', '声が登録されていません。右の声パネルから声を登録してください。'
            self.worker_state = 'Error'
            return
        if not is_meanvc2(self.parameters.model):self.prosody.load()
        with self._lifecycle_lock:
            if self.alive:
                return
            # A fresh process is always a fresh neural epoch, including retry
            # after Error without an intervening Stop. Never replay old output.
            self.generation += 1
            self.ack_generation = -1
            self._stop.clear()
            self.status, self.error = 'Loading', ''
            self.worker_state = 'Starting'
            self.last_response = self.last_output = self._pending_since = 0.0
            self.stats, self.model_stats = {}, {}
            try:
                self.input_ready.open()
            except OSError as error:
                self.status,self.error='Error',str(error)
                self.worker_state = 'Error'
                log.exception('Cannot initialize AI input notification')
                return
            self.thread = threading.Thread(target=self._run, name='ai-bridge', daemon=True)
            self.thread.start()

    def set_active(self, enabled):
        self.prosody.set_active(enabled and not is_meanvc2(self.parameters.model))
        if self.active != enabled:
            self.active = enabled
            self.generation += 1
            self.input_ready.signal()

    def prepare(self):
        self.prosody.set_active(False)
        was_active = self.active
        self.active = False
        self.generation += 1
        self._idle.clear()
        self.input_ready.signal()
        if self.alive and self.status == 'Ready' and not self._idle.wait(2):
            raise RuntimeError('AI worker did not reset safely')
        # No callback is running yet. The bridge is idle or still loading.
        self.input.reset(); self.output.reset(); self.reset_counters()
        self.stats = {}
        self.active = was_active

    def activate_prepared(self):
        """Control thread only; acknowledge a fresh input epoch before streaming."""
        self.set_active(True)
        until=time.monotonic()+2
        while self.ack_generation!=self.generation:
            if self.status=='Error' or self._stop.is_set() or time.monotonic()>=until:
                raise RuntimeError('AI start epoch did not reset safely')
            self._stop.wait(.001)

    def stop(self):
        try:
            self.prosody.stop()
        except Exception:
            # An analysis cleanup fault must not prevent the voice process
            # from being stopped during USB/error/window teardown.
            log.exception('Prosody stop failed; continuing AI worker cleanup')
        self.worker_state = 'Stopping'
        self.active = False
        self.generation += 1
        self._stop.set()
        self.input_ready.signal()
        client = self.client
        if client is not None:
            client.interrupt()  # wakes even a stalled model/pipe
        if self.alive and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
        if self.alive:
            self.status, self.error = 'Error', 'AI worker did not stop'
            raise RuntimeError(self.error)
        if self.watchdog is not None and self.watchdog is not threading.current_thread():
            self.watchdog.join(timeout=.2)
        self.status = 'Not Loaded'
        self.ack_generation = -1
        self.worker_state = 'Stopped' if not self.error else 'Error'
        log.info('AI worker stop: %s', self.worker_state)

    def _run(self):
        with audio_scheduling() as scheduling:
            self.scheduling = scheduling
            self._run_scheduled()

    def _run_scheduled(self):
        client = None
        try:
            client = self.client_factory(self.parameters)
            self.client = client
            self._load_started=time.monotonic()
            self.watchdog=threading.Thread(target=self._watch_deadlines,name='ai-watchdog',daemon=True)
            self.watchdog.start()
            client.start()
            log.info('AI worker start')
            while not self._stop.is_set():
                header, _ = client.read()
                self.status = header['status']
                log.info('AI model status: %s', self.status)
                if self.status == 'Ready':
                    self.worker_state = 'Ready'
                    self.last_response = time.monotonic()
                    self.model_stats = header.get('model', {})
                    self._load_started=0
                    break
            if self.parameters.delivery=='utterance':
                self._run_utterances()
                return
            seen_generation, seen_overrun = -1, self.overruns
            logged_high = logged_fades = 0
            next_log = 0.0
            while not self._stop.is_set():
                generation = self.generation
                if generation != seen_generation:
                    self.input.discard()
                    self._request({'op': 'reset'})
                    seen_generation = generation
                    self.ack_generation = generation
                if not self.active:
                    self.worker_state = 'Ready'
                    self._idle.set()
                    self._stop.wait(.002)
                    continue
                self._idle.clear()
                self.worker_state = 'Running'
                # Reserve output space BEFORE consuming another input chunk.
                # SPSC: only this producer can occupy it, only callback frees it.
                # Waiting after inference could drop a playable chunk when the
                # mode consumer paused. Bounded rings still cap total backlog.
                if self.output.capacity-self.output.available < self.chunk_frames:
                    # The callback already signals this event when input is
                    # available. It also consumes output in the same callback.
                    # Do not poll with a 1 ms sleep: Windows may round that wait
                    # to a timer tick when the GUI is occluded/backgrounded.
                    self.input_ready.wait(self._stop)
                    continue
                depth = self.input.available
                # A transient three-chunk burst can be drained faster than
                # realtime. Drop only on a full queue or a producer overrun,
                # rather than resetting the neural state at every such burst.
                drop = (max(0, depth-2*self.chunk_frames)
                        if (not is_meanvc2(self.parameters.model) and depth >= self.input.capacity) or self.overruns != seen_overrun else 0)
                if drop or self.overruns != seen_overrun:
                    self.dropped_frames += self.input.discard(drop)
                    # Preserve MeanVC2 state when skipping a live hop. Reset
                    # would insert its 480ms pre-roll and cascade dropouts.
                    if not is_meanvc2(self.parameters.model):self._request({'op': 'reset'})
                    seen_overrun = self.overruns
                if not self.input.read_into(self._chunk):
                    self.input_ready.wait(self._stop)
                    continue
                parameters = self.parameters
                delta_pitch, dynamic_gain = self.prosody.control() if not is_meanvc2(parameters.model) else (0.,0.)
                pitch=parameters.pitch+delta_pitch
                if self.prosody.parameters.enabled and self.prosody.status=='Running':
                    # Native parameter has 1/8-st steps. Clamp its quantized
                    # target INSIDE the requested delta bounds, not outside.
                    # v2 worker applies morphing preset bounds itself. Preserve
                    # that fade rather than abruptly applying new GUI bounds.
                    v2='quantized_pitch_delta' in self.prosody.stats.get('control',{})
                    lower=math.ceil((parameters.pitch+(-1.2 if v2 else self.prosody.parameters.min_pitch))*8)/8
                    upper=math.floor((parameters.pitch+(1.5 if v2 else self.prosody.parameters.max_pitch))*8)/8
                    pitch=max(lower,min(upper,round(pitch*8)/8))
                started = time.perf_counter_ns()
                header, audio = self._request(dict(op='process', brightness=parameters.brightness,
                    low_cut=parameters.low_cut, limiter=parameters.limiter, post_fx=parameters.post_fx,
                    pitch=max(-12.,min(12.,pitch)),
                    dynamic_gain_db=dynamic_gain), self._chunk)
                self.rpc_time.record(time.perf_counter_ns()-started, self.chunk_frames, 48000)
                if 'stats' in header:
                    self.stats = header['stats']
                if len(audio) != self.chunk_frames or not np.isfinite(audio).all():
                    raise RuntimeError('AI worker returned invalid audio')
                # Reject late results from an earlier mode/Start session.
                if generation != self.generation or not self.active:
                    continue
                # A fast catch-up burst must not discard playable output merely
                # because the callback has not consumed it yet. Wait only here,
                # bounded by one chunk; the callback never waits for this worker.
                until = time.monotonic()+self.chunk_frames/48000
                written = self.output.write(audio)
                while not written and generation == self.generation and self.active and not self._stop.is_set() and time.monotonic() < until:
                    self.input_ready.wait(self._stop)
                    written = self.output.write(audio)
                if not written and generation == self.generation and self.active:
                    self.output_overruns += 1
                    self.dropped_output_frames += len(audio)
                if written:
                    self.last_output = time.monotonic()
                if time.monotonic() >= next_log:
                    high = self.input.high_water_events+self.output.high_water_events
                    if high != logged_high:
                        log.info('Queue high-water events: %s', high)
                        logged_high = high
                    if self.crossfade_count != logged_fades:
                        log.info('Mode crossfade count: %s', self.crossfade_count)
                        logged_fades = self.crossfade_count
                    next_log = time.monotonic()+1
        except Exception as error:
            if not self._stop.is_set():
                self.status, self.error = 'Error', str(error)
                self.worker_state = 'Error'
                log.exception('AI Voice unavailable; Fallback: Original')
        finally:
            self._stop.set()
            self._idle.set()
            try:
                if client is not None:
                    client.close()
            except Exception as error:
                self.status, self.error = 'Error', 'AI process cleanup failed: '+str(error)
                log.exception(self.error)
            finally:
                self.client = None
                if self.watchdog is not None and self.watchdog is not threading.current_thread():
                    self.watchdog.join(timeout=.2)
                self.input_ready.close()
                if self.status == 'Error':
                    self.worker_state = 'Error'

    def _request(self, header, audio=None):
        if header.get('op')=='utterance':
            # LavaSR is the shipped restoration model, measured at RTF 0.043-0.064.
            # Budget 8x that worst case on top of a 30 s load allowance, capped so a
            # full-length utterance cannot hold the worker open indefinitely.
            self._request_timeout=min(270.,30.+len(audio)/48000*8)
        else:
            self._request_timeout=self.rpc_timeout_seconds
        self._request_started=time.monotonic()
        started=time.perf_counter_ns()
        try:
            result = self.client.request(header,audio)
            self.last_response = time.monotonic()
            return result
        finally:
            if header.get('op')=='reset':
                self.reset_time.record(time.perf_counter_ns()-started,self.chunk_frames,48000)
            self._request_started=0

    def _run_utterances(self):
        import queue
        from .utterance import UtteranceCollector, crossfade_blend, utterance_limit
        if self.parameters.experiment in ('all', 'natural', 'lowdelay', 'fastest'):
            # Integrated/natural comparison: shorter segmentation pauses, a lower endpoint
            # threshold, and a longer kept tail, plus an overlap crossfade below.
            collector=UtteranceCollector(silence_ms=500, threshold_db=-42.0, keep_ms=400,
                                         max_seconds=utterance_limit(self.parameters.enhancer))
            overlap=int(0.3*48000)
        else:
            collector=UtteranceCollector(max_seconds=utterance_limit(self.parameters.enhancer))
            overlap=0
        jobs=queue.Queue(maxsize=1)
        self.finish_utterance.clear()
        self.utterance_stats=dict(state='発話待ち',recording_seconds=0.,queued=0,completed=0,
                                  rejected=0,rejected_seconds=0.,limit_splits=0,message='')

        def enqueue(audio,generation,continues=False):
            if audio is None:return
            try:jobs.put_nowait((generation,audio,continues))
            except queue.Full:
                self.utterance_stats['rejected']+=1
                self.utterance_stats['rejected_seconds']+=len(audio)/48000
                self.utterance_stats['message']='変換待ちが満杯のため、この発話は受け付けられませんでした。'
            self.utterance_stats['queued']=jobs.qsize()

        def infer():
            try:
                tail = None
                tail_generation = -1
                while not self._stop.is_set():
                    try:generation,audio,continues=jobs.get(timeout=.05)
                    except queue.Empty:
                        # Flush a held-back tail once the epoch ends, even when
                        # no further utterance arrives to trigger it.
                        if tail is not None and (not self.active or self.generation != tail_generation):
                            if self.output.write(tail):
                                tail = None
                        continue
                    if generation!=self.generation or not self.active:
                        # A stale epoch drops its result, but a held-back tail from
                        # the previous utterance is still flushed best-effort.
                        if tail is not None:
                            self.output.write(tail)
                            tail = None
                        continue
                    self.utterance_stats.update(state='発話全体を変換中',queued=jobs.qsize())
                    header,result=self._request({'op':'utterance'},audio)
                    if len(result)!=len(audio) or not np.isfinite(result).all():raise RuntimeError('Invalid utterance result')
                    if generation!=self.generation or not self.active:
                        if tail is not None:
                            self.output.write(tail)
                            tail = None
                        continue
                    if overlap and tail is not None:
                        if len(result) >= overlap:
                            result = crossfade_blend(tail, result, overlap)
                        else:
                            # Too short to crossfade; join hard rather than drop audio.
                            result = np.concatenate((tail, result))
                        tail = None
                    if overlap and continues and len(result) > overlap:
                        # Hold back the tail and crossfade it into the next head
                        # only while the utterance continues past the length cap.
                        # A finished utterance emits whole immediately: its end
                        # no longer waits for the next speech, and no stale tail
                        # smears into the next head. The shipped streaming route
                        # has always emitted this way.
                        emit, tail = result[:-overlap], result[-overlap:].copy()
                        tail_generation = generation
                    else:
                        emit = result
                    self.stats=header.get('stats',{})
                    self.utterance_stats['state']='出力待ち'
                    output_deadline=time.monotonic()+90.+len(emit)/48000
                    while not self._stop.is_set() and generation==self.generation and self.active:
                        self.utterance_output_generation=generation
                        if self.output.write(emit):
                            self.last_output=time.monotonic()
                            self.utterance_stats['completed']+=1
                            break
                        if time.monotonic()>output_deadline:raise RuntimeError('発話出力が消費されないため停止しました。')
                        self._stop.wait(.01)
                    self.utterance_stats['state']='発話待ち'
            except Exception as error:
                if not self._stop.is_set():
                    self.status,self.error='Error',str(error);self.worker_state='Error';self._stop.set()

        self.utterance_thread=threading.Thread(target=infer,name='utterance-inference',daemon=True)
        self.utterance_thread.start()
        seen_generation=-1;seen_overrun=self.overruns
        try:
            while not self._stop.is_set():
                generation=self.generation
                if seen_generation!=generation:
                    self.input.discard();collector.reset();self.finish_utterance.clear()
                    while True:
                        try:jobs.get_nowait()
                        except queue.Empty:break
                    seen_generation=generation;self.ack_generation=generation
                    self.utterance_stats.update(state='発話待ち',recording_seconds=0.,queued=0)
                if not self.active:
                    self._idle.set();self._stop.wait(.005);continue
                self._idle.clear();self.worker_state='Running'
                if seen_overrun!=self.overruns:
                    collector.reset();seen_overrun=self.overruns
                    self.utterance_stats['message']='入力があふれたため、途中の発話を破棄しました。'
                    self.utterance_stats['rejected']+=1
                if self.input.read_into(self._chunk):
                    results = collector.feed(self._chunk)
                    flags = collector.take_result_flags()
                    for index, audio in enumerate(results):
                        enqueue(audio,generation,flags[index] if index < len(flags) else False)
                    self.utterance_stats.update(recording_seconds=collector.samples/48000,limit_splits=collector.limit_splits)
                    continue
                if self.finish_utterance.is_set():
                    self.finish_utterance.clear()
                    # Consume callback-sized tail before manually ending speech.
                    count=self.input.available
                    if count:
                        tail=np.empty(count,dtype=np.float32)
                        if self.input.read_into(tail):
                            results = collector.feed(tail)
                            flags = collector.take_result_flags()
                            for index, audio in enumerate(results):
                                enqueue(audio,generation,flags[index] if index < len(flags) else False)
                    enqueue(collector.flush(),generation)
                    self.utterance_stats['recording_seconds']=0.
                self.input_ready.wait(self._stop)
        finally:
            self._stop.set()
            self.client.interrupt()
            self.utterance_thread.join(timeout=2)

    def _watch_deadlines(self):
        """Bound a hung native process; no pipe/process operations on Qt/audio."""
        while not self._stop.wait(.05):
            now=time.monotonic()
            inference_expired=(self._request_started and now-self._request_started>(self._request_timeout or self.rpc_timeout_seconds))
            load_expired=(self._load_started and now-self._load_started>self.load_timeout_seconds)
            process = getattr(self.client, 'process', None)
            crashed = process is not None and process.poll() is not None
            pending = self.parameters.delivery!='utterance' and self.status == 'Ready' and self.active and self.input.available >= self.chunk_frames
            if pending:
                if not self._pending_since:
                    self._pending_since = now
                if self.last_output > self._pending_since:
                    self._pending_since = self.last_output
            else:
                self._pending_since = 0
            stalled = self._pending_since and now-self._pending_since > self.rpc_timeout_seconds
            resource_error=''
            if self.parameters.delivery=='utterance' and process is not None and process.poll() is None:
                import psutil
                try:
                    limit=9 if self.parameters.enhancer=='mossformer' else 7 if self.parameters.enhancer=='voicefixer' else 6
                    if psutil.Process(process.pid).memory_info().rss>limit*1024**3 or psutil.virtual_memory().available<1.5*1024**3:
                        resource_error='発話変換のRAM上限に達したため停止しました。'
                except psutil.Error:pass
            if inference_expired or load_expired or crashed or stalled or resource_error:
                if self._stop.is_set():
                    return  # intentional shutdown is not a worker crash
                self.status='Error'
                self.worker_state = 'Error'
                self.error = (resource_error or ('AI process crashed' if crashed else 'AI stalled' if stalled else
                              'AI inference timed out' if inference_expired else 'AI model load timed out'))
                log.error('%s; Fallback: Original',self.error)
                self._stop.set()
                client=self.client
                if client is not None:
                    client.interrupt()
                return

    def snapshot(self):
        count = self.incident_count
        start = max(0, count-len(self.incidents))
        incidents = [self.incidents[i % len(self.incidents)].tolist() for i in range(start,count)]
        input_stats, output_stats = self.input.snapshot(self.chunk_frames), self.output.snapshot(self.chunk_frames)
        return dict(status=self.status, error=self.error, quality=self.parameters.quality,
                    delivery=self.parameters.delivery,utterance=dict(self.utterance_stats),
                    prosody=self.prosody.snapshot(),
                    generation=self.generation, acknowledged_generation=self.ack_generation,
                    worker_pid=getattr(getattr(self.client,'process',None),'pid',None),
                    worker_state=self.worker_state, last_response_monotonic=self.last_response,
                    last_output_monotonic=self.last_output, mode_crossfades=self.crossfade_count,
                    input_queue=input_stats, output_queue=output_stats,
                    queue_wait=input_stats['wait'], output_wait=output_stats['wait'],
                    model=self.model_stats, **self.stats, rpc=asdict(self.rpc_time.snapshot()),
                    reset=asdict(self.reset_time.snapshot()),
                    bridge_scheduling=getattr(self, 'scheduling', {}),
                    queue_current=self.input.available/self.chunk_frames,
                    queue_max=self.input.maximum_depth/self.chunk_frames,
                    queue_capacity=self.input.capacity/self.chunk_frames,
                    output_queue_current=self.output.available/self.chunk_frames,
                    output_queue_max=self.output.maximum_depth/self.chunk_frames,
                    ai_underrun=self.underruns, ai_overrun=self.overruns+self.output_overruns,
                    missing_output_frames=self.missing_frames,
                    input_overruns=self.overruns, output_overruns=self.output_overruns,
                    dropped_chunks=(self.dropped_frames+self.dropped_output_frames)/self.chunk_frames,
                    buffering_target_ms=self.parameters.startup_frames/48,
                    preroll_observed_ms=self.preroll_observed_frames/48,
                    model_alignment_delay_ms=self.model_stats.get('model_alignment_delay_ms'),
                    resampler_delay_ms=self.model_stats.get('resampler_delay_ms',2),
                    worker_alive=self.alive, incident_count=count, incidents=incidents,
                    incident_columns=['monotonic_seconds', 'kind_1_underrun_2_overrun',
                        'input_chunks','output_chunks','rpc_last_ms','generation','inference_last_ms'])
