"""Nonblocking SPSC analysis tap -> local subprocess. Latest fresh context only."""
from dataclasses import asdict
import logging
import threading
import time
import numpy as np
from src.vc.ring import AudioRing
from src.vc.wakeup import InputWakeup
from .performance import ASRTimingStats as TimingStats
from .client import ASRClient
from .scheduler import update_interval

log=logging.getLogger(__name__)


class ASRBridge:
    load_timeout=30.
    request_timeout=3.

    def __init__(self,parameters,client_factory=ASRClient):
        self.parameters=parameters; self.client_factory=client_factory
        self.input=AudioRing(57600)  # hard 1200 ms; two 600 ms hops
        self.input.wait_time=TimingStats()
        self._chunk=np.empty(48000,dtype=np.float32)
        self.wakeup=InputWakeup(); self._stop=threading.Event(); self._lock=threading.Lock()
        self.thread=self.client=self.watchdog=None
        self.active=False; self.generation=0; self.status='Stopped'; self.error=''
        self.context={}; self.stats={}; self.drops=self.errors=self.restarts=self.fallbacks=0
        self._request_started=0.; self._loading=False; self.last_processed=0.
        self.lag=TimingStats(); self.rpc=TimingStats(); self.interval_ms=int(parameters.asr_update_ms)
        self._restart_thread=None
        self._stable_text=''; self._stable_since=0.
        self._load_permitted=True
        self._pending_since=0.
        self.phrase_id=0

    def set_phrase(self,phrase_id):
        if phrase_id!=self.phrase_id:
            self.phrase_id=phrase_id; self.generation+=1; self.context={}

    @property
    def alive(self): return self.thread is not None and self.thread.is_alive()

    def configure(self,p):
        restart=p.asr_threads!=self.parameters.asr_threads or p.asr_model!=self.parameters.asr_model
        self.parameters=p
        if restart:
            self.restart(); return
        if p.enabled and p.engine=='text_v3': self.load()
        else:
            self.context={}  # idle loaded worker retains model until Audio Stop

    def load(self):
        with self._lock:
            self._load_permitted=True
            if self.alive: return
            if self.status=='Error': self.restarts+=1
            self._stop.clear(); self.context={}; self.status='Loading'; self.error=''
            self.generation+=1
            self.thread=threading.Thread(target=self._run,name='asr-bridge',daemon=True); self.thread.start()

    def restart(self):
        if self._restart_thread is not None and self._restart_thread.is_alive(): return
        def run():
            try:
                active=self.active; self._shutdown(); self.active=active
                if self._load_permitted and self.parameters.enabled and self.parameters.engine=='text_v3': self.load()
            except Exception as error: self._fail(error)
        self._restart_thread=threading.Thread(target=run,name='asr-restart',daemon=True)
        self._restart_thread.start()

    def set_active(self,active):
        if self.active!=active:
            self.active=active; self.generation+=1; self.context={}

    def submit(self,audio):
        if not self.active or not self.parameters.enabled or self.parameters.engine!='text_v3' or self.status not in ('Ready','Listening','Degraded'): return
        if not self.input.write(audio): self.drops+=len(audio)
        if self.input.available>=self.interval_ms*48: self.wakeup.signal()

    def current(self,now=None):
        now=time.monotonic() if now is None else now
        if self.status not in ('Listening','Ready') or not self.active or not self.parameters.enabled or self.parameters.engine!='text_v3': return {}
        value=dict(self.context)
        age=max(0.,(now-value.get('audio_timestamp',0))*1000)
        if age>=1000 or value.get('phrase_id',self.phrase_id)!=self.phrase_id: return {}
        value['age_ms']=age
        value['stable_age_ms']=max(0.,(now-self._stable_since)*1000)
        return value

    def stop(self):
        self._load_permitted=False
        self._shutdown()
        if self._restart_thread is not None and self._restart_thread is not threading.current_thread():
            self._restart_thread.join(5)
        # A completed restart may have published a process before Stop was seen.
        if self.alive: self._shutdown()
        self.active=False; self.context={}

    def _shutdown(self):
        self.active=False; self.context={}; self._stop.set(); self.wakeup.signal()
        if self.client is not None: self.client.interrupt()
        if self.alive and self.thread is not threading.current_thread(): self.thread.join(5)
        if self.alive: raise RuntimeError('ASR worker failed to stop')
        if self.watchdog is not None and self.watchdog is not threading.current_thread(): self.watchdog.join(.2)
        if self.status!='Error': self.status='Stopped'

    def _fail(self,error):
        if self.status=='Error': return
        self.errors+=1; self.fallbacks+=1; self.context={}; self.error=str(error); self.status='Error'
        self._stop.set(); self.wakeup.signal()
        if self.client is not None:
            try: self.client.interrupt()
            except OSError: pass
        log.warning('ASR fallback: Rule v2 (%s)',self.error)

    def _watch(self):
        while not self._stop.wait(.05):
            process=getattr(self.client,'process',None)
            if process is not None and process.poll() is not None:
                self._fail('ASR worker crashed'); return
            deadline=self.load_timeout if self._loading else self.request_timeout
            if self._request_started and time.monotonic()-self._request_started>deadline:
                self._fail('ASR worker stalled'); return
            if self.active and self.parameters.enabled and self.parameters.engine=='text_v3' and self.input.available>=self.interval_ms*48:
                if not self._pending_since: self._pending_since=time.monotonic()
                self._pending_since=max(self._pending_since,self.last_processed)
                if not self._loading and time.monotonic()-self._pending_since>self.request_timeout:
                    self._fail('ASR queue stalled'); return
            else: self._pending_since=0.

    def _run(self):
        client=None
        try:
            self.wakeup.open(); self.input.reset()
            client=self.client_factory(self.parameters); self.client=client
            self._loading=True; self._request_started=time.monotonic()
            self.watchdog=threading.Thread(target=self._watch,name='asr-watchdog',daemon=True); self.watchdog.start()
            client.start(); header,_=client.read()
            self._request_started=0.; self._loading=False
            if self._stop.is_set(): return
            self.stats.update(header); self.status='Ready'; log.info('ASR worker Ready: ReazonSpeech INT8 rolling partial')
            seen=-1; last_drop=0
            while not self._stop.is_set():
                generation=self.generation
                if generation!=seen:
                    self.input.discard(); self.context={}; self._stable_text=''; self._stable_since=0.
                    self._request_started=time.monotonic(); client.request(dict(op='reset')); self._request_started=0.
                    seen=generation
                if not self.active or not self.parameters.enabled or self.parameters.engine!='text_v3':
                    self.input.discard(); self.context={}; self.status='Ready'; self._stop.wait(.02); continue
                frames=self.interval_ms*48
                excess=self.input.available-frames
                if excess>=frames or self.drops!=last_drop:
                    self.drops+=self.input.discard(max(0,excess)); last_drop=self.drops
                    self.context={}
                    self._request_started=time.monotonic(); client.request(dict(op='reset')); self._request_started=0.
                chunk=self._chunk[:frames]
                if not self.input.read_into(chunk): self.wakeup.wait(self._stop); continue
                # timestamp is the END of this audio chunk, including backlog age.
                stamp=time.monotonic()-max(0.,self.input.wait_time.last_ns/1e9-frames/48000)
                self._request_started=time.monotonic(); start=time.perf_counter_ns()
                phrase_id=self.phrase_id
                result,_=client.request(dict(op='process',audio_timestamp=stamp,window_ms=self.parameters.asr_window_ms,phrase_id=phrase_id),chunk)
                self._request_started=0.; now=time.monotonic()
                self.rpc.record(time.perf_counter_ns()-start,frames,48000)
                age=max(0.,now-stamp); self.lag.record(round(age*1e9),frames,48000)
                if generation!=self.generation or phrase_id!=self.phrase_id or not self.active: continue
                self.stats.update(result); self.last_processed=now
                self.stats['collection_wait_ms']=frames/48
                self.stats['backlog_wait_ms']=max(0.,self.input.wait_time.last_ns/1e6-frames/48)
                self.stats['delivery_overhead_estimate_ms']=max(0.,self.rpc.last_ns/1e6-
                    result.get('processing',{}).get('last_ms',0)-result.get('resample_ms',0)-result.get('classification_ms',0))
                ctx=result.get('context',{})
                signature=(ctx.get('phrase_type'),ctx.get('short_response_type'),
                    tuple(ctx.get(k,0)>=.65 for k in ('question_probability','exclamation_probability','callout_probability','farewell_probability')))
                if signature!=self._stable_text:
                    self._stable_text=signature; self._stable_since=now
                self.context=ctx if age<1. else {}
                self.status='Degraded' if age>1. else 'Listening'
                # Frequency is lowered when tail cost threatens scheduling. Ring
                # remains bounded; stale text fades to Rule v2 automatically.
                cost=self.rpc.snapshot().p95_ms
                self.interval_ms=update_interval(self.parameters.asr_update_ms,self.parameters.asr_window_ms,cost)
                if self.parameters.transcript_logging and ctx.get('partial_text'):
                    log.info('ASR transcript: %s',ctx['partial_text'])
        except Exception as error:
            if not self._stop.is_set(): self._fail(error)
        finally:
            self._stop.set(); self.context={}
            try:
                if client is not None: client.close()
            except Exception as error:
                self.status='Error'; self.error='ASR cleanup failed: '+str(error); self.errors+=1
            finally:
                self.client=None; self.wakeup.close()
            if self.status!='Error': self.status='Stopped'
            log.info('ASR worker stop: %s',self.status)

    def snapshot(self,include_transcript=False):
        context=self.current()
        if not include_transcript:
            for key in ('partial_text','final_text','stable_prefix','unstable_suffix'): context.pop(key,None)
        return dict(status=self.status,error=self.error,context=context,
            text_active=bool(context),fallback='Rule v2' if not context else '',
            processing=self.stats.get('processing',{}),rpc=asdict(self.rpc.snapshot()),
            lag=asdict(self.lag.snapshot()),rtf=self.stats.get('rtf',0.),
            queue=self.input.snapshot(self.interval_ms*48),dropped_frames=self.drops,
            dropped_chunks=self.drops/(self.interval_ms*48),errors=self.errors,restarts=self.restarts,
            fallbacks=self.fallbacks,ram_bytes=self.stats.get('ram_bytes',0),cpu_seconds=self.stats.get('cpu_seconds',0),
            worker_alive=self.alive,worker_pid=getattr(getattr(self.client,'process',None),'pid',None),
            load_seconds=self.stats.get('load_seconds',0),update_ms=self.interval_ms,
            transcript_logging=self.parameters.transcript_logging,model=self.parameters.asr_model,
            window_ms=self.parameters.asr_window_ms,
            phrase_id=self.phrase_id,resample_ms=self.stats.get('resample_ms',0),
            classification_ms=self.stats.get('classification_ms',0),
            collected_audio_ms=self.stats.get('collected_audio_ms',0),
            resample=self.stats.get('resample',{}),classification=self.stats.get('classification',{}),
            collection_wait_ms=self.stats.get('collection_wait_ms',0),
            backlog_wait_ms=self.stats.get('backlog_wait_ms',0),
            delivery_overhead_estimate_ms=self.stats.get('delivery_overhead_estimate_ms',0),
            backend='sherpa-onnx / ReazonSpeech simulated partial',last_processed=self.last_processed)
