"""SPSC audio tap and isolated analysis. Scalar controls are latest-only.

No callback locks, IPC waits, logging, inference, or resampling. Analysis ring
holds at most 160 ms; its consumer discards stale input before doing work.
"""
from dataclasses import asdict,replace
import logging
import threading
import time
import numpy as np
from src.vc.ring import AudioRing
from src.vc.wakeup import InputWakeup
from .parameters import ProsodyParameters
from .client import ProsodyClient

log=logging.getLogger(__name__)


class ProsodyBridge:
    timeout_seconds=.6
    load_timeout_seconds=30.
    def __init__(self,parameters=None,client_factory=ProsodyClient):
        self.parameters=parameters or ProsodyParameters()
        from src.asr.bridge import ASRBridge
        self.asr=ASRBridge(self.parameters)
        self._phrase_start=0.
        self.client_factory=client_factory
        self.input=AudioRing(1920*4)
        self._chunk=np.empty(2400,dtype=np.float32)
        self.wakeup=InputWakeup()
        self._stop=threading.Event()
        self._lock=threading.Lock()
        self.thread=self.watchdog=self.client=None
        self.status='Stopped'
        self.active=False
        self.generation=0
        self.errors=self.dropped_frames=0
        self.error=''
        self.stats={}
        self._control=(0.,0.,0.)  # analysis input time, pitch delta, gain
        self.last_f0=self.last_control=0.
        self._request_started=0.
        self._loading=False
        self._pending_since=0.
        self._fade_until=0.

    @property
    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def configure(self,parameters):
        if not isinstance(parameters,ProsodyParameters):
            raise TypeError('Expected validated ProsodyParameters')
        changed=self.parameters.enabled!=parameters.enabled
        if self.parameters.preset!=parameters.preset:
            log.info('Prosody preset change: %s',parameters.preset)
        self.parameters=parameters
        self.asr.configure(parameters)
        if changed:
            self._fade_until=time.monotonic()+.8 if not parameters.enabled else 0.
            log.info('Auto Intonation %s', 'on' if parameters.enabled else 'off')
        if parameters.enabled:
            self.load()

    def load(self):
        if not self.parameters.enabled:
            return
        self.asr.parameters=self.parameters
        if self.parameters.engine=='text_v3': self.asr.load()
        with self._lock:
            if self.alive:
                return
            self._stop.clear()
            self.status='Starting'; self.error=''; self.stats={}
            self._control=(0.,0.,0.)
            self.generation+=1
            self.thread=threading.Thread(target=self._run,name='prosody-bridge',daemon=True)
            self.thread.start()  # process construction/load is entirely off GUI

    def set_active(self,active):
        self.asr.set_active(active)
        if self.active!=active:
            self.active=active
            self.generation+=1
            self._control=(0.,0.,0.)

    def submit(self,audio):
        self.asr.submit(audio)
        if (not self.parameters.enabled and time.monotonic()>=self._fade_until) or not self.active or self.status not in ('Ready','Running'):
            return
        if not self.input.write(audio):
            self.dropped_frames+=len(audio)
        if self.input.available>=round(self.parameters.update_ms*48):
            self.wakeup.signal()

    def control(self,now=None):
        if (not self.parameters.enabled and time.monotonic()>=self._fade_until) or not self.active or self.status!='Running':
            return 0.,0.
        stamp,pitch,gain=self._control
        now=time.monotonic() if now is None else now
        if now-stamp>.250 or stamp<=0:
            return 0.,0.  # stale analysis never holds a stale pitch indefinitely
        return pitch,gain

    def stop(self):
        try:
            self.asr.stop()
        except Exception:
            log.exception('ASR stop failed; continuing audio-analysis shutdown')
        self.active=False
        self._control=(0.,0.,0.)
        self._stop.set()
        self.wakeup.signal()
        client=self.client
        if client is not None:
            client.interrupt()
        if self.alive and self.thread is not threading.current_thread():
            # Client cleanup has two bounded 2-second process waits. The
            # control-side join must cover both, including under CPU pressure.
            self.thread.join(5)
        if self.alive:
            raise RuntimeError('Prosody worker did not stop')
        if self.watchdog is not None and self.watchdog is not threading.current_thread():
            self.watchdog.join(.2)
        if self.status!='Error':
            self.status='Stopped'

    def _fail(self,error):
        if self.status=='Error':
            return
        self.errors+=1
        self.error=str(error); self.status='Error'
        self.parameters=replace(self.parameters,enabled=False)
        self.asr.parameters=self.parameters
        self.asr.context={}
        self._control=(0.,0.,0.)
        self._stop.set()
        if self.client is not None:
            try:
                self.client.interrupt()
            except OSError:
                log.exception('Prosody interrupt failed; cleanup will retry')
        log.error('Prosody fallback: Auto Intonation OFF, fixed base Pitch; %s',self.error)

    def _watch(self):
        while not self._stop.wait(.05):
            process=getattr(self.client,'process',None)
            if process is not None and process.poll() is not None:
                if self._stop.is_set():
                    return
                self._fail('Prosody worker crashed'); return
            deadline=self.load_timeout_seconds if self._loading else self.timeout_seconds
            if self._request_started and time.monotonic()-self._request_started>deadline:
                self._fail('Prosody worker stalled'); return
            pending=self.active and self.parameters.enabled and self.input.available>=round(self.parameters.update_ms*48)
            if pending:
                if not self._pending_since:
                    self._pending_since=time.monotonic()
                self._pending_since=max(self._pending_since,self.last_control)
                if time.monotonic()-self._pending_since>self.timeout_seconds:
                    self._fail('Prosody queue stalled'); return
            else:
                self._pending_since=0.

    def _run(self):
        client=None
        try:
            self.wakeup.open()
            self.input.reset()
            self.errors=self.dropped_frames=0
            client=self.client_factory(self.parameters); self.client=client
            self._loading=True
            self._request_started=time.monotonic()
            self.watchdog=threading.Thread(target=self._watch,name='prosody-watchdog',daemon=True)
            self.watchdog.start()
            client.start()
            # Startup imports get a longer deadline; streaming gets .6 seconds.
            header,_=client.read()
            self._request_started=0.
            self._loading=False
            if self._stop.is_set():
                return  # a late Ready must not overwrite watchdog Error/Stop
            self.status='Ready'
            log.info('Prosody worker start: FFT YIN')
            seen=-1; logged_drop=0; logged_invalid=0; next_log=0.
            text_reset=False
            while not self._stop.is_set():
                generation=self.generation
                if generation!=seen:
                    self.input.discard()  # only consumer mutates its read index
                    self._request_started=time.monotonic()
                    client.request(dict(op='reset'))
                    self._request_started=0.
                    seen=generation
                parameters=self.parameters
                if not self.active or (not parameters.enabled and time.monotonic()>=self._fade_until):
                    self.status='Ready'
                    self.input.discard()
                    self._stop.wait(.010)
                    continue
                frames=round(parameters.update_ms*48)
                excess=max(0,self.input.available-frames)
                if excess>=frames:
                    # Latest complete hop, bounded old-analysis latency.
                    self.dropped_frames+=self.input.discard((excess//frames)*frames)
                    self._control=(0.,0.,0.)
                    self._request_started=time.monotonic()
                    client.request(dict(op='reset'))  # discontinuity is analysis-only
                    self._request_started=0.
                chunk=self._chunk[:frames]
                if not self.input.read_into(chunk):
                    self.wakeup.wait(self._stop); continue
                stamp=time.monotonic()-self.input.wait_time.last_ns/1e9
                self._request_started=time.monotonic()
                text=self.asr.current()
                if self.asr.status in ('Error','Loading','Stopped'): text={'unavailable':True}
                text['source_generation']=self.asr.generation
                if not text.get('unavailable') and text.get('audio_timestamp',0)<self._phrase_start:
                    text={}
                result,_=client.request(dict(op='process',parameters=asdict(parameters),text_context=text),chunk)
                self._request_started=0.
                if generation!=self.generation or not self.active:
                    continue
                control=result['control']
                self.asr.set_phrase(control.get('phrase_id',0))
                if result.get('context',{}).get('audio',{}).get('onset'):
                    self._phrase_start=stamp
                if control.get('phrase_state')=='SILENCE' and (parameters.text_strategy!='events' or control.get('silence_duration',0)>.25):
                    self._phrase_start=time.monotonic()
                    if control.get('silence_duration',0)>=.4 and not text_reset:
                        self.asr.generation+=1; self.asr.context={}; text_reset=True
                else:
                    text_reset=False
                values=(control.get('quantized_pitch_delta',control['pitch_delta']),control['gain_db'])
                if not all(np.isfinite(v) for v in values):
                    raise ValueError('Non-finite prosody output')
                self.stats.update(result)  # timing/RAM sampled 1 Hz; control every hop
                state=control.get('phrase_state')
                if state and state!=getattr(self,'_logged_phrase',None):
                    log.debug('Phrase state %s -> %s',getattr(self,'_logged_phrase','SILENCE'),state)
                    self._logged_phrase=state
                self.last_f0=self.last_control=time.monotonic()
                minimum,maximum=(-1.2,1.5) if 'quantized_pitch_delta' in control else (parameters.min_pitch,parameters.max_pitch)
                self._control=(stamp,max(minimum,min(maximum,values[0])),
                    max(-parameters.max_gain_db,min(parameters.max_gain_db,values[1])))
                self.status='Running'
                if time.monotonic()>=next_log:
                    if self.dropped_frames!=logged_drop:
                        log.info('Prosody queue drop: %s frames',self.dropped_frames)
                        logged_drop=self.dropped_frames
                    invalid=self.stats.get('invalid_f0_count',0)
                    if invalid!=logged_invalid:
                        log.warning('Invalid F0 count: %s',invalid)
                        logged_invalid=invalid
                    next_log=time.monotonic()+1
        except Exception as error:
            if not self._stop.is_set():
                self._fail(error)
        finally:
            self._stop.set()
            self._control=(0.,0.,0.)
            try:
                self.asr.stop()
            except Exception:
                log.exception('ASR cleanup failed; continuing prosody cleanup')
            try:
                if client is not None:
                    client.close()
            except Exception as error:
                self.errors+=1; self.status='Error'; self.error='Prosody cleanup failed: '+str(error)
                self.parameters=replace(self.parameters,enabled=False)
                log.exception(self.error)
            finally:
                self.client=None
                self.wakeup.close()
            if self.status!='Error':
                self.status='Stopped'
            log.info('Prosody worker stop: %s',self.status)

    def snapshot(self):
        stats=dict(self.stats)
        if not self.parameters.transcript_logging and 'context' in stats:
            ctx=dict(stats['context']); text=dict(ctx.get('text',{}))
            for key in ('partial_text','final_text','stable_prefix','unstable_suffix'): text.pop(key,None)
            ctx['text']=text; stats['context']=ctx
        stats.update(status=self.status,error=self.error,enabled=self.parameters.enabled,
            worker_alive=self.alive,worker_pid=getattr(getattr(self.client,'process',None),'pid',None),
            method='FFT YIN',analysis_context_ms=80,control_update_ms=self.parameters.update_ms,
            queue=self.input.snapshot(round(self.parameters.update_ms*48)),
            queue_capacity_ms=self.input.capacity/48,queue_drop_frames=self.dropped_frames,
            queue_drop_chunks=self.dropped_frames/max(1,round(self.parameters.update_ms*48)),
            errors=self.errors,rule_fallback_count=self.errors,
            rule_version=3 if self.parameters.engine=='text_v3' else 2,asr=self.asr.snapshot(),
            quantization_st=.125,last_f0_monotonic=self.last_f0,last_control_monotonic=self.last_control,
            applied_pitch_delta=self.control()[0],applied_gain_db=self.control()[1])
        healthy=(self.status=='Running' and self.stats.get('analysis',{}).get('p95_ms',0)<self.parameters.update_ms*.8
                 and self.input.available<self.input.capacity*.75 and time.monotonic()-self._control[0]<.250)
        stats['health']=('Prosody Warning' if self.status=='Error' else 'Prosody Disabled' if
            not self.parameters.enabled or not self.active else 'Prosody Healthy' if healthy else 'Prosody Warning')
        return stats
