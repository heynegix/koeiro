"""Short opt-in Main callback -> CABLE capture delivery measurement.

Uses one existing WAV, not the physical microphone signal. Arrival timestamps
include capture delivery scheduling; this is not microphone/Discord E2E.
Analysis and WAV/JSON writes run only after both audio streams stop.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import sounddevice as sd
from src.audio.controller import AudioController
from src.audio.devices import enumerate_devices
from src.audio.engine import AudioEngine,EngineConfig
from src.processors.base import AudioProcessor
from src.processors.chain import ProcessorChain
from src.processors.ai_voice import AIVoiceProcessor
from src.processors.voice_router import VoiceRouter
from src.settings.manager import SettingsManager
from tools.compare_presets import read_wav,write_wav


class CallbackCapture:
    def __init__(self,seconds,block=256,rate=48000):
        self.audio=np.empty(round(seconds*rate),dtype=np.float32)
        self.starts=np.empty(len(self.audio)//block+100,dtype=np.int64)
        self.times=np.empty(len(self.starts),dtype=np.float64)
        self.samples=self.blocks=self.overruns=0

    def append(self,audio):
        n=len(audio)
        if self.samples+n>len(self.audio) or self.blocks==len(self.starts):
            self.overruns+=1
            return
        self.starts[self.blocks]=self.samples
        self.times[self.blocks]=time.perf_counter()
        np.copyto(self.audio[self.samples:self.samples+n],audio.reshape(-1))
        self.samples+=n
        self.blocks+=1

    def envelope(self):
        starts=self.starts[:self.blocks]
        lengths=np.diff(np.append(starts,self.samples))
        squared=self.audio[:self.samples].astype(np.float64)**2
        sums=np.add.reduceat(squared,starts)
        return self.times[:self.blocks].copy(),np.sqrt(sums/lengths)


def delay_estimate(source_time,source_rms,output_time,output_rms,maximum_ms=700):
    """Energy-envelope correlation. Weak/ambiguous matches are rejected.

    Absolute monotonic callback times allow independently started streams.
    Pitch-converted waveforms cannot be compared by raw sample correlation.
    """
    for times,energy in ((source_time,source_rms),(output_time,output_rms)):
        if (times.ndim!=1 or energy.ndim!=1 or len(times)!=len(energy) or len(times)<2
                or not np.isfinite(times).all() or not np.isfinite(energy).all()
                or not (np.diff(times)>0).all()):
            raise ValueError('Invalid envelope timestamps/values')
    if not 50<=maximum_ms<=1000:raise ValueError('Invalid delay search limit')
    step=.005
    start=max(source_time[0],output_time[0])
    stop=min(source_time[-1],output_time[-1])-maximum_ms/1000
    if stop-start<3:raise ValueError('Insufficient overlapping capture')
    timeline=np.arange(start,stop,step)
    source=np.interp(timeline,source_time,source_rms)
    # 20 ms averaging reduces pitch-cycle bias without removing phrase shape.
    source=np.convolve(source,np.ones(4)/4,mode='same')
    if np.std(source)<1e-5:raise ValueError('Silent reference')
    scores=[]
    for lag in np.arange(0,maximum_ms/1000+step/2,step):
        target=np.interp(timeline+lag,output_time,output_rms)
        target=np.convolve(target,np.ones(4)/4,mode='same')
        scores.append(float(np.corrcoef(source,target)[0,1]) if np.std(target)>1e-7 else -1.)
    peak=int(np.argmax(scores))
    if scores[peak]<.6 or peak in (0,len(scores)-1):
        raise ValueError('No reliable interior delay peak')
    distant=[score for index,score in enumerate(scores) if abs(index-peak)>10]
    margin=scores[peak]-max(distant,default=-1)
    return dict(delay_ms=peak*step*1000,correlation=scores[peak],
        distant_peak_margin=margin,grid_ms=step*1000,
        reliable=margin>.02,search_max_ms=maximum_ms,
        definition='Main callback input to CABLE capture callback delivery; excludes microphone input and Discord',
        uncertainty='5 ms grid plus callback scheduling and energy-envelope distortion; not sample-accurate E2E')


def window_estimates(source_time,source_rms,output_time,output_rms):
    """Independent four-second segments to expose inconsistent timing."""
    estimates=[]
    origin=source_time[0]
    for offset in (2.,6.,10.):
        mask=(source_time>=origin+offset)&(source_time<origin+offset+4.)
        try:
            result=delay_estimate(source_time[mask],source_rms[mask],output_time,output_rms)
            estimates.append(dict(start_seconds=offset,**result))
        except ValueError as error:
            estimates.append(dict(start_seconds=offset,error=str(error)))
    return estimates


def capture_safety(source_time,source_rms,output_time,output_rms,delay_ms):
    """Flag sustained near-silence during clearly active reference audio.

    This is a diagnostic heuristic, not an intelligibility or listening score.
    """
    target=np.interp(source_time+delay_ms/1000,output_time,output_rms)
    source_threshold=max(.005,float(np.percentile(source_rms,90))*.12)
    output_threshold=max(.0001,float(np.percentile(output_rms,90))*.025)
    suspicious=(source_rms>source_threshold)&(target<output_threshold)
    edges=np.diff(np.concatenate(([False],suspicious,[False])).astype(np.int8))
    starts,ends=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
    runs=[]
    for start,end in zip(starts,ends):
        duration=(source_time[min(end,len(source_time)-1)]-source_time[start])*1000
        if duration>=40:
            runs.append(dict(start_seconds=float(source_time[start]-source_time[0]),duration_ms=float(duration)))
    return dict(suspicious_silence_over_40ms=runs,source_rms_threshold=source_threshold,
        output_rms_threshold=output_threshold,definition='Energy heuristic only; does not prove audible dropout absence')


class RecordedReplay(AudioProcessor):
    def __init__(self,router,signal,capture):
        self.router,self.signal,self.capture=router,signal,capture
        self.position=0

    def prepare(self,rate,frames):
        self.position=0
        self.router.prepare(rate,frames)

    def process(self,audio,rate):
        n=len(audio);remaining=max(0,min(n,len(self.signal)-self.position))
        audio.fill(0)
        audio[:remaining,0]=self.signal[self.position:self.position+remaining]
        self.position+=remaining
        self.capture.append(audio[:,0])
        return self.router.process(audio,rate)

    def stop(self):
        self.router.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','output','cable-return'):parser.add_argument('--'+name,type=int,required=True)
    parser.add_argument('--wav',type=Path,required=True)
    parser.add_argument('--mode',choices=['original','ai_voice'],default='ai_voice')
    parser.add_argument('--preprocessing',action='store_true',help='Replay through saved Noise Gate/Gain before VC')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    sys.setswitchinterval(.001)
    source,rate=read_wav(args.wav)
    if rate!=48000 or len(source)>48000*60:parser.error('Use 48 kHz WAV <=60 seconds')
    source=np.concatenate((np.zeros(48000*2,dtype=np.float32),source,np.zeros(48000*3,dtype=np.float32)))
    seconds=len(source)/48000
    reference,returned=CallbackCapture(seconds+5),CallbackCapture(seconds+15)
    devices={d.index:d for d in enumerate_devices()}
    inp,out,ret=devices[args.input],devices[args.output],devices[args.cable_return]
    if 'cable input' not in out.name.lower() or 'cable output' not in ret.name.lower():
        parser.error('Only VB-CABLE output/return allowed')
    settings=SettingsManager(Path('settings.json')).load()
    ai=AIVoiceProcessor(settings.ai_parameters())
    ai.bridge.prosody.parameters=settings.prosody_parameters()
    router=VoiceRouter(ai=ai,mode=args.mode)
    processor=ProcessorChain(router,gain_db=settings.gain_db,threshold_db=settings.noise_gate_db) if args.preprocessing else router
    replay=RecordedReplay(processor,source,reference)
    engine=AudioEngine(ProcessorChain(replay));controller=AudioController(engine)
    returned_xruns=0
    def receive(audio,frames,timing,status):
        nonlocal returned_xruns
        returned.append(audio[:,0])
        returned_xruns+=int(bool(status))
    reader=sd.InputStream(device=ret.index,samplerate=48000,channels=1,dtype='float32',blocksize=256,
        extra_settings=sd.WasapiSettings(auto_convert=True),callback=receive)
    trial_start=time.monotonic();limit=trial_start+120
    report=dict(mode=args.mode,input_kind='Human WAV injected at Main; physical microphone signal discarded',
        parameters=asdict(settings.ai_parameters()),error=None)
    report['preprocessing']=dict(enabled=args.preprocessing,gain_db=settings.gain_db,noise_gate_db=settings.noise_gate_db)
    def wait_state(state,request=None):
        deadline=min(limit,time.monotonic()+30)
        while time.monotonic()<deadline:
            snapshot=controller.snapshot
            if snapshot.state=='Error':raise RuntimeError(snapshot.error)
            if snapshot.state==state and (request is None or snapshot.request_id==request):return
            time.sleep(.01)
        raise TimeoutError('Short trial controller timeout')
    try:
        if args.mode=='ai_voice':
            router.select(args.mode)
            while ai.bridge.status!='Ready' and time.monotonic()<limit-60:
                if ai.bridge.status=='Error':raise RuntimeError(ai.bridge.error)
                time.sleep(.02)
            if ai.bridge.status!='Ready':raise TimeoutError('Model load timeout')
        reader.start()
        wait_state('Running',controller.start(EngineConfig(inp,out,48000,256)))
        until=min(limit-10,time.monotonic()+seconds)
        while time.monotonic()<until:
            if controller.snapshot.state!='Running':raise RuntimeError(controller.snapshot.error)
            if args.mode=='ai_voice' and ai.bridge.status=='Error':raise RuntimeError(ai.bridge.error)
            time.sleep(.02)
        report.update(ai=ai.bridge.snapshot(),callback=asdict(engine.performance.snapshot()),
            underflow=engine.underflows,overflow=engine.overflows,io_estimate_ms=engine.reported_latency_ms,
            input_io_estimate_ms=engine.reported_input_latency_ms,output_io_estimate_ms=engine.reported_output_latency_ms)
        wait_state('Stopped',controller.stop())
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        reader.close();controller.shutdown()
        until=time.monotonic()+10
        while controller.alive and time.monotonic()<until:time.sleep(.01)
        ai.stop()
        report.update(wall_seconds=time.monotonic()-trial_start,returned_xruns=returned_xruns,
            capture_overruns=reference.overruns+returned.overruns,worker_stopped=not ai.bridge.alive)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        if not report['error']:
            reference_envelope=reference.envelope()
            returned_envelope=returned.envelope()
            try:
                report['measurement']=delay_estimate(*reference_envelope,*returned_envelope)
                report['windows']=window_estimates(*reference_envelope,*returned_envelope)
                report['capture_safety']=capture_safety(*reference_envelope,*returned_envelope,report['measurement']['delay_ms'])
            except ValueError as error:
                report['measurement_error']=str(error)
            folder=Path('recordings/latency')/args.report.stem;folder.mkdir(parents=True,exist_ok=True)
            np.savez(folder/'envelopes.npz',source_time=reference_envelope[0],source_rms=reference_envelope[1],
                output_time=returned_envelope[0],output_rms=returned_envelope[1])
            values=returned.audio[:returned.samples]
            finite=bool(np.isfinite(values).all())
            report['raw_capture']=dict(finite=finite,peak=float(np.max(np.abs(values))) if finite else None,
                samples=len(values),above_full_scale=int(np.count_nonzero(np.abs(values)>1)))
            write_wav(folder/'Main_Input.wav',reference.audio[:reference.samples],48000)
            write_wav(folder/'CABLE_Return.wav',returned.audio[:returned.samples],48000)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report.get('measurement',dict(error=report.get('measurement_error')))))


if __name__=='__main__':main()
