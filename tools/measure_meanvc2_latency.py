"""Bounded WASAPI replay at app input to CABLE; no microphone recording.

Observed envelope delay excludes acoustic microphone ADC and Discord/network.
All waveform copying is preallocated; analysis and disk writes are off callback.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.compare_presets import read_wav,write_wav


def envelope(audio,hop=480):
    n=len(audio)//hop
    return np.sqrt(np.mean(audio[:n*hop].reshape(n,hop).astype('float64')**2,axis=1))


def observed_delay(source,capture,input_start,capture_start,min_correlation=.6):
    a=np.log1p(envelope(source)/.002)
    b=np.log1p(envelope(capture)/.002)
    a=a-a.mean();scale=np.linalg.norm(a)
    if scale==0:raise ValueError('Silent reference cannot measure delay')
    offset=input_start-capture_start
    scores=[]
    for delay in np.arange(0,4.001,.01):
        i=round((offset+delay)*100)
        if i<0 or i+len(a)>len(b):continue
        v=b[i:i+len(a)];v=v-v.mean()
        score=float(np.dot(a,v)/max(scale*np.linalg.norm(v),1e-12))
        scores.append((score,float(delay)))
    score,delay=max(scores)
    return dict(observed_input_to_cable_ms=round(delay*1000),correlation=score,
                resolution_ms=10,valid=score>=min_correlation and 0<delay<4,
                scope='Controlled WAV at application input -> observed CABLE capture; excludes microphone ADC, acoustic path and Discord')


def make_signals():
    folder=ROOT/'recordings/v011_meanvc2_continuity/source'
    normal,rate=read_wav(folder/'source_normal.wav');long,_=read_wav(folder/'source_long.wav')
    if rate!=48000:raise ValueError('48kHz source required')
    energy=envelope(normal)
    start=max(0,int(np.argmax(energy))*480-12000)
    phrase=normal[start:start+72000]
    if len(phrase)<72000:phrase=normal[max(0,len(normal)-72000):]
    silence=lambda sec:np.zeros(round(sec*48000),dtype=np.float32)
    parts=[silence(3),normal,silence(3),long,silence(6),phrase[:33600],silence(5),phrase,silence(5),phrase*.15,silence(5)]
    return np.concatenate(parts),np.concatenate([phrase[:57600],silence(5),phrase[:33600],silence(5)])


class ReplayGate:
    """Test-only injection before the existing gate, preserving the real chain."""
    def __init__(self,gate,signal):self.gate=gate;self.signal=signal;self.offset=0;self.started=None
    def prepare(self,rate,frames):self.gate.prepare(rate,frames);self.reset()
    def reset(self):self.offset=0;self.started=None;self.gate.reset()
    def stop(self):self.gate.stop()
    def process(self,audio,rate):
        if self.started is None:self.started=time.perf_counter()
        n=min(len(audio),len(self.signal)-self.offset)
        audio.fill(0)
        if n>0:np.copyto(audio[:n,0],self.signal[self.offset:self.offset+n]);self.offset+=n
        return self.gate.process(audio,rate)


def main():
    import sounddevice as sd
    from PySide6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow
    from src.settings.manager import SettingsManager,AppSettings
    from src.vc.models import VOICE_PROFILES
    from tools.compare_meanvc2_continuity import require_idle_voice_worker
    require_idle_voice_worker()
    parser=argparse.ArgumentParser();parser.add_argument('--chunks',nargs='+',type=int,default=[8,6,4])
    parser.add_argument('--report',type=Path,default=ROOT/'validation/v011/meanvc2_latency_sweep.json')
    parser.add_argument('--gate',type=float,default=-50)
    parser.add_argument('--converted-reference',action='store_true')
    a=parser.parse_args()
    if len(a.chunks)>3 or any(n not in (4,6,8) for n in a.chunks):parser.error('At most three bounded configurations: 4/6/8')
    begin=time.monotonic();deadline=begin+600;sys.setswitchinterval(.001)
    signal,restart=make_signals();folder=ROOT/'recordings/v011_meanvc2_latency';folder.mkdir(parents=True,exist_ok=True)
    (folder/'index.html').write_text((ROOT/'tools/meanvc2_latency_listening.html').read_text('utf-8'),encoding='utf-8')
    references={}
    if a.converted_reference:
        for kind,wave in (('continuous',signal),('restart_short',restart)):
            reference,ref_rate=read_wav(folder/f'reference_{kind}_pcm.wav')
            if ref_rate!=48000 or len(reference)!=len(wave):raise ValueError('Reference alignment mismatch')
            references[kind]=reference
    write_wav(folder/'controlled_source.wav',signal,48000);write_wav(folder/'restart_source.wav',restart,48000)
    app=QApplication([]);rows=[];window=None;reader=None
    profile=VOICE_PROFILES['meanvc2_ref60'];original=dict(profile)
    report=dict(input_kind='Hardware WASAPI + controlled WAV replay at application input; microphone overwritten, never recorded',
                rows=rows,passed=False,gate_db=a.gate)
    def pump(predicate=None,seconds=0,timeout=120):
        until=min(deadline,time.monotonic()+(seconds if predicate is None else timeout))
        while time.monotonic()<until:
            app.processEvents()
            if predicate is not None and predicate():return
            time.sleep(.003)
        if time.monotonic()>=deadline:raise RuntimeError('Ten minute whole-trial bound reached')
        if predicate is not None:raise RuntimeError('GUI/worker timeout')
    try:
        for chunks in a.chunks:
            profile['startup_chunks']=chunks
            manager=SettingsManager(a.report.with_name(a.report.stem+f'_{chunks}.settings.json'))
            settings=AppSettings.from_dict(json.loads((ROOT/'settings.json').read_text('utf-8')))
            settings.ai_model='meanvc2_ref60';settings.ai_threads=4;settings.voice_mode='original'
            settings.ai_quality='low_latency';settings.noise_gate_db=a.gate;settings.gain_db=0
            settings.prosody['enabled']=False;settings.prosody['engine']='off';manager.save(settings)
            window=MainWindow(manager);window.show();pump(lambda:window.start_button.isEnabled())
            for combo,index in ((window.input_device,68),(window.output_device,58)):
                combo.setCurrentIndex(next(i for i in range(combo.count()) if combo.itemData(i).index==index))
            window.mode.setCurrentIndex(window.mode.findData('ai_voice'))
            bridge=window.router.ai.bridge;pump(lambda:bridge.status in ('Ready','Error'))
            if bridge.status=='Error':raise RuntimeError(bridge.error)
            engine=window.controller.engine
            for kind,wave in (('continuous',signal),('restart_short',restart)):
                injected=ReplayGate(engine.chain.gate,wave)
                engine.chain._stages=(injected,engine.chain.gain,engine.chain.main_processor,*engine.chain.post_processors)
                memory=np.zeros(round((len(wave)/48000+12)*48000),dtype=np.float32)
                count=0;capture_started=None;xruns=0
                def capture(audio,frames,timing,status):
                    nonlocal count,capture_started,xruns
                    if capture_started is None:
                        capture_started=time.perf_counter()-(timing.currentTime-timing.inputBufferAdcTime)
                    n=min(frames,len(memory)-count)
                    if n>0:np.copyto(memory[count:count+n],audio[:n,0]);count+=n
                    xruns+=int(bool(status))
                reader=sd.InputStream(device=74,channels=2,dtype='float32',samplerate=48000,blocksize=256,
                    extra_settings=sd.WasapiSettings(auto_convert=True),callback=capture)
                reader.start();window.start_button.click();pump(lambda:window.stop_button.isEnabled())
                pump(seconds=len(wave)/48000+4)
                if not engine.running or bridge.status=='Error':raise RuntimeError(bridge.error or 'Stream stopped')
                snapshot=bridge.snapshot();callback=asdict(engine.performance.snapshot())
                window.stop_button.click();pump(lambda:window.start_button.isEnabled())
                reader.close();reader=None
                reference=wave
                if a.converted_reference:
                    reference=references[kind]
                delay=observed_delay(reference,memory[:count],injected.started,capture_started,
                                     min_correlation=.95 if a.converted_reference else .6)
                delay['method']='Same-voice converted reference envelope' if a.converted_reference else 'Input envelope; preliminary only'
                dest=folder/f'pre{chunks}_{kind}.wav';write_wav(dest,memory[:count],48000)
                row=dict(startup_chunks=chunks,kind=kind,latency=delay,ai=snapshot,callback=callback,
                    input_start_monotonic=injected.started,capture_start_monotonic=capture_started,
                    capture_xruns=xruns,underflows=engine.underflows,overflows=engine.overflows,
                    worker_stopped=not bridge.alive,output=str(dest))
                row['passed']=delay['valid'] and not any([xruns,engine.underflows,engine.overflows,
                    snapshot['ai_underrun'],snapshot['ai_overrun'],snapshot['dropped_chunks']]) and row['worker_stopped']
                rows.append(row)
                print(json.dumps(dict(chunks=chunks,kind=kind,passed=row['passed'],latency=delay,rtf=snapshot['rtf']),ensure_ascii=False),flush=True)
                a.report.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
                if kind=='continuous':
                    window.controller.restart_ai();pump(lambda:bridge.status=='Ready' and bridge.alive)
            window.close();pump(lambda:not window.controller.alive,timeout=15);window=None
        report['passed']=all(r['passed'] for r in rows)
    except Exception as error:report['error']=str(error);print(str(error),flush=True)
    finally:
        if reader is not None:reader.close()
        if window is not None:
            window.close();pump(lambda:not window.controller.alive,timeout=15)
        profile.clear();profile.update(original)
        report['elapsed_wall_seconds']=time.monotonic()-begin
        report['passed']=report['passed'] and report['elapsed_wall_seconds']<900
        a.report.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__=='__main__':sys.exit(main())
