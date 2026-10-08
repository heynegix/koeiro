"""Bounded native app trial: controlled WAV latency, restart, then physical mic.

All hardware phases and setup/cleanup together are limited to eight minutes.
Physical microphone audio is not recorded. Controlled CABLE replay is saved.
"""
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time
import subprocess
import wave as wav_reader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import sounddevice as sd
    from PySide6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow
    from src.settings.manager import SettingsManager,AppSettings
    from tools.compare_presets import write_wav
    from tools.measure_meanvc2_latency import ReplayGate,envelope
    # psutil belongs to the isolated compute runtime, not the GUI venv.
    subprocess.run([str(ROOT/'vc_models/meanvc2/.venv/Scripts/python.exe'),'-c',
        'from tools.compare_meanvc2_continuity import require_idle_voice_worker; require_idle_voice_worker()'],
        cwd=ROOT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    out=ROOT/'validation/v011/phrase_app_trial.json'
    audiofolder=ROOT/'recordings/v011_phrase_repair'
    report=dict(kind='Native PySide6 + WASAPI 48k/256',rows=[],passed=False,
                physical_microphone_audio_saved=False,human_listening_performed=False)
    def read_wav(path):
        with wav_reader.open(str(path),'rb') as f:
            if f.getsampwidth()!=2 or f.getnchannels()!=1:
                raise ValueError('Expected mono PCM16 comparison file')
            return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768,f.getframerate()
    start=time.monotonic();deadline=start+480
    sys.setswitchinterval(.001)
    settings=AppSettings.from_dict(json.loads((ROOT/'settings.json').read_text('utf-8')))
    settings.ai_model='meanvc2_ref20_phrase';settings.ai_threads=2;settings.voice_mode='original'
    settings.noise_gate_db=-55;settings.gain_db=0;settings.prosody['enabled']=False;settings.prosody['engine']='off'
    settings.sample_rate=48000;settings.buffer_size=256
    manager=SettingsManager(out.with_suffix('.settings.json'));manager.save(settings)
    app=QApplication([]);window=MainWindow(manager);window.show();reader=None
    def pump(predicate=None,seconds=0,timeout=120):
        until=min(deadline,time.monotonic()+(seconds if predicate is None else timeout))
        while time.monotonic()<until:
            app.processEvents()
            if predicate is not None and predicate():return
            time.sleep(.003)
        if time.monotonic()>=deadline:raise RuntimeError('Eight-minute whole-trial limit')
        if predicate is not None:raise RuntimeError('App/worker timeout')
    try:
        pump(lambda:window.start_button.isEnabled())
        devices=sd.query_devices()
        def find(name,inputs=False):
            return next(i for i,d in enumerate(devices) if name in d['name'] and
                sd.query_hostapis(d['hostapi'])['name']=='Windows WASAPI' and
                d['max_input_channels' if inputs else 'max_output_channels']>0)
        inputid=find('Senary',True);outputid=find('CABLE Input');returnid=find('CABLE Output',True)
        for combo,index in ((window.input_device,inputid),(window.output_device,outputid)):
            combo.setCurrentIndex(next(i for i in range(combo.count()) if combo.itemData(i).index==index))
        window.mode.setCurrentIndex(window.mode.findData('ai_voice'))
        bridge=window.router.ai.bridge
        pump(lambda:bridge.status in ('Ready','Error'))
        if bridge.status=='Error':raise RuntimeError(bridge.error)
        engine=window.controller.engine;original_stages=engine.chain._stages
        for name in ('long','normal'):
            src,rate=read_wav(audiofolder/f'{name}_source.wav')
            reference,rr=read_wav(audiofolder/f'{name}_baseline.wav')
            if rate!=16000 or rr!=16000:raise ValueError('16kHz offline comparison required')
            wave=np.pad(np.repeat(src,3),(48000,48000))
            ref=np.pad(np.repeat(reference,3),(48000,48000))
            injected=ReplayGate(engine.chain.gate,wave)
            engine.chain._stages=(injected,engine.chain.gain,engine.chain.main_processor,*engine.chain.post_processors)
            memory=np.zeros(len(wave)+8*48000,np.float32)
            count=0;capture_start=None;xruns=0
            def capture(audio,frames,timing,status):
                nonlocal count,capture_start,xruns
                if capture_start is None: capture_start=time.perf_counter()-(timing.currentTime-timing.inputBufferAdcTime)
                n=min(frames,len(memory)-count)
                if n>0:np.copyto(memory[count:count+n],audio[:n,0]);count+=n
                xruns+=int(bool(status))
            reader=sd.InputStream(device=returnid,channels=2,dtype='float32',samplerate=48000,
                blocksize=256,extra_settings=sd.WasapiSettings(auto_convert=True),callback=capture)
            reader.start();window.start_button.click();pump(lambda:window.stop_button.isEnabled())
            pump(seconds=len(wave)/48000+7)
            snapshot=bridge.snapshot()
            row=dict(kind='Controlled WAV replay through app -> hardware CABLE capture',source=name,
                     ai=snapshot,callback=asdict(engine.performance.snapshot()),
                     underflows=engine.underflows,overflows=engine.overflows,cable_xruns=xruns)
            window.stop_button.click();pump(lambda:window.start_button.isEnabled())
            reader.close();reader=None
            write_wav(audiofolder/f'{name}_cable_capture.wav',memory[:count],48000)
            a=np.log1p(envelope(ref)/.002);a-=a.mean()
            b=np.log1p(envelope(memory[:count])/.002)
            offset=injected.started-capture_start;scores=[]
            for delay in np.arange(.5,5.501,.01):
                at=round((offset+delay)*100)
                if at<0 or at+len(a)>len(b):continue
                c=b[at:at+len(a)];c=c-c.mean()
                scores.append((float(np.dot(a,c)/max(np.linalg.norm(a)*np.linalg.norm(c),1e-12)),float(delay)))
            correlation,delay=max(scores)
            row.update(observed_input_to_cable_ms=round(delay*1000),correlation=correlation,
                       measurement_valid=correlation>=.6 and .5<delay<5.5,
                       scope='Excludes microphone ADC/acoustics and Discord/network')
            report['rows'].append(row);print(json.dumps(row),flush=True)
            out.write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
            window._restart_ai();pump(lambda:not window._pending and bridge.status in ('Ready','Error'))
            if bridge.status=='Error':raise RuntimeError(bridge.error)
        engine.chain._stages=original_stages
        window.start_button.click();pump(lambda:window.stop_button.isEnabled())
        pump(seconds=60)
        report['physical']=dict(kind='Physical Senary microphone -> native app -> CABLE; no audio saved',
            active_seconds=60,ai=bridge.snapshot(),callback=asdict(engine.performance.snapshot()),
            underflows=engine.underflows,overflows=engine.overflows)
        window.stop_button.click();pump(lambda:window.start_button.isEnabled())
        report['passed']=all(r['measurement_valid'] and r['observed_input_to_cable_ms']<=5000 and
            not any((r['ai']['ai_underrun'],r['ai']['ai_overrun'],r['ai']['dropped_chunks'],
                     r['underflows'],r['overflows'],r['cable_xruns'])) for r in report['rows'])
        physical=report['physical']
        report['passed'] &= not any((physical['ai']['ai_underrun'],physical['ai']['ai_overrun'],
                                    physical['ai']['dropped_chunks'],physical['underflows'],physical['overflows']))
    except Exception as error:
        report['error']=str(error)
    finally:
        if reader is not None:reader.close()
        window.close();pump(lambda:not window.controller.alive,timeout=15)
        report.update(wall_seconds=time.monotonic()-start,worker_stopped=not window.router.ai.bridge.alive,
                      controller_stopped=not window.controller.alive)
        report['passed'] &= report['worker_stopped'] and report['controller_stopped'] and report['wall_seconds']<480
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
        print(json.dumps({k:report.get(k) for k in ('passed','error','wall_seconds')}),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':sys.exit(main())
