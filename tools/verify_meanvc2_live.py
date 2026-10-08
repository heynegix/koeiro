"""Bounded native PySide6 + physical microphone -> CABLE test; no mic recording."""
import argparse,json,sys,time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sounddevice as sd
from PySide6.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.settings.manager import SettingsManager,AppSettings
from src.audio.meters import peak_level
from src.vc.models import VOICE_PROFILES,is_meanvc2

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=int,required=True);p.add_argument('--output',type=int,required=True)
    p.add_argument('--cable-return',type=int,required=True);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--report',type=Path,default=ROOT/'validation/v011/meanvc2_live.json')
    p.add_argument('--model',choices=[key for key in VOICE_PROFILES if is_meanvc2(key)],default='meanvc2_120')
    p.add_argument('--switch-model',choices=[key for key in VOICE_PROFILES if is_meanvc2(key)])
    a=p.parse_args()
    if not 1<=a.seconds<=720:p.error('Use at most 12 minutes playback, leaving setup/cleanup within 15 minutes')
    beginning=time.monotonic();deadline=beginning+870;sys.setswitchinterval(.001)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    manager=SettingsManager(a.report.with_suffix('.settings.json'))
    settings=AppSettings.from_dict(json.loads((ROOT/'settings.json').read_text('utf-8')))
    settings.ai_model=a.model;settings.ai_quality='low_latency';settings.ai_threads=4
    # This tool measures the streaming backend. Inheriting a saved utterance /
    # enhancer setting would silently test a different path (and report a clean
    # run with no inference at all), so both are pinned here and recorded.
    settings.ai_delivery='streaming';settings.ai_enhancer='none'
    settings.ai_pitch=0.;settings.ai_brightness=50.;settings.ai_post_fx=settings.ai_low_cut=settings.ai_limiter=False
    settings.voice_mode='original';settings.prosody['enabled']=False;settings.prosody['engine']='off'
    settings.sample_rate=48000;settings.buffer_size=256;manager.save(settings)
    app=QApplication([]);window=MainWindow(manager);window.show()
    report=dict(input_kind='Physical microphone + native PySide6 GUI + CABLE capture statistics',
                microphone_audio_saved=False,seconds=a.seconds,observations=[],profile_runs=[],passed=False,
                delivery=settings.ai_delivery,enhancer=settings.ai_enhancer,threads=settings.ai_threads)
    reader=None;return_peak=0.;return_xruns=0
    def capture(audio,frames,timing,status):
        nonlocal return_peak,return_xruns
        return_peak=max(return_peak,peak_level(audio));return_xruns+=int(bool(status))
    def pump(predicate=None,timeout=30):
        until=min(time.monotonic()+timeout,deadline)
        while time.monotonic()<until:
            app.processEvents()
            if predicate is not None and predicate():return
            time.sleep(.002)
        if predicate is not None:raise RuntimeError('GUI/worker timed out')
    try:
        pump(lambda:window.start_button.isEnabled())
        for combo,index in [(window.input_device,a.input),(window.output_device,a.output)]:
            combo.setCurrentIndex(next(i for i in range(combo.count()) if combo.itemData(i).index==index))
        for combo in (window.input_device,window.output_device):
            if combo.currentData().host_api!='Windows WASAPI':raise ValueError('WASAPI required')
        if 'cable input' not in window.output_device.currentData().name.lower():raise ValueError('CABLE Input required')
        report.update(input=window.input_device.currentData().label,output=window.output_device.currentData().label)
        window.mode.setCurrentIndex(window.mode.findData('ai_voice'))
        bridge=window.router.ai.bridge
        pump(lambda:bridge.status in ('Ready','Error'),timeout=120)
        if bridge.status=='Error':raise RuntimeError(bridge.error)
        reader=sd.InputStream(device=a.cable_return,channels=2,dtype='float32',samplerate=48000,blocksize=256,
                             extra_settings=sd.WasapiSettings(auto_convert=True),callback=capture)
        reader.start();window.start_button.click();pump(lambda:window.stop_button.isEnabled())
        started=time.monotonic();next_observation=started;switched=False
        active_started=started;active_model=a.model
        def save_profile():
            engine=window.controller.engine
            report['profile_runs'].append(dict(profile=active_model,active_seconds=time.monotonic()-active_started,
                ai=bridge.snapshot(),callback=asdict(engine.performance.snapshot()),model=bridge.model_stats,
                underflows=engine.underflows,overflows=engine.overflows))
        while time.monotonic()-started<a.seconds:
            if time.monotonic()>=deadline:raise RuntimeError('15 minute trial limit; stopping')
            app.processEvents()
            if not window.controller.engine.running or bridge.status=='Error':raise RuntimeError(bridge.error or window.controller.snapshot.error)
            if a.switch_model and not switched and time.monotonic()-started>=a.seconds/2:
                save_profile();window.stop_button.click();pump(lambda:window.start_button.isEnabled())
                if bridge.alive:raise RuntimeError('Previous profile worker did not stop')
                window.ai_model.setCurrentIndex(window.ai_model.findData(a.switch_model))
                pump(lambda:not window._pending and bridge.parameters.model==a.switch_model,timeout=120)
                pump(lambda:bridge.status in ('Ready','Error'),timeout=120)
                if bridge.status=='Error':raise RuntimeError(bridge.error)
                window.start_button.click();pump(lambda:window.stop_button.isEnabled())
                active_model=a.switch_model;active_started=time.monotonic();switched=True
            if time.monotonic()>=next_observation:
                engine=window.controller.engine
                snapshot=dict(seconds=time.monotonic()-started,profile=active_model,ai=bridge.snapshot(),callback=asdict(engine.performance.snapshot()),
                              underflows=engine.underflows,overflows=engine.overflows)
                report['observations'].append(snapshot)
                a.report.with_suffix('.progress.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
                print(json.dumps({k:snapshot['ai'].get(k) for k in ['rtf','ai_underrun','ai_overrun','dropped_chunks']}|{'seconds':round(snapshot['seconds'],1)}),flush=True)
                next_observation+=60
            time.sleep(.005)
        save_profile();engine=window.controller.engine
        report.update(ai=bridge.snapshot(),callback=asdict(engine.performance.snapshot()),diagnostics=engine.diagnostics.snapshot(),
                      underflows=engine.underflows,overflows=engine.overflows,model=bridge.model_stats)
        window.stop_button.click();pump(lambda:window.start_button.isEnabled())
        report['worker_stopped']=not bridge.alive
        # A run where the worker never received a request proves nothing about
        # realtime conversion; refuse to record it as a pass.
        report['worker_requests']=report['ai']['rpc']['count']
        report['inference_observed']=report['worker_requests']>0
        report['passed']=all(not any([r['ai']['ai_underrun'],r['ai']['ai_overrun'],r['ai']['dropped_chunks'],r['underflows'],r['overflows']]) for r in report['profile_runs']) and not return_xruns and report['inference_observed']
        if not report['inference_observed']:
            report['error']='Worker processed no audio; this run measures the audio path only, not realtime conversion'
    except Exception as error:
        report['error']=str(error);print('ERROR: '+str(error),flush=True)
    finally:
        if reader is not None:reader.close()
        window.close();pump(lambda:not window.controller.alive,timeout=15);app.processEvents()
        report.update(elapsed_wall_seconds=time.monotonic()-beginning,controller_stopped=not window.controller.alive,
                      worker_stopped=not window.router.ai.bridge.alive,cable_return_peak=return_peak,cable_return_xruns=return_xruns)
        report['passed']=report['passed'] and report['controller_stopped'] and report['worker_stopped'] and report['elapsed_wall_seconds']<900
        a.report.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps({k:report.get(k) for k in ['passed','error','elapsed_wall_seconds','worker_stopped','controller_stopped']}),flush=True)
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
