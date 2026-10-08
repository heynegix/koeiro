"""Opt-in real WASAPI AI test; recordings replay is explicitly labelled."""
import argparse
import math
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
from src.audio.engine import AudioEngine, EngineConfig
from src.audio.meters import peak_level
from src.processors.ai_voice import AIVoiceProcessor
from src.processors.voice_router import VoiceRouter
from src.processors.female_dsp import FemaleDSPProcessor
from src.processors.chain import ProcessorChain
from src.presets.female_presets import load_preset
from src.vc.config import AIParameters
from src.prosody.parameters import ProsodyParameters,load_preset as load_prosody_preset
from tools.verify_audio import VerificationSignal
from src.utils.logging import configure_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=int, required=True)
    parser.add_argument('--output', type=int, required=True)
    parser.add_argument('--cable-return', type=int, required=True)
    parser.add_argument('--wav', type=Path)
    parser.add_argument('--buffer', type=int, default=256)
    parser.add_argument('--quality', choices=['low_latency','balanced','stable'], default='balanced')
    parser.add_argument('--threads', type=int, default=1)
    from src.vc.models import VOICE_PROFILES
    parser.add_argument('--model',choices=tuple(VOICE_PROFILES),default='girl_01')
    parser.add_argument('--pitch',type=float,default=4.)
    parser.add_argument('--wait-ms', type=float, default=39)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--switches', type=int, default=0, help='Three-mode live changes before each timed cycle')
    parser.add_argument('--prosody',action='store_true')
    parser.add_argument('--prosody-preset',choices=['Natural','Anime Light','Anime Expressive'],default='Anime Light')
    parser.add_argument('--prosody-update',type=int,choices=[20,30,40,50],default=30)
    parser.add_argument('--asr-window',type=int,choices=[250,400,600,800,1000],default=400)
    parser.add_argument('--text-strategy',choices=['legacy','direct','modulation','hybrid','events'],default='hybrid')
    parser.add_argument('--crash-prosody',action='store_true',help='Kill only this test-owned analysis process')
    parser.add_argument('--text-aware',action='store_true')
    parser.add_argument('--crash-asr',action='store_true',help='Kill/restart only this test-owned ASR process')
    parser.add_argument('--crash-after-event',action='store_true',help='Wait for an active event; abort wait after 25 seconds')
    parser.add_argument('--report', type=Path, default=Path('test-results/v03-ai.json'))
    args = parser.parse_args()
    trial_started=time.monotonic()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.cycles < 1 or args.switches < 0:
        parser.error('Invalid test duration/cycles')
    if args.seconds*args.cycles+args.switches*args.cycles*.12>900:
        parser.error('Project policy: hardware trials must be 15 minutes or less')
    sys.setswitchinterval(.001)
    configure_logging(Path('test-results/ai-logs'))
    devices = {d.index:d for d in enumerate_devices()}
    input_device, output_device, cable = devices[args.input],devices[args.output],devices[args.cable_return]
    if 'cable input' not in output_device.name.lower() or 'cable output' not in cable.name.lower():
        parser.error('Output and return must be CABLE Input / CABLE Output')
    ai = AIVoiceProcessor(AIParameters(model=args.model,pitch=args.pitch,quality=args.quality,threads=args.threads,output_wait_ms=args.wait_ms))
    ai.bridge.prosody.parameters=load_prosody_preset(args.prosody_preset,ProsodyParameters(
        enabled=args.prosody or args.text_aware,engine='text_v3' if args.text_aware else 'rule_v2',update_ms=args.prosody_update,
        asr_window_ms=args.asr_window,text_strategy=args.text_strategy))
    router = VoiceRouter(FemaleDSPProcessor(load_preset('Anime Test')), ai, 'ai_voice')
    main_processor = VerificationSignal(router,args.wav) if args.wav else router
    engine = AudioEngine(ProcessorChain(main_processor))
    # Forward teardown through the developer replay wrapper.
    if args.wav:
        main_processor.stop = router.stop
    controller = AudioController(engine)
    return_peak = 0.0
    return_xruns = 0
    def receive(audio,frames,timing,status):
        nonlocal return_peak,return_xruns
        return_peak=max(return_peak,peak_level(audio))
        return_xruns+=int(bool(status))
    reader = sd.InputStream(device=cable.index,channels=min(2,cable.max_input_channels),dtype='float32',
        samplerate=48000,blocksize=args.buffer,extra_settings=sd.WasapiSettings(auto_convert=True),callback=receive)
    report = dict(input_kind='Human WAV replay at Main' if args.wav else 'Physical microphone',
                  input_wave=str(args.wav), input=input_device.label,output=output_device.label,
                  sample_rate=48000,buffer=args.buffer,quality=args.quality,threads=args.threads,
                  seconds=args.seconds,wait_ms=args.wait_ms,ai_parameters=asdict(ai.bridge.parameters),
                  prosody_parameters=asdict(ai.bridge.prosody.parameters),cycles=[],error=None)
    trial_end=time.monotonic()+870  # reserve 30 s cleanup within 15 min hard limit
    def wait_request(request,state):
        until=min(time.monotonic()+20,trial_end)
        while time.monotonic()<until:
            snapshot=controller.snapshot
            if snapshot.request_id==request:
                if snapshot.state in ('Error','Restart required'):
                    raise RuntimeError(snapshot.error)
                if snapshot.state==state:
                    return
            time.sleep(.01)
        raise RuntimeError('Controller timed out')
    try:
        reader.start()
        for cycle in range(args.cycles):
            router.select('ai_voice')
            until=time.monotonic()+30
            while ai.bridge.status!='Ready' and time.monotonic()<min(until,trial_end):
                if ai.bridge.status=='Error':
                    raise RuntimeError(ai.bridge.error)
                time.sleep(.02)
            if ai.bridge.status!='Ready':
                raise RuntimeError('AI load timed out')
            wait_request(controller.start(EngineConfig(input_device,output_device,48000,args.buffer)),'Running')
            for switch in range(args.switches):
                router.select(('original','female_dsp','ai_voice')[switch%3])
                time.sleep(.12)
            router.select('ai_voice')
            started=time.monotonic()
            observations=[]
            targets=[t for t in (0,5,20,120,300,600) if t<args.seconds]
            next_report=time.monotonic()
            crashed=False; asr_crashed=False; asr_restarted=False
            asr_pid_before=asr_pid_after=None; crash_event_active=False; crash_elapsed=None; crash_event_trace=[]
            while time.monotonic()-started<args.seconds:
                if time.monotonic()>=trial_end:
                    raise RuntimeError('Hardware trial time limit reached; stopping')
                if controller.snapshot.state!='Running' or ai.bridge.status=='Error':
                    raise RuntimeError(controller.snapshot.error or ai.bridge.error)
                if args.prosody and not args.crash_prosody and ai.bridge.prosody.status=='Error':
                    raise RuntimeError('Prosody trial failed: '+ai.bridge.prosody.error)
                elapsed=time.monotonic()-started
                if args.crash_asr:
                    asr=ai.bridge.prosody.asr
                    process=getattr(asr.client,'process',None)
                    event_active=bool(ai.bridge.prosody.stats.get('control',{}).get('text_events',{}).get('active_events'))
                    eligible=event_active or not args.crash_after_event or elapsed>25
                    if not asr_crashed and elapsed>7 and process is not None and eligible:
                        crash_event_active=event_active; crash_elapsed=elapsed
                        asr_pid_before=process.pid; process.terminate(); asr_crashed=True
                    if asr_crashed and not asr_restarted and elapsed>crash_elapsed+2 and not asr.alive:
                        asr.restart(); asr_restarted=True
                    if asr_restarted and process is not None and process.pid!=asr_pid_before:
                        asr_pid_after=process.pid
                    if asr_crashed and elapsed<crash_elapsed+1 and len(crash_event_trace)<30:
                        control=ai.bridge.prosody.stats.get('control',{})
                        crash_event_trace.append(dict(seconds=elapsed,status=asr.status,
                            pitch=control.get('quantized_pitch_delta',0),phrase_id=control.get('phrase_id'),
                            events=control.get('text_events',{}).get('active_events',[])))
                if args.crash_prosody and not crashed and elapsed>3:
                    process=getattr(ai.bridge.prosody.client,'process',None)
                    if process is not None:
                        process.terminate()
                        crashed=True
                if targets and elapsed>=targets[0]:
                    ai_snapshot=ai.bridge.snapshot()
                    observations.append(dict(seconds=elapsed,ai=ai_snapshot,
                        callback=asdict(engine.performance.snapshot()),
                        underflow=engine.underflows,overflow=engine.overflows,
                        parent_cpu_seconds=time.process_time()))
                    targets.pop(0)
                if time.monotonic()>=next_report:
                    snapshot=ai.bridge.snapshot()
                    print(json.dumps(dict(seconds=round(elapsed,1),status=snapshot['status'],rtf=snapshot.get('rtf'),
                        queue=snapshot['queue_current'],max_queue=snapshot['queue_max'],underrun=snapshot['ai_underrun'],
                        overrun=snapshot['ai_overrun'],dropped=snapshot['dropped_chunks'],
                        rpc_max_ms=snapshot['rpc']['maximum_ms'],reset_max_ms=snapshot['reset']['maximum_ms'],
                        missing_frames=snapshot['missing_output_frames'])),flush=True)
                    # Test-only checkpoint: file I/O runs on this control thread.
                    args.report.parent.mkdir(parents=True,exist_ok=True)
                    args.report.with_suffix('.progress.json').write_text(json.dumps(dict(
                        test=report,cycle=cycle+1,seconds=elapsed,ai=snapshot,
                        callback=asdict(engine.performance.snapshot()),
                        diagnostics=engine.diagnostics.snapshot()),ensure_ascii=False,indent=2),encoding='utf-8')
                    next_report=time.monotonic()+60
                time.sleep(.05)
            observations.append(dict(seconds=time.monotonic()-started,ai=ai.bridge.snapshot(),
                callback=asdict(engine.performance.snapshot()),
                underflow=engine.underflows,overflow=engine.overflows,
                parent_cpu_seconds=time.process_time()))
            record=dict(ai=ai.bridge.snapshot(),observations=observations,callback=asdict(engine.performance.snapshot()),
                        main=asdict(engine.chain.performance.snapshot()),underflow=engine.underflows,overflow=engine.overflows,
                        io_estimate_ms=engine.reported_latency_ms,diagnostics=engine.diagnostics.snapshot(),switches=args.switches)
            record['asr_crash_recovery']=dict(requested=args.crash_asr,crashed=asr_crashed,restart_requested=asr_restarted,
                pid_before=asr_pid_before,pid_after=asr_pid_after,event_active_at_crash=crash_event_active,crash_seconds=crash_elapsed,
                event_trace=crash_event_trace)
            record['input_io_estimate_ms']=engine.reported_input_latency_ms
            record['output_io_estimate_ms']=engine.reported_output_latency_ms
            wait_request(controller.stop(),'Stopped')
            record['worker_stopped']=not ai.bridge.alive
            record['asr_worker_stopped']=not ai.bridge.prosody.asr.alive
            report['cycles'].append(record)
            print(json.dumps(dict(cycle=cycle+1,rtf=record['ai'].get('rtf'),
                inference=record['ai'].get('inference'),underrun=record['ai']['ai_underrun'],
                overrun=record['ai']['ai_overrun'],dropped=record['ai']['dropped_chunks'],
                callback=record['callback'],underflow=record['underflow'],overflow=record['overflow'],
                worker_stopped=record['worker_stopped'])),flush=True)
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        controller.shutdown()
        until=time.monotonic()+10
        while controller.alive and time.monotonic()<until:
            time.sleep(.01)
        ai.stop()
        reader.close()
        report.update(cable_return_peak=return_peak,cable_return_xruns=return_xruns,controller_stopped=not controller.alive,
            elapsed_wall_seconds=time.monotonic()-trial_started)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
