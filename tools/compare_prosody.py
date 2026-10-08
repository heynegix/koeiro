"""Causal same-input OFF/Natural/Light/Expressive comparison and native pitch check."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.compare_presets import read_wav,write_wav
from src.vc.beatrice_vst import BeatriceVSTBackend
from src.vc.post_fx import LightPostFX
from src.vc.resampler import StreamingResampler
from src.prosody.f0 import YinEstimator
from src.prosody.controller import ProsodyController
from src.prosody.parameters import ProsodyParameters,load_preset


def render(source,parameters,backend,text_partials=None):
    backend.reset()
    for _ in range(100):
        backend.set_pitch(4.)
    fx=LightPostFX(); resampler=StreamingResampler(48000,16000)
    controller=ProsodyController(); estimator=YinEstimator()
    context=np.zeros(1280,dtype=np.float32)
    padded=np.pad(source,(0,(-len(source))%624))
    result=np.empty_like(padded); trace=[]
    control_pitch=control_gain=0.
    hop=round(parameters.update_ms*48)
    analysis_input=[]; analysis_frames=0
    analysis_seconds=0.; text_index=0; current_text={}; phrase_start=0.
    started=time.perf_counter()
    for offset in range(0,len(padded),624):
        chunk=padded[offset:offset+624]
        # Audio first, using a previous completed analysis control. No FCPE
        # wait/look-ahead, same causal 30 ms update as the realtime path.
        target=max(4+np.ceil(parameters.min_pitch*8)/8,
            min(4+np.floor(parameters.max_pitch*8)/8,round((4+control_pitch)*8)/8))
        backend.set_pitch(target)
        converted=backend.process_chunk(chunk)
        result[offset:offset+624]=fx.process(converted,47,True,True,True,dynamic_gain_db=control_gain)
        analysis_input.append(chunk); analysis_frames+=len(chunk)
        if analysis_frames>=hop:
            pending=np.concatenate(analysis_input)
            while len(pending)>=hop:
                reduced=resampler.process(pending[:hop]); pending=pending[hop:]
                context[:-len(reduced)]=context[len(reduced):];context[-len(reduced):]=reduced
                f0,voiced,energy=estimator.estimate(context)
                analysis_seconds+=parameters.update_ms/1000
                if text_partials:
                    while text_index<len(text_partials) and text_partials[text_index]['ready_seconds']<=analysis_seconds:
                        event=text_partials[text_index]; current_text=dict(event['context']); text_index+=1
                        current_text['received_seconds']=event['ready_seconds']
                    age=(analysis_seconds-current_text.get('source_seconds',-10))*1000
                    current_text['age_ms']=max(0.,age)
                    current_text['stable_age_ms']=max(0.,(analysis_seconds-current_text.get('received_seconds',analysis_seconds))*1000)
                used=current_text if current_text.get('source_seconds',-1)>=phrase_start else {}
                control=controller.update(f0,voiced,energy,parameters.update_ms/1000,parameters,used)
                if controller.context.audio.onset: phrase_start=analysis_seconds
                if control.phrase_state=='SILENCE' and (parameters.text_strategy!='events' or control.silence_duration>.25): phrase_start=analysis_seconds
                control_pitch,control_gain=control.quantized_pitch_delta,control.gain_db
                trace.append(dict(seconds=(offset+624-len(pending))/48000,input_f0=f0,
                    input_voiced=voiced,input_energy=energy,**asdict(control)))
            analysis_input=[pending]; analysis_frames=len(pending)
    return result[:len(source)],dict(rtf=(time.perf_counter()-started)/(len(source)/48000),
        trace=trace,parameters=asdict(parameters),invalid_f0_count=controller.invalid_f0_count)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('recordings/v03/input.wav'))
    parser.add_argument('--output-dir',type=Path,default=Path('recordings/v05/prosody_comparison'))
    parser.add_argument('--update-ms',type=int,choices=(30,40,50),default=30)
    args=parser.parse_args()
    source,rate=read_wav(args.input)
    if rate!=48000: parser.error('Use 48 kHz input')
    source=np.pad(source,(0,4800))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    backend=BeatriceVSTBackend(); backend.load(Path('models/girl_01'));backend.warmup()
    report=dict(source=str(args.input.resolve()),sample_rate=48000,tail_ms=100,
        model_delay='Retained, not measured',human_quality='Pending user comparison',comparisons={})
    try:
        # Real native parameter and real non-zero inference checks, not a mock.
        checks=[]
        energy=np.convolve(source.astype(np.float64)**2,np.ones(4800)/4800,mode='valid')
        peak=int(np.argmax(energy))
        probe=np.pad(source[max(0,peak-4800):min(len(source),peak+48000)],(0,4800))
        probe=np.pad(probe,(0,(-len(probe))%624))
        for target in (4.,4.1,4.2,4.3):
            backend.reset()
            # Explicitly exercise the native automation endpoint as preset
            # component state and the plugin controller readback differ.
            backend.set_pitch(target-.125)
            for _ in range(10): backend.set_pitch(target)
            output=np.concatenate([backend.process_chunk(probe[i:i+624]) for i in range(0,len(probe),624)])
            checks.append(dict(requested=target,actual=backend.get_stats()['pitch'],
                native_parameter=float(backend.plugin.pitch_shift_st),finite=bool(np.isfinite(output).all()),
                output_peak=float(np.max(np.abs(output))),output_sha256=hashlib.sha256(output.tobytes()).hexdigest()))
        report['native_pitch_checks']=checks
        for name in ('OFF','Natural','Anime Light','Anime Expressive'):
            parameters=ProsodyParameters() if name=='OFF' else load_preset(name,ProsodyParameters(enabled=True))
            parameters=replace(parameters,update_ms=args.update_ms)
            output,stats=render(source,parameters,backend)
            filename='prosody_off.wav' if name=='OFF' else 'prosody_on.wav' if name=='Anime Light' else name.replace(' ','_')+'.wav'
            write_wav(args.output_dir/filename,output,rate)
            stats.update(file=str((args.output_dir/filename).resolve()),peak=float(np.max(np.abs(output))),finite=bool(np.isfinite(output).all()),
                voiced_frames=sum(t['f0']>0 for t in stats['trace']),
                pitch_min=min(t['pitch_delta'] for t in stats['trace']),pitch_max=max(t['pitch_delta'] for t in stats['trace']))
            report['comparisons'][name]=stats
            pitch=np.array([t['quantized_pitch_delta'] for t in stats['trace']])
            gain=np.array([t['gain_db'] for t in stats['trace']])
            stats['distribution']=dict(pitch_mean=float(pitch.mean()),pitch_std=float(pitch.std()),
                pitch_abs_p95=float(np.percentile(abs(pitch),95)),pitch_abs_max=float(abs(pitch).max()),
                quantized_values=np.unique(pitch).tolist(),gain_mean=float(gain.mean()),gain_std=float(gain.std()),
                transitions=stats['trace'][-1]['state_transition_count'],endings=stats['trace'][-1]['ending_trigger_count'],
                raw_std=float(np.std([t['raw_pitch_delta'] for t in stats['trace']])),
                smoothed_std=float(np.std([t['pitch_delta'] for t in stats['trace']])),
                max_quantized_step=float(np.max(abs(np.diff(pitch)))),
                resets=stats['trace'][-1]['state_reset_count'],holds=stats['trace'][-1]['hysteresis_hold_count'],
                ending_emphasis_frames=stats['trace'][-1]['ending_emphasis_frames'])
            print(filename,flush=True)
    finally:
        backend.unload()
    (args.output_dir/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

if __name__=='__main__': main()
