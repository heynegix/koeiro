"""Independent causal analysis process. Never waits on or owns VC/audio output."""
from dataclasses import asdict
import sys
import time
import numpy as np
from src.audio.performance import TimingStats
from src.vc.protocol import send,receive
from src.vc.resampler import StreamingResampler
from .f0 import YinEstimator
from .controller import ProsodyController
from .parameters import ProsodyParameters


def main():
    source,output=sys.stdin.buffer,sys.stdout.buffer
    try:
        import psutil
        estimator=YinEstimator()
        resampler=StreamingResampler(48000,16000)
        controller=ProsodyController()
        count_keys=('state_transition_count','ending_trigger_count','state_reset_count',
            'quantization_change_count','hysteresis_hold_count','ending_emphasis_frames')
        totals={key:0 for key in count_keys}; last_counts=dict(totals)
        invalid_total=analysis_resets=0
        context=np.zeros(estimator.window_samples,dtype=np.float32)
        analysis,total=TimingStats(),TimingStats()
        next_stats=0.; statistics={}
        send(output,dict(status='Ready',method='FFT YIN',context_ms=80,resample_delay_ms=1))
        while True:
            header,audio=receive(source)
            if header['op']=='stop':
                break
            if header['op']=='reset':
                for key in count_keys: totals[key]+=last_counts[key]
                invalid_total+=controller.invalid_f0_count
                last_counts={key:0 for key in count_keys}; analysis_resets+=1
                resampler.reset(); controller.reset(preserve_phrase=True); context.fill(0)
                send(output,dict(status='Ready'))
                continue
            if header['op']!='process' or not 0<len(audio)<=2400 or not np.isfinite(audio).all():
                raise ValueError('Invalid analysis chunk')
            parameters=ProsodyParameters.from_dict(header.get('parameters'))
            t0=time.perf_counter_ns()
            reduced=resampler.process(audio)
            n=len(reduced)
            context[:-n]=context[n:]
            context[-n:]=reduced
            t1=time.perf_counter_ns()
            f0,voiced,energy=estimator.estimate(context)
            t2=time.perf_counter_ns()
            control=controller.update(f0,voiced,energy,len(audio)/48000,parameters,header.get('text_context'))
            analysis.record(t2-t1,len(audio),48000)
            total.record(time.perf_counter_ns()-t0,len(audio),48000)
            control_data=asdict(control)
            last_counts={key:control_data[key] for key in count_keys}
            for key in count_keys: control_data[key]+=totals[key]
            response=dict(status='Ready',control=control_data,context=asdict(controller.context),
                invalid_f0_count=invalid_total+controller.invalid_f0_count,analysis_reset_count=analysis_resets)
            if time.monotonic()>=next_stats:
                statistics=dict(analysis=asdict(analysis.snapshot()),process_total=asdict(total.snapshot()),
                    cpu_seconds=time.process_time(),ram_bytes=psutil.Process().memory_info().rss)
                response.update(statistics)
                next_stats=time.monotonic()+1
            send(output,response)
    except Exception as error:
        send(output,dict(status='Error',error=str(error)))
        raise

if __name__=='__main__':
    main()
