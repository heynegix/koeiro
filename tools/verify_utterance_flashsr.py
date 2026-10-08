"""Guarded actual app-worker WAV verification; no audio devices or listening."""
import json
from pathlib import Path
import sys
import threading
import time
import wave
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vc.bridge import AIBridge
from src.vc.config import AIParameters
from src.vc.client import ServiceClient


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=0,help='0 uses the saved ai_threads setting')
    parser.add_argument('--enhancer',choices=('flashsr','lavasr','mossformer','voicefixer'),default='flashsr')
    args=parser.parse_args()
    from tools.compare_meanvc2_continuity import require_idle_voice_worker
    require_idle_voice_worker()
    settings=json.loads((ROOT/'settings.json').read_text('utf-8'))
    selected=settings['ai_model']
    threads=args.threads or int(settings.get('ai_threads',4))
    params=AIParameters(model=selected,threads=threads,delivery='utterance',enhancer=args.enhancer)
    client=ServiceClient(params);done=threading.Event();report=dict(kind='OFFLINE actual app-worker WAV RPC; no devices, no human listening',profile=selected,threads=threads,enhancer=args.enhancer,peak_ram_bytes=0,cases=[])
    deadline=time.monotonic()+300
    def guard():
        while not done.wait(.1):
            try:
                if client.process:
                    rss=psutil.Process(client.process.pid).memory_info().rss
                    report['peak_ram_bytes']=max(report['peak_ram_bytes'],rss)
                    if rss>6*1024**3 or psutil.virtual_memory().available<1.5*1024**3 or time.monotonic()>deadline:
                        report['guard']='RAM/time guard';client.interrupt();return
            except psutil.Error:pass
    folder=ROOT/'recordings/v011_utterance_flashsr';folder.mkdir(parents=True,exist_ok=True)
    client.start();monitor=threading.Thread(target=guard,daemon=True);monitor.start()
    try:
        while True:
            header,_=client.read()
            if header['status']=='Ready':report['model']=header['model'];break
        for case,path,seconds in [('normal',ROOT/'recordings/v011_mega_tournament/source/source_normal.wav',8),
                                  ('long',ROOT/'recordings/v011_meanvc2_continuity/source/source_long.wav',25)]:
            with wave.open(str(path),'rb') as source:
                assert source.getframerate()==48000 and source.getsampwidth()==2
                audio=np.frombuffer(source.readframes(seconds*48000),dtype='<i2').astype(np.float32)/32768
            print('CONVERT '+case,flush=True);start=time.monotonic()
            header,result=client.request({'op':'utterance'},audio)
            assert len(result)==len(audio) and np.isfinite(result).all() and np.max(abs(result))>.001
            assert not header['stats'].get('enhancer_error')
            assert np.max(abs(result))<=.981
            dest=folder/(case+'.wav')
            with wave.open(str(dest),'wb') as output:
                output.setnchannels(1);output.setsampwidth(2);output.setframerate(48000)
                output.writeframes((np.clip(result,-1,1)*32767).astype('<i2').tobytes())
            report['cases'].append(dict(case=case,output=str(dest),seconds=len(audio)/48000,wall_seconds=time.monotonic()-start,stats=header['stats']))
            print(json.dumps(report['cases'][-1],ensure_ascii=False),flush=True)
        # Cancel a genuine long model request; Stop must close its owner process.
        stop_errors=[]
        def pending():
            try:client.request({'op':'utterance'},audio)
            except (RuntimeError,EOFError,OSError) as error:stop_errors.append(str(error))
        pending_thread=threading.Thread(target=pending);pending_thread.start();time.sleep(.5)
        start=time.monotonic();client.interrupt();pending_thread.join(3)
        report['stop_seconds']=time.monotonic()-start
        assert not pending_thread.is_alive() and stop_errors
        report['status']='SUCCESS'
    finally:
        done.set();client.close();monitor.join(2)
        (ROOT/'validation/v011/utterance_flashsr_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
