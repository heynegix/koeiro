"""Guarded actual app-worker WAV verification for the VoiceFixer2 (Q012) mode; no audio devices or listening."""
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
    from tools.compare_meanvc2_continuity import require_idle_voice_worker
    require_idle_voice_worker()
    selected=json.loads((ROOT/'settings.json').read_text('utf-8'))['ai_model']
    params=AIParameters(model=selected,threads=1,delivery='utterance',enhancer='voicefixer')
    client=ServiceClient(params);done=threading.Event();report=dict(kind='OFFLINE actual app-worker WAV RPC; no devices, no human listening',profile=selected,enhancer='voicefixer',peak_ram_bytes=0,cases=[])
    deadline=time.monotonic()+900
    def guard():
        while not done.wait(.1):
            try:
                if client.process:
                    rss=psutil.Process(client.process.pid).memory_info().rss
                    report['peak_ram_bytes']=max(report['peak_ram_bytes'],rss)
                    if rss>7*1024**3 or psutil.virtual_memory().available<1.5*1024**3 or time.monotonic()>deadline:
                        report['guard']='RAM/time guard';client.interrupt();return
            except psutil.Error:pass
    folder=ROOT/'recordings/v011_utterance_voicefixer';folder.mkdir(parents=True,exist_ok=True)
    client.start();monitor=threading.Thread(target=guard,daemon=True);monitor.start()
    try:
        while True:
            header,_=client.read()
            if header['status']=='Ready':report['model']=header['model'];break
        for name,source,seconds in [('normal',ROOT/'recordings/v011_mega_tournament/source/source_normal.wav',8),
                                    ('long',ROOT/'recordings/v011_meanvc2_continuity/source/source_long.wav',25)]:
            with wave.open(str(source),'rb') as stream:
                assert stream.getframerate()==48000 and stream.getsampwidth()==2
                audio=np.frombuffer(stream.readframes(seconds*48000),dtype='<i2').astype(np.float32)/32768
            print('CONVERT',name,flush=True);start=time.monotonic()
            header,result=client.request({'op':'utterance'},audio)
            wall=time.monotonic()-start
            assert len(result)==len(audio) and np.isfinite(result).all() and np.max(abs(result))>.001
            assert not header['stats'].get('enhancer_error'),header['stats'].get('enhancer_error')
            assert header['stats'].get('enhancer')=='voicefixer'
            dest=folder/f'{name}.wav'
            with wave.open(str(dest),'wb') as output:
                output.setnchannels(1);output.setsampwidth(2);output.setframerate(48000)
                output.writeframes((np.clip(result,-1,1)*32767).astype('<i2').tobytes())
            report['cases'].append(dict(case=name,output=str(dest),seconds=len(audio)/48000,
                wall_seconds=wall,rtf=wall/(len(audio)/48000),
                utterance_vc_seconds=header['stats'].get('utterance_vc_seconds'),
                utterance_post_seconds=header['stats'].get('utterance_post_seconds'),
                output_peak=float(np.max(abs(result))),stats=header['stats']))
            print(json.dumps(report['cases'][-1],ensure_ascii=False),flush=True)
        report['status']='SUCCESS'
    finally:
        done.set();client.interrupt();client.stop()
    (ROOT/'validation/v011/utterance_voicefixer_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),'utf-8')
    print(json.dumps({k:report[k] for k in ('status','peak_ram_bytes')},ensure_ascii=False))

if __name__=='__main__':main()
