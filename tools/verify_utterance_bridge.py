"""Wall-clock-paced WAV replay through the real callback/bridge/worker; no devices."""
import json
from pathlib import Path
import sys
import time
import wave
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.vc.bridge import AIBridge
from src.vc.config import AIParameters
from src.processors.ai_voice import AIVoiceProcessor
from src.processors.voice_router import VoiceRouter


def main():
    from tools.compare_meanvc2_continuity import require_idle_voice_worker
    require_idle_voice_worker()
    settings=json.loads((ROOT/'settings.json').read_text('utf-8'))
    bridge=AIBridge(AIParameters(model=settings['ai_model'],threads=settings['ai_threads'],delivery='utterance',enhancer='flashsr'))
    processor=AIVoiceProcessor(bridge=bridge)
    class UnusedDSP:
        def prepare(self,*args):pass
        def reset(self):pass
        def process(self,audio,*args):return audio
    router=VoiceRouter(dsp=UnusedDSP(),ai=processor,mode='ai_voice')
    folder=ROOT/'recordings/v011_utterance_flashsr';folder.mkdir(parents=True,exist_ok=True)
    report=dict(kind='OFFLINE wall-clock WAV callback replay, 48kHz/256; no audio devices or listening')
    try:
        bridge.load();until=time.monotonic()+120
        while bridge.status!='Ready':
            if bridge.status=='Error' or time.monotonic()>until:raise RuntimeError(bridge.error or 'Load timeout')
            time.sleep(.01)
        router.prepare(48000,256)
        with wave.open(str(ROOT/'recordings/v011_mega_tournament/source/source_bright.wav'),'rb') as source:
            audio=np.frombuffer(source.readframes(48000*3),dtype='<i2').astype(np.float32)/32768
        start=time.monotonic();next_callback=start;parts=[];sent=False;finish_time=first_output=None
        while time.monotonic()-start<100:
            pos=len(parts)*256;block=np.zeros((256,1),dtype=np.float32)
            piece=audio[pos:pos+256];block[:len(piece),0]=piece
            router.process(block,48000);parts.append(block[:,0].copy())
            if np.max(abs(block))>.0001 and first_output is None:first_output=time.monotonic()
            if not sent and pos+256>=len(audio):
                bridge.finish_utterance.set();bridge.input_ready.signal();sent=True;finish_time=time.monotonic()
            if bridge.status=='Error':raise RuntimeError(bridge.error)
            if bridge.utterance_stats.get('completed',0)>=1 and bridge.output.available==0:break
            next_callback+=256/48000;time.sleep(max(0,next_callback-time.monotonic()))
        assert first_output is not None and bridge.utterance_stats['completed']>=1
        assert bridge.overruns==bridge.output_overruns==bridge.underruns==0
        result=np.concatenate(parts)
        assert np.max(abs(result[:round((finish_time-start)*48000)]))==0
        path=folder/'callback_replay.wav'
        with wave.open(str(path),'wb') as output:
            output.setnchannels(1);output.setsampwidth(2);output.setframerate(48000)
            output.writeframes((np.clip(result,-1,1)*32767).astype('<i2').tobytes())
        report.update(status='SUCCESS',output=str(path),after_finish_first_output_seconds=first_output-finish_time,
                      wall_seconds=time.monotonic()-start,snapshot=bridge.snapshot())
    finally:
        start_stop=time.monotonic();bridge.stop();report['stop_seconds']=time.monotonic()-start_stop
        (ROOT/'validation/v011/utterance_bridge_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='snapshot'},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
