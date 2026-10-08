"""Device-free reference for matching the same converted voice on CABLE."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from tools.measure_meanvc2_latency import make_signals
from tools.audit_input_gate import audit
from src.vc.meanvc2 import MeanVC2Backend
from src.vc.resampler import StreamingResampler
from tools.compare_meanvc2_continuity import sha,require_idle_voice_worker
from tools.compare_presets import write_wav


def main():
    require_idle_voice_worker()
    folder=ROOT/'recordings/v011_meanvc2_latency';folder.mkdir(parents=True,exist_ok=True)
    b=MeanVC2Backend(4);b.load(ROOT/'models/meanvc2_ref60');b.warmup()
    for kind,wave in zip(('continuous','restart_short'),make_signals()):
        wave,_=audit(wave,-55)
        b.reset();down=StreamingResampler(48000,16000);up=StreamingResampler(16000,48000)
        n=len(wave);wave=np.pad(wave,(0,(-n)%7680+12*7680));parts=[]
        for i in range(0,len(wave),7680):parts.append(up.process(b.process_chunk(down.process(wave[i:i+7680]))))
        trim=round((b.algorithmic_buffer_ms+40+2)*48)
        result=np.concatenate(parts)[trim:trim+n]
        sf.write(folder/f'reference_{kind}.wav',result,48000,subtype='FLOAT')
        write_wav(folder/f'reference_{kind}_pcm.wav',result,48000)
        (folder/f'reference_{kind}.json').write_text(json.dumps(dict(
            input_kind='Offline replay with actual production backend and causal streaming resamplers',
            trim_ms=trim/48,gate_db=-55,threads=4,runtime_sha256=sha(ROOT/'models/meanvc2_ref60/runtime.json'),
            source_kind=kind,model_stats=b.get_stats()),indent=2),encoding='utf-8')
    b.unload()


if __name__=='__main__':main()
