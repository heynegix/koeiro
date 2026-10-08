"""Bounded real-model smoke check for the quality-mode MossFormerSR wrapper."""
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vc.mossformer_sr import MossFormerSR

report=dict(kind='OFFLINE real-model smoke; no audio devices, no human listening')
process=psutil.Process()
peak={'rss':0}
def sample():
    try:peak['rss']=max(peak['rss'],process.memory_info().rss)
    except psutil.Error:pass

started=time.monotonic()
enhancer=MossFormerSR(threads=1)
report['load_seconds']=time.monotonic()-started
report['checkpoint_sha256']=enhancer.sha256
sample()

source=ROOT/'recordings/v011_mega_tournament/source/source_normal.wav'
with wave.open(str(source),'rb') as handle:
    assert handle.getframerate()==48000 and handle.getsampwidth()==2
    full=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').astype(np.float32)/32768
# Speech-bearing segment; the leading two seconds of this source are silence.
audio=full[2*48000:6*48000]
started=time.monotonic()
result=enhancer.process(audio)
elapsed=time.monotonic()-started
sample()
report['case']=dict(input_seconds=len(audio)/48000,wall_seconds=elapsed,
                    rtf=elapsed/(len(audio)/48000),
                    output_seconds=len(result)/48000,
                    peak_rss_bytes=peak['rss'],
                    input_peak=float(np.max(abs(audio))),
                    output_peak=float(np.max(abs(result))),
                    finite=bool(np.isfinite(result).all()))
assert len(result)==len(audio) and report['case']['finite']
assert report['case']['output_peak']>.001
# Degenerate near-silent input must not reach the model (hallucination guard).
silent=np.zeros(48000,dtype=np.float32);silent[::100]=3e-4
guard=enhancer.process(silent)
report['silence_guard']=dict(output_peak=float(np.max(abs(guard))),zeros=bool(not guard.any()))
assert report['silence_guard']['zeros']
report['status']='SUCCESS'
dest=ROOT/'validation/v011/mossformer_smoke_verification.json'
dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
