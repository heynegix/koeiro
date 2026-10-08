"""Synthetic ASR position-boundary equivalence; no audio devices or recordings."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.vc.meanvc2 import MeanVC2Backend

torch.set_num_threads(4)
with (ROOT/'vc_models/meanvc2/repo/preprocess/ckpts/fastu2pp_160ms.pt').open('rb') as f:
    model=torch.jit.load(f).eval()
model=torch.jit.freeze(model,preserved_attrs=['m.encoder.embed.pos_enc.pe'])
torch.manual_seed(110)
xs=torch.randn(1,19,80);att=torch.randn(6,4,8,128);cnn=torch.randn(6,1,256,8)
frequencies=torch.exp(torch.arange(0,256,2,dtype=torch.float32)*(-__import__('math').log(10000.)/256))
positions=torch.arange(6000,dtype=torch.float32)
phase=positions[:,None]*frequencies[None,:]
full=torch.empty(1,6000,256);full[0,:,0::2]=torch.sin(phase);full[0,:,1::2]=torch.cos(phase)
with torch.inference_mode():
    original=model(xs,torch.tensor(100),torch.tensor(8),att,cnn)
    model.m.encoder.embed.pos_enc.pe=full
    expanded=model(xs,torch.tensor(100),torch.tensor(8),att,cnn)
    expected=model(xs,torch.tensor(5000),torch.tensor(8),att,cnn)
    backend=MeanVC2Backend(4);backend.torch=torch;backend.position_table=torch.empty(1,5000,256)
    model.m.encoder.embed.pos_enc.pe=backend.position_table
    backend.position_frequencies=frequencies;backend.asr_offset=5000;backend.position_base=0;backend.position_rolls=0
    backend._ensure_asr_positions()
    actual=model(xs,torch.tensor(backend.asr_offset),torch.tensor(8),att,cnn)
    errors=[float((a-b).abs().max()) for a,b in zip(expected,actual)]
    unchanged=[float((a-b).abs().max()) for a,b in zip(original,expanded)]
    passed=max(errors+unchanged)<1e-5
report=dict(input_kind='Synthetic feature tensors; no audio devices',passed=passed,
            feature_and_cache_max_errors=errors,original_range_max_errors=unchanged,
            position_base=backend.position_base,position_offset=backend.asr_offset)
(ROOT/'validation/v011/meanvc2_position_equivalence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report));sys.exit(0 if passed else 1)
