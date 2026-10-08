"""Offline post-VC adapters for the second comparison (v2 conditions).

Runs in vc_models/meanvc2/.venv with PYTHONPATH to post_lavasr / post_flashsr /
post_voicefixer2 vendor+repo. One model per bounded offline subprocess; no audio
devices. Baseline flashsr/lavasr/apbwe conditions stay in tools/post_vc_worker.py
with their original hashes; this file must not change them.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--chain',required=True,choices=['flashsr','lavasr2','lavasrden','voicefixer2','flashsr_voicefixer2','lavasrden_flashsr'])
 parser.add_argument('--input',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--metadata',type=Path,required=True)
 args=parser.parse_args()
 import numpy as np
 import soundfile as sf
 from scipy.signal import resample_poly
 adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 original,sr=sf.read(args.input,dtype='float32')
 if sr not in (16000,48000) or original.ndim!=1:raise ValueError('Expected 16k/48k mono input')
 x16=resample_poly(original,1,3) if sr==48000 else original
 start=time.monotonic();provenance={};output_sr=48000
 if args.chain=='flashsr':
  sys.path.insert(0,str(ROOT/'vc_models/post_flashsr/vendor'))
  import onnxruntime as ort
  options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
  session=ort.InferenceSession(str(ROOT/'vc_models/post_flashsr/repo/models/model.onnx'),options,providers=['CPUExecutionProvider'])
  flash=session.run(None,{'x':x16[None,None,:].astype(np.float32)})[0].reshape(-1)
  if not np.isfinite(flash).all():raise RuntimeError('Invalid FlashSR output')
  infer_result=flash/(np.max(abs(flash))+1e-7)*.999
  def infer():return infer_result
  provenance=dict(checkpoint='Official FlashSR models/model.onnx 16k->48k; official peak normalization retained')
  output_sr=48000
 elif args.chain=='lavasr2':
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  sys.path.insert(0,str(ROOT/'vc_models/post_lavasr/vendor'))
  sys.path.insert(0,str(ROOT/'vc_models/post_lavasr/repo'))
  from LavaSR.model import LavaEnhance2
  model_path=ROOT/'vc_models/post_lavasr/weights'
  model=LavaEnhance2(str(model_path),'cpu')
  from LavaSR.enhancer.linkwitz_merge import FastLRMerge
  model.bwe_model.lr_refiner=FastLRMerge(device='cpu',cutoff=8000,transition_bins=1024)
  tensor=torch.from_numpy(x16.copy()).float().unsqueeze(0)
  infer=lambda:model.enhance(tensor,denoise=True,batch=False)
  provenance=dict(checkpoint='YatharthS/LavaSR enhancer_v2 + denoiser (local weights)',denoise=True,cutoff_hz=8000)
 elif args.chain=='lavasrden':
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  sys.path.insert(0,str(ROOT/'vc_models/post_lavasr/vendor'))
  sys.path.insert(0,str(ROOT/'vc_models/post_lavasr/repo'))
  from LavaSR.model import LavaEnhance2
  model=LavaEnhance2(str(ROOT/'vc_models/post_lavasr/weights'),'cpu')
  tensor=torch.from_numpy(x16.copy()).float().unsqueeze(0)
  with torch.inference_mode():
   denoised=model.denoiser_model.infer(tensor)
  infer_result=denoised.detach().cpu().float().reshape(-1).numpy()
  def infer():return infer_result
  provenance=dict(stage='post_lavasr denoiser at 16kHz; BWE deliberately not applied (cascade stage 1)')
  output_sr=16000
 elif args.chain=='voicefixer2' or args.chain=='flashsr_voicefixer2':
  sys.path.insert(0,str(ROOT/'vc_models/post_voicefixer2/vendor'))
  sys.path.insert(0,str(ROOT/'vc_models/post_voicefixer2/repo'))
  # The upstream hf:// checkpoints (voicefixer/voicefixer, voicefixer/vocoder)
  # are unavailable; resolve to the byte-identical official mirror instead.
  import cached_path as cached_path_module
  weights=ROOT/'vc_models/post_voicefixer2/weights'
  expected={'hf://voicefixer/voicefixer/vf.ckpt':('vf.ckpt','748411b70089cadf34a6c11054f95f3a454e614af562c23b13a82f6cb413109f'),
            'hf://voicefixer/vocoder/model.ckpt-1490000_trimed.pt':('vocoder_44100.pt','9410d0b528c10a251ae947bd299d1939b0b3247df680c81c4164e94f5d87dc45')}
  def local_cached_path(url,*args,**kwargs):
   if str(url) in expected:
    name,want=expected[str(url)];path=weights/name
    if not path.exists():raise FileNotFoundError('Missing VoiceFixer checkpoint: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as stream:
     for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=want:raise ValueError('VoiceFixer checkpoint checksum mismatch: '+name)
    return str(path)
   return cached_path_module.cached_path(url,*args,**kwargs)
  cached_path_module.cached_path=local_cached_path
  if args.chain=='flashsr_voicefixer2':
   sys.path.insert(0,str(ROOT/'vc_models/post_flashsr/vendor'))
   import onnxruntime as ort
   options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
   session=ort.InferenceSession(str(ROOT/'vc_models/post_flashsr/repo/models/model.onnx'),options,providers=['CPUExecutionProvider'])
   flash=session.run(None,{'x':x16[None,None,:].astype(np.float32)})[0].reshape(-1)
   if not np.isfinite(flash).all():raise RuntimeError('Invalid FlashSR output')
   flash=flash/(np.max(abs(flash))+1e-7)*.999
   provenance['stage1']='Official FlashSR models/model.onnx 16k->48k; official peak normalization retained'
   source_48k=flash
  else:
   source_48k=original
  temp=Path(tempfile.mkdtemp(prefix='vf2_',dir=os.environ.get('TMP')))
  work=temp/'in.wav'
  sf.write(work,source_48k,48000,subtype='FLOAT')
  from voicefixer import VoiceFixer
  vf=VoiceFixer()
  out_path=temp/'out.wav'
  vf.restore(str(work),str(out_path),cuda=False,mode=0)
  restored,rate=sf.read(out_path,dtype='float32')
  if restored.ndim!=1 or not np.isfinite(restored).all():raise RuntimeError('Invalid VoiceFixer output')
  output_sr=rate
  provenance['stage2']='Render-AI-Team voicefixer2 vf.ckpt + vocoder model.ckpt-1490000_trimed.pt (official Zenodo mirror, sha256-verified); mode=0'
  infer_result=restored
  def infer():return infer_result
 else:
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  sys.path.insert(0,str(ROOT/'vc_models/post_lavasr/vendor'))
  sys.path.insert(0,str(ROOT/'vc_models/post_lavasr/repo'))
  sys.path.insert(0,str(ROOT/'vc_models/post_flashsr/vendor'))
  from LavaSR.model import LavaEnhance2
  model=LavaEnhance2(str(ROOT/'vc_models/post_lavasr/weights'),'cpu')
  tensor=torch.from_numpy(x16.copy()).float().unsqueeze(0)
  with torch.inference_mode():
   denoised=model.denoiser_model.infer(tensor)
  denoised=denoised.detach().cpu().float().reshape(-1).numpy()
  if not np.isfinite(denoised).all() or len(denoised)!=len(x16):raise RuntimeError('Invalid denoiser output')
  import onnxruntime as ort
  options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
  session=ort.InferenceSession(str(ROOT/'vc_models/post_flashsr/repo/models/model.onnx'),options,providers=['CPUExecutionProvider'])
  flash=session.run(None,{'x':denoised[None,None,:].astype(np.float32)})[0].reshape(-1)
  if not np.isfinite(flash).all():raise RuntimeError('Invalid FlashSR output')
  flash=flash/(np.max(abs(flash))+1e-7)*.999
  infer_result=flash
  def infer():return infer_result
  provenance=dict(stage1='post_lavasr denoiser at 16kHz (local weights); no BWE',stage2='Official FlashSR models/model.onnx 16k->48k; official peak normalization retained')
  output_sr=48000
 if args.chain in ('flashsr','lavasrden','voicefixer2','flashsr_voicefixer2','lavasrden_flashsr'):
  output=infer_result
 else:
  with torch.inference_mode():output=infer()
 if hasattr(output,'detach'):output=output.detach().float().cpu().numpy()
 output=np.asarray(output,dtype=np.float32).squeeze()
 if output.ndim!=1 or not np.isfinite(output).all() or np.max(abs(output))<1e-5:raise ValueError('Invalid/silent output')
 if abs(len(output)/output_sr-len(original)/sr)>.1:raise ValueError('Output duration changed >100ms')
 args.output.parent.mkdir(parents=True,exist_ok=True);sf.write(args.output,output,output_sr,subtype='FLOAT')
 args.metadata.parent.mkdir(parents=True,exist_ok=True)
 elapsed=time.monotonic()-start
 args.metadata.write_text(json.dumps(dict(output=str(args.output),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_revision='postvc2_conditions_v1',chain=args.chain,load_and_download_seconds=None,generation_seconds=elapsed,rtf=elapsed/(len(original)/sr),input_seconds=len(original)/sr,output_seconds=len(output)/output_sr,output_sample_rate=output_sr,output_clipping=int(np.sum(abs(output)>=.999)),output_peak=float(np.max(abs(output))),precision='CPU FP32',provenance=provenance),indent=2),'utf-8')

if __name__=='__main__':main()
