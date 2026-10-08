"""One model per bounded offline subprocess; no audio devices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--metadata',type=Path,required=True);args=parser.parse_args()
 import torch
 import numpy as np
 import soundfile as sf
 from scipy.signal import resample_poly
 from huggingface_hub import hf_hub_download,snapshot_download
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 folder=ROOT/'vc_models'/('post_'+args.model);start=time.monotonic()
 original,sr=sf.read(args.input,dtype='float32');x=resample_poly(original,1,3) if sr==48000 else original
 if sr not in (16000,48000):raise ValueError('Expected 16/48k mono input')
 if original.ndim!=1:raise ValueError('Expected mono')
 tensor=torch.from_numpy(x.copy()).float().unsqueeze(0)
 provenance={};output_sr=48000
 if args.model=='lavasr':
  from LavaSR.model import LavaEnhance2
  model_path=snapshot_download('YatharthS/LavaSR',allow_patterns=['enhancer_v2/*','denoiser/*'],local_dir=folder/'weights')
  model=LavaEnhance2(model_path,'cpu')
  from LavaSR.enhancer.linkwitz_merge import FastLRMerge
  model.bwe_model.lr_refiner=FastLRMerge(device='cpu',cutoff=8000,transition_bins=1024)
  infer=lambda:model.enhance(tensor,denoise=False,batch=False)
  provenance.update(denoise=False,cutoff_hz=8000,checkpoint='YatharthS/LavaSR enhancer_v2')
 elif args.model=='flashsr':
  import onnxruntime as ort
  options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
  checkpoint=folder/'repo/models/model.onnx'
  model=ort.InferenceSession(str(checkpoint),options,providers=['CPUExecutionProvider'])
  def infer():
   result=model.run(None,{'x':x[None,None,:].astype(np.float32)})[0].reshape(-1)
   return result/(np.max(abs(result))+1e-7)*.999
  provenance['checkpoint']='Official repository models/model.onnx; official peak normalization retained'
 elif args.model=='novasr':
  from NovaSR import FastSR
  ckpt=hf_hub_download('YatharthS/NovaSR','pytorch_model_v1.bin',local_dir=folder/'weights')
  model=FastSR(ckpt,half=False);infer=lambda:model.infer(tensor.unsqueeze(1))
  provenance['checkpoint']='YatharthS/NovaSR pytorch_model_v1.bin'
 elif args.model=='apbwe':
  import gdown
  weights=folder/'weights';weights.mkdir(exist_ok=True)
  if not list(weights.rglob('g_16kto48k*')):
   listing=gdown.download_folder('https://drive.google.com/drive/folders/1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF',output=str(weights),quiet=False,skip_download=True,use_cookies=False)
   for item in listing:
    if '16kto48k' not in item.path.lower():continue
    target=Path(item.local_path).resolve()
    if not target.is_relative_to(weights.resolve()):raise ValueError('Unsafe checkpoint path')
    if not (target.name.startswith('g_') or target.name=='config.json'):continue
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():gdown.download(id=item.id,output=str(target),quiet=False,use_cookies=False)
  candidates=list(weights.rglob('g_16kto48k*'))
  if not candidates:raise FileNotFoundError('Official 16kto48k checkpoint missing')
  sys.path.insert(0,str(folder/'repo'));sys.path.insert(0,str(folder/'repo/models'))
  from env import AttrDict
  from models.model import APNet_BWE_Model
  from datasets.dataset import amp_pha_stft,amp_pha_istft
  import torchaudio.functional as af
  checkpoint=sorted(candidates)[-1];config=AttrDict(json.loads((checkpoint.parent/'config.json').read_text('utf-8')))
  if config.lr_sampling_rate!=16000 or config.hr_sampling_rate!=48000:raise ValueError('Wrong AP-BWE rate configuration')
  model=APNet_BWE_Model(config).eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu')['generator'])
  def infer():
   upsampled=af.resample(tensor,16000,48000)
   amplitude,phase,_=amp_pha_stft(upsampled,config.n_fft,config.hop_size,config.win_size)
   amp,pha,_=model(amplitude,phase)
   return amp_pha_istft(amp,pha,config.n_fft,config.hop_size,config.win_size)
  provenance['checkpoint']=str(checkpoint)
 elif args.model=='resemble':
  from resemble_enhance.enhancer.inference import enhance
  infer=lambda:enhance(torch.from_numpy(original.copy()),sr,'cpu',nfe=16,solver='midpoint',lambd=0.1,tau=0.5)
  output_sr=44100;provenance['checkpoint']='Resemble Enhance official auto-download; nfe16'
 elif args.model=='mossformer_sr':
  # ClearVoice can silently continue with random weights after a failed download.
  # Require every inference checkpoint before constructing the public wrapper.
  checkpoint_dir=folder/'checkpoints/MossFormer2_SR_48K'
  for filename in ('last_best_checkpoint','last_best_checkpoint_m.pt','last_best_checkpoint_g.pt'):
   hf_hub_download('alibabasglab/MossFormer2_SR_48K',filename,revision='39eb1f25ea84f5e0315ade9ac0070fff216fc690',local_dir=checkpoint_dir)
  for filename in (checkpoint_dir/'last_best_checkpoint').read_text('utf-8').splitlines():
   if not filename:continue
   if Path(filename).name!=filename or not (checkpoint_dir/filename).is_file():raise FileNotFoundError('Required pretrained weights missing: '+filename)
  os.chdir(folder)
  from clearvoice import ClearVoice
  model=ClearVoice(task='speech_super_resolution',model_names=['MossFormer2_SR_48K'])
  infer=lambda:model(input_path=str(args.input),online_write=False)
  provenance['checkpoint']='MossFormer2_SR_48K official auto-download'
 else:raise ValueError(args.model)
 loaded=time.monotonic()
 with torch.inference_mode():output=infer()
 elapsed=time.monotonic()-loaded
 if args.model=='resemble':output,output_sr=output
 if isinstance(output,dict):
  if len(output)!=1:raise ValueError('Ambiguous model output')
  output=next(iter(output.values()))
 if isinstance(output,torch.Tensor):output=output.detach().float().cpu().numpy()
 output=np.asarray(output,dtype=np.float32).squeeze()
 if output.ndim!=1 or not np.isfinite(output).all() or np.max(abs(output))<1e-5:raise ValueError('Invalid/silent output')
 if abs(len(output)/output_sr-len(original)/sr)>.1:raise ValueError('Output duration changed >100ms')
 args.output.parent.mkdir(parents=True,exist_ok=True);sf.write(args.output,output,output_sr,subtype='FLOAT')
 model_bytes=sum(p.stat().st_size for p in (folder/'weights').rglob('*') if p.is_file())
 adapter_revision='moss_prefetch_v2' if args.model=='mossformer_sr' else '2d4c0b484747c0f92fb1de782d736f450b5edca50e79a62a307b803ad34bea33'
 args.metadata.write_text(json.dumps(dict(output=str(args.output),adapter_sha256=adapter_sha256,adapter_revision=adapter_revision,model_download_bytes=model_bytes,load_and_download_seconds=loaded-start,generation_seconds=elapsed,rtf=elapsed/(len(original)/sr),input_seconds=len(original)/sr,output_seconds=len(output)/output_sr,output_clipping=int(np.sum(abs(output)>=.999)),output_peak=float(np.max(abs(output))),precision='CPU FP32',provenance=provenance),indent=2),'utf-8')

if __name__=='__main__':main()
