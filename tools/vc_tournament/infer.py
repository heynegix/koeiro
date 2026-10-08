"""One isolated, CPU-only offline inference process per source/candidate.

Adapters follow checked-out official inference code. No microphone/GUI/training.
Raw outputs are float WAV where APIs expose arrays; upstream CLI output is retained.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MODEL,SOURCE,REFERENCE,OUTPUT=sys.argv[1:5]
SOURCE=Path(SOURCE);REFERENCE=Path(REFERENCE);OUTPUT=Path(OUTPUT)
REPO=Path.cwd().resolve();BASE=REPO.parent
os.environ['HF_HUB_CACHE']=str(ROOT/'vc_models/cache/hf')
sys.path[:]=[p for p in sys.path if p and Path(p).resolve()!=ROOT]
sys.path.insert(0,str(REPO))
metrics={'model_load_seconds':None,'reference_seconds':None,'generation_seconds':None,
         'first_audio_latency_seconds':None,'stage':'imports','device':'cpu'}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def mark(stage):
    metrics['stage']=stage
    sidecar=OUTPUT.with_suffix('.metrics.json')
    sidecar.parent.mkdir(parents=True,exist_ok=True)
    temporary=sidecar.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    temporary.replace(sidecar)
    print('[stage] '+stage,flush=True)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def wav(path,sr=16000):
    import librosa
    return librosa.load(str(path),sr=sr)[0]

def write(x,sr):
    import soundfile as sf
    if hasattr(x,'detach'):x=x.detach().cpu().float().squeeze().numpy()
    sf.write(OUTPUT,x,sr,subtype='FLOAT')

def meanvc2():
    import numpy as np
    import torch
    code=REPO/'src/infer/infer_e2e.py'
    sys.path.insert(0,str(code.parent))
    # Upstream package __init__ imports Trainer/evaluation dependencies even for
    # offline inference. Expose its unchanged submodules without those exports.
    import types
    package=types.ModuleType('src.model')
    package.__path__=[str(REPO/'src/model')]
    sys.modules['src.model']=package
    e=module(code,'meanvc2_e2e')
    preset=e.MODEL_PRESETS['40ms' if MODEL=='meanvc2_40' else '120ms']
    mark('model_load');t=time.perf_counter()
    asr=e.load_asr_model(preset['asr_ckpt'],'cpu')
    with open(preset['vc_config']) as f:config=json.load(f)
    vc=e.DiT(**config['model']).to('cpu')
    vc=e.load_checkpoint(vc,preset['vc_ckpt'],device='cpu',use_ema=True).float().eval()
    vocos=torch.jit.load(str(REPO/'ckpts/vocos/vocos.pt'),map_location='cpu').eval()
    metrics['model_load_seconds']=time.perf_counter()-t
    mark('reference_embedding');t=time.perf_counter()
    # Shared fixed speaker embedding between 40ms/120ms and all source clips.
    key=sha(REFERENCE)+sha(REPO/'preprocess/ckpts/wavlm_large_finetune.pth')+sha(REPO/'preprocess/ckpts/wavlm_large.pt')+sha(REPO/'preprocess/models/ecapa_tdnn.py')+sha(BASE/'requirements.lock.txt')
    cache=BASE/'reference_embeddings';cache.mkdir(exist_ok=True)
    embfile=cache/(hashlib.sha256(key.encode()).hexdigest()+'.npy')
    metrics['fixed_embedding_cache_hit']=embfile.exists()
    if embfile.exists():spk=np.load(embfile,allow_pickle=False)
    else:
        encoder=e.load_spk_model('cpu');spk=e.extract_spk_emb(str(REFERENCE),encoder,'cpu')
        np.save(embfile,spk);del encoder
    metrics['reference_seconds']=time.perf_counter()-t
    mark('generation');t=time.perf_counter()
    torch.manual_seed(110) # Embedding cache misses must not change VC noise.
    bn=e.extract_bn(str(SOURCE),asr,window=preset['bn_window'],stride=preset['bn_stride'],offset_init=preset['offset_init'],offset_step=preset['offset_step'],req_cache=preset['req_cache'])
    _,audio=e.vc_inference(vc,vocos,bn,spk,preset['chunk_size'],preset['block_size'],3,'cpu')
    metrics['generation_seconds']=time.perf_counter()-t
    write(audio,16000)

def meanvc():
    import numpy as np
    sys.path.insert(0,str(REPO/'src/runtime'))
    e=module(REPO/'src/runtime/run_rt.py','meanvc_rt_file')
    mark('model_load');t=time.perf_counter()
    # Equivalent official s3prl WavLM expert, with explicit shared checkpoint.
    # Avoid torch.hub re-downloading the same encoder under a second filename.
    import torch
    from s3prl.upstream.wavlm.expert import UpstreamExpert
    old_hub_load=torch.hub.load
    def local_wavlm(repo_name,model_name,*args,**kwargs):
        if repo_name=='s3prl/s3prl' and model_name=='wavlm_large':
            return UpstreamExpert(ckpt=str(ROOT/'vc_models/cache/torch/hub/checkpoints/WavLM-Large.pt'))
        return old_hub_load(repo_name,model_name,*args,**kwargs)
    torch.hub.load=local_wavlm
    try:engine=e.VCRunner(str(REFERENCE),steps=2)
    finally:torch.hub.load=old_hub_load
    engine.init_cache()
    metrics['model_load_seconds']=time.perf_counter()-t
    audio=wav(SOURCE);mark('generation');t=time.perf_counter();chunks=[]
    # Mirror the file-independent chunk primitive; never invoke run()/PyAudio().
    first=engine.CHUNK+engine.samples_cache_len
    start=0
    while start<len(audio):
        size=first if start==0 else engine.CHUNK
        data=audio[start:start+size]
        if len(data)<size:data=np.pad(data,(0,size-len(data)))
        out=engine.inference_one_chunk(data);chunks.append(out)
        if len(chunks)==1:metrics['first_audio_latency_seconds']=time.perf_counter()-t
        start+=size
    metrics['generation_seconds']=time.perf_counter()-t
    write(np.concatenate(chunks),16000)

def knnvc():
    import torch
    import numpy as np
    mark('model_load');t=time.perf_counter()
    hub=module(REPO/'hubconf.py','local_knn_hub')
    model=hub.knn_vc(pretrained=True,prematched=True,device='cpu')
    metrics['model_load_seconds']=time.perf_counter()-t
    mark('reference_features');t=time.perf_counter()
    # Bound WavLM attention memory; retain every reference sample in 20s shards.
    cache=BASE/'reference_features';cache.mkdir(exist_ok=True)
    encoder=ROOT/'vc_models/cache/torch/hub/checkpoints/WavLM-Large.pt'
    key=hashlib.sha256((sha(REFERENCE)+sha(REPO/'matcher.py')+sha(encoder)+sha(BASE/'requirements.lock.txt')+'20s-v1').encode()).hexdigest()
    cached=cache/(key+'.npy')
    metrics['reference_features_cache_hit']=cached.exists()
    if cached.exists():
        ref=torch.from_numpy(np.load(cached,allow_pickle=False))
    else:
        audio=torch.from_numpy(wav(REFERENCE,16000))
        shards=[audio[start:start+20*16000] for start in range(0,len(audio),20*16000)]
        parts=cache/key;parts.mkdir(exist_ok=True)
        features=[]
        metrics['reference_shards_total']=len(shards)
        for index, shard in enumerate(shards):
            part=parts/f'{index:03d}.npy'
            if part.exists():feature=torch.from_numpy(np.load(part,allow_pickle=False))
            else:
                feature=model.get_matching_set([shard])
                temporary=part.with_suffix('.tmp.npy');np.save(temporary,feature.numpy());temporary.replace(part)
            features.append(feature)
            metrics['reference_shards_complete']=index+1;mark('reference_features')
        ref=torch.cat(features,dim=0)
        temporary=cached.with_suffix('.tmp.npy');np.save(temporary,ref.numpy());temporary.replace(cached)
    metrics['reference_seconds']=time.perf_counter()-t
    mark('generation');t=time.perf_counter()
    with torch.inference_mode():
        q=model.get_features(str(SOURCE));audio=model.match(q,ref,topk=4,tgt_loudness_db=None)
    metrics['generation_seconds']=time.perf_counter()-t;write(audio,16000)

def facodec():
    import torch
    import numpy as np
    from models.codec.ns3_codec import FACodecEncoderV2,FACodecDecoderV2
    mark('model_load');t=time.perf_counter()
    enc=FACodecEncoderV2(ngf=32,up_ratios=[2,4,5,5],out_channels=256)
    dec=FACodecDecoderV2(in_channels=256,upsample_initial_channel=1024,ngf=32,up_ratios=[5,5,4,2],vq_num_q_c=2,vq_num_q_p=1,vq_num_q_r=3,vq_dim=256,codebook_dim=8,codebook_size_prosody=10,codebook_size_content=10,codebook_size_residual=10,use_gr_x_timbre=True,use_gr_residual_f0=True,use_gr_residual_phone=True)
    enc.load_state_dict(torch.load(REPO/'ckpts/facodec/ns3_facodec_encoder_v2.bin',map_location='cpu',weights_only=False))
    dec.load_state_dict(torch.load(REPO/'ckpts/facodec/ns3_facodec_decoder_v2.bin',map_location='cpu',weights_only=False))
    enc.eval();dec.eval();metrics['model_load_seconds']=time.perf_counter()-t
    def tensor(path):
        x=wav(path);x=np.pad(x,(0,(-len(x))%200));return torch.from_numpy(x)[None,None,:]
    ref=tensor(REFERENCE);src=tensor(SOURCE)
    with torch.inference_mode():
        mark('reference_features');t=time.perf_counter()
        _,_,_,_,speaker=dec(enc(ref),enc.get_prosody_feature(ref),eval_vq=False,vq=True)
        metrics['reference_seconds']=time.perf_counter()-t
        mark('generation');t=time.perf_counter()
        _,codes,_,_,_=dec(enc(src),enc.get_prosody_feature(src),eval_vq=False,vq=True)
        embedded=dec.vq2emb(codes,use_residual=False)
        audio=dec.inference(embedded,speaker)
        metrics['generation_seconds']=time.perf_counter()-t
    write(audio,16000)

def vevo():
    import torch
    from models.vc.vevo.vevo_utils import VevoInferencePipeline
    from models.vc.flow_matching_transformer import llama_nar
    # Vevo requirements pin Transformers 4.41.2, while the checked-in attention
    # calls its newer config-based rotary constructor. Preserve the same RoPE
    # dimensions/base with the pinned version's constructor signature.
    legacy_rope=llama_nar.LlamaRotaryEmbedding
    def configured_rope(*args,config=None,**kwargs):
        if config is not None:
            if config.rope_scaling is not None:
                raise RuntimeError('Vevo scaled RoPE is not covered by the pinned-version adapter')
            kwargs.update(dim=getattr(config,'head_dim',config.hidden_size//config.num_attention_heads),
                          max_position_embeddings=config.max_position_embeddings,base=config.rope_theta)
        return legacy_rope(*args,**kwargs)
    llama_nar.LlamaRotaryEmbedding=configured_rope
    mark('model_load');t=time.perf_counter()
    ck=REPO/'ckpts/Vevo-local'
    model=VevoInferencePipeline(content_style_tokenizer_ckpt_path=str(ck/'tokenizer/vq8192'),
        fmt_cfg_path='models/vc/vevo/config/Vq8192ToMels.json',fmt_ckpt_path=str(ck/'acoustic_modeling/Vq8192ToMels'),
        vocoder_cfg_path='models/vc/vevo/config/Vocoder.json',vocoder_ckpt_path=str(ck/'acoustic_modeling/Vocoder'),device=torch.device('cpu'))
    metrics['model_load_seconds']=time.perf_counter()-t
    mark('generation');t=time.perf_counter()
    audio=model.inference_fm(src_wav_path=str(SOURCE),timbre_ref_wav_path=str(REFERENCE),flow_matching_steps=32)
    metrics['generation_seconds']=time.perf_counter()-t
    # Preserve the raw float waveform; apply the comparison gain only in blind/.
    write(audio,24000)

def freevc():
    import torch
    import librosa
    import utils
    from models import SynthesizerTrn
    from wavlm import WavLM,WavLMConfig
    from speaker_encoder.voice_encoder import SpeakerEncoder
    mark('model_load');t=time.perf_counter()
    config=REPO/'configs/freevc.json'
    if not config.exists():config=REPO/'logs/freevc.json'
    hp=utils.get_hparams_from_file(str(config))
    vc=SynthesizerTrn(hp.data.filter_length//2+1,hp.train.segment_size//hp.data.hop_length,**hp.model).cpu().eval()
    utils.load_checkpoint(str(REPO/'checkpoints/freevc.pth'),vc,None,True)
    state=torch.load(REPO/'wavlm/WavLM-Large.pt',map_location='cpu',weights_only=False)
    encoder=WavLM(WavLMConfig(state['cfg']));encoder.load_state_dict(state['model']);encoder.eval()
    speaker=SpeakerEncoder(str(REPO/'speaker_encoder/ckpt/pretrained_bak_5805000.pt'),device='cpu')
    metrics['model_load_seconds']=time.perf_counter()-t
    mark('reference_features');t=time.perf_counter()
    ref,_=librosa.effects.trim(wav(REFERENCE,hp.data.sampling_rate),top_db=20)
    emb=torch.from_numpy(speaker.embed_utterance(ref))[None]
    metrics['reference_seconds']=time.perf_counter()-t
    mark('generation');t=time.perf_counter()
    with torch.inference_mode():
        content=utils.get_content(encoder,torch.from_numpy(wav(SOURCE,hp.data.sampling_rate))[None])
        audio=vc.infer(content,g=emb)[0][0]
    metrics['generation_seconds']=time.perf_counter()-t;write(audio,hp.data.sampling_rate)

def openvoice():
    from openvoice.api import ToneColorConverter,OpenVoiceBaseClass
    mark('model_load');t=time.perf_counter();ck=BASE/'checkpoints_v2/converter'
    # Upstream forwards enable_watermark to a base constructor that rejects it.
    # Initialize the identical converter model while disabling watermark loading.
    class OfflineConverter(ToneColorConverter):
        def __init__(self,config):
            OpenVoiceBaseClass.__init__(self,config,device='cpu')
            self.watermark_model=None;self.version=getattr(self.hps,'_version_','v1')
    converter=OfflineConverter(str(ck/'config.json'))
    converter.load_ckpt(str(ck/'checkpoint.pth'))
    metrics['model_load_seconds']=time.perf_counter()-t
    mark('reference_features');t=time.perf_counter()
    src_se=converter.extract_se(str(SOURCE));ref_se=converter.extract_se(str(REFERENCE))
    metrics['reference_seconds']=time.perf_counter()-t
    mark('generation');t=time.perf_counter()
    audio=converter.convert(str(SOURCE),src_se,ref_se,output_path=None)
    metrics['generation_seconds']=time.perf_counter()-t;write(audio,converter.hps.data.sampling_rate)

def cli(argv,generated_folder=None):
    mark('upstream_cli');t=time.perf_counter()
    process=subprocess.Popen([str(a) for a in argv],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,encoding='utf-8',errors='replace',bufsize=1)
    captured=[]
    for line in process.stdout:
        print(line,end='',flush=True);captured.append(line)
    if process.wait():raise subprocess.CalledProcessError(process.returncode,argv)
    metrics['generation_including_load_seconds']=time.perf_counter()-t
    if generated_folder:
        found=list(Path(generated_folder).rglob('*.wav'))
        if len(found)!=1:raise RuntimeError(f'Expected one upstream output, found {len(found)}')
        shutil.copy2(found[0],OUTPUT)
    return ''.join(captured)

def seedvc():
    # Initialize Transformers' cached constants before upstream inference.py
    # replaces HF_HUB_CACHE with its own relative directory. Setup has already
    # downloaded the official encoder/vocoder snapshots; inference is offline.
    os.environ['HF_HUB_CACHE']=str(ROOT/'vc_models/cache/hf')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import transformers
    import huggingface_hub
    import io
    import re
    from types import SimpleNamespace
    tiny=MODEL=='seedvc_tiny'
    name='DiT_uvit_tat_xlsr_ema.pth' if tiny else 'DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth'
    conf='config_dit_mel_seed_uvit_xlsr_tiny.yml' if tiny else 'config_dit_mel_seed_uvit_whisper_small_wavenet.yml'
    os.environ['HF_HUB_CACHE']=str(ROOT/'vc_models/cache/hf')
    # XLS-R explicitly uses FP16 in the official script, which falls back to
    # very slow CPU kernels on N150. Only those two dtype expressions differ.
    source_code=(REPO/'inference.py').read_text('utf-8')
    replacements={
        'wav2vec_model = wav2vec_model.half()':'wav2vec_model = wav2vec_model.float()',
        'ori_outputs = wav2vec_model(\n                    ori_inputs.input_values.half(),':'ori_outputs = wav2vec_model(\n                    ori_inputs.input_values.to(wav2vec_model.dtype),',
    }
    for old,new in replacements.items():
        if source_code.count(old)!=1:raise RuntimeError('Seed-VC upstream CPU adaptation no longer matches source')
        source_code=source_code.replace(old,new)
    import types
    e=types.ModuleType('seedvc_offline');e.__file__=str(REPO/'inference.py')
    exec(compile(source_code,e.__file__,'exec'),e.__dict__)
    metrics['encoder_precision']='FP32 CPU; pretrained XLS-R/Whisper weights unchanged'
    os.environ['HF_HUB_CACHE']=str(ROOT/'vc_models/cache/hf')
    def cached_model(repo_id,model_filename='pytorch_model.bin',config_filename=None):
        def get(n):
            local=REPO/'checkpoints'/n
            return str(local) if local.exists() else huggingface_hub.hf_hub_download(repo_id,n,cache_dir=str(ROOT/'vc_models/cache/hf'))
        return (get(model_filename),get(config_filename)) if config_filename else get(model_filename)
    e.load_custom_model_from_hf=cached_model
    original_load=e.load_models
    def timed_load(args):
        import torch
        # Official loader hardcodes half precision for Whisper even on CPU.
        # Use FP32 on this CPU-only slate without changing VC controls.
        from transformers import WhisperModel
        original_whisper=WhisperModel.from_pretrained
        def cpu_whisper(*a,**kw):
            import torch
            kw.update(torch_dtype=torch.float32,cache_dir=str(ROOT/'vc_models/cache/hf'),local_files_only=True)
            return original_whisper(*a,**kw)
        WhisperModel.from_pretrained=cpu_whisper
        # XLS-R's official checkpoint uses legacy weight_g/weight_v names.
        # Keep that mathematically identical parameterization while loading,
        # avoiding Transformers treating pretrained convolution keys as absent.
        native_weight_norm=torch.nn.utils.parametrizations.weight_norm
        torch.nn.utils.parametrizations.weight_norm=torch.nn.utils.weight_norm
        mark('model_load');t=time.perf_counter()
        try:value=original_load(args)
        finally:
            WhisperModel.from_pretrained=original_whisper
            torch.nn.utils.parametrizations.weight_norm=native_weight_norm
        metrics['model_load_seconds']=time.perf_counter()-t
        model,semantic,f0,vocoder,camp,mel,mel_args=value
        semantic_calls=0
        def timed_semantic(audio):
            nonlocal semantic_calls
            phase='source_features' if semantic_calls==0 else 'reference_features'
            semantic_calls+=1;mark(phase);started=time.perf_counter()
            result=semantic(audio);metrics[phase+'_seconds']=time.perf_counter()-started
            return result
        def timed_vocoder(audio):
            mark('vocoder');started=time.perf_counter();result=vocoder(audio)
            metrics['vocoder_seconds']=time.perf_counter()-started;return result
        original_diffusion=model.cfm.inference
        def timed_diffusion(*a,**kw):
            mark('diffusion');started=time.perf_counter();result=original_diffusion(*a,**kw)
            metrics['diffusion_seconds']=time.perf_counter()-started;return result
        model.cfm.inference=timed_diffusion
        mark('generation')
        return model,timed_semantic,f0,timed_vocoder,camp,mel,mel_args
    e.load_models=timed_load
    captured=[]
    class Tee:
        def __init__(self,underlying):self.underlying=underlying
        def write(self,s):captured.append(s);return self.underlying.write(s)
        def flush(self):self.underlying.flush()
    with tempfile.TemporaryDirectory(prefix='vc-',dir=BASE) as d:
        args=SimpleNamespace(source=str(SOURCE),target=str(REFERENCE),output=d,checkpoint=str(REPO/'checkpoints'/name),config=str(REPO/'checkpoints'/conf),diffusion_steps=10 if tiny else 30,f0_condition=False,auto_f0_adjust=False,semi_tone_shift=0,fp16=False,length_adjust=1.0,inference_cfg_rate=.7)
        old_stdout=sys.stdout;sys.stdout=Tee(old_stdout)
        original_save=e.torchaudio.save
        def float_wav_save(path,audio,sample_rate,**kwargs):
            kwargs.update(encoding='PCM_F',bits_per_sample=32)
            return original_save(path,audio,sample_rate,**kwargs)
        e.torchaudio.save=float_wav_save
        try:e.main(args)
        finally:
            sys.stdout=old_stdout;e.torchaudio.save=original_save
        found=list(Path(d).glob('*.wav'))
        if len(found)!=1:raise RuntimeError('Seed-VC did not generate exactly one WAV')
        shutil.copy2(found[0],OUTPUT)
        match=re.search(r'RTF:\s*([0-9.eE+-]+)',''.join(captured))
        if match:
            import soundfile as sf
            metrics['generation_seconds']=float(match.group(1))*sf.info(str(OUTPUT)).duration
            metrics['generation_measurement']='Official time_vc_end - time_vc_start; includes source/reference feature extraction, excludes model load'

def xvc():
    import yaml
    from bins.infer_utils import load_xvc,load_pair_as_tensors,run_offline
    config=yaml.safe_load((REPO/'configs/xvc.yaml').read_text('utf-8'))
    config.update(highpass_cutoff_freq=0,volume_normalize=False)
    cfg=config['model']['generator']
    cfg['semantic_encoder']['encoder']['from_pretrained']['local_ckpt']=str(REPO/'pretrained/glm')
    cfg['semantic_encoder']['cfg']['local_ckpt']=str(REPO/'pretrained/glm')
    cfg['speaker_encoder']['pretrained_dir']=str(REPO/'pretrained/eres2net')
    cfgpath=BASE/'tournament_xvc.yaml';cfgpath.write_text(yaml.safe_dump(config),encoding='utf-8')
    mark('model_load');t=time.perf_counter()
    cfg,model,device=load_xvc(str(cfgpath),str(REPO/'ckpts/xvc.pt'),0,False)
    metrics['model_load_seconds']=time.perf_counter()-t
    mark('input_features');t=time.perf_counter()
    src,ref,ref_cond=load_pair_as_tensors(str(SOURCE),str(REFERENCE),cfg,device,1280,False)
    metrics['reference_seconds']=time.perf_counter()-t
    mark('generation');t=time.perf_counter();audio=run_offline(model,src,ref,ref_cond)
    metrics['generation_seconds']=time.perf_counter()-t;write(audio,int(cfg['sample_rate']))

def fragmentvc():
    ck=BASE/'checkpoints';files=list(ck.rglob('*.pt'))
    vc=next((p for p in files if 'fragment' in p.name.lower() or p.name=='model.pt'),None)
    voc=next((p for p in files if 'vocoder' in p.name.lower() or 'wavernn' in p.name.lower()),None)
    if not vc or not voc:raise FileNotFoundError('FragmentVC / WaveRNN release checkpoints unavailable')
    cli([sys.executable,'convert.py','-w',ck/'wav2vec_small.pt','-c',vc,'-v',voc,SOURCE,REFERENCE,'-o',OUTPUT])

def conan():
    import types
    from utils.commons.hparams import set_hparams,hparams
    # The upstream package eagerly imports an unused NSF vocoder whose
    # parallel_wavegan implementation is absent. Register only shipped HiFiGAN.
    package=types.ModuleType('tasks.tts.vocoder_infer')
    package.__path__=[str(REPO/'tasks/tts/vocoder_infer')]
    sys.modules['tasks.tts.vocoder_infer']=package
    import tasks.tts.vocoder_infer.hifigan
    from inference.Conan import StreamingVoiceConversion
    fast=MODEL=='conan_fast';ck=BASE/'checkpoints'
    main=next(iter(ck.rglob('Conan_fast' if fast else 'Conan')),None)
    emformer=next(iter(ck.rglob('Emformer_fast' if fast else 'Emformer')),None)
    voc=next(iter(ck.rglob('hifigan_vc')),None)
    if not all([main,emformer,voc]):raise FileNotFoundError('Official Conan/Emformer/HiFiGAN checkpoint folders missing')
    conf=main/'config.yaml'
    set_hparams(config=str(REPO/'egs/conan_emformer.yaml'),exp_name='',print_hparams=False)
    checkpoint_config=set_hparams(config=str(conf),exp_name='',print_hparams=False,global_hparams=False)
    hparams.update(checkpoint_config)
    hparams.update(work_dir=str(main),emformer_ckpt=str(emformer),vocoder_ckpt=str(voc),infer=True)
    if fast:hparams['right_context']=0
    mark('model_load');t=time.perf_counter();engine=StreamingVoiceConversion(hparams)
    metrics['model_load_seconds']=time.perf_counter()-t
    mark('generation');t=time.perf_counter();audio,_=engine.infer_once({'src_wav':str(SOURCE),'ref_wav':str(REFERENCE)})
    metrics['generation_seconds']=time.perf_counter()-t;write(audio,hparams['audio_sample_rate'])

def ezvc():
    if not all((BASE/'checkpoints'/name).exists() for name in ('model_2700000.safetensors','vocab.txt')):
        mark('checkpoint_access')
        raise FileNotFoundError('Official EZ-VC model/vocabulary unavailable; SPRINGLab/EZ-VC requires approved authenticated access. See setup download error; no substitute model used.')
    sys.path.insert(0,str(REPO/'src'))
    import cached_path
    cached_path.set_cache_dir(ROOT/'vc_models/cache/cached-path')
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_model,load_vocoder,infer_process
    from f5_tts.infer.utils_xeus import ApplyKmeans,load_xeus_model,extract_units
    mark('model_load');t=time.perf_counter()
    xeus=load_xeus_model('cpu').eval();kmeans=ApplyKmeans('cpu')
    vocoder=load_vocoder(vocoder_name='bigvgan',device='cpu')
    config=OmegaConf.load(REPO/'src/f5_tts/configs/F5TTS_Base_EZ-VC.yaml')
    vc=load_model(get_class('f5_tts.model.'+config.model.backbone),config.model.arch,str(BASE/'checkpoints/model_2700000.safetensors'),mel_spec_type='bigvgan',vocab_file=str(BASE/'checkpoints/vocab.txt'),device='cpu')
    metrics['model_load_seconds']=time.perf_counter()-t
    mark('reference_units');t=time.perf_counter();ref=extract_units(str(REFERENCE),xeus,kmeans,'cpu')
    metrics['reference_seconds']=time.perf_counter()-t
    mark('generation');t=time.perf_counter();src=extract_units(str(SOURCE),xeus,kmeans,'cpu')
    audio,sr,_=infer_process(str(REFERENCE),ref,src,vc,vocoder,mel_spec_type='bigvgan',nfe_step=12,speed=1,device='cpu')
    metrics['generation_seconds']=time.perf_counter()-t;write(audio,sr)

def audiocpp():
    # Upstream canonicalizes junctions before narrow-character fopen. Use real
    # ASCII directories and same-volume hardlinks, with no extra model copies.
    native_base=Path(os.environ.get('VC_NATIVE_DIR') or (ROOT/'native_staging'))
    native_base.mkdir(parents=True,exist_ok=True)
    if native_base.resolve()!=native_base:
        raise RuntimeError('Native staging directory must be a physical D: path')
    for folder in ('bin','models'):
        target=native_base/folder;target.mkdir(exist_ok=True)
        for source in (BASE/folder).rglob('*'):
            if not source.is_file():continue
            linked=target/source.relative_to(BASE/folder)
            linked.parent.mkdir(parents=True,exist_ok=True)
            if linked.exists():
                if not os.path.samefile(source,linked):
                    raise RuntimeError(f'Unexpected native staging file: {linked}')
            else:os.link(source,linked)
    executables=list((native_base/'bin').rglob('audiocpp_cli.exe'))
    models=[p for p in (native_base/'models').glob('*.gguf') if ('q4_k' in p.name if MODEL.endswith('_q4') else 'fp32' in p.name)]
    if not executables or not models:raise FileNotFoundError('Windows CPU binary / MeanVC2 GGUF missing')
    with tempfile.TemporaryDirectory(prefix='vc-',dir=native_base) as directory:
        temp=Path(directory);source=temp/'source.wav';reference=temp/'reference.wav';output=temp/'output.wav'
        shutil.copy2(SOURCE,source);shutil.copy2(REFERENCE,reference)
        captured=cli([executables[0],'--task','vc','--family','meanvc2','--model',models[0],'--backend','cpu','--threads','4',
                      '--seed','110','--metrics','--log','--audio',source,'--voice-ref',reference,'--out',output,'--out-format','float32'])
        metrics['sampler_configuration']='Native fixed sampling schedule; MeanVC2 family rejects num_inference_steps override'
        import re
        measured=re.search(r'^metrics\.wall_ms=([0-9.eE+-]+)',captured,re.M)
        if measured:
            metrics['generation_seconds']=float(measured.group(1))/1000
            metrics['generation_measurement']='Native offline session prepare + run; excludes initial model load and WAV disk write'
            metrics['rtf_scope']='session_prepare_and_generation'
        shutil.copy2(output,OUTPUT)

failure_dir=BASE/'preflight_failures';failure_dir.mkdir(exist_ok=True)
failure_inputs=[Path(__file__),REFERENCE,BASE/'requirements.lock.txt',BASE/'artifacts.json']
failure_key=hashlib.sha256((MODEL+'|'+ '|'.join(sha(p) if p.exists() else 'MISSING' for p in failure_inputs)).encode()).hexdigest()
failure_file=failure_dir/(failure_key+'.json')
try:
    if failure_file.exists() and '--retry-failed' not in sys.argv[5:]:
        prior=json.loads(failure_file.read_text('utf-8'))
        metrics.update(stage=prior['stage'],failure_cache_hit=True,error=prior['error'])
        raise RuntimeError('Cached source-independent model-load failure: '+prior['error'])
    if MODEL.startswith('meanvc2_cpu'):audiocpp()
    else:
        import torch
        # Windows libtorch's filename fopen can reject Japanese workspace paths.
        # Python opens the identical bytes; TorchScript accepts binary file objects.
        native_jit_load=torch.jit.load
        def unicode_jit_load(value,*args,**kwargs):
            if isinstance(value,(str,os.PathLike)):
                with open(value,'rb') as stream:return native_jit_load(stream,*args,**kwargs)
            return native_jit_load(value,*args,**kwargs)
        torch.jit.load=unicode_jit_load
        torch.set_num_threads(4);torch.manual_seed(110)
        if MODEL.startswith('meanvc2_'):meanvc2()
        elif MODEL=='meanvc':meanvc()
        elif MODEL.startswith('knnvc'):knnvc()
        elif MODEL.startswith('seedvc'):seedvc()
        elif MODEL.startswith('conan'):conan()
        elif MODEL=='facodec':facodec()
        elif MODEL=='vevo':vevo()
        elif MODEL=='freevc':freevc()
        elif MODEL=='fragmentvc':fragmentvc()
        elif MODEL=='openvoice':openvoice()
        elif MODEL=='xvc':xvc()
        elif MODEL=='ezvc':ezvc()
        elif MODEL=='noro':
            mark('upstream_cpu_compatibility')
            raise RuntimeError('N150 production unavailable: official Noro inference requires CUDA and outputs mel only; standalone pretrained BigVGAN WAV path unverified. No training or CPU optimization attempted.')
        else:raise ValueError('Unknown candidate '+MODEL)
    mark('complete')
except Exception as e:
    metrics.setdefault('error',f'{type(e).__name__}: {e}');traceback.print_exc()
    if not metrics.get('failure_cache_hit') and (
        metrics['stage'] in ('model_load','upstream_cpu_compatibility','checkpoint_access') or
        (metrics['stage']=='imports' and isinstance(e,(ImportError,FileNotFoundError)))):
        # Model initialization never consumed the source; do not repeat the same
        # fatal load for all three recordings. Changed weights/locks/code retry.
        failure_file.write_text(json.dumps({'stage':metrics['stage'],'error':metrics['error']}),encoding='utf-8')
finally:
    if os.name=='nt':
        try:
            import ctypes
            from ctypes import wintypes
            class MemoryCounters(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
            counters=MemoryCounters();counters.cb=ctypes.sizeof(counters)
            kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
            get_memory=ctypes.WinDLL('psapi').GetProcessMemoryInfo
            get_memory.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
            if get_memory(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
                metrics['worker_process_ram_bytes']=counters.ws
                metrics['worker_process_peak_ram_bytes']=counters.peak_ws
        except Exception as memory_error:metrics['memory_measurement_error']=str(memory_error)
    OUTPUT.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
if metrics.get('error'):sys.exit(1)
