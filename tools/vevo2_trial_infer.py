"""Offline staged Vevo2 trial: cache tokens, then AR, then FM, then vocoder.

Official inference maths/weights stay unchanged. Models are released between
stages to avoid holding Whisper+AR+FM+vocoder concurrently on the N150.
"""
import gc, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'vc_models/vevo2'
OUT=ROOT/'recordings/v011_vevo2_style_trial'
sys.path.insert(0,str(ROOT/'vc_models/amphion/repo'))
from models.svc.vevo2 import vevo2_utils as v
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
def stamp(stage):
    print(stage,flush=True)
    (OUT/'stage.json').write_text(json.dumps({'stage':stage,'time':time.time()}))
def clear():
    gc.collect()
def checkpoint(builder,cfg,path):
    # Construct the official CPU model so nonpersistent rotary/STFT buffers
    # remain valid. Assign mapped weights rather than copying a second state.
    import safetensors.torch
    model=builder(cfg,torch.device('cpu'))
    state=safetensors.torch.load_file(str(path/'model.safetensors'),device='cpu')
    mismatch=model.load_state_dict(state,strict=False,assign=True)
    if mismatch.missing_keys:raise RuntimeError('Missing checkpoint weights: '+repr(mismatch.missing_keys))
    if any(t.device.type=='meta' for t in list(model.parameters())+list(model.buffers())):
        raise RuntimeError('Unmaterialized checkpoint tensors')
    del state
    model.eval()
    return model
class EncoderOnly:
    def __init__(self):
        import whisper
        from whisper.model import AudioEncoder
        state=torch.load(BASE/'whisper/medium.pt',map_location='cpu',mmap=True,weights_only=True)
        d=state['dims']
        with torch.device('meta'):
            self.encoder=AudioEncoder(d['n_mels'],d['n_audio_ctx'],d['n_audio_state'],d['n_audio_head'],d['n_audio_layer'])
        self.encoder.load_state_dict({k[len('encoder.'):]:x.float() for k,x in state['model_state_dict'].items() if k.startswith('encoder.')},assign=True)
        self.encoder.eval()
        del state
    def embed_audio(self,mel):return self.encoder(mel)
def main():
    stage=sys.argv[sys.argv.index('--stage')+1] if '--stage' in sys.argv else 'all'
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((OUT/'inputs.json').read_text('utf-8'))
    ck=BASE/'checkpoints'
    fmt_path=ck/'acoustic_modeling/fm_emilia101k_singnet7k_repa'
    ar_path=ck/'contentstyle_modeling/posttrained'
    voc_path=ck/'vocoder'
    v.load_checkpoint=lambda builder,cfg,path,device:checkpoint(builder,cfg,Path(path))
    p=v.Vevo2InferencePipeline.__new__(v.Vevo2InferencePipeline)
    p.device=torch.device('cpu');p.use_vllm=False
    p.fmt_cfg=v.load_config(str(fmt_path/'config.json'))
    p.ar_cfg=v.load_config(str(ar_path/'amphion_config.json'))
    p.fmt_use_text_as_condition=False
    token_file=OUT/'tokens.pt'
    if not token_file.exists():
        stamp('WHISPER_AND_CONTENT_STYLE_TOKENS')
        p.whisper_model=EncoderOnly()
        p.use_normed_whisper=True
        stats=torch.load(fmt_path/'whisper_stats.pt',map_location='cpu',weights_only=True)
        p.whisper_mean=stats['mean'];p.whisper_std=stats['std']
        p.content_style_tokenizer=checkpoint(v.build_coco_model,p.fmt_cfg.model.coco,ck/'tokenizer/contentstyle_fvq16384_12.5hz')
        tokens={}
        with torch.inference_mode():
            for name in ['style','timbre']:
                saved=OUT/(name+'_tokens.pt')
                if saved.exists():
                    tokens[name]=torch.load(saved,weights_only=True)
                    continue
                speech,_,s16=v.load_wav(cfg[name],p.device)
                tokens[name]=p.extract_coco_codec('content_style',s16,speech)
                torch.save(tokens[name],saved)
        torch.save(tokens,token_file)
        del p.whisper_model,p.content_style_tokenizer
        clear()
    if stage=='tokens':return
    tokens=torch.load(token_file,weights_only=True)
    codefile=OUT/'generated_codes.pt'
    if not codefile.exists():
        stamp('AR_STYLE_CONVERSION')
        # Keep the official stored BF16 dtype to fit the current available
        # RAM. Latency is explicitly not a selection criterion for this trial.
        p.ar_model=v.AutoModelForCausalLM.from_pretrained(str(ar_path),device_map='cpu',
            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True)
        p.ar_tokenizer=v.AutoTokenizer.from_pretrained(str(ar_path),local_files_only=True)
        # Style-converted speech VC uses source transcript and a new style
        # reference; copying source prosody would defeat the purpose here.
        prompt=p.get_llm_prompt_text(cfg['style_text']+' '+cfg['target_text'],False)
        prompt+='<|content_style_start|>'+''.join(f'<|content_style_{i}|>' for i in tokens['style'][0].tolist())
        ids=p.ar_tokenizer.encode(prompt,add_special_tokens=True)
        ids=torch.tensor([ids],dtype=torch.long)
        torch.manual_seed(11)
        with torch.inference_mode():
            generated=p.ar_model.generate(input_ids=ids,min_new_tokens=15,max_new_tokens=200,eos_token_id=p.ar_tokenizer.eos_token_id,do_sample=True,top_k=25,top_p=.8,temperature=1.)
        codes=p.parse_llm_generated_ids(generated,ids,logging=True)
        if codes.shape[1]==0:raise RuntimeError('AR generated no audio tokens')
        (OUT/'ar_generation.json').write_text(json.dumps({'generated_token_count':int(generated.shape[1]-ids.shape[1]),
            'audio_token_count':int(codes.shape[1]),'max_new_tokens':200,
            'ended_with_eos':int(generated[0,-1])==p.ar_tokenizer.eos_token_id,
            'seed':11,'target_text':cfg['target_text']},ensure_ascii=False,indent=2),encoding='utf-8')
        torch.save(codes,codefile)
        (OUT/'ar_decoded.txt').write_text(p.ar_tokenizer.decode(generated[0]),encoding='utf-8')
        del p.ar_model,p.ar_tokenizer,generated
        clear()
    if stage=='ar':return
    codes=torch.load(codefile,weights_only=True)
    melfile=OUT/'generated_mel.pt'
    if not melfile.exists():
        stamp('FLOW_MATCHING_32_STEPS')
        torch.manual_seed(11)
        p.fmt_model=checkpoint(v.build_fmt_model,p.fmt_cfg,fmt_path)
        p.vocoder_cfg=v.load_config(str(voc_path/'config.json'))
        p.mel_model=v.build_mel_model(p.vocoder_cfg,p.device)
        # Same official code2mel, with the already cached reference tokens.
        p.extract_coco_codec=lambda *args,**kwargs:tokens['timbre']
        with torch.inference_mode():mel=p.code2mel(codes,cfg['timbre'],flow_matching_steps=32)
        torch.save(mel,melfile)
        del p.fmt_model,p.mel_model
        clear()
    if stage=='fm':return
    stamp('VOCODER')
    p.vocoder_cfg=v.load_config(str(voc_path/'config.json'))
    p.vocoder_model=checkpoint(v.build_vocoder_model,p.vocoder_cfg,voc_path)
    mel=torch.load(melfile,weights_only=True)
    with torch.inference_mode():audio=p.mel2audio(mel).numpy().reshape(-1)
    if not len(audio) or not np.isfinite(audio).all():raise RuntimeError('Invalid output waveform')
    sf.write(OUT/'vevo2_style_raw.wav',audio,24000,subtype='FLOAT')
    stamp('COMPLETE')
if __name__=='__main__':main()
