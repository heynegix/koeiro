"""Reuse the verified tournament reference/embedding; no downloads or training."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def install_approved_profiles():
    source=ROOT/'recordings/v011_meanvc2_improvements'
    feedback=json.loads((ROOT/'validation/v011/meanvc2_improvement_ratings.json').read_text('utf-8'))
    manifest=json.loads((source/'metadata/blind_manifest.json').read_text('utf-8'))
    if feedback['package']!=manifest['package']:raise ValueError('Human ratings package mismatch')
    original=json.loads((ROOT/'models/meanvc2_120/runtime.json').read_text('utf-8'))
    from_profile=[('meanvc2_ref20','ref20','legacy','legacy','自分の声 · メイン'),
                  ('meanvc2_ref60','ref60_aligned','aligned','fixed_linear','自分の声 · 第2音声')]
    for ident,variant,frontend,interpolation,label in from_profile:
        accepted=[c for c in feedback['candidates'] if c['variant']==variant]
        if len(accepted)!=3 or any(c['ratings']['decision']!='Keep' for c in accepted):
            raise ValueError('Expected three human Keep ratings for '+variant)
        ref=variant.split('_')[0];stamp=json.loads((source/'reference'/(ref+'.json')).read_text('utf-8'))
        embedding=source/'reference'/(ref+'.npy');reference=source/'reference'/(ref+'.wav')
        if sha(embedding)!=stamp['embedding_sha256'] or sha(reference)!=stamp['reference_sha256']:
            raise ValueError('Approved reference/embedding checksum mismatch')
        vector=np.load(embedding,allow_pickle=False)
        if vector.shape!=(256,) or not np.isfinite(vector).all():raise ValueError('Invalid approved speaker vector')
        folder=ROOT/'models'/ident;folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(embedding,folder/'fixed_embedding.npy');shutil.copy2(reference,folder/'reference.wav')
        assets=[a.copy() for a in original['assets'] if a['path'].startswith('vc_models/')]
        for name in ('reference.wav','fixed_embedding.npy'):
            assets.append(dict(path='models/'+ident+'/'+name,sha256=sha(folder/name)))
        runtime=dict(backend='meanvc2',preset='120ms',steps=2,assets=assets,default_threads=4,
                     feature_frontend=frontend,bn_interpolation=interpolation,vocoder_context=36,
                     display_name=label,approved_variant=variant,human_approved_offline=True,
                     human_approved_blind_ids=[c['blind_id'] for c in accepted],ratings_package=feedback['package'],
                     reference_origin='combined_67clips.wav; '+ref+'; fixed 5s-window centroid',
                     reference_embedding_method=stamp['method'],retained_reference_offsets_seconds=[stamp['offsets_seconds'][i] for i in stamp['retained_indices']])
        if ident=='meanvc2_ref60':
            runtime.update(vocoder_batch_frames=1,vc_group_chunks=6,continuity_variant='vc720',
                           continuity_package='5886837dd9012654ca79a1cd2553eaad85ba1a22b12d1ebca074f9a3c99b3fee',
                           continuity_human_approval='User approved all five long candidates and authorized selection by measurements.')
        (folder/'runtime.json').write_text(json.dumps(runtime,indent=2,ensure_ascii=False),encoding='utf-8')
        print('Installed '+ident+' from '+variant)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,choices=(2,3),default=2);parser.add_argument('--install-approved-profiles',action='store_true');args=parser.parse_args()
    if args.install_approved_profiles:return install_approved_profiles()
    base=ROOT/'vc_models/meanvc2';repo=base/'repo';folder=ROOT/'models/meanvc2_120';folder.mkdir(parents=True,exist_ok=True)
    ref=ROOT/'recordings/v011_mega_tournament/reference/reference_10s.wav'
    inventory=json.loads((base/'artifacts.json').read_text('utf-8'))
    hashes={r['path'].replace('\\','/'):r['sha256'] for r in inventory['files']}
    key=sha(ref)+hashes['repo/preprocess/ckpts/wavlm_large_finetune.pth']+hashes['repo/preprocess/ckpts/wavlm_large.pt']+sha(repo/'preprocess/models/ecapa_tdnn.py')+sha(base/'requirements.lock.txt')
    embedding=base/'reference_embeddings'/(hashlib.sha256(key.encode()).hexdigest()+'.npy')
    x=np.load(embedding,allow_pickle=False)
    if x.shape!=(256,) or not np.isfinite(x).all():raise ValueError('Invalid existing speaker embedding')
    shutil.copy2(ref,folder/'reference.wav');shutil.copy2(embedding,folder/'fixed_embedding.npy')
    assets=[]
    for relative in ['repo/preprocess/ckpts/fastu2pp_160ms.pt','repo/ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors','repo/ckpts/vocos/vocos.pt']:
        assets.append(dict(path='vc_models/meanvc2/'+relative,sha256=hashes[relative]))
    for name in ['reference.wav','fixed_embedding.npy']:
        assets.append(dict(path='models/meanvc2_120/'+name,sha256=sha(folder/name)))
    runtime=dict(backend='meanvc2',preset='120ms',steps=args.steps,assets=assets,embedding_cache_key=embedding.stem,
        reference_origin='combined_67clips.wav; tournament 10s reference',default_threads=4)
    (folder/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    print(json.dumps(runtime,indent=2))
if __name__=='__main__':main()
