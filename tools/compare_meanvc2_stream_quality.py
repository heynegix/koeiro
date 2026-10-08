"""Compare cached offline/streaming outputs; no inference, training or devices."""
import hashlib,json,random
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'recordings/v011_meanvc2_qualitycheck'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def normalize(source,destination):
    x,sr=sf.read(source,dtype='float64',always_2d=True)
    if not np.isfinite(x).all() or x.shape[1]!=1:raise ValueError('Finite mono audio required')
    dc=float(x.mean());x-=dc
    rms=float(np.sqrt(np.mean(x*x)));peak=float(np.max(np.abs(x)))
    gain=min(.08/max(rms,1e-12),.98/max(peak,1e-12))
    sf.write(destination,x*gain,sr,subtype='PCM_24')
    return dict(input_sha256=sha(source),output_sha256=sha(destination),duration=len(x)/sr,
                sample_rate=sr,dc_removed=dc,scalar_gain=gain)

def main():
    blind=OUTPUT/'blind';private=OUTPUT/'metadata'
    blind.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'recordings/v011_mega_tournament/raw/meanvc2_120/source_normal.wav'
    original=json.loads((ROOT/'recordings/v011_mega_tournament/metadata/blind_manifest.json').read_text('utf-8'))
    row=next(x for x in original['candidates'] if x['blind_id']=='A029')
    if row['model']!='meanvc2_120' or sha(baseline)!=row['normalization']['raw']['sha256']:
        raise ValueError('Previously listened baseline mismatch')
    variants=[dict(condition='Offline 3-step; previous A029 Keep',path=baseline)]
    for name,steps in [('meanvc2_stream_final_cpu2step',2),('meanvc2_stream_sdpa3',3)]:
        base=ROOT/'validation/v011'/name
        m=json.loads(base.with_suffix('.json').read_text('utf-8'))
        if m['stage']!='complete' or m['steps']!=steps or 'source_normal.wav' not in m['source']:
            raise ValueError('Cached streaming condition mismatch')
        if m['reference_sha256']!='453589ccbbcb9472d3e02b739fbe92b2861696fc14dc3e8b63d746be2682869b':
            raise ValueError('Reference mismatch')
        variants.append(dict(condition=f'Streaming {steps}-step; local causal BN interpolation',path=base.with_suffix('.wav')))
    random.Random(110).shuffle(variants)
    mapping=[]
    for i,v in enumerate(variants,1):
        ident=f'B{i:03}'
        mapping.append(dict(blind_id=ident,condition=v['condition'],input=str(v['path']),
                            normalization=normalize(v['path'],blind/(ident+'.wav'))))
    normalize(ROOT/'models/meanvc2_120/reference.wav',blind/'target.wav')
    normalize(ROOT/'recordings/v011_mega_tournament/source/source_normal.wav',blind/'source.wav')
    package=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
    (private/'blind_manifest.json').write_text(json.dumps(dict(package=package,candidates=mapping),indent=2,ensure_ascii=False),encoding='utf-8')
    page='''<!doctype html><meta charset="utf-8"><title>MeanVC2 声質の切り分け</title>
<style>body{font:17px system-ui;background:#161921;color:#eee;max-width:920px;margin:30px auto;padding:0 20px}section{background:#242936;padding:20px;margin:20px 0;border-radius:10px}audio{display:block;width:100%;margin:10px 0}label{display:inline-block;margin:8px}select,button,textarea{font:inherit;padding:6px}textarea{width:95%}button{cursor:pointer}</style>
<h1>同じSource・Referenceで声質を比較</h1><p>3候補です。モデル条件の対応表はmetadataへ分離しています。補正はDC除去・一定音量の調整だけで、Pitch・EQ・Prosodyは使用していません。</p>
<p>Target Reference</p><audio controls src="target.wav"></audio><p>Source 通常声</p><audio controls src="source.wav"></audio>
<div id="cards"></div><button id="export">評価JSONを保存</button><span id="status"></span>
<script>
const ids=IDS, packageId=PACKAGE, key='meanvc2-quality-'+packageId;
let ratings={};try{ratings=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const fields=[['similarity','Targetへの似具合'],['naturalness','自然さ'],['femininity','女声らしさ'],['male_residue','元声残り'],['mechanical','機械感'],['childlike','声変わり前の子供っぽさ']];
function save(){try{localStorage.setItem(key,JSON.stringify(ratings));document.querySelector('#status').textContent=' 保存済み'}catch(e){document.querySelector('#status').textContent=' JSON保存ボタンを使ってください'}}
for(const id of ids){const s=document.createElement('section');s.innerHTML='<h2>'+id+'</h2><audio controls src="'+id+'.wav"></audio>';ratings[id]??={};
for(const [field,label] of fields){const l=document.createElement('label');l.textContent=label+' ';const v=document.createElement('select');v.innerHTML='<option value="">未評価</option>'+[1,2,3,4,5].map(n=>'<option>'+n+'</option>').join('');v.value=ratings[id][field]??'';v.onchange=()=>{ratings[id][field]=v.value?Number(v.value):null;save()};l.append(v);s.append(l)}
const decision=document.createElement('select');decision.innerHTML='<option value="">未選択</option><option>Keep</option><option>Maybe</option><option>Reject</option>';decision.value=ratings[id].decision??'';decision.onchange=()=>{ratings[id].decision=decision.value;save()};s.append(document.createElement('br'),decision);
const comment=document.createElement('textarea');comment.placeholder='気になった声質';comment.value=ratings[id].comment??'';comment.oninput=()=>{ratings[id].comment=comment.value;save()};s.append(document.createElement('br'),comment);document.querySelector('#cards').append(s)}
document.addEventListener('play',e=>{for(const a of document.querySelectorAll('audio'))if(a!==e.target)a.pause()},true);
document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify({package:packageId,exported_at:new Date().toISOString(),ratings},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='meanvc2-quality-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script><p>類似・自然さ・女声らしさは5が良い評価。元声残り・機械感・子供っぽさは1が少ない評価です。評価はこのブラウザへ保存し、JSONでも書き出せます。</p>'''
    page=page.replace('IDS',json.dumps([v['blind_id'] for v in mapping])).replace('PACKAGE',json.dumps(package))
    (blind/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(page=str(blind/'index.html'),candidates=len(mapping),package=package)))

if __name__=='__main__':main()
