"""Blind page: standard-voice utterance output with no post-processing vs LavaSR vs FlashSR.

Compares only cached offline WAV renders; no inference, no audio devices, no
training. Condition names stay in metadata so the page reveals nothing.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'recordings/standard_voice_utterance'
OUTPUT = INPUT / 'blind'

SOURCES = ['normal', 'long']
CONDITIONS = [('none', '後段なし（16kHzのまま）'),
              ('lavasr', 'LavaSR'),
              ('flashsr', 'FlashSR')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(source, destination):
    """DC removal plus one constant RMS scalar; no EQ, compressor or pitch change."""
    x, sr = sf.read(source, dtype='float64', always_2d=True)
    if not np.isfinite(x).all() or x.shape[1] != 1:
        raise ValueError('Finite mono audio required')
    dc = float(x.mean())
    x -= dc
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    gain = min(.08 / max(rms, 1e-12), .98 / max(peak, 1e-12))
    sf.write(destination, x * gain, sr, subtype='PCM_24')
    return dict(input_sha256=sha(source), output_sha256=sha(destination),
                duration=len(x) / sr, sample_rate=sr, dc_removed=dc,
                scalar_gain=gain, peak_after=float(np.max(np.abs(x * gain))))


def main():
    blind = OUTPUT
    blind.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in SOURCES:
        for key, label in CONDITIONS:
            wav = INPUT / 'raw' / key / f'{source}_0.wav'
            if not wav.exists():
                raise RuntimeError(f'missing render {wav}')
            entries.append(dict(source=source, key=key, label=label, path=wav))
    random.Random(110).shuffle(entries)

    mapping = []
    for index, entry in enumerate(entries, 1):
        ident = f'L{index:03}'
        mapping.append(dict(blind_id=ident, source=entry['source'],
                            enhancer=entry['key'], condition=entry['label'],
                            input=str(entry['path']),
                            normalization=normalize(entry['path'], blind / (ident + '.wav'))))
    for source in SOURCES:
        normalize(INPUT / 'raw' / 'none' / f'{source}_0.wav', blind / f'source_{source}.wav')

    timing = {}
    for name in ('q_none', 'q_lavasr', 'q_flashsr'):
        report = ROOT / f'validation/v011/rtfopt/{name}.json'
        if report.exists():
            data = json.loads(report.read_text(encoding='utf-8'))
            for row in data['rows']:
                timing[f"{row['case']}_{data['enhancer']}"] = row
    for row in mapping:
        row['rtf'] = timing.get(f"{row['source']}_{row['enhancer']}", {}).get('rtf')
        row['enhancer_rtf'] = timing.get(f"{row['source']}_{row['enhancer']}", {}).get('enhancer_rtf')

    package = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    private = INPUT / 'metadata'
    private.mkdir(parents=True, exist_ok=True)
    (private / 'blind_manifest.json').write_text(
        json.dumps(dict(package=package, sources=SOURCES, candidates=mapping),
                   indent=2, ensure_ascii=False), encoding='utf-8')

    page = '''<!doctype html><meta charset="utf-8"><title>標準ボイス 発話単位 後段比較</title>
 <style>body{font:17px system-ui;background:#161921;color:#eee;max-width:920px;margin:30px auto;padding:0 20px}
 section{background:#242936;padding:20px;margin:20px 0;border-radius:10px}
 audio{display:block;width:100%;margin:10px 0}label{display:inline-block;margin:8px}
 select,button,textarea{font:inherit;padding:6px}textarea{width:95%}button{cursor:pointer}
 h2{color:#9fd3ff}</style>
 <h1>標準ボイス · 発話単位変換の「後段処理」を比較</h1>
 <p>同じ MeanVC2（標準ボイス の固定Reference・固定Speaker Embedding）で変換した発話に対し、
 後段の帯域復元だけを切り替えたものです。<b>Voice・モデル重み・変換条件は全候補で同一</b>です。
 補正はDC除去と一定音量の調整だけで、Pitch・EQ・Prosodyは使用していません。</p>
 <p>以下はOffline WAV再生です。マイク／Discordの実機試験や実測遅延ではありません。</p>
 <div id="sources"></div><button id="export">評価JSONを保存</button><span id="status"></span>
 <script>
 const groups=GROUPS, packageId=PACKAGE, key='standard-voice-utterance-'+packageId;
 let ratings={};try{ratings=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
 const fields=[['bandwidth','帯域の自然さ（高域のDIN）'],['naturalness','自然さ'],['similarity','標準ボイスへの似具合'],
   ['femininity','女声らしさ'],['male_residue','元声残り'], ['mechanical','機械感'],['artefact','DAC的なノイズ']];
 const sources={'normal':'通常声（8秒）','long':'長文（25秒）'};
 function save(){try{localStorage.setItem(key,JSON.stringify(ratings));document.querySelector('#status').textContent=' 保存済み'}catch(e){document.querySelector('#status').textContent=' JSON保存ボタンを使ってください'}}
 for(const g of groups){
  const box=document.createElement('section');box.innerHTML='<h2>Source: '+sources[g]+'</h2><audio controls src="source_'+g+'.wav"></audio>';
  for(const id of g){const s=document.createElement('div');
   s.innerHTML='<h3>'+id+'</h3><audio controls src="'+id+'.wav"></audio>';ratings[id]??={};
   for(const [field,label] of fields){const l=document.createElement('label');l.textContent=label+' ';
    const v=document.createElement('select');v.innerHTML='<option value="">未評価</option>'+[1,2,3,4,5].map(n=>'<option>'+n+'</option>').join('');
    v.value=ratings[id][field]??'';v.onchange=()=>{ratings[id][field]=v.value?Number(v.value):null;save()};l.append(v);s.append(l)}
   const d=document.createElement('select');d.innerHTML='<option value="">未選択</option><option>Keep</option><option>Maybe</option><option>Reject</option>';
   d.value=ratings[id].decision??'';d.onchange=()=>{ratings[id].decision=d.value;save()};s.append(document.createElement('br'),d);
   const c=document.createElement('textarea');c.placeholder='気になった声質・り返り';c.value=ratings[id].comment??'';
   c.oninput=()=>{ratings[id].comment=c.value;save()};s.append(document.createElement('br'),c);box.append(s)}
  document.querySelector('#sources').append(box)}
 document.addEventListener('play',e=>{for(const a of document.querySelectorAll('audio'))if(a!==e.target)a.pause()},true);
 document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify({package:packageId,exported_at:new Date().toISOString(),ratings},null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='standard-voice-utterance-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
 </script>
 <p>Voiceは5が良い評価、元声残り・機械感・DINは1が少ない評価です。採点はこのブラウザへ自動保存され、JSONでも書き出せます。</p>'''
    groups = {source: [c['blind_id'] for c in mapping if c['source'] == source]
              for source in SOURCES}
    page = page.replace('GROUPS', json.dumps(groups)).replace('PACKAGE', json.dumps(package))
    (blind / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(page=str(blind / 'index.html'), candidates=len(mapping),
                          package=package), ensure_ascii=False))


if __name__ == '__main__':
    main()