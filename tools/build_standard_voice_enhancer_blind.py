"""Blind page for the standard voice's utterance-route post-processing candidates.

Cached offline renders only; no inference, no audio device, no training. The
condition names and their RTF stay in metadata so the page itself reveals nothing.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'recordings/standard_voice_enhancer'
OUTPUT = INPUT / 'blind'

LABELS = {
    'meanvc2_only': '後段なし（MeanVC2のみ）',
    'lavasr_cutoff8000': 'LavaSR cutoff 8kHz（現状）',
    'lavasr_cutoff8000_denoise': 'LavaSR cutoff 8kHz + denoise',
    'lavasr_cutoff6000': 'LavaSR cutoff 6kHz',
    'lavasr_cutoff10000': 'LavaSR cutoff 10kHz',
    'novasr': 'NovaSR',
}


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
    report = json.loads((ROOT / 'validation/v011/enhancer_candidates.json').read_text('utf-8'))
    metrics = {row['candidate']: row for row in report['rows']}

    entries = []
    for key in LABELS:
        wav = INPUT / 'raw' / f'{key}.wav'
        if not wav.exists():
            raise RuntimeError(f'missing render {wav}')
        entries.append((key, wav))
    random.Random(110).shuffle(entries)

    mapping = []
    for index, (key, wav) in enumerate(entries, 1):
        ident = f'E{index:03}'
        row = metrics[key]
        mapping.append(dict(blind_id=ident, candidate=key, condition=LABELS[key],
                            input=str(wav),
                            enhancer_rtf=row.get('enhancer_rtf'),
                            total_rtf=row.get('total_rtf'),
                            share_4_8k=row.get('share_4_8k'),
                            share_8k_plus=row.get('share_8k_plus'),
                            tilt_4_8k_db=row.get('tilt_4_8k_db'),
                            normalization=normalize(wav, blind / (ident + '.wav'))))
    normalize(INPUT / 'raw' / 'meanvc2_only.wav', blind / 'source.wav')

    package = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    (INPUT / 'metadata').mkdir(parents=True, exist_ok=True)
    (INPUT / 'metadata/blind_manifest.json').write_text(
        json.dumps(dict(package=package, vc_rtf=report['vc_rtf'], candidates=mapping),
                   indent=2, ensure_ascii=False), encoding='utf-8')

    page = '''<!doctype html><meta charset="utf-8"><title>標準ボイス 後段処理の比較</title>
 <style>body{font:17px system-ui;background:#161921;color:#eee;max-width:920px;margin:30px auto;padding:0 20px}
 section{background:#242936;padding:20px;margin:20px 0;border-radius:10px}
 audio{display:block;width:100%;margin:10px 0}label{display:inline-block;margin:8px}
 select,button,textarea{font:inherit;padding:6px}textarea{width:95%}button{cursor:pointer}
 h2{color:#9fd3ff}</style>
 <h1>標準ボイス ・ 発話単位変換の「後段処理」を比較</h1>
 <p>同じ MeanVC2 出力（標準ボイス・固定Reference・固定Speaker Embedding・同じ推論条件）に対して、
 後段だけを変えたものです。<b>Voice・重み・変換条件は全候補で同一</b>です。
 補正はDC除去と一定音量の調整だけで、Pitch・EQ・Prosodyは使用していません。</p>
 <p>25秒長文のOffline WAV再生です。マイク／Discord実機試験や実測遅延ではありません。</p>
 <p>参照：後段なし</p><audio controls src="source.wav"></audio>
 <div id="cards"></div><button id="export">評価JSONを保存</button><span id="status"></span>
 <script>
 const ids=IDS, packageId=PACKAGE, key='standard-voice-enhancer-'+packageId;
 let ratings={};try{ratings=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
 const fields=[['clarity','明瞭さ・聞き取りやすさ'],['bandwidth','帯域の自然さ（高域のDIN）'],
   ['naturalness','自然さ'],['similarity','標準ボイスへの似具合'],['femininity','女声らしさ'],
   ['male_residue','元声残り'],['mechanical','機械感'],['artefact','DAC的なノイズ']];
 function save(){try{localStorage.setItem(key,JSON.stringify(ratings));document.querySelector('#status').textContent=' 保存済み'}catch(e){document.querySelector('#status').textContent=' JSON保存ボタンを使ってください'}}
 for(const id of ids){const s=document.createElement('section');s.innerHTML='<h2>'+id+'</h2><audio controls src="'+id+'.wav"></audio>';ratings[id]??={};
 for(const [field,label] of fields){const l=document.createElement('label');l.textContent=label+' ';
  const v=document.createElement('select');v.innerHTML='<option value="">未評価</option>'+[1,2,3,4,5].map(n=>'<option>'+n+'</option>').join('');
  v.value=ratings[id][field]??'';v.onchange=()=>{ratings[id][field]=v.value?Number(v.value):null;save()};l.append(v);s.append(l)}
 const d=document.createElement('select');d.innerHTML='<option value="">未選択</option><option>Keep</option><option>Maybe</option><option>Reject</option>';
 d.value=ratings[id].decision??'';d.onchange=()=>{ratings[id].decision=d.value;save()};s.append(document.createElement('br'),d);
 const c=document.createElement('textarea');c.placeholder='気になった声質・り返り';c.value=ratings[id].comment??'';
 c.oninput=()=>{ratings[id].comment=c.value;save()};s.append(document.createElement('br'),c);document.querySelector('#cards').append(s)}
 document.addEventListener('play',e=>{for(const a of document.querySelectorAll('audio'))if(a!==e.target)a.pause()},true);
 document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify({package:packageId,exported_at:new Date().toISOString(),ratings},null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='standard-voice-enhancer-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
 </script>
 <p>Voiceは5が良い評価、元声残り・機械感・DINは1が少ない評価です。採点はこのブラウザへ自動保存され、JSONでも書き出せます。</p>'''
    page = page.replace('IDS', json.dumps([m['blind_id'] for m in mapping]))
    page = page.replace('PACKAGE', json.dumps(package))
    (blind / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(page=str(blind / 'index.html'), candidates=len(mapping),
                          vc_rtf=report['vc_rtf'], package=package), ensure_ascii=False))


if __name__ == '__main__':
    main()