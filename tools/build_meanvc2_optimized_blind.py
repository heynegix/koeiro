"""Blind listening page for the approved vs optimized MeanVC2 streaming voices.

Compares only cached offline WAV renders; no inference, no audio devices, no
training. Condition names stay in metadata so the page itself reveals nothing.
"""
import hashlib, json, random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'recordings/v011_meanvc2_optimized'
OUTPUT = INPUT / 'blind'

SOURCES = ['normal', 'low', 'bright', 'long']
SOURCE_PATHS = {
    'normal': ROOT / 'recordings/v011_mega_tournament/source/source_normal.wav',
    'low': ROOT / 'recordings/v011_mega_tournament/source/source_low.wav',
    'bright': ROOT / 'recordings/v011_mega_tournament/source/source_bright.wav',
    'long': ROOT / 'recordings/v011_post_vc2/source/long.wav',
}
CONDITIONS = [
    ('meanvc2_ref20', '自分の声 · メイン（20秒Reference）· 現行'),
    ('meanvc2_ref20_optimized', '自分の声 · メイン（20秒Reference）· 超高速'),
    ('meanvc2_120', 'MeanVC2 · 旧B003 · 現行'),
    ('meanvc2_120_optimized', 'MeanVC2 · 旧B003 · 超高速'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(source, destination):
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
    comparison = json.loads((INPUT / 'metadata/optimized_comparison.json').read_text('utf-8'))
    stems = {path.stem: name for name, path in SOURCE_PATHS.items()}
    timing = {row['profile'] + '_' + stems[Path(row['source']).stem]: row
              for row in comparison['rows']}

    entries = []
    for source in SOURCES:
        for profile, label in CONDITIONS:
            wav = INPUT / 'raw' / f'{profile}_{source}.wav'
            if not wav.exists():
                raise RuntimeError(f'missing render {wav}')
            entries.append(dict(source=source, profile=profile, label=label,
                                path=wav, timing=timing[f'{profile}_{source}']))
    random.Random(110).shuffle(entries)

    mapping = []
    for index, entry in enumerate(entries, 1):
        ident = f'E{index:03}'
        mapping.append(dict(blind_id=ident, source=entry['source'],
                            profile=entry['profile'], condition=entry['label'],
                            input=str(entry['path']),
                            rtf=entry['timing']['rtf'],
                            chunk_p95_ms=entry['timing']['chunk_p95_ms'],
                            chunk_max_ms=entry['timing']['chunk_max_ms'],
                            algorithmic_buffer_ms=entry['timing']['algorithmic_buffer_ms'],
                            vc_group_chunks=entry['timing']['vc_group_chunks'],
                            vocoder_batch_frames=entry['timing']['vocoder_batch_frames'],
                            normalization=normalize(entry['path'], blind / (ident + '.wav'))))

    for source in SOURCES:
        normalize(INPUT / 'raw' / f'meanvc2_ref20_{source}.wav',
                  blind / f'source_{source}.wav')
    normalize(ROOT / 'models/meanvc2_ref20/reference.wav', blind / 'reference_ref20.wav')
    normalize(ROOT / 'models/meanvc2_120/reference.wav', blind / 'reference_b003.wav')

    package = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    (INPUT / 'metadata/blind_manifest.json').write_text(
        json.dumps(dict(package=package, sources=SOURCES, candidates=mapping),
                   indent=2, ensure_ascii=False), encoding='utf-8')

    page = '''<!doctype html><meta charset="utf-8"><title>MeanVC2 超高速版 声質比較</title>
 <style>body{font:17px system-ui;background:#161921;color:#eee;max-width:920px;margin:30px auto;padding:0 20px}
 section{background:#242936;padding:20px;margin:20px 0;border-radius:10px}
 audio{display:block;width:100%;margin:10px 0}label{display:inline-block;margin:8px}
 select,button,textarea{font:inherit;padding:6px}textarea{width:95%}button{cursor:pointer}
 h2{color:#9fd3ff}</style>
 <h1>承認済みの声と「超高速」版を比較</h1>
 <p>同じチェックポイント・同じReference・同じ固定Speaker Embeddingです。違うのは推論のまとめ方だけで、
 <b>モデル重み・Reference・埋め込みは変更していません</b>。補正はDC除去と一定音量の調整だけです。
 Pitch・EQ・Prosody・後段帯域復元は使用していません。</p>
 <p>以下のVoiceはOffline WAV再生です。マイク/Discordの実機試験や実測遅延ではありません。</p>
 <p>Target Reference（メイン20秒）</p><audio controls src="reference_ref20.wav"></audio>
 <p>Target Reference（旧B003 10秒）</p><audio controls src="reference_b003.wav"></audio>
 <div id="sources"></div><button id="export">評価JSONを保存</button><span id="status"></span>
 <script>
 const groups=GROUPS, packageId=PACKAGE, key='meanvc2-optimized-'+packageId;
 let ratings={};try{ratings=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
 const fields=[['similarity','Referenceへの似具合'],['naturalness','自然さ'],['femininity','女声らしさ'],
   ['male_residue','元声残り'],['mechanical','機械感'],['childlike','声変わり前の子供っぽさ']];
 const sources={'normal':'通常声','low':'低め声','bright':'明るめ声','long':'25秒長文'};
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
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='meanvc2-optimized-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
 </script>
 <p>Voiceは5が良い評価、元声残り・機械感・子供っぽさは1が少ない評価です。採点はこのブラウザへ自動保存され、JSONでも書き出せます。</p>'''
    groups = {source: [c['blind_id'] for c in mapping if c['source'] == source]
              for source in SOURCES}
    page = page.replace('GROUPS', json.dumps(groups)).replace('PACKAGE', json.dumps(package))
    (blind / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(page=str(blind / 'index.html'), candidates=len(mapping),
                          package=package), ensure_ascii=False))


if __name__ == '__main__':
    main()