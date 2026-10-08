"""Collect existing VC output history on D: without inference or device access."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'recordings'
OUT=REC/'version_audio_history'
VERSIONS={'v03':'v0.3','v04':'v0.4','v05':'v0.5','v08':'v0.8','v09':'v0.9',
          'v091':'v0.9.1','v092':'v0.9.2','v010_tournament':'v0.10',
          'v0102_clone':'v0.10.2','v011_mega_tournament':'v0.11 · 一括モデル比較',
          'v011_meanvc2_qualitycheck':'v0.11 · 初期Streaming',
          'v011_meanvc2_improvements':'v0.11 · Reference・発音改良',
          'v011_meanvc2_continuity':'v0.11 · 長文改良 / D009',
          'v011_meanvc2_latency':'v0.11 · 遅延・弱音・起動改善',
          'voice_quality':'旧音声候補（バージョン未確定）'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def local(value):
    if not isinstance(value,str):return None
    value=value.replace('\\','/')
    p=Path(value)
    if not p.is_absolute():p=ROOT/p
    try:
        p=p.resolve()
        return p if p.is_relative_to(ROOT) and p.is_file() else None
    except OSError:return None


def metadata():
    by_path={};by_sha={};blind=set()
    for folder in VERSIONS:
        base=REC/folder
        for manifest in base.rglob('*manifest.json'):
            try:data=read(manifest)
            except (OSError,ValueError):continue
            if not isinstance(data,dict):continue
            for c in data.get('candidates',[]):
                if not isinstance(c,dict):continue
                norm=c.get('normalization',{})
                raw=norm.get('raw',{})
                ident=c.get('blind_id',c.get('id'))
                info=dict(blind_id=ident,source=c.get('source'),reference=c.get('reference'),
                          label=c.get('model',c.get('variant',c.get('condition'))))
                if folder=='v011_meanvc2_qualitycheck':
                    info.update(version=VERSIONS[folder],major=ident=='B003',
                                source=str(base/'blind/source.wav'),reference=str(base/'blind/target.wav'))
                if raw.get('sha256'):by_sha[raw['sha256']]=info
                origin=local(c.get('input'))
                if origin:by_path[str(origin)]=info
                for p in base.rglob((ident or '__missing__')+'.wav'):
                    if origin or raw.get('sha256'):blind.add(p)
                    else:by_path[str(p)]=info
    clone=REC/'v0102_clone/comparison_metadata.json'
    if clone.exists():
        for c in read(clone):
            p=local(c.get('output'))
            if p:by_path[str(p)]=dict(source=c.get('source'),blind_id=c.get('blind_id'),label=c.get('candidate'))
        blind.update((REC/'v0102_clone/blind').glob('*.wav'))
    tour=REC/'v010_tournament/metadata/tournament_manifest.json'
    if tour.exists():
        for c in read(tour).get('renders',[]):
            p=REC/'v010_tournament'/c['output'];cfg=c.get('candidate',{})
            by_path[str(p)]=dict(source=c.get('source_wav'),label=f"Beatrice {cfg.get('model')} / Voice {cfg.get('voice')} / Pitch {cfg.get('pitch')}",
                                 note='旧モデル声候補。10秒等の切出し条件は元manifest参照。')
        blind.update((REC/'v010_tournament/blind').rglob('*.wav'))
    return by_path,by_sha,blind


def describe(p,by_path,by_sha,digest):
    rel=p.relative_to(ROOT);parts=rel.parts
    folder=parts[1] if parts[0]=='recordings' else 'validation'
    data=dict(by_sha.get(digest,{}));data.update(by_path.get(str(p),{}))
    version=VERSIONS.get(folder,data.get('version') or 'v0.11 · アプリWorker検証')
    label=data.get('label') or p.stem
    if '/raw/' in p.as_posix():label=p.parent.name
    source=local(data.get('source'));reference=local(data.get('reference'))
    note=data.get('note','');kind='Offline変換済みWAV'
    side=p.with_suffix('.json')
    if side.exists():
        try:
            extra=read(side)
            if isinstance(extra,dict):
                source=source or local(extra.get('source'))
                if isinstance(extra.get('config'),dict):
                    c=extra['config'];label=c.get('variant',label)
                    note+=f" {c.get('steps','?')} steps / {c.get('threads','?')} threads"
                if isinstance(extra.get('steps'),int):note+=f" {extra['steps']} steps"
        except (OSError,ValueError):pass
    comp=p.parent/'comparison.json'
    if comp.exists():
        try:
            c=read(comp)
            if isinstance(c,dict):source=source or local(c.get('source',c.get('input')))
        except (OSError,ValueError):pass
    if folder=='v011_meanvc2_qualitycheck':
        source=REC/folder/'blind/source.wav';reference=REC/folder/'blind/target.wav'
    elif folder=='v011_meanvc2_continuity':
        source=source or REC/folder/'source'/p.name
        reference=reference or REC/folder/'reference/current_reference.wav'
    elif folder=='v011_meanvc2_improvements':
        source=source or REC/folder/'source'/p.name
        ref=p.parent.name.split('_')[0]
        reference=reference or (REC/folder/'reference'/f'{ref}.wav')
    elif folder=='v011_mega_tournament':source=source or REC/folder/'source'/p.name
    elif folder=='v011_meanvc2_latency':
        kind='実機WASAPI＋既存録音replay → CABLE（新規マイク録音ではありません）'
        source=REC/folder/('restart_source.wav' if 'restart' in p.stem else 'controlled_source.wav')
        label={'pre4':'640ms buffer','pre6':'960ms buffer','pre8':'1280ms buffer'}.get(p.stem.split('_')[0],p.stem)
        label+=' / '+('短い発話・Start/Stop' if 'restart' in p.stem else '長文・短い発話・弱音')
        note+=' 旧処理' if 'preliminary' in p.parts else ' Gate −55dB / 起動時通常声混入対策'
        if p.stem.startswith('pre6'):note='予備測定 Gate −50dB'
    elif folder=='voice_quality':
        source=REC/'v03/input.wav'
        note+=' 入力は旧compare_voice_quality.pyの固定録音。正式バージョンは未確定。'
    if folder=='validation' and p.stem.startswith('meanvc2_adopted720_'):
        label='採用D009 · VC 720ms / 2 steps / 4 threads'
    major=False
    if folder in ('v03','v04'):major=p.stem in ('AI_Voice','Beatrice_Pitch4_Final','Beatrice_JVS002_Pitch4','Female_Soft','Female_Bright')
    elif folder=='v05':major=p.parent.name=='prosody_verified' and p.stem in ('prosody_off','prosody_on')
    elif folder=='v08':major=p.parent.name=='update40' and p.stem in ('prosody_off','prosody_on')
    elif folder in ('v09','v091','v092'):major='Natural' in p.stem or p.stem in ('prosody_off','v09_current','v091')
    elif folder=='v0102_clone':major=p.stem in ('baseline_jvs002','custom_5000')
    elif folder=='v011_mega_tournament':major=p.parent.name=='meanvc2_120'
    elif folder=='v011_meanvc2_qualitycheck':major=data.get('blind_id')=='B003'
    elif folder=='v011_meanvc2_improvements':major=p.parent.name in ('baseline','ref20','ref60_aligned')
    elif folder=='v011_meanvc2_continuity':major=p.parent.name in ('current','vc720')
    elif folder=='v011_meanvc2_latency':major='preliminary' not in p.parts and p.stem.startswith('pre4')
    if folder in ('validation','v011_meanvc2_qualitycheck'):major=data.get('major',major)
    label={
        'baseline':'初期MeanVC2 · 旧B003基準','ref20':'20秒Reference · メイン',
        'ref60_aligned':'60秒Reference · 第2音声（長文改良前）',
        'current':'長文改良前 · VC 120msずつ','decode360':'波形復元360ms · 声を維持した効率化',
        'vc360':'VC 360msまとめ','vc720':'D009採用 · VC 720msまとめ / 2 steps',
        'vc720_steps3':'VC 720msまとめ / 3 steps','meanvc2_120':'MeanVC2 120ms · 初回Offline',
        'meanvc2_40':'MeanVC2 40ms · 初回Offline'}.get(label,label)
    return dict(version=version,label=label,source=source if source and source.exists() else None,
                reference=reference if reference and reference.exists() else None,note=note.strip(),kind=kind,
                major=major,blind_id=data.get('blind_id'),archive_path=str(rel))


def main():
    import numpy as np
    import psutil
    import soundfile as sf
    if sys.platform=='win32':
        process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        process.cpu_affinity(process.cpu_affinity()[:1])
    for d in ('audio','original','metadata'): (OUT/d).mkdir(parents=True,exist_ok=True)
    by_path,by_sha,blind=metadata()
    files=[]
    for folder in VERSIONS:
        for p in (REC/folder).rglob('*.wav'):
            if p in blind or any(x in ('source','reference') for x in p.relative_to(REC/folder).parts[:-1]):continue
            if folder=='v011_meanvc2_continuity' and p.parent.name=='baseline':continue
            if p.stem.lower() in ('original','input','target','source','target_reference','gate_default'):continue
            if p.name.startswith(('reference_','All_Candidates')) or p.name in ('controlled_source.wav','restart_source.wav'):continue
            files.append(p)
    # These contain original streaming variants referenced by B001/B003 and
    # production replays; keep distinct versions, not repeated audition controls.
    for p in (ROOT/'validation/v011').glob('*.wav'):
        if 'audition4' not in p.stem:files.append(p)
    rows=[];failures=[];source_cache={};processed={}
    def copy_source(p):
        digest=sha(p)
        dest=OUT/'original'/f'{digest}.wav'
        if not dest.exists():shutil.copy2(p,dest)
        return dict(id=digest,url=f'original/{digest}.wav',label=p.name)
    for p in sorted(files):
        try:
            digest=sha(p);desc=describe(p,by_path,by_sha,digest)
            original=copy_source(p)
            saved=OUT/'metadata'/f'{digest}.json';dest=OUT/'audio'/f'{digest}.wav'
            if saved.exists() and dest.exists():stats=read(saved)
            else:
                if psutil.disk_usage(str(OUT)).free<2.5*1024**3:raise RuntimeError('D disk reserve < 2.5GiB')
                audio,rate=sf.read(p,dtype='float32',always_2d=True)
                if not np.isfinite(audio).all() or not len(audio):raise ValueError('Invalid WAV')
                raw_clip=int(np.count_nonzero(np.abs(audio)>=1))
                dc=np.mean(audio,axis=0,dtype='float64');audio=audio-dc
                rms=float(np.sqrt(np.mean(audio**2)));peak=float(np.max(np.abs(audio)))
                gain=min(.1/max(rms,1e-12),.98/max(peak,1e-12)) if rms>1e-8 else 1
                sf.write(dest,audio*gain,rate,subtype='PCM_16')
                stats=dict(duration=len(audio)/rate,sample_rate=rate,channels=audio.shape[1],
                           raw_clipping_samples=raw_clip,dc_removed=dc.tolist(),scalar_gain=gain,
                           method='DC removal + one constant gain to RMS .1 / peak <= .98; no EQ, compressor, pitch, trim or resampling')
                saved.write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
            for key in ('source','reference'):
                origin=desc[key]
                if origin:
                    if str(origin) not in source_cache:source_cache[str(origin)]=copy_source(origin)
                    desc[key]=source_cache[str(origin)]
            row=dict(desc,id=hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest()[:16],
                     output_sha256=digest,audio=f'audio/{digest}.wav',original=original['url'],**stats)
            rows.append(row)
        except Exception as error:failures.append(dict(path=str(p),error=str(error)))
    order=list(VERSIONS.values())+['v0.11 · アプリWorker検証']
    rows.sort(key=lambda r:(order.index(r['version']),r['label'],r['source']['label'] if r['source'] else ''))
    # Sources share a group only when their saved WAV bytes really match.
    groups={}
    for r in rows:
        s=r['source'];key=s['id'] if s else 'unknown'
        groups.setdefault(key,dict(id=key,label=s['label'] if s else '入力不明',count=0))['count']+=1
    result=dict(schema=1,created='2026-10-04',rows=rows,groups=list(groups.values()),
                failures=failures,total=len(rows),major=sum(r['major'] for r in rows),
                unique_audio=len({r['output_sha256'] for r in rows}),versions=list(dict.fromkeys(r['version'] for r in rows)),
                missing_versions=['v0.1','v0.2','v0.6','v0.7'],
                note='保存済み結果を整理。再推論・音声デバイス・設定変更なし。Unknown sourceは同一入力と見なさない。実機版は初期無音を保持。')
    (OUT/'metadata/history_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    template=(ROOT/'tools/version_audio_history.html').read_text('utf-8')
    payload=json.dumps(result,ensure_ascii=False).replace('<','\\u003c')
    (OUT/'index.html').write_text(template.replace('__HISTORY_DATA__',payload),encoding='utf-8')
    report=ROOT/'validation/v011/version_audio_history_report.md'
    report.write_text('# 保存済み変換音声のバージョン比較\n\n'+
        f"比較カード {len(rows)}、主要 {result['major']}、固有音声 {result['unique_audio']}、失敗 {len(failures)}。\n\n"+
        '元WAV・アプリ設定を変更せず、既存の変換済み音声だけを整理。音量合わせはDC除去と一定倍率のみ。\n\n'+
        '\n'.join(f"- {v}: {sum(r['version']==v for r in rows)}件" for v in result['versions'])+
        '\n\nv0.1 / v0.2 / v0.6 / v0.7はバージョンを確認できる変換後WAVが見つからず。旧音声候補は別表示。\n'+
        '\n同一入力は保存WAVのSHAでgroup化。入力不明・別入力・Offline/実機の差を表示。モデル候補の実験を正式採用と同一視しない。\n'+
        '\nページ: recordings/version_audio_history/index.html。対応表: metadata/history_manifest.json。\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('total','major','unique_audio','versions','failures')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
