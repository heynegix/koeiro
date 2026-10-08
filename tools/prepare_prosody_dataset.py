"""Validate 500 sentence pairs, FCPE features, DTW and leak-free sample splits.

Offline analysis only. Original WAVs/metadata are never overwritten. Feature
caches are bound to source SHA256 + extractor version. No training occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.dataset.features import validate_pairs,extract_features,metrics
from src.dataset.alignment import align,training_sample,split_ids
from src.prosody.f0 import FCPEEstimator
from src.audio.performance import TimingStats

EXTRACTOR='torchfcpe-0.0.4-65-650-threshold0.006-rms25ms-hop10ms-v1'


def write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('voice_sample/prosody_dataset'))
    parser.add_argument('--output',type=Path,default=Path('voice_sample'))
    parser.add_argument('--report-copy',type=Path,default=Path('validation/v0.5/dataset-results.json'))
    args=parser.parse_args()
    rows=[json.loads(line) for line in (args.root/'sentences.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    ids=[row['id'] for row in rows]
    if len(set(ids))!=len(ids) or any(not isinstance(i,str) or not i.isdigit() for i in ids):
        raise ValueError('Invalid/duplicate sentence IDs')
    validation=validate_pairs(args.root,ids)
    write_json(args.output/'dataset_validation.json',validation)
    estimator=FCPEEstimator(threads=1)
    extraction=TimingStats(); extraction_ms=[]; started=time.perf_counter()
    records=[]; eligible=[]; excluded=list(validation['problems'])
    duplicate_ids={r['id'] for r in validation['duplicate_audio']}|{r['matches']['id'] for r in validation['duplicate_audio']}
    duration_limits={style:max(12.,4*float(np.median([p[style]['duration'] for p in validation['valid_pairs']])))
                     for style in ('neutral','anime')}
    for pair in validation['valid_pairs']:
        identifier=pair['id']; features={}; pair_metrics={}
        for style in ('neutral','anime'):
            source=Path(pair[style]['path'])
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            target=args.output/'features'/style/(identifier+'.npz')
            cached=False
            if target.exists():
                with np.load(target,allow_pickle=False) as data:
                    cached=str(data.get('source_sha256',''))==digest and str(data.get('extractor',''))==EXTRACTOR
                    if cached:
                        features[style]={k:data[k] for k in data.files}
            if not cached:
                t=time.perf_counter_ns()
                features[style]=extract_features(source,estimator)
                elapsed=time.perf_counter_ns()-t
                extraction.record(elapsed,round(pair[style]['duration']*16000),16000)
                extraction_ms.append(elapsed/1e6)
                target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,**features[style],source_sha256=np.array(digest),extractor=np.array(EXTRACTOR))
            pair_metrics[style]=metrics(features[style])
        n,a=pair_metrics['neutral'],pair_metrics['anime']
        reasons=[]
        if identifier in duplicate_ids:
            reasons.append('duplicate audio content')
        ratio=a['duration']/n['duration']
        if not .5<=ratio<=2:
            reasons.append('duration ratio outside 0.5..2')
        if any(pair_metrics[style]['duration']>duration_limits[style] for style in duration_limits):
            reasons.append('extreme absolute duration candidate')
        if min(n['voiced_ratio'],a['voiced_ratio'])<.20:
            reasons.append('voiced ratio below 20%')
        if min(n['pitch_st']['std'],a['pitch_st']['std'])<.05:
            reasons.append('pitch extraction flat/failure candidate')
        if max(n['pitch_st']['range'],a['pitch_st']['range'])>24:
            reasons.append('extreme pitch range/detection error candidate')
        if a['pitch_st']['std']<.8*n['pitch_st']['std'] and a['pitch_st']['range']<.8*n['pitch_st']['range']:
            reasons.append('anime pitch is substantially flatter than neutral')
        try:
            mapping,alignment=align(features['neutral'],features['anime'])
            if alignment['longest_flat_seconds']>1.0 or alignment['mean_cost']>2.5 or alignment['normalized_endpoint_error']>.02:
                reasons.append('alignment collapse/high cost candidate')
            if not reasons:
                sample=training_sample(features['neutral'],features['anime'],mapping)
                folder=args.output/'training_samples'; folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/(identifier+'.npz'),**sample,sentence_id=np.array(identifier))
                eligible.append(identifier)
        except ValueError as error:
            alignment=dict(error=str(error)); reasons.append('DTW failed')
        if reasons:
            excluded.append(dict(id=identifier,reasons=reasons))
        records.append(dict(id=identifier,neutral=n,anime=a,duration_ratio=ratio,alignment=alignment,excluded=reasons))
        if len(records)%20==0:
            print(json.dumps(dict(pairs=len(records),training_samples=len(eligible),seconds=round(time.perf_counter()-started,1))),flush=True)
            write_json(args.output/'dataset_progress.json',dict(completed_pairs=len(records),eligible=len(eligible)))
    comparisons={
        'anime_f0_std_greater':sum(r['anime']['pitch_st']['std']>r['neutral']['pitch_st']['std'] for r in records),
        'anime_f0_range_greater':sum(r['anime']['pitch_st']['range']>r['neutral']['pitch_st']['range'] for r in records),
        'anime_energy_range_greater':sum(r['anime']['energy_db']['range']>r['neutral']['energy_db']['range'] for r in records)}
    comparisons['anime_raw_f0_hz_std_greater']=sum(r['anime']['f0']['std']>r['neutral']['f0']['std'] for r in records)
    comparisons['anime_raw_f0_hz_range_greater']=sum(r['anime']['f0']['range']>r['neutral']['f0']['range'] for r in records)
    splits=split_ids(eligible)
    # Preserve superseded generated samples outside the active training root.
    # Never move source WAVs or files with unknown schema/ownership.
    sample_folder=(args.output/'training_samples').resolve()
    if not sample_folder.is_relative_to(args.output.resolve()):
        raise ValueError('Training destination outside output root')
    for path in sample_folder.glob('*.npz'):
        if path.stem in ids and path.stem not in eligible:
            with np.load(path,allow_pickle=False) as old:
                owned=int(old.get('schema_version',0))==1 and str(old.get('sentence_id',''))==path.stem
            if owned:
                excluded_folder=sample_folder/'excluded'; excluded_folder.mkdir(exist_ok=True)
                path.replace(excluded_folder/path.name)
    write_json(args.output/'training_samples/splits.json',dict(seed=500,group='sentence ID',**splits))
    write_json(args.output/'exclusion_list.json',excluded)
    write_json(args.output/'pair_metrics.json',records)
    checks=[]
    for index in np.random.default_rng(500).choice(len(records),min(20,len(records)),replace=False):
        r=records[int(index)]
        checks.append(dict(id=r['id'],duration_ratio=r['duration_ratio'],alignment=r['alignment'],excluded=r['excluded']))
    write_json(args.output/'alignment_checks.json',dict(seed=500,method='20 random numerical checks; not human-reviewed',pairs=checks))
    def stats(values):
        return dict(mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.percentile(values,95)),
            minimum=float(np.min(values)),maximum=float(np.max(values))) if values else {}
    summary=dict(expected_pairs=len(ids),valid_wav_pairs=len(records),training_pairs=len(eligible),
        excluded_pairs=len(ids)-len(eligible),wav_counts=validation['wav_counts'],comparisons=comparisons,
        duplicate_audio_count=len(validation['duplicate_audio']),
        absolute_duration_limits_seconds=duration_limits,
        splits={k:len(v) for k,v in splits.items()},dtw_success=sum('error' not in r['alignment'] for r in records),
        alignment_checks=checks,extraction=asdict(extraction.snapshot()),elapsed_seconds=time.perf_counter()-started,
        neutral={key:stats([r['neutral'][key] if key=='duration' else r['neutral']['pitch_st'][key] for r in records]) for key in ('duration','median','std','range')},
        anime={key:stats([r['anime'][key] if key=='duration' else r['anime']['pitch_st'][key] for r in records]) for key in ('duration','median','std','range')},
        f0_hz={style:stats([r[style]['f0']['median'] for r in records]) for style in ('neutral','anime')},
        energy_range_db={style:stats([r[style]['energy_db']['range'] for r in records]) for style in ('neutral','anime')},
        duration_ratio=stats([r['duration_ratio'] for r in records]),extractor=EXTRACTOR,
        teacher_quality='Automatic screening only; perceptual quality and training utility not established',
        comparison_units='f0_std/range: semitones after speaker-register normalization; raw_f0_hz: Hz; energy: dBFS')
    if extraction_ms:
        summary['extraction'].update({f'p{q}_ms':float(np.percentile(extraction_ms,q)) for q in (50,95,99)})
    else:
        summary['extraction']={'count':0,'note':'Source-hash caches reused; no new FCPE inference measured'}
    write_json(args.output/'dataset_summary.json',summary); write_json(args.report_copy,summary)
    lines=['# Prosody Dataset Foundation — v0.5','',
        f"総Pair {len(ids)} / WAV読込・基本検証合格 {len(records)} / 学習候補 {len(eligible)} / 除外候補 {len(ids)-len(eligible)}。",'',
        'FCPE公式同梱モデル0.0.4、CPU1 thread、16 kHz、10 ms hop。voicedは二値判定で確率ではありません。Energyは25 ms RMSのdBFS。元WAVは変更していません。','',
        '| 比較 | animeが大きいPair |','|---|---:|']
    for k,v in comparisons.items():
        lines.append(f'| {k} | {v} / {len(records)} |')
    lines+=['','## 統計','', '```json',json.dumps({k:summary[k] for k in ('neutral','anime','f0_hz','energy_range_db','duration_ratio')},ensure_ascii=False,indent=2),'```','',
        f"DTW成功 {summary['dtw_success']}。Energy・相対F0・voicedの複合cost、20 ms grid、25% band。20ランダムPairの数値検証はalignment_checks.json。人による対応箇所の聴取確認は未実施。",'',
        f"Training samples {len(eligible)} / Train {len(splits['train'])} / Validation {len(splits['validation'])} / Test {len(splits['test'])}。seed500、文章ID単位で分離。",'',
        '## 問題サンプル','']
    lines += [f"- {r.get('id')}: {', '.join(r.get('reasons',[r.get('reason','')]))}" for r in excluded]
    lines+=['','## 教師データとしての総評','',
        '全Pairを無条件で学習に使いません。Animeが平坦なPair、低voiced、異常duration、DTW崩壊を候補から除外しています。F0のstd/range比較は声域正規化後の半音単位、raw_f0_hz項目はHzです。改善割合は上表を参照。AnimeのPitch変動がNeutralより大きい割合が低ければ、抑揚拡大の教師として不十分です。特徴と対応データは準備できても、v0.6の学習開始前に教師の聴取・選別や再生成が必要です。候補合格はAnimeらしさの認定ではありません。','',
        'target_pitch_offset_stは声域medianを各音声で除いたaligned anime relative pitch − neutral relative pitch。target_energy_offset_dbも各話者のvoiced energy medianを除いた差で、声量差を除きます。target_energy_raw_offset_dbは未正規化の差。*_f0_deltaは隣接frameの半音差、無声境界は0。valid_pitch_targetで両側voicedをマスクします。声質や絶対F0をProsody教師へ混ぜないための形式です。v0.5では学習していません。']
    (args.output/'prosody_dataset_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
