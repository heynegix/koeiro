"""Offline, bounded CPU registration into the local MeanVC2 library; no training.

The worker runs in two phases so a person can look at what was selected before the
voice is published:

  1. ``--source`` decodes, analyses, selects clips and encodes them, then writes an
     unpublished bundle under ``.pending_<id>`` and reports it. Nothing is visible in
     the voice library yet.
  2. ``--finalize <bundle>`` publishes that bundle, optionally restricted to a
     reviewed subset of the clips. Because the per-clip embeddings already exist, this
     second phase is fast and never re-runs the encoder.

``--add-to`` folds new audio into an existing voice instead of creating one.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.vc import voice_embedding, voice_quality
from src.vc.voice_library import MAX_PROFILE_BYTES, MAX_REFERENCE_SECONDS, digest
from src.vc.voice_library import folder_for as _validated_folder

# Kept long enough to analyse whole recordings, short enough to stay bounded. The
# retained reference is capped separately by voice_quality.DEFAULT_BUDGET_SECONDS.
DECODE_SECONDS = voice_quality.MAX_DECODE_SECONDS
BUNDLE_VERSION = 2
BUNDLE_PREFIX = '.pending_'
MAX_SOURCE_FILES = 8
# High-pass the reference is measured with. 80 Hz removes handling rumble and HVAC
# noise, which otherwise get pooled into the speaker embedding; DC removal alone leaves
# them. 0 disables it. Registered voices are only ever *read* by the runtime, so this
# does not affect any existing profile.
DEFAULT_HIGHPASS_HZ = 80.0
# 'ns_mean' is the default: with clips that come from a single speaker, a mild trim
# protects against an unusual recording without discarding the spread a plain mean uses.
DEFAULT_AGGREGATION = 'ns_mean'
AGGREGATIONS = ('centroid', 'ns_mean', 'geometric_median', 'mean')
# Registered time is not runtime RTF, so an exhaustive search over clip subsets is
# affordable here. It only runs for small candidate counts; see search_subset.
EXHAUSTIVE_SUBSET_LIMIT = 14
# Reference audio plus per-clip embeddings stay small; this only guards a hostile path.
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
CLIP_GAP_SECONDS = 0.4


def progress(message):
    print(json.dumps(dict(message=message), ensure_ascii=False), flush=True)


def report_line(**payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def select_segments(audio, rate=16000, budget_seconds=voice_quality.DEFAULT_BUDGET_SECONDS):
    """Pick reference material at natural speech boundaries; identity and gender aren't inferred."""
    parts, selections, _ = voice_quality.select_reference(audio, rate, budget_seconds=budget_seconds)
    return parts, selections


# Kept importable for callers that already used the original helper.
centroid = voice_embedding.centroid


def load_sources(paths):
    """Decode every source and join them with a short gap, as one continuous recording."""
    paths = [Path(item).resolve() for item in paths]
    if not paths:
        raise ValueError('音声ファイルを選んでください。')
    if len(paths) > MAX_SOURCE_FILES:
        raise ValueError('同時に登録できる音声ファイルは%d件までです。' % MAX_SOURCE_FILES)
    decoded = []
    scales = []
    for path in paths:
        audio, scale = voice_quality.decode_audio(path, sample_rate=16000, max_seconds=DECODE_SECONDS)
        decoded.append(audio)
        scales.append(scale)
    if len(decoded) == 1:
        return decoded[0], [path.name for path in paths], scales
    gap = np.zeros(int(CLIP_GAP_SECONDS*16000), dtype=np.float32)
    joined = []
    for index, audio in enumerate(decoded):
        if index:
            joined.append(gap)
        joined.append(audio)
    return np.concatenate(joined).astype(np.float32), [path.name for path in paths], scales


def load_spk_encoder():
    repo = ROOT/'vc_models/meanvc2/repo'
    for filename in ('wavlm_large_finetune.pth', 'wavlm_large.pt'):
        if not (repo/'preprocess/ckpts'/filename).is_file():
            raise RuntimeError('声登録用モデルがありません: '+filename)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo/'src/infer'))
    package = types.ModuleType('src.model')
    package.__path__ = [str(repo/'src/model')]
    sys.modules['src.model'] = package
    spec = importlib.util.spec_from_file_location('voice_registration_upstream', repo/'src/infer/infer_e2e.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    return torch, official.load_spk_model('cpu')


def library_root():
    """The library root this worker writes into. Honours the module ROOT so tests can
    redirect it; the voice id itself is still validated by folder_for."""
    return Path(ROOT)/'models/user_voices'


def bundle_path(identifier):
    return library_root()/(BUNDLE_PREFIX+identifier)


def encode_selection(parts, selections, torch, encoder, passes=None,
                     aggregation=DEFAULT_AGGREGATION, subset_search=True,
                     highpass_hz=DEFAULT_HIGHPASS_HZ):
    """High-pass, TTA-encode every candidate clip, then report self-agreement.

    This is the only step that needs the encoder, and it runs once per registration:
    nothing here is on the realtime path, so the pass count is chosen for stability
    rather than for speed. The agreement figures come back so the dialog can show how
    consistent the clips actually were instead of asserting a quality.
    """
    passes = voice_embedding.DEFAULT_TTA_PASSES if passes is None else int(passes)
    if aggregation not in AGGREGATIONS:
        raise ValueError('Unknown embedding aggregation: '+str(aggregation))
    rng = np.random.default_rng(0)  # Deterministic: the same file must register the same.
    prepared = []
    for part in parts:
        filtered = voice_quality.high_pass(np.asarray(part, dtype=np.float32), 16000, highpass_hz)
        prepared.append(np.asarray(filtered, dtype=np.float32))
    vectors, norms = voice_embedding.encode_clips(
        encoder, prepared, torch, passes,
        progress=lambda index, total: progress('声の特徴を抽出中 %d/%d…' % (index+1, total)),
        rng=rng)
    agreement = {'tta_passes': passes, 'tta_axes': [axis for axis, _ in voice_embedding.tta_variants(passes)],
                 'highpass_hz': float(highpass_hz), 'aggregation': aggregation,
                 'clip_count': len(prepared)}
    mean_agreement, sharpest = voice_embedding.consistency(vectors)
    agreement['mean_clip_agreement'] = round(mean_agreement, 4)
    agreement['sharpest_clip_agreement'] = round(sharpest, 4)
    return vectors, norms, agreement


def aggregate_embedding(vectors, norms, weights, aggregation=DEFAULT_AGGREGATION):
    """Reduce per-clip directions to one embedding under the chosen rule."""
    unit_vectors, _ = voice_embedding.unit(vectors)
    if aggregation == 'centroid':
        embedding, keep = voice_embedding.centroid(vectors, weights)
        return embedding, keep
    if aggregation == 'ns_mean':
        direction = voice_embedding.ns_mean(vectors, trim=0.25)
    elif aggregation == 'geometric_median':
        direction = voice_embedding.geometric_median(vectors)
    elif aggregation == 'mean':
        direction = unit_vectors.mean(axis=0)
    else:
        raise ValueError('Unknown embedding aggregation: '+str(aggregation))
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise RuntimeError('Inconsistent speaker embedding')
    # Keep the published norm contract: the median of the retained per-clip norms.
    keep = list(np.argsort(unit_vectors @ direction)[-min(2, len(unit_vectors)):])
    direction = direction/length
    median = float(np.median(np.asarray(norms)[keep]))
    return (direction*median).astype(np.float32), [int(index) for index in keep]


def write_bundle(identifier, parts, selections, vectors, norms, report, sources,
                 existing=None):
    """Persist an unpublished selection so it can be reviewed and published later."""
    staging = bundle_path(identifier)
    staging.mkdir(parents=True, exist_ok=True)
    np.save(staging/'clips.npy', np.asarray(parts, dtype=object), allow_pickle=True)
    np.save(staging/'clip_embeddings.npy', np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    np.save(staging/'clip_norms.npy', np.asarray(norms, dtype=np.float64), allow_pickle=False)
    payload = dict(bundle_version=BUNDLE_VERSION, id=identifier, selections=selections,
                   sources=list(sources), report=report,
                   tta_passes=voice_embedding.DEFAULT_TTA_PASSES,
                   existing=existing, created=time.time())
    (staging/'bundle.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), 'utf-8')
    return staging


def read_bundle(path):
    staging = Path(path).resolve()
    manifest = staging/'bundle.json'
    if not manifest.is_file():
        raise ValueError('確認用の選択データが見つかりません。最初から登録し直してください。')
    payload = json.loads(manifest.read_text('utf-8'))
    if payload.get('bundle_version') != BUNDLE_VERSION:
        raise ValueError('選択データの形式が一致しません。最初から登録し直してください。')
    # Only the pending folder of a real identifier may be read, and it must be a folder.
    identifier = payload.get('id')
    if not isinstance(identifier, str) or staging != bundle_path(identifier).resolve():
        raise ValueError('選択データの保存先が不正です。')
    parts = np.load(staging/'clips.npy', allow_pickle=True)
    vectors = np.load(staging/'clip_embeddings.npy')
    norms = np.load(staging/'clip_norms.npy')
    if not (len(parts) == len(vectors) == len(norms) == len(payload['selections'])):
        raise ValueError('選択データが壊れています。最初から登録し直してください。')
    for item in (staging/'clips.npy', staging/'clip_embeddings.npy', staging/'clip_norms.npy'):
        if item.stat().st_size > MAX_BUNDLE_BYTES:
            raise ValueError('選択データのサイズが上限を超えています。')
    return payload, [np.asarray(part, dtype=np.float32) for part in parts], vectors, norms


def resolve_indices(requested, available):
    """Turn the reviewed subset into indices, keeping chronological order."""
    if requested is None:
        return list(range(available))
    if isinstance(requested, str):
        requested = [item for item in requested.replace(',', ' ').split() if item]
    chosen = []
    for item in requested:
        try:
            index = int(item)
        except (TypeError, ValueError):
            raise ValueError('区間番号が不正です。')
        if not 0 <= index < available:
            raise ValueError('区間番号が範囲外です。')
        chosen.append(index)
    unique = sorted(set(chosen))
    if not unique:
        raise ValueError('発話区間を1つ以上選んでください。')
    return unique


def publish(identifier, name, parts, selections, embedding, keep, report, sources,
            existing_id=None, existing_reference=None, existing_selections=None,
            tta_passes=voice_embedding.DEFAULT_TTA_PASSES):
    """Write a complete voice atomically. Source recordings are never modified.

    Extending an existing voice replaces the published folder: the previous copy is
    moved aside first and restored if the swap fails, so a failure can never leave a
    half-written voice in the library.
    """
    _validated_folder(identifier, ROOT)
    final = library_root()/identifier
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() and not existing_id:
        raise ValueError('同じIDの声が既に存在します。')
    staging = final.parent/(BUNDLE_PREFIX+identifier+'.publishing')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    backup = None
    try:
        import soundfile as sf
        reference = np.concatenate(parts) if len(parts) > 1 else parts[0]
        sf.write(staging/'reference.wav', reference, 16000, subtype='PCM_16')
        np.save(staging/'fixed_embedding.npy', np.asarray(embedding, dtype=np.float32), allow_pickle=False)
        original = json.loads((ROOT/'models/meanvc2_ref20/runtime.json').read_text('utf-8'))
        assets = [asset.copy() for asset in original['assets'] if asset['path'].startswith('vc_models/')]
        for asset in assets:
            path = (ROOT/asset['path']).resolve()
            if not path.is_relative_to((ROOT/'vc_models/meanvc2').resolve()) or digest(path) != asset['sha256']:
                raise ValueError('共有モデルの検証に失敗しました。')
        for filename in ('reference.wav', 'fixed_embedding.npy'):
            assets.append(dict(path=(final/filename).relative_to(ROOT).as_posix(),
                               sha256=digest(staging/filename)))
        runtime = {key: original[key] for key in ('backend', 'preset', 'steps', 'default_threads',
                       'feature_frontend', 'bn_interpolation', 'vocoder_context')}
        runtime.update(assets=assets, display_name=name, human_approved_offline=False,
                       reference_origin='User-added local audio',
                       reference_embedding_method='vad-quality-diversity-tta-centroid-v3')
        (staging/'runtime.json').write_text(json.dumps(runtime, ensure_ascii=False, indent=2), 'utf-8')
        warnings = ['BGM・複数話者・声の性別は完全には自動判定していません。'
                    'contaminationの判定はプロキシであり、誤検出と検出漏れの両方が起こります。']
        if report.get('rejected'):
            warnings.append('品質基準を満たさず不採用の区間: '
                            + ', '.join('%s×%s' % item for item in sorted(report['rejected'].items())))
        warnings.extend(voice_quality.contamination_advice(report.get('contamination')))
        seconds = sum(len(part) for part in parts)/16000
        if not voice_quality.MIN_REFERENCE_SECONDS <= seconds <= MAX_REFERENCE_SECONDS:
            raise ValueError('登録後の音声長が%g〜%g秒の範囲外です。'
                             % (voice_quality.MIN_REFERENCE_SECONDS, MAX_REFERENCE_SECONDS))
        info = dict(schema=1, id=identifier, name=name, source_filename='、'.join(sources),
                    selections=selections, retained_indices=keep,
                    reference_seconds=seconds, tta_passes=tta_passes,
                    registration_seconds=report.get('elapsed_seconds'),
                    embedding_sha256=digest(staging/'fixed_embedding.npy'),
                    report={key: value for key, value in report.items()
                            if key not in ('selections', 'elapsed_seconds')},
                    warnings=warnings)
        if existing_id:
            info['extended_from'] = existing_id
            info['previous_reference_seconds'] = report.get('previous_reference_seconds')
        if existing_selections is not None:
            info['selections'] = list(existing_selections) + list(selections)
        text = json.dumps(info, ensure_ascii=False, indent=2)
        if len(text.encode('utf-8')) > MAX_PROFILE_BYTES:
            raise ValueError('登録情報が上限を超えたため保存できません。区間数を減らしてください。')
        (staging/'profile.json').write_text(text, 'utf-8')
        if final.exists():
            backup = final.parent/(BUNDLE_PREFIX+identifier+'.replaced')
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(final, backup)
        try:
            os.replace(staging, final)
        except OSError:
            if backup is not None and not final.exists():
                os.replace(backup, final)
                backup = None
            raise
        return info
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)


def analyse(name, sources, add_to=None, started=None, highpass_hz=DEFAULT_HIGHPASS_HZ,
            aggregation=DEFAULT_AGGREGATION, subset_search=True, passes=None):
    """Phase 1: decode, select, encode, and leave an unpublished bundle behind."""
    started = started or time.perf_counter()
    if not name or len(name) > 80:
        raise ValueError('声の名前は1〜80文字にしてください。')
    audio, source_names, scales = load_sources(sources)
    if add_to:
        return analyse_existing(name, add_to, audio, source_names, scales, started)
    parts, selections, report = voice_quality.select_reference(
        audio, 16000, progress=lambda stage, done, total: progress(
            '%s %d/%d' % (stage, done, total) if total > 1 else stage))
    report['peak_scale'] = round(max(scales), 6)
    progress('声の特徴を抽出しています。初回登録は数分かかる場合があります…')
    torch, encoder = load_spk_encoder()
    vectors, norms, agreement = encode_selection(
        parts, selections, torch, encoder, aggregation=aggregation, passes=passes)
    report.update(agreement)
    if subset_search and len(parts) > 1:
        progress('最も整合する区間の組み合わせを探索しています…')
        quality = [float(row.get('quality') or 1.0) for row in selections]
        search = voice_embedding.search_subset(
            vectors, norms, budget=len(parts), quality=quality,
            exhaustive=len(parts) <= EXHAUSTIVE_SUBSET_LIMIT)
        report['subset_search'] = {
            'indices': search['indices'], 'score': round(search['score'], 4),
            'mean_agreement': round(search['mean_agreement'], 4),
            'exhaustive': bool(len(parts) <= EXHAUSTIVE_SUBSET_LIMIT),
            'dropped': [index for index in range(len(parts)) if index not in search['indices']],
        }
    report['elapsed_seconds'] = time.perf_counter()-started
    identifier = 'user_'+uuid.uuid4().hex
    write_bundle(identifier, parts, selections, vectors, norms, report, source_names)
    return dict(bundle=str(bundle_path(identifier)), identifier=identifier,
                report=report, sources=source_names)


def analyse_existing(name, identifier, audio, source_names, scales, started):
    """Phase 1 for Phase 5: new clips are described, but nothing is merged yet."""
    _validated_folder(identifier, ROOT)
    existing = library_root()/identifier
    for filename in ('profile.json', 'reference.wav', 'fixed_embedding.npy', 'runtime.json'):
        if not (existing/filename).is_file():
            raise ValueError('既存の声が見つからないか、完成していません。')
    info = json.loads((existing/'profile.json').read_text('utf-8'))
    previous_seconds = float(info.get('reference_seconds') or 0.0)
    remaining = MAX_REFERENCE_SECONDS - previous_seconds
    if remaining < voice_quality.MIN_REFERENCE_SECONDS:
        raise ValueError('この声はすでに上限（%g秒）に達しています。別の声として登録してください。'
                         % MAX_REFERENCE_SECONDS)
    # Only ask for what still fits, so extending a voice cannot silently overflow.
    parts, selections, report = voice_quality.select_reference(
        audio, 16000, budget_seconds=remaining, progress=lambda stage, done, total: progress(
            '%s %d/%d' % (stage, done, total) if total > 1 else stage))
    report['peak_scale'] = round(max(scales), 6)
    report['previous_reference_seconds'] = round(previous_seconds, 2)
    report['added_seconds'] = report['retained_seconds']
    progress('声の特徴を抽出しています…')
    torch, encoder = load_spk_encoder()
    vectors, norms = voice_embedding.encode_clips(
        encoder, parts, torch, voice_embedding.DEFAULT_TTA_PASSES,
        progress=lambda index, total: progress('声の特徴を抽出中 %d/%d…' % (index+1, total)))
    report['elapsed_seconds'] = time.perf_counter()-started
    bundle_identifier = 'user_'+uuid.uuid4().hex
    write_bundle(bundle_identifier, parts, selections, vectors, norms, report, source_names,
                 existing=dict(id=identifier, name=info.get('name') or name,
                               reference_seconds=previous_seconds,
                               embedding_sha256=info.get('embedding_sha256')))
    return dict(bundle=str(bundle_path(bundle_identifier)), identifier=bundle_identifier,
                report=report, sources=source_names, existing=identifier)


def finalize(name, path, clips=None):
    """Phase 2: publish the reviewed bundle, restricted to the chosen clips."""
    payload, parts, vectors, norms = read_bundle(path)
    indices = resolve_indices(clips, len(parts))
    chosen_parts = [parts[index] for index in indices]
    chosen_selections = [payload['selections'][index] for index in indices]
    report = dict(payload['report'])
    report['retained_seconds'] = round(sum(len(part) for part in chosen_parts)/16000, 2)
    report['selections'] = chosen_selections
    quality = [float(row.get('quality') or 1.0) for row in chosen_selections]
    aggregation = (payload.get('report') or {}).get('aggregation', DEFAULT_AGGREGATION)
    existing = payload.get('existing')
    if existing:
        import soundfile as sf
        folder = _validated_folder(existing['id'], ROOT)
        embedding = np.load(folder/'fixed_embedding.npy')
        merged, keep = voice_embedding.merge(embedding, vectors[indices], norms[indices], quality)
        reference, existing_rate = sf.read(folder/'reference.wav', dtype='float32')
        existing_selections = json.loads((folder/'profile.json').read_text('utf-8')).get('selections')
        if existing_rate != 16000:
            raise ValueError('既存のリファレンスのサンプルレートが想定と異なります。')
        all_parts = [np.asarray(reference, dtype=np.float32)] + chosen_parts
        identifier = existing['id']
        published = publish(identifier, existing.get('name') or name, all_parts, chosen_selections,
                            merged, keep, report, list(payload['sources']),
                            existing_id=existing['id'], existing_selections=existing_selections,
                            tta_passes=payload.get('tta_passes', voice_embedding.DEFAULT_TTA_PASSES))
        shutil.rmtree(Path(path), ignore_errors=True)
        return dict(registered=identifier, report=report, extended_from=existing['id'],
                    profile=published, message=_success_message(published))
    embedding, keep = aggregate_embedding(vectors[indices], norms[indices], quality, aggregation)
    identifier = payload['id']
    published = publish(identifier, name, chosen_parts, chosen_selections, embedding, keep,
                        report, list(payload['sources']),
                        tta_passes=payload.get('tta_passes', voice_embedding.DEFAULT_TTA_PASSES))
    shutil.rmtree(Path(path), ignore_errors=True)
    return dict(registered=identifier, report=report, profile=published,
                message=_success_message(published))


def _success_message(info):
    parts = info.get('report', {}).get('candidate_count', 0)
    if info.get('extended_from'):
        return ('既存の声に音声を追加しました。合計 %.1f秒（追加 %d 区間）です。'
                % (info['reference_seconds'], len(info['selections'])))
    return ('登録が完了しました。%.1f秒（%d区間）を登録しました。'
            % (info['reference_seconds'], len(info['selections'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--name')
    parser.add_argument('--finalize')
    parser.add_argument('--clips')
    parser.add_argument('--add-to')
    parser.add_argument('--list-bundles', action='store_true')
    parser.add_argument('--tta-passes', type=int, default=voice_embedding.DEFAULT_TTA_PASSES)
    parser.add_argument('--highpass', type=float, default=DEFAULT_HIGHPASS_HZ,
                        help='Reference high-pass cutoff in Hz; 0 disables it')
    parser.add_argument('--aggregation', choices=AGGREGATIONS, default=DEFAULT_AGGREGATION)
    parser.add_argument('--no-subset-search', action='store_true')
    args = parser.parse_args()
    cache = ROOT/'vc_models/cache'
    for key, subdir in dict(TEMP='temp', TMP='temp', TORCH_HOME='torch', HF_HOME='hf-home',
                           HF_HUB_CACHE='hf', MPLCONFIGDIR='matplotlib',
                           NUMBA_CACHE_DIR='numba').items():
        folder = cache/subdir
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                      HF_HUB_OFFLINE='1')
    try:
        import psutil
        process = psutil.Process()
        if sys.platform == 'win32':
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            process.cpu_affinity(process.cpu_affinity()[:1])
        if args.list_bundles:
            report_line(bundles=[str(item.parent) for item in
                                 sorted(library_root().glob(BUNDLE_PREFIX+'user_*/bundle.json'))])
            return 0
        started = time.perf_counter()
        if args.finalize:
            result = finalize(args.name, args.finalize, args.clips)
        else:
            if not args.source:
                raise ValueError('音声ファイルを選んでください。')
            result = analyse(args.name, args.source, args.add_to, started,
                         highpass_hz=args.highpass, aggregation=args.aggregation,
                         subset_search=not args.no_subset_search,
                         passes=args.tta_passes)
        report_line(**result)
        return 0
    except Exception as error:
        report_line(error=str(error), message='登録できませんでした。')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())