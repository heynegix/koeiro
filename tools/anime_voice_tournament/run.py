"""Stage 1 first-pass auditions, or reviewed subsequent recipes, with resume."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.anime_voice_tournament.core import (
    Candidate, ROOT, OUTPUT, MODEL_FOLDERS, blind_mapping, cache_valid,
    digest, inspect_audio, load_inventory, save_json,
)


def stage1_candidates(inventory):
    ids = [1,3,5,7,9,11,15,19,23,27,31,35,39,47,55,63,71,79,87,95]
    available = {(x['model'], x['voice']) for x in inventory}
    return [Candidate(model, voice) for model, voice in
            [('jvs', i) for i in ids] + [('character', i) for i in range(4)]
            if (model, voice) in available]


def assets(candidate):
    base = ROOT / MODEL_FOLDERS[candidate.model]
    manifest = json.loads((base / 'beatrice-manifest.json').read_text(encoding='utf-8'))
    # Cache actual installed assets, not only a stale manifest.
    return {name: digest(base / 'beatrice' / name) for name in manifest['files']}


def package_review(folder, manifest, stage):
    entries = [x for x in manifest['renders'] if x['stage'] == stage and x['status'] == 'ok']
    keys = sorted({x['recipe_key'] for x in entries}); mapping = blind_mapping(keys)
    package_id = hashlib.sha256(json.dumps(sorted((x['recipe_key'],x['source_id'],x['source_sha256'],x['output_sha256']) for x in entries)).encode()).hexdigest()
    blind = folder / 'blind' / stage; blind.mkdir(parents=True, exist_ok=True)
    items = []
    for key in keys:
        candidate_id = mapping[key]; files = []
        for entry in entries:
            if entry['recipe_key'] != key: continue
            name = candidate_id + '_' + entry['source_id'] + '.wav'
            shutil.copyfile(folder / entry['output'], blind / name)
            files.append(dict(source_id=entry['source_id'], path=name))
        items.append(dict(candidate_id=candidate_id, files=files))
    public = dict(stage=stage, package_id=package_id, items=sorted(items, key=lambda x:x['candidate_id']))
    # Never place revealing mappings in blind directory or HTML.
    save_json(folder/'metadata'/f'{stage}_blind_mapping.json', dict(package_id=package_id, mapping=mapping))
    template = Path(__file__).with_name('review_template.html').read_text(encoding='utf-8')
    (blind/'index.html').write_text(template.replace('__PACKAGE_DATA__', json.dumps(public, ensure_ascii=False).replace('<','\\u003c')), encoding='utf-8')
    ratings = blind/'ratings.csv'
    if not ratings.exists():
        with ratings.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f); writer.writerow(['candidate_id','anime','natural','female','mechanical','preference','comment'])
            writer.writerows([[x['candidate_id']]+['']*6 for x in public['items']])
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(OUTPUT)); parser.add_argument('--sources', required=True)
    parser.add_argument('--recipes', help='Explicit reviewed recipe list; stage2/3 must contain review evidence')
    parser.add_argument('--stage', choices=['stage1_voice','stage2_pitch_formant','stage3_merge'], default='stage1_voice')
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--all-sources', action='store_true', help='Default first pass uses first source only')
    args = parser.parse_args()
    if not 8 <= args.seconds <= 15: parser.error('Use 8–15 seconds')
    folder = Path(args.output).resolve(); folder.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path(args.sources).read_text(encoding='utf-8'))['sources']
    if not sources or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',x['source_id']) for x in sources):
        raise ValueError('Source IDs must be safe short identifiers')
    if len({x['source_id'] for x in sources})!=len(sources):
        raise ValueError('Duplicate source ID')
    if not args.all_sources: sources = sources[:1]
    inventory = load_inventory(); save_json(folder/'metadata/inventory.json', inventory)
    if args.recipes:
        plan = json.loads(Path(args.recipes).read_text(encoding='utf-8'))
        if args.stage != 'stage1_voice' and not plan.get('human_shortlist'):
            raise ValueError('Human shortlist required before pitch/formant/merge sweep')
        candidates = [Candidate(**{**row, 'merge':tuple(tuple(x) for x in row.get('merge',()))}) for row in plan['candidates']]
    else:
        if args.stage != 'stage1_voice': parser.error('Reviewed recipes required')
        candidates = stage1_candidates(inventory)
    manifest_path = folder/'metadata/tournament_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else dict(
        schema=1, baseline=Candidate('jvs',1,4.,0.).recipe(), winner=None,
        decision='NO CLEAR WINNER', human_review_pending=True, renders=[])
    asset_hashes = {}
    for candidate in candidates:
        for source in sources:
            path = Path(source['path']).resolve()
            key = candidate.key(); output = Path(args.stage)/(key+'_'+source['source_id']+'.wav')
            record = dict(stage=args.stage,recipe_key=key,candidate=candidate.recipe(),source_id=source['source_id'],
                          source_wav=str(path),output=output.as_posix(),backend='official Beatrice VST3 CPU',status='error')
            try:
                info = inspect_audio(path, source=True)
                if candidate.model not in asset_hashes: asset_hashes[candidate.model] = assets(candidate)
                record['source_sha256'] = info['sha256']
                payload = dict(candidate=candidate.recipe(), source_sha256=info['sha256'],seconds=args.seconds,
                               assets=asset_hashes[candidate.model], renderer_sha256=digest(Path(__file__).with_name('render.py')),
                               core_sha256=digest(Path(__file__).with_name('core.py')))
                cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                previous = next((x for x in manifest['renders'] if x['stage']==args.stage and x['recipe_key']==key and x['source_id']==source['source_id']),{})
                if cache_valid(previous, cache_key, folder/output):
                    print('cached', key, flush=True); continue
                job_path = folder/'metadata/job.json'; result = folder/'metadata/job-result.json'
                save_json(job_path, dict(candidate=candidate.recipe(),source=str(path),seconds=args.seconds,
                                         output=str(folder/output),result=str(result)))
                process = subprocess.run([sys.executable,str(Path(__file__).with_name('render.py')),str(job_path)],
                                         capture_output=True,timeout=180)
                if process.returncode:
                    raise RuntimeError(process.stderr.decode('utf-8',errors='replace')[-1500:])
                stats = json.loads(result.read_text(encoding='utf-8'))
                output_info = inspect_audio(folder/output)
                if output_info['clip_count'] or output_info['frames'] != min(info['frames'],int(args.seconds*48000)):
                    raise ValueError('Output length/clipping validation failed')
                record.update(status='ok', cache_key=cache_key, output_sha256=output_info['sha256'],
                              validation=output_info, **stats)
            except (OSError,ValueError,RuntimeError,wave.Error,EOFError,subprocess.TimeoutExpired) as exc:
                record['error'] = str(exc)
            manifest['renders'] = [x for x in manifest['renders'] if not(x['stage']==args.stage and x['recipe_key']==key and x['source_id']==source['source_id'])] + [record]
            save_json(manifest_path,manifest)
            print(record['status'],key,flush=True)
    count = package_review(folder,manifest,args.stage)
    print('Blind candidates:',count,'Review:',folder/'blind'/args.stage/'index.html')


if __name__ == '__main__': main()
