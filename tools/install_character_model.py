"""Import consented official archive locally; never redistribute model assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def install(archive,root):
    folder=root/'models/character_01'
    assets=folder/'beatrice';model=assets/'model'
    prefix='beatrice_2.0.0-rc.0_official_model_1/'
    required={'beatrice_2.0.0-rc.0_official_model_1.toml','embedding_setter.bin',
        'phone_extractor.bin','pitch_estimator.bin','speaker_embeddings.bin','waveform_generator.bin','README.txt'}
    prepared={}
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir():continue
            if not item.filename.startswith(prefix):raise ValueError('Unexpected official archive layout')
            name=item.filename[len(prefix):]
            if any(c in name for c in '/\\:') or name in ('','..') or item.file_size>100_000_000:
                raise ValueError('Invalid model archive entry')
            if name in prepared:raise ValueError('Duplicate model archive entry')
            prepared[name]=z.read(item)
    if not required<=prepared.keys():raise ValueError('Incomplete model archive')
    old=root/'models/girl_01'
    old_runtime=json.loads((old/'runtime.json').read_text(encoding='utf-8'))
    plugin_relative=Path(old_runtime['plugin']).relative_to('beatrice')
    source_plugin=(old/old_runtime['plugin']).resolve()
    if not source_plugin.is_relative_to((old/'beatrice').resolve()) or not source_plugin.is_file():
        raise ValueError('Existing Beatrice plugin missing/invalid')
    model.mkdir(parents=True,exist_ok=True)
    for name,data in prepared.items():(model/name).write_bytes(data)
    plugin=assets/plugin_relative
    plugin.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source_plugin,plugin)
    files={}
    for path in sorted(assets.rglob('*')):
        if path.is_file():files[path.relative_to(assets).as_posix()]=dict(bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    terms='https://prj-beatrice.com/2.0.0-rc.0-official-model-1-terms'
    (folder/'runtime.json').write_text(json.dumps(dict(backend='beatrice_vst',voice=0,pitch=7.,
        name='Anime Cute beta / Tsukuyomi',plugin='beatrice/'+plugin_relative.as_posix(),
        model='beatrice/model/beatrice_2.0.0-rc.0_official_model_1.toml'),indent=2),encoding='utf-8')
    (folder/'beatrice-manifest.json').write_text(json.dumps(dict(version='2.0.0-rc.0 official model 1',
        files=files,license='Official model 1; target voice terms apply; no model redistribution',
        terms_url=terms,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    return folder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--accept-official-model-terms',action='store_true')
    a=p.parse_args()
    if not a.accept_official_model_terms:p.error('Official model terms must be accepted before import')
    print(install(a.archive,Path(__file__).resolve().parents[1]))


if __name__=='__main__':main()
