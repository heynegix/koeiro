"""Freeze existing human sources and prepare an honest missing-category inventory."""
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.anime_voice_tournament.core import OUTPUT, ROOT, digest, inspect_audio, read_audio, write_audio, save_json


def main():
    folder = OUTPUT/'source'; folder.mkdir(parents=True,exist_ok=True)
    first = ROOT/'recordings/v03/input.wav'
    second = folder/'recording_02.wav'  # Locally decoded existing M4A; preserve original.
    sources = []
    for i,path in enumerate((first,second),1):
        if not path.exists(): continue
        audio,rate,channels = read_audio(path)
        target = folder/f'S{i:02}.wav'
        if rate!=48000 or channels!=1: raise ValueError('Convert to mono 48k PCM16 first')
        if len(audio)>15*rate: write_audio(target,audio[:15*rate])
        else: shutil.copyfile(path,target)
        info = inspect_audio(target,source=True)
        sources.append(dict(source_id=f'S{i:02}',path=str(target.resolve()),origin=str(path.resolve()),
                            origin_sha256=digest(path),category='unannotated_existing_human_recording',
                            category_verified=False,validation=info,trim_seconds=[0,info['duration']]))
    save_json(OUTPUT/'metadata/source_manifest.json', dict(sources=sources,
        independent_recordings=len(sources),minimum_five_met=len(sources)>=5,
        missing_categories=['ordinary','bright','low','question','excited','soft'],
        note='Two actual recordings; category coverage not verified. No artificial variants counted as independent sources.'))
    save_json(OUTPUT/'metadata/source_categories_template.json', dict(
        sources=[dict(source_id=f'S{i+1:02}',path='',category=category,category_verified=False) for i,category in
                 enumerate(['ordinary','bright','low','question','excited','soft'])]))


if __name__=='__main__': main()
