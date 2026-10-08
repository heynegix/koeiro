"""Decode user-provided M4A sources once; preserve the originals and provenance."""
import json
import shutil
import subprocess
from pathlib import Path
from .audio import digest, stats

def decode_sources(source_dir, output):
    source_dir=Path(source_dir);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in ('normal','low','bright'):
        original=source_dir/f'source_{name}.m4a';dest=output/f'source_{name}.wav'
        if not original.exists():continue
        stamp=dest.with_suffix('.origin.json')
        old=json.loads(stamp.read_text('utf-8')) if stamp.exists() else {}
        src_hash=digest(original)
        if not (dest.exists() and old.get('origin_sha256')==src_hash and old.get('wav_sha256')==digest(dest)):
            ffmpeg=shutil.which('ffmpeg')
            if not ffmpeg:raise RuntimeError('FFmpeg required to decode M4A; place WAV sources directly if unavailable')
            temp=dest.with_name(dest.stem+'.partial.wav')
            cmd=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-i',str(original),'-map','0:a:0','-vn','-ac','1','-ar','48000','-c:a','pcm_s16le',str(temp)]
            subprocess.run(cmd,check=True,timeout=60)
            info=stats(temp)
            if not 5<=info['duration']<=10:raise ValueError(f'{name}: {info["duration"]:.3f}s; expected 5–10s; original not automatically truncated')
            temp.replace(dest)
            old=dict(origin=str(original),origin_sha256=src_hash,wav_sha256=digest(dest),command=cmd,
                     decode='PCM16 48kHz mono; no EQ, gain, pitch or compression',stats=info)
            stamp.write_text(json.dumps(old,indent=2),encoding='utf-8')
        records.append(old)
    return records
