"""Check real campaign bindings and browser script syntax, without playback."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vc.voice_library import digest


def main():
    root=ROOT/'recordings/v011_naturalness'
    manifest=json.loads((root/'metadata/blind_manifest.json').read_text('utf-8'))
    rows=manifest['candidates']
    assert len(rows)>=60 and len({r['id'] for r in rows})==len(rows)
    for row in rows:
        assert digest(row['output'])==row['sha256']
        with wave.open(str(root/'blind'/(row['id']+'.wav'))) as stream:
            assert stream.getnframes()>0 and stream.getnchannels()==1 and stream.getsampwidth()==2
    html=(root/'blind/index.html').read_text('utf-8')
    assert '__DATA__' not in html
    script=root/'metadata/listening-script.js'
    script.write_text(html.split('<script>',1)[1].split('</script>',1)[0],'utf-8')
    node=shutil.which('node')
    if not node:raise RuntimeError('Node needed for listening-page syntax check')
    subprocess.run([node,'--check',str(script)],check=True)
    print(str(len(rows))+' WAV bindings + listening JavaScript syntax verified; no human listening')


if __name__=='__main__':main()
