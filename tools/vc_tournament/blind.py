import hashlib
import html
import json
import random
from pathlib import Path
from .audio import digest, normalize

def build(out, rows):
    folder=out/'blind';folder.mkdir(parents=True,exist_ok=True)
    successful=[r for r in rows if r['status']=='SUCCESS']
    random.Random(110).shuffle(successful)
    mapping=[];public=[]
    for i,row in enumerate(successful,1):
        bid=f'A{i:03d}';dest=folder/f'{bid}.wav'
        correction=normalize(row['output'],dest)
        mapping.append(dict(blind_id=bid,model=row['model'],source=row['source'],
                            reference=row['reference'],fingerprint=row['fingerprint'],normalization=correction))
        # No model ID, speed, or revealing filenames in the audition page.
        public.append(dict(id=bid,audio=dest.name,source='../source/'+Path(row['source']).name,
                           reference='../reference/reference_10s.wav',sha256=digest(dest)))
    package=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()
    (out/'metadata/blind_manifest.json').write_text(json.dumps(dict(package=package,candidates=mapping),indent=2),encoding='utf-8')
    template=Path(__file__).with_name('listening.html').read_text('utf-8')
    data=json.dumps(dict(package=package,candidates=public),ensure_ascii=False).replace('<','\\u003c')
    (folder/'index.html').write_text(template.replace('__DATA__',data),encoding='utf-8')
    return dict(package=package,count=len(public),page=str(folder/'index.html'))
