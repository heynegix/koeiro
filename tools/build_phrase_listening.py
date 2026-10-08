"""Blind comparison of preserved voice and bounded phrase repairs; no inference."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'recordings/v011_phrase_repair'


def main():
    from tools.vc_tournament.audio import normalize
    blind=OUT/'blind';metadata=OUT/'metadata'
    blind.mkdir(exist_ok=True);metadata.mkdir(exist_ok=True)
    settings=json.loads((ROOT/'settings.json').read_text('utf-8'))
    model=settings['ai_model'].removesuffix('_phrase')
    shutil.copy2(ROOT/'models'/model/'reference.wav',OUT/'reference.wav')
    rows=[(source,variant) for source in ('normal','low','bright','long')
          for variant in ('baseline','energy','pitch','combined')]
    random.Random(511).shuffle(rows)
    public=[];private=[]
    for i,(source,variant) in enumerate(rows,1):
        ident=f'P{i:03}'
        original=OUT/f'{source}_{variant}.wav'
        dest=blind/f'{ident}.wav'
        correction=normalize(original,dest)
        public.append(dict(id=ident,audio=dest.name,source=f'../{source}_source.wav',
                           reference='../reference.wav',source_kind=source))
        private.append(dict(id=ident,source=source,variant=variant,normalization=correction))
    package=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()
    (metadata/'blind_manifest.json').write_text(json.dumps(dict(package=package,model=model,candidates=private),
        ensure_ascii=False,indent=2),'utf-8')
    template=(ROOT/'tools/vc_tournament/listening.html').read_text('utf-8')
    template=template.replace('__DATA__',json.dumps(dict(package=package,candidates=public),ensure_ascii=False).replace('<','\\u003c'))
    template=template.replace('v0.11 Blind Voice Comparison','今の声を保つ・抑揚補完比較')
    template=template.replace("['mechanical','機械感']","['mechanical','機械感'],['prosody','抑揚の自然さ'],['voice_changed','今の声からの変化']")
    template=template.replace('元声残り・機械感：5が多い','元声残り・機械感・声の変化：5が多い')
    template=template.replace("[['Candidate',c.audio],['Target Reference',c.reference],['Source',c.source]]",
        "[['Candidate',c.audio],['現在の声','../'+c.source_kind+'_baseline.wav'],['Source',c.source]]")
    template=template.replace('v011-ratings.json','phrase-repair-ratings.json')
    template=template.replace('<main id="candidates">','<label>Source <select id="sourcefilter"><option value="long">長文</option><option value="normal">通常声</option><option value="low">低め</option><option value="bright">明るめ</option><option value="">全て</option></select></label><main id="candidates">')
    template=template.replace('for(const c of data.candidates){',"for(const c of data.candidates.filter(c=>!document.getElementById('sourcefilter').value||c.source_kind===document.getElementById('sourcefilter').value))){")
    template=template.replace('render();\n</script>',"document.getElementById('sourcefilter').onchange=render;\nrender();\n</script>")
    (blind/'index.html').write_text(template,'utf-8')
    print(str(blind/'index.html'))


if __name__=='__main__':main()
