"""Use explicit human selections to plan the next stage; never choose a winner."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.anime_voice_tournament.core import Candidate, OUTPUT, PITCHES, FORMANTS, save_json


def reviewed_candidates(folder, stage, ratings, selected):
    mapping = json.loads((folder/'metadata'/f'{stage}_blind_mapping.json').read_text(encoding='utf-8'))
    if ratings.get('package_id')!=mapping['package_id'] or ratings.get('human_review') is not True:
        raise ValueError('Human review for this exact package is required')
    if len(selected)!=len(set(selected)) or not 1<=len(selected)<=5:
        raise ValueError('Select 1–5 distinct candidates')
    rows = {x['candidate_id']:x for x in ratings['ratings']}
    reverse = {v:k for k,v in mapping['mapping'].items()}
    renders = json.loads((folder/'metadata/tournament_manifest.json').read_text(encoding='utf-8'))['renders']
    chosen=[]
    for candidate_id in selected:
        if candidate_id not in rows or candidate_id not in reverse or not rows[candidate_id].get('first_pass'):
            raise ValueError('Selected candidate has no first-pass listening rating')
        row = next(x for x in renders if x['recipe_key']==reverse[candidate_id] and x['status']=='ok' and x['stage']==stage)
        spec=row['candidate'].copy();spec['merge']=tuple(tuple(x) for x in spec.get('merge',()))
        chosen.append(Candidate(**spec))
    return chosen


def plan(chosen, phase):
    if phase=='pitch': return [Candidate(c.model,c.voice,p,0) for c in chosen for p in PITCHES]
    if phase=='formant': return [Candidate(c.model,c.voice,c.pitch,f) for c in chosen for f in FORMANTS]
    if len(chosen)!=2 or chosen[0].model!=chosen[1].model or chosen[0].voice==chosen[1].voice:
        raise ValueError('Native merge requires 2 distinct voices from the same model')
    a,b=chosen
    return [Candidate(a.model,a.voice,a.pitch,a.formant,((a.voice,w),(b.voice,1-w))) for w in (.8,.7,.6,.5)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',default=str(OUTPUT));parser.add_argument('--ratings',required=True)
    parser.add_argument('--stage',required=True);parser.add_argument('--select',nargs='+',required=True)
    parser.add_argument('--phase',choices=['pitch','formant','merge'],required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args()
    ratings=json.loads(Path(args.ratings).read_text(encoding='utf-8'))
    chosen=reviewed_candidates(Path(args.folder),args.stage,ratings,args.select)
    candidates=plan(chosen,args.phase)
    save_json(args.out,dict(human_shortlist=args.select,rating_package=ratings['package_id'],
        candidates=[c.recipe() for c in candidates],winner=None))


if __name__=='__main__':main()
