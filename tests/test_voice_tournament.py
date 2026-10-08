import json
from pathlib import Path
import wave
import numpy as np
import pytest

from tools.anime_voice_tournament.core import (
    Candidate, blind_mapping, cache_valid, digest, inspect_audio, normalize,
    save_json, write_audio,
)
from tools.anime_voice_tournament.run import package_review, stage1_candidates
from tools.anime_voice_tournament.shortlist import reviewed_candidates, plan
from tools.anime_voice_tournament.render import pair_cursor


@pytest.mark.parametrize('spec', [dict(pitch=float('nan')),dict(pitch=13),dict(pitch=.01),
                                 dict(formant=.25),dict(formant=3),dict(voice=-1),dict(voice=True)])
def test_invalid_native_candidate(spec):
    with pytest.raises(ValueError): Candidate(**(dict(model='jvs',voice=1)|spec))


def test_identity_and_deterministic_blind_mapping():
    a=Candidate('jvs',1);b=Candidate('jvs',9)
    assert a.key()==Candidate('jvs',1,4,0).key()
    assert a.key()!=b.key()
    assert blind_mapping([a.key(),b.key()])==blind_mapping([b.key(),a.key()])


def test_missing_model_and_invalid_merge():
    with pytest.raises(ValueError): Candidate('missing',1)
    for weights in (((1,.8),(2,.8)),((1,.5),(1,.5)),((1,-.1),(2,1.1))):
        with pytest.raises(ValueError): Candidate('jvs',1,merge=weights)


def test_merge_cursor_matches_native_ratio():
    for target in (.8,.7,.6,.5):
        x,weight=pair_cursor(target)
        assert .2<=x<=.5
        assert abs(weight-target)<.004


def test_normalize_preserves_shape_and_dynamics():
    audio=np.array([0.,.1,.2,-.3,1.2],np.float32)
    result,stats=normalize(audio)
    assert np.max(abs(result))<=.951
    assert np.allclose(result,audio*stats['gain'])
    assert stats['raw_over_one']==1
    with pytest.raises(ValueError): normalize(np.array([np.nan]))
    with pytest.raises(ValueError): normalize(np.zeros(10))


def test_source_and_output_validation(tmp_path):
    path=tmp_path/'source.wav';audio=(.1*np.sin(np.arange(480000)*.02)).astype(np.float32)
    write_audio(path,audio)
    info=inspect_audio(path,source=True)
    assert info['duration']==10 and info['clip_count']==0
    write_audio(path,audio[:20])
    with pytest.raises(ValueError): inspect_audio(path,source=True)


def test_cache_detects_corruption_and_input_change(tmp_path):
    path=tmp_path/'output.wav';write_audio(path,np.array([.1,.2],np.float32))
    entry=dict(status='ok',cache_key='input-key',output_sha256=digest(path))
    assert cache_valid(entry,'input-key',path)
    assert not cache_valid(entry,'different-input',path)
    path.write_bytes(b'broken')
    assert not cache_valid(entry,'input-key',path)


def test_atomic_metadata_no_nan(tmp_path):
    path=tmp_path/'manifest.json';save_json(path,dict(winner=None))
    assert json.loads(path.read_text())==dict(winner=None)
    with pytest.raises(ValueError): save_json(path,dict(winner=float('nan')))
    assert json.loads(path.read_text())==dict(winner=None)


def test_sweep_only_existing_voices():
    candidates=stage1_candidates([dict(model='jvs',voice=1),dict(model='character',voice=0)])
    assert len(candidates)==2 and Candidate('jvs',1) in candidates


def test_blind_package_never_exposes_voice_settings(tmp_path):
    audio=tmp_path/'stage1_voice/a.wav';write_audio(audio,np.array([.1,.2],np.float32))
    manifest=dict(renders=[dict(stage='stage1_voice',status='ok',recipe_key='key',source_id='S01',
        source_sha256='source',output_sha256=digest(audio),output='stage1_voice/a.wav',
        candidate=Candidate('jvs',1).recipe())])
    manifest['renders'].append(dict(manifest['renders'][0],source_id='S02'))
    package_review(tmp_path,manifest,'stage1_voice')
    first_mapping=json.loads((tmp_path/'metadata/stage1_voice_blind_mapping.json').read_text())
    manifest['renders'].reverse()
    package_review(tmp_path,manifest,'stage1_voice')
    assert json.loads((tmp_path/'metadata/stage1_voice_blind_mapping.json').read_text())==first_mapping
    html=(tmp_path/'blind/stage1_voice/index.html').read_text(encoding='utf-8')
    assert 'jvs' not in html and 'applied_pitch' not in html
    assert 'A001_S01.wav' in html
    assert 'mechanical' in html and '未評価' in html
    assert not list((tmp_path/'blind/stage1_voice').glob('*mapping*'))
    mapping=json.loads((tmp_path/'metadata/stage1_voice_blind_mapping.json').read_text())
    save_json(tmp_path/'metadata/tournament_manifest.json',manifest)
    ratings=dict(human_review=True,package_id=mapping['package_id'],ratings=[dict(candidate_id='A001',first_pass='アニメっぽい')])
    assert reviewed_candidates(tmp_path,'stage1_voice',ratings,['A001'])==[Candidate('jvs',1)]
    ratings['package_id']='old'
    with pytest.raises(ValueError): reviewed_candidates(tmp_path,'stage1_voice',ratings,['A001'])


def test_next_stage_bounded_without_winner():
    chosen=[Candidate('jvs',1),Candidate('jvs',9)]
    pitch=plan(chosen,'pitch');formant=plan(chosen,'formant');merge=plan(chosen,'merge')
    assert len(pitch)==16 and len(formant)==10 and len(merge)==4
    assert all(c.merge for c in merge)
    with pytest.raises(ValueError): plan([chosen[0],Candidate('character',0)],'merge')
