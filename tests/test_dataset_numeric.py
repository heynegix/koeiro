"""SciPy-dependent dataset fixtures run in isolated AI/analysis environment."""
import numpy as np
import pytest
pytest.importorskip('scipy',reason='Dataset numeric tests use isolated environment')
from scipy.io import wavfile
from src.dataset.features import validate_pairs,read_wave,extract_features,metrics,relative_f0,voiced_delta
from src.dataset.alignment import align,training_sample,split_ids


def wave_file(path,frequency=160,samples=3200):
    path.parent.mkdir(parents=True,exist_ok=True)
    wavfile.write(path,16000,(.1*np.sin(2*np.pi*frequency*np.arange(samples)/16000)).astype(np.float32))


def test_pair_validator_and_missing(tmp_path):
    wave_file(tmp_path/'generated/neutral/000001.wav')
    result=validate_pairs(tmp_path,['000001'])
    assert not result['valid_pairs'] and 'missing' in result['problems'][0]['reason']
    wave_file(tmp_path/'generated/anime/000001.wav')
    result=validate_pairs(tmp_path,['000001'])
    assert len(result['valid_pairs'])==1 and not result['problems']


def test_duplicate_and_unexpected_ids(tmp_path):
    for style in ('neutral','anime'):
        wave_file(tmp_path/f'generated/{style}/000001.wav')
    wave_file(tmp_path/'generated/neutral/other/000001.wav')
    wave_file(tmp_path/'generated/anime/999999.wav')
    result=validate_pairs(tmp_path,['000001'])
    assert not result['valid_pairs']
    assert any(p['reason']=='Duplicate ID' for p in result['problems'])
    assert any(p['reason']=='Unexpected ID' for p in result['problems'])


@pytest.mark.parametrize('audio',[np.zeros(1000,np.float32),np.full(1000,np.nan,np.float32),np.empty(0,np.float32)])
def test_bad_wave(tmp_path,audio):
    path=tmp_path/'bad.wav'; wavfile.write(path,16000,audio)
    with pytest.raises(ValueError): read_wave(path)


def test_invalid_wave_format(tmp_path):
    path=tmp_path/'bad.wav'; path.write_bytes(b'bad')
    with pytest.raises(ValueError): read_wave(path)


class FixtureEstimator:
    def extract(self,audio):
        n=len(audio)//160+1
        return np.full(n,160,np.float32),np.ones(n,np.float32)


def test_extract_features_and_metrics(tmp_path):
    path=tmp_path/'voice.wav'; wave_file(path)
    f=extract_features(path,FixtureEstimator())
    assert len(f['f0'])==len(f['energy'])==len(f['time'])==21
    assert f['f0'].dtype==np.float32 and f['duration']==.2
    assert all(np.isfinite(f[k]).all() for k in ('f0','energy','rms','time'))
    m=metrics(f)
    assert m['f0']['median']==160 and m['voiced_ratio']>0
    assert m['pitch_st']['std']==0


def features(n=50):
    x=np.linspace(0,1,n)
    return dict(f0=(160*2**(.15*np.sin(x*6))).astype(np.float32),voiced=np.ones(n,np.float32),
        energy=(-30+8*np.sin(x*6)).astype(np.float32),time=x*.5,silence_mask=np.zeros(n,bool),
        duration=np.float64(.5),hop_seconds=np.float64(.01))


def test_relative_register_and_unvoiced_delta():
    f=np.array([0,100,110,0,120],np.float32)
    a=relative_f0(f); b=relative_f0(f*2)
    np.testing.assert_allclose(a,b)
    assert a[0]==0 and a[3]==0
    delta=voiced_delta(a,(f>0).astype(np.float32))
    assert delta[0]==delta[1]==delta[3]==delta[4]==0 and delta[2]>0


def test_dtw_stretch_not_equal_index_and_training_samples(tmp_path):
    neutral=features(50); anime=features(80)
    mapping,stats=align(neutral,anime)
    assert len(mapping)==50 and np.all(np.diff(mapping)>=0)
    assert mapping[-1]>70 and stats['monotonic'] and stats['mean_cost']<.2
    sample=training_sample(neutral,anime,mapping)
    assert all(len(sample[k])==50 for k in ('neutral_f0','target_f0','alignment','target_pitch_offset_st','target_energy_offset_db'))
    path=tmp_path/'sample.npz'; np.savez_compressed(path,**sample)
    with np.load(path,allow_pickle=False) as loaded:
        assert loaded['schema_version']==1
        assert np.isfinite(loaded['target_f0_delta']).all()


def test_dtw_empty_and_invalid_mapping():
    with pytest.raises(ValueError): align(features(0),features(20))
    with pytest.raises(ValueError): training_sample(features(10),features(20),np.arange(11))
    with pytest.raises(ValueError): training_sample(features(10),features(20),np.full(10,-1))


def test_split_by_sentence_id_no_leak():
    ids=[f'{i:06}' for i in range(500)]
    a=split_ids(ids); b=split_ids(ids[::-1])
    assert a==b and [len(a[k]) for k in ('train','validation','test')]==[400,50,50]
    assert not set(a['train'])&set(a['validation']) and not set(a['test'])&set(a['train'])
    assert set().union(*map(set,a.values()))==set(ids)
    with pytest.raises(ValueError): split_ids(['1','1'])


def test_nonfinite_estimator_output_safe(tmp_path):
    class BadEstimator:
        def extract(self,audio):
            return np.full(len(audio)//160+1,np.nan,np.float32),np.ones(len(audio)//160+1,np.float32)
    path=tmp_path/'voice.wav'; wave_file(path)
    f=extract_features(path,BadEstimator())
    assert np.all(f['f0']==0) and np.all(f['voiced']==0)


def test_content_duplicates_are_reported(tmp_path):
    for style in ('neutral','anime'):
        wave_file(tmp_path/f'generated/{style}/000001.wav')
    result=validate_pairs(tmp_path,['000001'])
    assert len(result['duplicate_audio'])==1


def test_energy_target_removes_speaker_level_difference():
    a=features(50); b=features(50)
    b['energy']=b['energy']+12
    sample=training_sample(a,b,np.arange(50))
    np.testing.assert_allclose(sample['target_energy_offset_db'],0,atol=1e-5)
    np.testing.assert_allclose(sample['target_energy_raw_offset_db'],12,atol=1e-5)
