import json
from types import SimpleNamespace

import pytest

from src.vc.config import AIParameters
from src.vc.meanvc2_phrase import MeanVC2PhraseBackend
from src.vc.models import DEFAULT_VOICE_ID, profile
from src.vc.naturalness_profiles import discover


def test_the_shipped_profile_keeps_neutral_fx_and_phrase_repair():
    # The energy-only and combined experimental candidates are gone; the one shipped
    # profile keeps the full repair and the neutral control set.
    entry = profile(DEFAULT_VOICE_ID)
    current = AIParameters()
    assert current.model == DEFAULT_VOICE_ID
    assert current.startup_frames == entry['startup_chunks'] * current.chunk_frames
    assert current.queue_chunks == entry['queue_chunks']
    assert entry.get('phrase_repair')
    assert 'repair_mode' not in entry
    assert current.pitch == 0 and not current.post_fx
    for name in ('meanvc2_ref20_energy', 'meanvc2_ref60_combined'):
        from src.settings.manager import AppSettings
        assert AppSettings.from_dict({'ai_model': name}).ai_model == DEFAULT_VOICE_ID


def test_select_energy_disables_only_pitch_edit():
    backend = MeanVC2PhraseBackend(1)
    backend.repair = SimpleNamespace(pitch=True, energy=True, focus=None,
                                     lookahead_hops=10)
    backend.select_profile({'repair_mode':'energy'})
    assert not backend.repair.pitch and backend.repair.energy
    assert backend.stats['phrase_pitch_limit_st'] == 0
    backend.select_profile({})
    assert backend.repair.pitch and backend.repair.energy
    with pytest.raises(ValueError):backend.select_profile({'repair_mode':'unknown'})


def test_experimental_profiles_require_complete_local_package(tmp_path):
    folder=tmp_path/'models/naturalness_candidates/natural_calm'
    folder.mkdir(parents=True)
    (folder/'profile.json').write_text(json.dumps(dict(schema=1,status='COMPLETE',name='test')))
    assert not discover(tmp_path)
    for name in ('reference.wav','fixed_embedding.npy','runtime.json'):(folder/name).write_bytes(b'fixture')
    found=discover(tmp_path)
    assert len(found)==3
    assert found['natural_calm_none']['extra_delay_ms']==0
    assert not found['natural_calm_none']['phrase_repair']
    assert found['natural_calm_energy']['repair_mode']=='energy'
    assert found['natural_calm_combined']['extra_delay_ms']==1600
