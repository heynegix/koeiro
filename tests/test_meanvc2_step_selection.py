from types import SimpleNamespace

import numpy as np
import pytest

from src.vc.meanvc2 import MeanVC2Backend
from src.vc.config import AIParameters
from src.vc.models import DEFAULT_VOICE_ID, profile
from src.settings.manager import AppSettings


def test_shipped_profile_pins_reference_buffering_and_neutral_controls():
    entry=profile(DEFAULT_VOICE_ID)
    # Phrase repair and the grouped condition live on the one shipped profile, so a
    # variant name no longer selects a different reference or buffering.
    assert entry['folder'] and entry.get('phrase_repair')
    parameters=AIParameters()
    assert parameters.model==DEFAULT_VOICE_ID
    assert parameters.startup_frames==entry['startup_chunks']*parameters.chunk_frames
    assert parameters.queue_chunks==entry['queue_chunks']
    assert parameters.pitch==0 and not parameters.post_fx
    # A variant name from an older build falls back instead of loading a second model.
    restored=AppSettings.from_dict({'ai_model':'meanvc2_ref20_phrase_3step'})
    assert restored.ai_model==DEFAULT_VOICE_ID


def test_only_the_two_step_solver_is_accepted():
    # 2 steps is the only condition this build runs: 3 and 4 measured RTF 1.130 and
    # 1.343, both over the realtime budget.
    backend=MeanVC2Backend(1)
    backend.torch=SimpleNamespace(tensor=lambda x:np.array(x))
    resets=[];backend.reset=lambda:resets.append(True)
    backend.stats={'human_approved_offline':True,'reference_sha256':'same'}
    backend.stats['steps']=2
    backend.select_profile({'steps':2})
    assert backend.stats['steps']==2 and resets==[]
    assert backend.stats['human_approved_offline']
    assert backend.stats['reference_sha256']=='same'
    # The shipped profile changes only the inference grouping, so selecting it discards
    # the previous schedule once and then settles.
    backend.select_profile(profile(DEFAULT_VOICE_ID))
    assert backend.stats['reference_sha256']=='same'
    count=len(resets)
    backend.select_profile(profile(DEFAULT_VOICE_ID))
    assert len(resets)==count


@pytest.mark.parametrize('steps',[True,1,3,4,5,3.0])
def test_non_two_step_selection_is_rejected(steps):
    with pytest.raises(ValueError):MeanVC2Backend(1).select_profile({'steps':steps})
