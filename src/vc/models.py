"""Built-in profiles plus completed, locally registered MeanVC2 voices.

This build offers:

* ``DEFAULT_VOICE_ID`` — the standard voice this build ships and starts on.
* Every other completed local registration under ``models/user_voices``.
* ``streaming`` — convert chunk by chunk while speaking.
* ``utterance_lavasr`` — wait for the end of an utterance, then restore bandwidth.

The routes that were available during development (the 3/4-step variants, the
experimental naturalness candidates and the Beatrice/UTAU character voices) are
deliberately absent. Their model files may still be on disk; this module is what
decides what the app offers.
"""
from .voice_library import DEFAULT_VOICE_ID, DEFAULT_VOICE_NAME, discover as discover_user_voices

# Fallback if the default registration is missing, so the app still starts.
FALLBACK_VOICE_ID = DEFAULT_VOICE_ID

# Grouped inference is the tuned condition for this CPU: six 120 ms blocks per call and
# 36-frame Vocos decoding. It leaves the reference, the fixed embedding, the checkpoint
# and the 120 ms attention mask untouched.
_GROUPED = dict(backend='meanvc2', voice=0, pitch=0., formant=0.,
                vc_group_chunks=6, vocoder_batch_frames=36,
                model_buffer_ms=1440, grid_delay_ms=0, extra_delay_ms=1600,
                startup_chunks=6, queue_chunks=10, mute_during_startup=True,
                optimized_condition='720ms grouped VC + 36-frame Vocos')

VOICE_PROFILES = {
    DEFAULT_VOICE_ID: dict(_GROUPED, name=DEFAULT_VOICE_NAME,
                           optimized_condition=_GROUPED['optimized_condition']),
}

# Delivery modes are chosen in the UI, not as separate voices, so they are not profiles.
# `streaming` and `utterance_lavasr` are the two shipped routes. `utterance_x_all`
# is the listening-comparison variant of the utterance+LavaSR route: same voice,
# same checkpoint, every experimental improvement applied at once, so the shipped
# routes stay untouched.
DELIVERY_MODES = {
    'streaming': dict(label='逐次変換',
                      detail='話しながら順に処理します',
                      delivery='streaming', enhancer='none', experiment='none'),
    'utterance_lavasr': dict(label='一括変換',
                             detail='話が終わってから変換し、LavaSRで帯域を復元します',
                             delivery='utterance', enhancer='lavasr', experiment='none'),
    'utterance_x_all': dict(label='比較用：一括＋全部入り',
                            detail='休止リフレッシュ・文末補正・高域ブレンド等を全部適用します',
                            delivery='utterance', enhancer='lavasr', experiment='all'),
    'utterance_x_natural': dict(label='比較用：一括＋自然寄せ',
                                detail='入力整音・抑揚・子音・呼気床入りの自然寄せです',
                                delivery='utterance', enhancer='lavasr', experiment='natural'),
}
DEFAULT_DELIVERY = 'utterance_lavasr'


def default_voice_id():
    """The registered voice to start on, or None when it is not present."""
    return DEFAULT_VOICE_ID if DEFAULT_VOICE_ID in VOICE_PROFILES else None


def standard_voice_id():
    """The voice this build ships, when it is available."""
    return default_voice_id()


def is_standard_voice(name):
    return name == DEFAULT_VOICE_ID


def refresh_user_profiles():
    """Re-read the library.

    Every completed local registration is offered. `discover` only returns
    folders with a valid profile, reference audio and embedding, and it pins
    the grouped inference tuning, so nothing half-written can appear.
    """
    for key in list(VOICE_PROFILES):
        if VOICE_PROFILES[key].get('user_voice') or 'folder' not in VOICE_PROFILES[key]:
            del VOICE_PROFILES[key]
    found = discover_user_voices()
    for identifier, entry in found.items():
        # The discovered entry is the source of truth for `folder` and `user_voice`;
        # _GROUPED supplies the inference tuning for the default voice entry.
        VOICE_PROFILES[identifier] = dict(entry)
    if DEFAULT_VOICE_ID in VOICE_PROFILES:
        VOICE_PROFILES[DEFAULT_VOICE_ID]['name'] = DEFAULT_VOICE_NAME


def apply_bundle_restriction():
    """Drop profiles the packaged build does not ship.

    A bundle marker lists the profiles it contains so a restricted build cannot offer a
    voice whose model or reference was deliberately left out. Unfrozen there is no
    marker and every discovered profile stays available.
    """
    from ..runtime_paths import bundle_profiles
    allowed = bundle_profiles()
    if allowed is None:
        return
    for key in list(VOICE_PROFILES):
        if key not in allowed:
            del VOICE_PROFILES[key]


refresh_user_profiles()
apply_bundle_restriction()


def is_meanvc2(name):
    selected = VOICE_PROFILES.get(name)
    return selected is not None and selected.get('backend') == 'meanvc2'


def profile(name):
    if name not in VOICE_PROFILES:
        raise ValueError('Unsupported AI voice profile')
    return dict(VOICE_PROFILES[name])