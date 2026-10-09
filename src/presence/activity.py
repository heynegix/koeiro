"""What Discord is told about the session, as a pure function.

Discord renders Rich Presence, called "activity" in current Discord docs, in three
lines: the registered application name, then the activity's ``details``, then its
``state``. The first line is owned by the Discord developer portal, so this module
only decides the other two.

Two facts are deliberately the whole content: whether voice conversion is running,
and which voice is selected -- naming the one this build ships as the standard voice
and anything registered later as an added voice, because "標準ボイス" and a
self-registered name read the same otherwise.

The activity's image is listed twice, under ``large_image`` and ``small_image``:
the "playing" card draws the first and its corner (and the call row) the second, so
one without the other leaves a "?" in the other slot.

Keeping the activity a pure function of a small frozen snapshot means what Discord
would show can be tested without Discord, and the GUI thread never builds payloads.
"""
import os
from dataclasses import dataclass

# The Discord application this build reports to. Discord shows the name registered
# for this id, not anything sent here.
APPLICATION_ID = '1558051751137910794'

# Discord's activity image field takes either an uploaded application asset (a key
# from the developer portal, which Discord resolves itself) or an external image URL
# that Discord fetches through its media proxy. This is the external route: the
# project's own committed icon, so no portal step can be forgotten when publishing.
#
# It must be a format Discord renders -- PNG, JPEG or WebP. assets/icon.ico is only
# that: an icon file, which Discord's proxy does not turn into a card image. The
# proxy is also where the "?" comes from: it reports a fetch failure as a missing
# asset, and its cache can keep an old failure, so a fixed URL can need a moment and
# a fresh client start before the image appears.
#
# Swapping to the portal route is this one line: the asset key is what the portal
# shows next to the uploaded image.
ICON_IMAGE = 'https://raw.githubusercontent.com/heynegix/koeiro/refs/heads/main/assets/discord/icon.png'

# Hover text, and the one place a deployed build can be pointed somewhere else
# without editing code (a portal key, or an icon hosted elsewhere).
ICON_TEXT = 'Koeiro（声彩）'
ICON_ENV = 'KOEIRO_ICON'

# Discord caps details/state at 128 characters. Measured in UTF-8 bytes so a
# Japanese voice name can never overflow, and characters are never split.
LINE_LIMIT_BYTES = 128

ACTIVITY_TYPE_PLAYING = 0

VOICE_STANDARD = 'standard'
VOICE_USER = 'user'

MODE_LABELS = {
    'original': 'Original（無変換）',
    'female_dsp': 'Female DSP（軽い加工）',
    'ai_voice': 'AI Voice',
}


@dataclass(frozen=True)
class PresenceState:
    """The session facts the activity is built from.

    ``started_at`` is the Unix time the current run began, or None while stopped:
    Discord would otherwise count up from a start that never happened.
    """

    running: bool = False
    mode: str = 'original'
    voice: str = ''
    voice_kind: str = ''
    started_at: int | None = None
    failed: bool = False


def _fit(text):
    """Truncate to Discord's line limit on a character boundary."""
    encoded = text.encode('utf-8')[:LINE_LIMIT_BYTES]
    return encoded.decode('utf-8', 'ignore')


def describe_voice(name, kind):
    """Name the voice, saying whether the app ships it or the user added it."""
    name = (name or '').strip()
    if kind == VOICE_STANDARD:
        return name or '標準ボイス'
    if kind == VOICE_USER:
        return f'{name}（追加した声）' if name else '追加した声'
    return name


def icon_image():
    """The image Discord is told to show, or '' when this build names none."""
    return os.environ.get(ICON_ENV, ICON_IMAGE).strip()


def icon_assets():
    """The ``assets`` object: the same image as the card and as its corner icon."""
    image = icon_image()
    if not image:
        return {}
    # Discord caps the hover text like the detail lines, so it is fitted the same way.
    text = _fit(ICON_TEXT)
    return {'large_image': image, 'large_text': text,
            'small_image': image, 'small_text': text}


def details_line(state: PresenceState):
    if state.running:
        return 'ボイス変換中'
    if state.failed:
        return 'エラー（変換停止中）'
    return '待機中（変換していません）'


def state_line(state: PresenceState):
    """The selected route and voice, whether or not conversion is running."""
    if state.mode == 'ai_voice':
        voice = describe_voice(state.voice, state.voice_kind)
        return f'声: {voice} · AI Voice' if voice else '声が未選択 · AI Voice'
    return MODE_LABELS.get(state.mode, str(state.mode or '停止中'))


def build_activity(state: PresenceState):
    """The ``activity`` object for a SET_ACTIVITY command."""
    activity = {
        'type': ACTIVITY_TYPE_PLAYING,
        'details': _fit(details_line(state)),
        'state': _fit(state_line(state)),
    }
    # An empty assets object would be sent as-is, and Discord has nothing to draw.
    assets = icon_assets()
    if assets:
        activity['assets'] = assets
    if state.running and state.started_at:
        activity['timestamps'] = {'start': int(state.started_at)}
    return activity
