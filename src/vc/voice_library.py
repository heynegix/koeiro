"""Local voice library. Only completed registrations enter the live profile list."""
import hashlib
import json
import math
from pathlib import Path
import re

from ..runtime_paths import asset_root

ROOT = asset_root()
VOICE_ID = re.compile(r'user_[0-9a-f]{32}\Z')
# The voice this build ships and starts on. It is listed first in the voice combo
# under the app's own label rather than the id it was registered with, and the
# shipped profile.json names it the same way, so the standard voice is never
# presented as one of the user's own registrations.
DEFAULT_VOICE_ID = 'user_3b11387911244c6e9a013b42fa88a458'
DEFAULT_VOICE_NAME = '標準ボイス'
# The shipped profile records this as its source; a user registration always names
# the audio files it was built from.
STANDARD_VOICE_SOURCE = 'built-in'
# Registration retains up to 60s of speech. The ceiling leaves room for a future
# larger budget while still bounding what a hand-edited profile.json can claim.
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 120.0
# A profile now carries per-clip measurements, so the guard is larger than a flat
# metadata file but still far too small to be audio or a model.
MAX_PROFILE_BYTES = 65536


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def folder_for(identifier, root=ROOT):
    if not isinstance(identifier, str) or not VOICE_ID.fullmatch(identifier):
        raise ValueError('Invalid voice identifier')
    folder = Path(root) / 'models/user_voices' / identifier
    if not folder.resolve().is_relative_to((Path(root)/'models/user_voices').resolve()):
        raise ValueError('Voice path escapes library')
    return folder


def discover(root=ROOT):
    """Read small metadata only; no audio/model work on the GUI or callback."""
    found = {}
    for path in sorted((Path(root)/'models/user_voices').glob('user_*/profile.json')):
        try:
            identifier = path.parent.name
            folder = folder_for(identifier, root)
            if path.stat().st_size > MAX_PROFILE_BYTES:
                continue
            data = json.loads(path.read_text('utf-8'))
            if data.get('schema') != 1 or data.get('id') != identifier:
                continue
            name = data.get('name')
            if not isinstance(name, str) or not name.strip() or len(name) > 80:
                continue
            seconds = data.get('reference_seconds')
            if type(seconds) not in (float, int) or not math.isfinite(seconds) \
                    or not MIN_REFERENCE_SECONDS <= seconds <= MAX_REFERENCE_SECONDS:
                continue
            filename = data.get('source_filename')
            if not isinstance(filename, str) or not filename or len(filename) > 1024:
                continue
            if not all((folder/filename).is_file() for filename in ('runtime.json', 'reference.wav', 'fixed_embedding.npy')):
                continue
            # Timing and backend cannot be supplied by an arbitrary JSON file.
            # Grouped inference is the tuned condition for this CPU: six 120ms
            # blocks per call plus 36-frame vocoder decoding. It leaves the
            # reference, the fixed embedding, the checkpoint and the 120ms
            # attention mask untouched and only widens the fixed algorithmic
            # buffer from 960ms to 1440ms. No user voice is human-approved, so
            # this is the default rather than an additional profile.
            found[identifier] = dict(folder='user_voices/'+identifier,
                name=DEFAULT_VOICE_NAME if identifier == DEFAULT_VOICE_ID else name,
                backend='meanvc2', user_voice=True, phrase_repair=True,
                standard=identifier == DEFAULT_VOICE_ID,
                vc_group_chunks=6, vocoder_batch_frames=36,
                model_buffer_ms=1440, grid_delay_ms=0, extra_delay_ms=1600,
                startup_chunks=6, queue_chunks=10, mute_during_startup=True,
                optimized_condition='user voice: 720ms grouped VC + 36-frame Vocos',
                voice=0, pitch=0., formant=0.)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return found


def metadata(identifier, root=ROOT):
    return json.loads((folder_for(identifier, root)/'profile.json').read_text('utf-8'))


def is_standard(identifier):
    """True for the voice this build ships, false for every user registration."""
    return identifier == DEFAULT_VOICE_ID
