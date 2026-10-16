"""Model files and the AudioSet event-list for the nonvoice passthrough gate.

Silero VAD (v6, MIT) vetoes speech; YAMNet (Apache-2.0, AudioSet ontology
CC BY 4.0) scores environmental events. Both ONNX sessions run on the
analysis thread with one thread each; nothing here touches the callback.
"""
import csv
import logging
from pathlib import Path

import numpy as np

from src.runtime_paths import asset_root, install_root, is_frozen

log = logging.getLogger(__name__)

SILERO_FILE = 'silero_vad.onnx'
YAMNET_FILE = 'yamnet.onnx'
CLASS_MAP_FILE = 'yamnet_class_map.csv'

# AudioSet classes treated as environmental events. Names must exist verbatim
# in yamnet_class_map.csv; "Rattle (instrument)" is deliberately absent, it is
# a musical instrument class.
EVENT_NAMES = frozenset({
    "Hands", "Finger snapping", "Clapping", "Cheering", "Applause", "Laughter",
    "Knock", "Tap", "Thump, thud", "Slam", "Door", "Sliding door",
    "Typing", "Computer keyboard", "Mouse", "Tick", "Tick-tock", "Clock",
    "Squeak", "Cupboard open or close", "Drawer open or close",
    "Dishes, pots, and pans", "Cutlery, silverware", "Chopping (food)",
    "Zipper (clothing)", "Scissors", "Coin (dropping)", "Camera",
    "Burst, pop", "Crack", "Smash, crash", "Breaking", "Glass", "Shatter",
    "Crunch", "Creak", "Scrape", "Rub", "Rustle", "Tearing", "Filing (rasp)",
    "Beep, bleep", "Ding", "Ding-dong", "Doorbell", "Telephone bell ringing",
    "Ringtone", "Alarm clock", "Buzzer", "Fire alarm", "Car alarm",
    "Reversing beeps", "Sewing machine", "Pour", "Fill (with liquid)",
    "Boiling", "Water tap, faucet", "Firecracker", "Gurgling", "Crackle",
    "Sizzle", "Squish", "Rustling leaves", "Whistling", "Snort", "Fart",
    "Hiccup", "Burping, eructation", "Baby cry, infant cry",
    "Clang", "Clatter", "Bang", "Slap, smack", "Chink, clink", "Thunk",
})

# The user's own acute body sounds. They are NOT environmental events: a raw
# cough or sniffle is the microphone owner's voice leaking unchanged, so a
# high score vetoes routing the way speech does. "Sigh"/"Gasp" are
# deliberately absent: YAMNet reads the decay of claps and knocks as Sigh
# (measured 0.36 on the synthetic clap the tests use), and a veto that fires
# on legitimate events silences them. "Snoring" is long-form; acute leak
# protection stays with these four classes. Measured on the reference
# recording (knocks, taps, typing) all four stay under 0.03.
VOCAL_NAMES = frozenset({
    "Cough", "Throat clearing", "Sneeze", "Sniff",
})


def models_dir():
    """First directory holding the full model set, or None."""
    roots = [Path(asset_root()) / 'models' / 'nonvoice']
    if is_frozen():
        # A frozen install may have been set up (setup_release) beside the exe.
        roots.append(Path(install_root()) / 'models' / 'nonvoice')
    for root in roots:
        if all((root / name).is_file() for name in (SILERO_FILE, YAMNET_FILE, CLASS_MAP_FILE)):
            return root
    return None


def class_map_names(path):
    """AudioSet index -> display name from the shipped class map."""
    with open(path, encoding='utf-8', newline='') as stream:
        rows = list(csv.reader(stream))[1:]
    return {int(row[0]): row[2] for row in rows}


def _class_ids(path, wanted_names, label):
    """Sorted class indexes for a name set; unknown names are reported once."""
    names = class_map_names(path)
    wanted = {name for name in names.values() if name in wanted_names}
    missing = wanted_names - wanted
    if missing:
        log.warning('nonvoice class map lacks %d %s classes: %s',
                    len(missing), label, ', '.join(sorted(missing)))
    return np.array(sorted(index for index, name in names.items() if name in wanted), dtype=np.int64)


def event_ids(path):
    """Sorted class indexes for EVENT_NAMES."""
    return _class_ids(path, EVENT_NAMES, 'event')


def vocal_ids(path):
    """Sorted class indexes for VOCAL_NAMES; their scores veto routing."""
    return _class_ids(path, VOCAL_NAMES, 'vocal')


def load_sessions(directory):
    """Open the Silero and YAMNet sessions, pinned to a single thread each."""
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    providers = ['CPUExecutionProvider']
    silero = ort.InferenceSession(str(Path(directory) / SILERO_FILE),
                                  sess_options=options, providers=providers)
    yamnet = ort.InferenceSession(str(Path(directory) / YAMNET_FILE),
                                  sess_options=options, providers=providers)
    class_map = Path(directory) / CLASS_MAP_FILE
    return silero, yamnet, event_ids(class_map), vocal_ids(class_map)
