"""ASR-only bounded framing. VC's smaller packet contract remains unchanged."""
import json
import struct
import numpy as np
from src.vc.protocol import _read_exact

MAX_AUDIO_BYTES=48000*4  # max 1 second at 48 kHz


def send(stream,header,audio=None):
    encoded=json.dumps(header,allow_nan=False).encode('utf-8')
    payload=b'' if audio is None else np.asarray(audio,dtype='<f4').tobytes()
    if len(encoded)>32768 or len(payload)>MAX_AUDIO_BYTES:
        raise ValueError('ASR packet exceeds bounded size')
    stream.write(struct.pack('<II',len(encoded),len(payload))+encoded+payload); stream.flush()


def receive(stream):
    hs,ps=struct.unpack('<II',_read_exact(stream,8))
    if hs>32768 or ps>MAX_AUDIO_BYTES or ps%4: raise ValueError('Invalid ASR packet size')
    header=json.loads(_read_exact(stream,hs))
    audio=np.frombuffer(_read_exact(stream,ps),dtype='<f4').copy()
    return header,audio
