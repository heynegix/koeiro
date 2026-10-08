"""Small framed worker protocol. Used only by the AI bridge/service threads."""
import json
import struct
import numpy as np

MAX_PAYLOAD = 60 * 48000 * 4


def _payload_limit(header):
    return MAX_PAYLOAD if header.get('op')=='utterance' else 65536


def _read_exact(stream, size):
    parts = bytearray()
    while len(parts) < size:
        part = stream.read(size-len(parts))
        if not part:
            raise EOFError('AI worker pipe closed')
        parts.extend(part)
    return parts


def send(stream, header, audio=None):
    encoded = json.dumps(header, allow_nan=False).encode('utf-8')
    payload = b'' if audio is None else np.asarray(audio, dtype='<f4').tobytes()
    if len(encoded) > 32768 or len(payload) > _payload_limit(header):
        raise ValueError('AI packet exceeds bounded size')
    # One packet/write avoids waking the other process between header and body.
    # Packet allocation happens only in worker threads, never audio callbacks.
    stream.write(struct.pack('<II', len(encoded), len(payload))+encoded+payload)
    stream.flush()


def receive(stream):
    header_size, payload_size = struct.unpack('<II', _read_exact(stream, 8))
    if header_size > 32768 or payload_size > MAX_PAYLOAD or payload_size % 4:
        raise ValueError('Invalid AI packet size')
    header = json.loads(_read_exact(stream, header_size))
    if not isinstance(header,dict) or payload_size>_payload_limit(header):
        raise ValueError('Invalid AI operation/payload')
    audio = np.frombuffer(_read_exact(stream, payload_size), dtype='<f4').copy()
    return header, audio
