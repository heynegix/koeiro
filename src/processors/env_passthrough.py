"""Callback-safe routing for non-voice environmental sound on the AI route.

The AI slot converts whatever reaches it, so a clap, a keypress or a knock
arrives at the listener re-synthesised as speech-like artefacts. This module
supplies the two pieces the passthrough needs without putting anything
blocking on the callback:

* ``DelayRing`` — the input history the converted output is aligned against.
  The worker's output stream describes the same content as its input stream,
  exactly ``model_alignment_delay_ms`` behind, so a fixed read offset in the
  sample-index domain keeps a raw copy that lines up with the converted block
  being replaced; pipeline backlog never has to be measured.
* ``TransientDetector`` — the cheap per-hop envelope flag fed to the routing
  gate. It only reports whether a block carried a broadband transient
  (claps, keyboard, taps, percussive hits); the model-backed decision
  (speech veto, event classes) lives in ``src/nonvoice``.
"""
import math

import numpy as np


class DelayRing:
    """Fixed-capacity history of one stream, addressed by absolute sample index."""

    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Positive ring capacity required')
        self.capacity = capacity
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._position = 0
        self._written = 0

    @property
    def written(self):
        return self._written

    def reset(self):
        self._buffer.fill(0)
        self._position = 0
        self._written = 0

    def write(self, block):
        n = len(block)
        if n > self.capacity:
            raise ValueError('Block does not fit the history ring')
        end = self._position + n
        if end <= self.capacity:
            self._buffer[self._position:end] = block
        else:
            split = self.capacity - self._position
            self._buffer[self._position:] = block[:split]
            self._buffer[:end - self.capacity] = block[split:]
        self._position = end % self.capacity
        self._written += n

    def read_into(self, out, start):
        """Copy ``out`` from index ``start``; positions never written read zero.

        ``start`` may be negative: output content older than the first input
        sample is the worker's silence prefill, and zeros are its raw
        counterpart.
        """
        out.fill(0)
        begin = max(start, 0)
        end = min(start + len(out), self._written)
        if begin >= end:
            return
        offset = begin - start
        position = begin % self.capacity
        first = min(end - begin, self.capacity - position)
        out[offset:offset + first] = self._buffer[position:position + first]
        if end - begin > first:
            out[offset + first:offset + end - begin] = self._buffer[:end - begin - first]


class TransientDetector:
    """Streaming broadband-transient flags on 512-sample hops (10.7 ms @48 kHz).

    ``process`` returns True when any hop in the block carried a fast envelope
    jump over the slow one. Deliberately dumb on its own: speech onsets and
    lone mouse clicks look the same here, and the nonvoice gate is what
    separates them.
    """

    HOP = 512
    RATE = 48000.0
    SILENCE_RMS = 10.0 ** (-50.0 / 20.0)
    TRANSIENT_RATIO = 3.2
    FAST_TAU = 0.010
    SLOW_TAU = 0.150

    def __init__(self):
        self._hop = np.zeros(self.HOP, dtype=np.float32)
        self._fill = 0
        self._fast_alpha = 1.0 - math.exp(-self.HOP / (self.RATE * self.FAST_TAU))
        self._slow_alpha = 1.0 - math.exp(-self.HOP / (self.RATE * self.SLOW_TAU))
        self.reset()

    def reset(self):
        self._hop.fill(0)
        self._fill = 0
        self._fast = 0.0
        self._slow = 0.0

    def process(self, block):
        """Consume one block of any length and report any transient hop."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        offset = 0
        found = False
        while offset < len(block):
            take = min(self.HOP - self._fill, len(block) - offset)
            self._hop[self._fill:self._fill + take] = block[offset:offset + take]
            self._fill += take
            offset += take
            if self._fill == self.HOP:
                found = self._analyse() or found
                self._fill = 0
        return found

    def _analyse(self):
        hop = self._hop
        rms = float(np.sqrt(np.mean(np.multiply(hop, hop))))
        self._fast += self._fast_alpha * (rms - self._fast)
        self._slow += self._slow_alpha * (rms - self._slow)
        return (rms > self.SILENCE_RMS * 3.0
                and self._fast > max(self._slow, self.SILENCE_RMS) * self.TRANSIENT_RATIO)
