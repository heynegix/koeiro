"""Overlapped feature frontend for the streaming MeanVC2 backend.

Same checkpoint, fixed reference, fixed speaker embedding, 120ms attention mask,
24-frame KV cache, grouped inference and vocoder context as the parent backend.
The only difference is scheduling: the fbank/ASR stage runs on its own thread and
passes finished (cond, noise) frames to the conversion stage through a bounded
queue, so each stage can use cores the other one leaves idle. This CPU sustains
about 1.9 of 4 cores in the serial route.

Thread ownership is strict and there is no lock held while either stage computes.
The producer thread alone touches the waveform tail, fbank history, attention and
CNN caches, ASR offset, position table, previous BN frame and noise generator.
The consumer thread alone touches the KV cache, mel buffer, decode cursor and
pending waveform. Completed results are bounded, so the frontend can never run
unboundedly ahead of the converted audio.
"""
import collections
import threading
import time

import numpy as np

from .meanvc2 import MeanVC2Backend

MAX_QUEUE_CHUNKS = 6


class MeanVC2ParallelBackend(MeanVC2Backend):
    """Producer/consumer backend with bounded backpressure on the frontend."""

    def __init__(self, threads=4, queue_chunks=1, device='cpu'):
        if type(queue_chunks) is not int or not 1 <= queue_chunks <= MAX_QUEUE_CHUNKS:
            raise ValueError(f'Parallel frontend queue must be 1..{MAX_QUEUE_CHUNKS} chunks')
        self.queue_chunks = queue_chunks
        self._condition = threading.Condition()
        self._work = collections.deque()
        self._ready = collections.deque()
        self._failure = None
        self._stop = threading.Event()
        self._thread = None
        self._producer_calls = 0
        self._consumer_waits = 0
        super().__init__(threads, device)

    def load(self, model_path):
        super().load(model_path)
        self.stats.update(parallel_frontend=True, parallel_queue_chunks=self.queue_chunks,
                          parallel_bounded_queue=True,
                          human_approved_offline=False,
                          streaming_quality_verified=False,
                          parallel_streaming_quality_verified=False)
        self.reset()

    def unload(self):
        self._stop_producer()
        super().unload()

    def reset(self):
        self._stop_producer()
        super().reset()
        with self._condition:
            self._work.clear()
            self._ready.clear()
            self._failure = None
        self._producer_calls = 0
        self._consumer_waits = 0
        if getattr(self, 'vc', None) is not None:
            self._start_producer()

    def get_stats(self):
        stats = super().get_stats()
        stats.update(parallel_frontend=True, parallel_queue_chunks=self.queue_chunks,
                     parallel_producer_calls=self._producer_calls,
                     parallel_consumer_waits=self._consumer_waits,
                     parallel_producer_alive=self._thread is not None)
        return stats

    # -- producer ------------------------------------------------------------
    def _start_producer(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._produce, name='meanvc2-frontend',
                                        daemon=True)
        self._thread.start()

    def _stop_producer(self):
        thread, self._thread = self._thread, None
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=30)
        self._stop.clear()

    def _produce(self):
        try:
            while True:
                with self._condition:
                    # Bound the producer: never hold more than queue_chunks
                    # finished-but-unconsumed results.
                    while (not self._stop.is_set()
                           and (not self._work or len(self._ready) >= self.queue_chunks)):
                        self._condition.wait(0.02)
                    if self._stop.is_set():
                        return
                    audio = self._work.popleft()
                begin = time.perf_counter()
                with self.torch.inference_mode():
                    fragments = self._frontend_fragments(audio)
                elapsed = (time.perf_counter() - begin) * 1000
                with self._condition:
                    self._ready.append((fragments, elapsed))
                    self._producer_calls += 1
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._failure = error
                self._condition.notify_all()

    # -- consumer ------------------------------------------------------------
    def process_chunk(self, audio):
        if self.vc is None:
            raise RuntimeError('MeanVC2 is not loaded')
        if audio.shape != (2560,) or audio.dtype != np.float32 or not np.isfinite(audio).all():
            raise ValueError('MeanVC2 expects 160ms finite float32 mono')
        begin = time.perf_counter()
        with self._condition:
            self._work.append(audio.copy())
            self._condition.notify_all()
        with self._condition:
            while not self._ready:
                if self._failure is not None:
                    failure, self._failure = self._failure, None
                    raise RuntimeError('MeanVC2 feature frontend failed') from failure
                self._consumer_waits += 1
                self._condition.wait(0.02)
            fragments, feature_ms = self._ready.popleft()
            self._condition.notify_all()
        waited = (time.perf_counter() - begin) * 1000
        t = self.torch
        with self.torch.inference_mode():
            for cond, noise in fragments:
                self.cond = t.cat((self.cond, cond), dim=1)
                self.noise = t.cat((self.noise, noise), dim=1)
            convert = time.perf_counter()
            self._convert_blocks()
            after_vc = time.perf_counter()
            self._decode_ready()
            finish = time.perf_counter()
        self.stats['last_stage_ms'] = dict(
            features=feature_ms,
            frontend_wait=max(0.0, waited - feature_ms),
            vc=(after_vc - convert) * 1000,
            vocoder=(finish - after_vc) * 1000)
        if len(self.pending_audio) < 2560:
            raise RuntimeError('MeanVC2 streaming output alignment exhausted')
        result = self.pending_audio[:2560].copy()
        self.pending_audio = self.pending_audio[2560:]
        if not np.isfinite(result).all():
            raise RuntimeError('Nonfinite MeanVC2 output')
        self.calls += 1
        return result