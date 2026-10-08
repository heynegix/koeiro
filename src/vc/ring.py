"""Bounded single-producer/single-consumer storage within GIL-enabled CPython.

Each endpoint alone owns its counter; publish after copying the samples. No
locks, blocking queue calls or allocations for sample storage in the callback.
Reset both counters only while the producer/consumer are stopped.
"""
import numpy as np
import time
from dataclasses import asdict
from src.audio.performance import TimingStats


class AudioRing:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Positive ring capacity required')
        self.data = np.empty(capacity, dtype=np.float32)
        self.capacity = capacity
        self.read_position = self.write_position = 0
        self.maximum_depth = 0
        self.timestamps = np.empty(capacity, dtype=np.int64)
        self.wait_time = TimingStats()
        self.depth_histogram = np.zeros(101, dtype=np.uint64)
        self.depth_count = self.depth_total = self.high_water_events = 0
        self._high = False

    @property
    def available(self):
        return self.write_position-self.read_position

    def write(self, audio):
        n = len(audio)
        if n > self.capacity-self.available:
            return False
        offset = self.write_position % self.capacity
        first = min(n, self.capacity-offset)
        np.copyto(self.data[offset:offset+first], audio[:first])
        stamp = time.perf_counter_ns()
        self.timestamps[offset:offset+first].fill(stamp)
        if n > first:
            np.copyto(self.data[:n-first], audio[first:])
            self.timestamps[:n-first].fill(stamp)
        self.write_position += n
        self.maximum_depth = max(self.maximum_depth, self.available)
        self.observe_depth()
        return True

    def read_into(self, audio):
        n = len(audio)
        if self.available < n:
            return False
        offset = self.read_position % self.capacity
        if n:
            self.wait_time.record(max(0,time.perf_counter_ns()-int(self.timestamps[offset])), n, 48000)
        first = min(n, self.capacity-offset)
        np.copyto(audio[:first], self.data[offset:offset+first])
        if n > first:
            np.copyto(audio[first:], self.data[:n-first])
        self.read_position += n
        self._high = self.available > self.capacity*.75
        return True

    def discard(self, count=None):
        """Consumer only: discard stale input or output without touching writer."""
        n = self.available if count is None else min(max(0, count), self.available)
        self.read_position += n
        self._high = self.available > self.capacity*.75
        return n

    def reset(self):
        self.read_position = self.write_position = self.maximum_depth = 0
        self.wait_time.reset()
        self.depth_histogram.fill(0)
        self.depth_count = self.depth_total = self.high_water_events = 0
        self._high = False

    def observe_depth(self):
        """Producer-side observations after each successful publication."""
        depth = self.available
        self.depth_count += 1
        self.depth_total += depth
        self.depth_histogram[min(100, max(0, round(depth*100/self.capacity)))] += 1
        high = depth > self.capacity*.75
        if high and not self._high:
            self.high_water_events += 1
        self._high = high

    def snapshot(self, chunk_frames):
        cumulative = np.cumsum(self.depth_histogram)
        bucket = min(100,int(np.searchsorted(cumulative,max(1,int(np.ceil(self.depth_count*.95))))))
        return dict(current=self.available/chunk_frames, maximum=self.maximum_depth/chunk_frames,
                    average=self.depth_total/max(1,self.depth_count)/chunk_frames,
                    p95=min(self.maximum_depth,bucket*self.capacity/100)/chunk_frames,
                    samples=self.depth_count, high_water_events=self.high_water_events,
                    wait=asdict(self.wait_time.snapshot()))
