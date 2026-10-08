from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TimingSnapshot:
    count: int
    average_ms: float
    maximum_ms: float
    last_ms: float
    deadline_exceeded: int
    p95_ms: float = 0.0
    p50_ms: float = 0.0
    p99_ms: float = 0.0


class TimingStats:
    """Scalar callback counters; snapshots and formatting happen off callback."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = self.total_ns = self.maximum_ns = self.last_ns = self.deadline_exceeded = 0
        # Preserve callback precision; a second range covers slow model/RPC
        # timings without mapping every 160ms inference to the maximum.
        self._histogram = np.zeros(14901, dtype=np.uint64)

    def record(self, elapsed_ns, frames, sample_rate):
        self.count += 1
        self.total_ns += elapsed_ns
        self.last_ns = elapsed_ns
        self.maximum_ns = max(self.maximum_ns, elapsed_ns)
        bucket = elapsed_ns // 10000 if elapsed_ns < 100000000 else 10000+(elapsed_ns-100000000)//1000000
        self._histogram[min(14900, max(0, bucket))] += 1
        if elapsed_ns > frames * 1e9 / sample_rate:
            self.deadline_exceeded += 1

    def snapshot(self):
        count = self.count
        cumulative = np.cumsum(self._histogram)
        def percentile(q):
            bucket = int(np.searchsorted(cumulative, max(1, int(np.ceil(count * q)))))
            upper = (bucket+1)/100 if bucket < 10000 else 101+(bucket-10000)
            return self.maximum_ns / 1e6 if bucket >= 14900 else min(upper, self.maximum_ns/1e6)
        return TimingSnapshot(count, self.total_ns / max(1, count) / 1e6,
                              self.maximum_ns / 1e6, self.last_ns / 1e6, self.deadline_exceeded,
                              percentile(.95), percentile(.5), percentile(.99))
