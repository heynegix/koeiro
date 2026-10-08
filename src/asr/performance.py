"""ASR-tail histogram up to four seconds; callback's existing stats unchanged."""
import math
import numpy as np
from src.audio.performance import TimingSnapshot


class ASRTimingStats:
    bin_ns=100000
    def __init__(self): self.reset()

    def reset(self):
        self.count=self.total_ns=self.maximum_ns=self.last_ns=self.deadline_exceeded=0
        self.histogram=np.zeros(40001,dtype=np.uint64)

    def record(self,elapsed_ns,frames,sample_rate):
        self.count+=1; self.total_ns+=elapsed_ns; self.last_ns=elapsed_ns
        self.maximum_ns=max(self.maximum_ns,elapsed_ns)
        self.histogram[min(40000,max(0,elapsed_ns//self.bin_ns))]+=1
        if elapsed_ns>frames*1e9/sample_rate: self.deadline_exceeded+=1

    def snapshot(self):
        cumulative=np.cumsum(self.histogram)
        def q(value):
            if not self.count: return 0.
            bucket=int(np.searchsorted(cumulative,math.ceil(self.count*value)))
            return self.maximum_ns/1e6 if bucket>=40000 else min((bucket+1)*.1,self.maximum_ns/1e6)
        return TimingSnapshot(self.count,self.total_ns/max(1,self.count)/1e6,self.maximum_ns/1e6,
            self.last_ns/1e6,self.deadline_exceeded,q(.95),q(.5),q(.99))
