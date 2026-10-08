"""Bounded callback observations; formatting/logging stays on control threads."""
import gc
import time

import numpy as np


class AudioDiagnostics:
    def __init__(self):
        self._gc_observer = self._observe_gc
        self.attached = False
        self.reset()

    def reset(self):
        self.late_arrivals = self.maximum_gap_ms = 0
        self.gc_count = self.gc_maximum_ms = self.gc_last_stop_ns = self._gc_start = 0
        self.gui_last_action_ns = 0
        self._last_entry = self._last_host_time = 0
        self._events = np.zeros((128, 7), dtype=np.float64)
        self._event_count = 0

    def attach(self):
        if not self.attached:
            gc.callbacks.append(self._gc_observer)
            self.attached = True

    def detach(self):
        if self.attached:
            gc.callbacks.remove(self._gc_observer)
            self.attached = False

    def _observe_gc(self, phase, info):
        now = time.perf_counter_ns()
        if phase == "start":
            self._gc_start = now
        elif self._gc_start:
            self.gc_count += 1
            self.gc_maximum_ms = max(self.gc_maximum_ms, (now-self._gc_start)/1e6)
            self.gc_last_stop_ns = now
            self._gc_start = 0

    def note_gui_action(self):
        self.gui_last_action_ns = time.perf_counter_ns()

    def entered(self, now, timing, deadline_ms):
        gap = (now-self._last_entry)/1e6 if self._last_entry else 0
        host = getattr(timing, "currentTime", 0.0) if timing is not None else 0.0
        host_gap = (host-self._last_host_time)*1000 if self._last_host_time and host else 0
        self._last_entry, self._last_host_time = now, host
        self.maximum_gap_ms = max(self.maximum_gap_ms, gap)
        # Shared WASAPI often calls fixed-size callbacks in 10 ms bursts.
        # A 1-block gap is therefore not itself evidence of a glitch.
        late = gap > max(20, 4*deadline_ms)
        self.late_arrivals += int(late)
        return gap, host_gap, late

    def completed(self, now, gap, host_gap, late, elapsed_ms, deadline_ms, gate):
        if late or elapsed_ms > deadline_ms:
            row = self._events[self._event_count % len(self._events)]
            row[0], row[1], row[2], row[3] = now/1e9, gap, host_gap, elapsed_ms
            row[4] = (now-self.gui_last_action_ns)/1e6 if self.gui_last_action_ns else -1
            row[5] = (now-self.gc_last_stop_ns)/1e6 if self.gc_last_stop_ns else -1
            row[6] = gate._level
            self._event_count += 1

    def snapshot(self):
        count = min(len(self._events), self._event_count)
        start = (self._event_count-count) % len(self._events)
        events = [self._events[(start+i) % len(self._events)].tolist() for i in range(count)]
        return dict(late_arrivals=self.late_arrivals, maximum_callback_gap_ms=self.maximum_gap_ms,
                    gc_count=self.gc_count, gc_maximum_ms=self.gc_maximum_ms,
                    event_count=self._event_count, events=events,
                    event_columns=["monotonic_seconds", "entry_gap_ms", "host_gap_ms", "callback_ms",
                                   "gui_action_age_ms", "gc_stop_age_ms", "gate_level"])
