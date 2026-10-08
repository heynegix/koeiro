"""Audio-derived causal context. No phrase length prediction or text semantics."""
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics


class PhraseState(str, Enum):
    SILENCE = 'SILENCE'
    ONSET = 'ONSET'
    EARLY = 'EARLY'
    MIDDLE = 'MIDDLE'
    LATE = 'LATE'
    ENDING_CANDIDATE = 'ENDING_CANDIDATE'


@dataclass(frozen=True)
class PhraseContext:
    state: str = 'SILENCE'
    phrase_duration: float = 0.
    silence_duration: float = 0.
    last_pause: float = 0.
    last_phrase_duration: float = 0.
    short_pause_count: int = 0
    pitch_slope: float = 0.
    pitch_velocity: float = 0.
    pitch_acceleration: float = 0.
    pitch_variance: float = 0.
    energy_slope: float = 0.
    local_energy_mean: float = -120.
    energy_deviation: float = 0.
    ending_probability: float = 0.
    onset: bool = False
    long_reset: bool = False


@dataclass(frozen=True)
class ProsodyContext:
    audio: PhraseContext
    # Separate delayed provider; v0.9 uses local partial ASR, audio remains primary.
    text: dict = field(default_factory=dict)


def slope(points):
    if len(points) < 3:
        return 0.
    ts, ys = zip(*points)
    mt, my = statistics.mean(ts), statistics.mean(ys)
    denominator = sum((t-mt)**2 for t in ts)
    return sum((t-mt)*(y-my) for t,y in points)/denominator if denominator else 0.


class EndingProbability:
    def __init__(self):
        self.value = 0.

    def update(self, energy_slope, voiced, duration, quiet_time, pitch_slope, pause_count, dt):
        falling = min(1., max(0., -energy_slope/24.))
        decay = min(1., max(0., (1.-voiced)/.6))
        mature = min(1., max(0., (duration-.4)/.5))
        quiet = min(1., quiet_time/.18)
        terminal = min(1., abs(pitch_slope)/8.)
        # Stable energy and voicing cannot become an ending just by being long.
        evidence = max(falling, decay, quiet)
        raw = mature * evidence * min(1., .55*falling + .25*decay + .25*quiet
            + .15*terminal + .03*min(3,pause_count))
        self.value += (1-math.exp(-dt/.080))*(raw-self.value)
        return self.value


class PhraseStateTracker:
    def __init__(self):
        self.state = PhraseState.SILENCE
        self.time = self.onset_time = self.state_since = 0.
        self.silence = 1.5
        self.last_energy_time = self.last_voiced_time = -1.5
        self.last_pause = self.last_phrase = 0.
        self.short_pauses = self.transitions = self.ending_count = self.reset_count = 0
        self.pitch_history = deque(maxlen=32)
        self.energy_history = deque(maxlen=32)
        self.ending = EndingProbability()
        self.previous_velocity = 0.
        self._reset_done = True

    def update(self, pitch_st, voiced, energy_db, good, dt, history_ms=200., min_silence_ms=180.):
        self.time += dt
        onset = good and (self._reset_done or self.silence >= min_silence_ms/1000 and self.time-self.last_energy_time >= min_silence_ms/1000)
        new_phrase = good and (self._reset_done or self.silence >= .4)
        if good:
            if onset:
                self.last_pause = self.silence
                if self.silence < .4:
                    self.short_pauses += 1
                if new_phrase:
                    self.last_phrase = max(0., self.time-self.onset_time-self.silence)
                    self.onset_time = self.time
                    self.pitch_history.clear(); self.energy_history.clear()
                    self.ending.value = 0.
            self._reset_done = False
            self.silence = 0.
            self.last_voiced_time = self.time
            self.pitch_history.append((self.time,pitch_st))
        else:
            self.silence += dt
        if energy_db > -60:
            self.last_energy_time = self.time
        self.energy_history.append((self.time,energy_db))
        for history in (self.pitch_history,self.energy_history):
            while history and self.time-history[0][0] > history_ms/1000:
                history.popleft()
        long_reset = self.silence >= 1.2 and self.time-self.last_energy_time >= .4 and not self._reset_done
        if long_reset:
            self.last_phrase = max(0.,self.time-self.onset_time-self.silence)
            self.pitch_history.clear(); self.energy_history.clear()
            self.previous_velocity = self.ending.value = 0.
            self.short_pauses = 0
            self.reset_count += 1
            self._reset_done = True
        duration = 0. if self._reset_done else max(0., self.time-self.onset_time)
        ps = max(-36.,min(36.,slope(self.pitch_history)))
        es = max(-120.,min(120.,slope(self.energy_history)))
        acceleration = (ps-self.previous_velocity)/dt
        self.previous_velocity = ps
        mean_energy = statistics.mean(v for _,v in self.energy_history) if self.energy_history else energy_db
        probability = self.ending.update(es,voiced,duration,self.time-self.last_energy_time,
            ps,self.short_pauses,dt) if not self._reset_done else 0.
        if self._reset_done or self.silence >= .21 and self.time-self.last_energy_time >= .15:
            desired = PhraseState.SILENCE
        elif onset and new_phrase:
            desired = PhraseState.ONSET
        elif duration < .15:
            desired = PhraseState.ONSET
        elif duration < .5:
            desired = PhraseState.EARLY
        elif probability > (.55 if self.state == PhraseState.ENDING_CANDIDATE else .70):
            desired = PhraseState.ENDING_CANDIDATE
        elif probability > (.30 if self.state == PhraseState.LATE else .40) and duration > .65:
            desired = PhraseState.LATE
        else:
            desired = PhraseState.MIDDLE
        urgent = desired == PhraseState.SILENCE or onset and new_phrase
        if desired != self.state and (urgent or self.time-self.state_since >= .12):
            self.state = desired; self.state_since = self.time; self.transitions += 1
            if desired == PhraseState.ENDING_CANDIDATE:
                self.ending_count += 1
        variance = statistics.pvariance(v for _,v in self.pitch_history) if self.pitch_history else 0.
        return PhraseContext(self.state.value,duration,self.silence,self.last_pause,self.last_phrase,
            self.short_pauses,ps,ps,acceleration,variance,es,mean_energy,energy_db-mean_energy,
            probability,onset,long_reset)
