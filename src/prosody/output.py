"""Continuous controls and Beatrice's 1/8 semitone automation grid."""
import math


class OutputControl:
    def __init__(self):
        self.pitch = self.gain = self.quantized = 0.
        self.pending = None
        self.pending_time = 0.
        self.changes = self.holds = 0

    def update(self, target, gain, dt, p):
        tau = (p.attack_ms if abs(target)>abs(self.pitch) else p.release_ms)/1000
        step = (1-math.exp(-dt/tau))*(target-self.pitch)
        self.pitch += max(-4*dt,min(4*dt,step))
        self.gain += (1-math.exp(-dt/.12))*(gain-self.gain)
        low,high = math.ceil(p.min_pitch*8)/8, math.floor(p.max_pitch*8)/8
        candidate = max(low,min(high,round(self.pitch*8)/8))
        if candidate != self.quantized:
            # A .025-st deadband beyond the midpoint plus 60 ms dwell.
            distance = abs(self.pitch-self.quantized)
            direction=1 if candidate>self.quantized else -1
            if direction != self.pending:
                self.pending = direction; self.pending_time = 0.
            self.pending_time += dt
            if distance >= .0875 and self.pending_time >= .06:
                self.quantized += direction*.125; self.changes += 1
            elif candidate == 0 and abs(self.pitch)<.025:
                self.quantized = 0.; self.changes += 1
            else:
                self.holds += 1
        else:
            self.pending = None; self.pending_time = 0.
        # Clamp changes during preset morphing also remain slew limited; only
        # the native output must lie on the legal grid immediately.
        self.quantized = max(low,min(high,self.quantized))
        return self.quantized, self.gain
