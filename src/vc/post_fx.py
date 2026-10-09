"""Light worker-only EQ/high-pass and smoothed peak limiter; no pitch processing."""
import math
import numpy as np


class LightPostFX:
    def __init__(self, rate=48000):
        from scipy.signal import butter, sosfilt
        self.sosfilt = sosfilt
        self.hp = butter(2, 90, fs=rate, btype='highpass', output='sos')
        self.rate = rate
        self.reset()

    def reset(self):
        self.hp_state = np.zeros((1, 2))
        self.eq_state = np.zeros((1, 2))
        self.brightness = 50.0
        self.low_mix = 0.0
        self.gain = 1.0
        self.dynamic_gain_db = 0.0
        self._shelf_brightness = None
        self._shelf_sos = None

    def _shelf(self, brightness):
        # The coefficients depend only on the requested brightness, so cache
        # them instead of rebuilding the identical filter on every chunk.
        if self._shelf_brightness != brightness:
            # RBJ high shelf: bounded +/-4 dB, presence stays subtle.
            A = 10**(4*(brightness-50)/50/40)
            w = 2*math.pi*3800/self.rate
            c, beta = math.cos(w), math.sqrt(2*A)*math.sin(w)
            b = [A*((A+1)+(A-1)*c+beta), -2*A*((A-1)+(A+1)*c), A*((A+1)+(A-1)*c-beta)]
            a = [(A+1)-(A-1)*c+beta, 2*((A-1)-(A+1)*c), (A+1)-(A-1)*c-beta]
            self._shelf_sos = np.array([[*(x/a[0] for x in b), 1, a[1]/a[0], a[2]/a[0]]])
            self._shelf_brightness = brightness
        return self._shelf_sos

    def process(self, audio, brightness=50, low_cut=True, limiter=True, enabled=True, dynamic_gain_db=0.):
        audio = np.nan_to_num(audio, nan=0, posinf=0, neginf=0).astype(np.float32)
        if enabled:
            alpha = 1-math.exp(-len(audio)/(self.rate*.030))
            self.brightness += alpha*(brightness-self.brightness)
            self.low_mix += alpha*(float(low_cut)-self.low_mix)
            hp, self.hp_state = self.sosfilt(self.hp, audio, zi=self.hp_state)
            audio = audio+self.low_mix*(hp-audio)
            audio, self.eq_state = self.sosfilt(self._shelf(self.brightness), audio, zi=self.eq_state)
        if not math.isfinite(dynamic_gain_db):
            dynamic_gain_db=0.
        target_gain=max(-1.5,min(1.5,dynamic_gain_db))
        if target_gain or self.dynamic_gain_db:
            # Ramp gain before limiter, including when EQ is bypassed. Zero
            # control keeps the previous fixed-voice arithmetic unchanged.
            end=self.dynamic_gain_db+(1-math.exp(-len(audio)/(self.rate*.050)))*(target_gain-self.dynamic_gain_db)
            audio*=10**(np.linspace(self.dynamic_gain_db,end,len(audio))/20)
            self.dynamic_gain_db=end if abs(end)>1e-8 else 0.
        if limiter:
            peak = float(np.max(np.abs(audio), initial=0))
            target = min(1.0, .95/max(peak, 1e-12))
            if target < self.gain:
                self.gain = target
            else:
                self.gain += (1-math.exp(-len(audio)/(self.rate*.080)))*(target-self.gain)
            audio *= self.gain
        return np.clip(audio, -.95 if limiter else -1, .95 if limiter else 1).astype(np.float32)
