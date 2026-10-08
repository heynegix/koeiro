"""Best-effort local monitoring of the converted output.

The main duplex callback offers each finished block to a bounded tap; a second,
output-only stream plays it on the monitor device (typically headphones while
the main output feeds a virtual cable). The two devices run on independent
clocks, so the tap absorbs drift in a ~2 s ring: overruns drop the oldest
audio and underruns play silence, both counted and reported. Monitoring is
therefore a listening aid, not a measurement path; use 試聴 for exact checks.
The player can be started, stopped and re-levelled while the main stream keeps
running, so the monitor never forces a Stop/Start cycle. Hearing your own output
on speakers will howl: use headphones.
"""
import logging
import time

import numpy as np

log = logging.getLogger(__name__)

# Kept inside the callback's budget: a single multiply plus a clamp on the block.
MAX_MONITOR_GAIN = 4.0

try:
    import sounddevice as sd
except Exception:  # Documentation/test environments without PortAudio.
    sd = None


class MonitorTap:
    """Single-producer/single-consumer ring between the two callbacks.

    The audio callback is the only writer, the monitor callback the only
    reader. Indices are plain ints mutated under the GIL; offer() never
    blocks, allocates, or raises on a full ring (oldest audio is dropped).
    """

    def __init__(self, seconds=2.0, rate=48000):
        capacity = max(4800, int(seconds * rate))
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.head = 0
        self.available = 0
        self.active = False
        self.offered = 0
        self.dropped = 0

    def reset(self):
        self.head = 0
        self.available = 0
        self.offered = 0
        self.dropped = 0

    def offer(self, audio):
        """Copy one finished mono block; drop oldest on overflow, never block."""
        if not self.active:
            return
        block = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        count = len(block)
        self.offered += count
        if count >= self.capacity:
            self.buffer[:] = block[-self.capacity:]
            self.dropped += self.available
            self.head = 0
            self.available = self.capacity
            return
        overflow = self.available + count - self.capacity
        if overflow > 0:
            self.dropped += overflow
            self.available = self.capacity
        else:
            self.available += count
        first = min(count, self.capacity - self.head)
        self.buffer[self.head:self.head + first] = block[:first]
        if first < count:
            rest = count - first
            self.buffer[:rest] = block[first:]
        self.head = (self.head + count) % self.capacity

    def read_into(self, destination):
        """Fill destination with the oldest audio; return frames actually read."""
        want = len(destination)
        take = min(want, self.available)
        if take <= 0:
            return 0
        tail = (self.head - self.available) % self.capacity
        first = min(take, self.capacity - tail)
        destination[:first] = self.buffer[tail:tail + first]
        if first < take:
            rest = take - first
            destination[first:] = self.buffer[:rest]
        self.available -= take
        return take


class MonitorPlayer:
    """Output-only stream that plays the tap on the monitor device."""

    def __init__(self, backend=None):
        self.backend = backend if backend is not None else sd
        self.tap = None
        self._stream = None
        self._scratch = None
        self.device_label = ""
        self.played = 0
        self.zeroed = 0
        self.gain = 1.0

    @property
    def running(self):
        return self._stream is not None

    def set_gain_db(self, db):
        """Set the monitor level. Only this float changes live; no restart."""
        try:
            value = float(db)
        except (TypeError, ValueError):
            return
        if not np.isfinite(value):
            return
        self.gain = min(MAX_MONITOR_GAIN, max(0.0, 10.0**(value/20.0)))

    def start(self, device, rate, blocksize, tap):
        """Open the monitor device and start playing. Raises AudioEngineError."""
        from .engine import AudioEngineError
        if self.backend is None:
            raise AudioEngineError("MonitorにはPortAudioが必要です。")
        try:
            info = self.backend.query_devices(device.index)
            channels = 2 if info["max_output_channels"] >= 2 else 1
            self._scratch = np.empty(blocksize, dtype=np.float32)
            self.tap = tap
            tap.active = True
            self._stream = self.backend.Stream(
                device=device.index, samplerate=rate, blocksize=blocksize,
                channels=channels, dtype="float32", latency="high",
                callback=self._callback,
                finished_callback=self._on_finished)
            self._stream.start()
        except Exception as error:
            self.stop()
            raise AudioEngineError(f"Monitorを開始できません。デバイスを確認してください。\n{error}") from error
        self.device_label = device.label
        log.info("Monitor start: %s", device.label)

    def _on_finished(self):
        pass

    def _callback(self, outdata, frames, timing, status):
        del timing
        if status is not None and getattr(status, "output_underflow", False):
            self.zeroed += frames
        scratch = self._scratch[:frames] if self._scratch is not None and len(self._scratch) >= frames else np.empty(frames, dtype=np.float32)
        got = self.tap.read_into(scratch) if self.tap is not None else 0
        if got < frames:
            scratch[got:frames] = 0.0
            self.zeroed += frames - got
        self.played += got
        gain = self.gain
        if gain != 1.0:
            # In-place, allocation free: the monitor is a listening aid, so a
            # bounded clamp is safer than letting a hot level fold over.
            np.multiply(scratch, gain, out=scratch)
            np.clip(scratch, -1.0, 1.0, out=scratch)
        flat = np.ascontiguousarray(scratch, dtype=np.float32)
        if outdata.shape[1] > 1:
            outdata[:, 0] = flat
            outdata[:, 1] = flat
        else:
            outdata[:, 0] = flat

    def stop(self):
        tap, stream = self.tap, self._stream
        self.tap = None
        self._stream = None
        self._scratch = None
        if tap is not None:
            tap.active = False
        if stream is not None:
            try:
                stream.abort(ignore_errors=True)
            except Exception:
                log.exception("Monitor abort failed")
            try:
                stream.close(ignore_errors=True)
            except Exception:
                log.exception("Monitor close failed")
        log.info("Monitor stop: played=%d zeroed=%d", self.played, self.zeroed)


def test_tone(rate=48000, seconds=1.2, gain=0.25):
    """A short two-note chime, used to confirm a monitor device by ear."""
    count = max(1, int(rate*float(seconds)))
    times = np.arange(count, dtype=np.float64)/rate
    tone = np.empty(count, dtype=np.float32)
    half = count//2
    tone[:half] = np.sin(2.0*np.pi*660.0*times[:half])
    tone[half:] = np.sin(2.0*np.pi*880.0*times[half:])
    # 20 ms fades so the chime cannot click at either edge.
    fade = np.minimum(1.0, np.minimum(times, float(seconds)-times)*50.0)
    return (tone*np.clip(fade, 0.0, 1.0)*float(gain)).astype(np.float32)


def play_test_tone(backend, device, rate=48000, seconds=1.2, gain_db=-12.0,
                   blocksize=256, timeout=None):
    """Play the chime once on ``device``. Blocking; call from a worker thread.

    The chime is pushed through the same bounded tap the live monitor uses, so a
    device that cannot play the chime is exactly a device the monitor cannot use.
    """
    from .engine import AudioEngineError
    if backend is None:
        raise AudioEngineError("MonitorにはPortAudioが必要です。")
    level = min(MAX_MONITOR_GAIN, max(0.0, 10.0**(float(gain_db)/20.0)))
    tone = test_tone(rate, seconds, 0.25*level)
    tap = MonitorTap(seconds=max(1.0, float(seconds)+0.5), rate=rate)
    tap.active = True
    player = MonitorPlayer(backend)
    try:
        player.start(device, rate, blocksize, tap)
    except Exception as error:
        raise AudioEngineError(
            f"Monitorテスト音を再生できません。デバイスを確認してください。\n{error}") from error
    drain = timeout if timeout is not None else float(seconds)+1.0
    started = time.monotonic()
    try:
        for start in range(0, len(tone), blocksize):
            tap.offer(tone[start:start+blocksize])
        # Wait for the device to consume the ring, bounded so a stalled device
        # can never hang the caller.
        while tap.available > 0 and time.monotonic()-started < drain:
            time.sleep(0.02)
        return player.device_label
    finally:
        player.stop()
