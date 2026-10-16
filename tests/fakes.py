from types import SimpleNamespace

import numpy as np

from src.audio.devices import AudioDevice

INPUT = AudioDevice(0, "Physical Microphone", "Windows WASAPI", 2, 0, True, False)
OUTPUT = AudioDevice(1, "CABLE Input (VB-Audio Virtual Cable)", "Windows WASAPI", 0, 2, False, True)


def pixel_colours(image):
    """Every distinct colour a rendered widget shows, lower-case hex."""
    return {image.pixelColor(x, y).name().lower()
            for y in range(image.height()) for x in range(image.width())}


def dominant_colour(image):
    """The colour a rendered widget is mostly filled with, ignoring its glyphs."""
    counts = {}
    for y in range(image.height()):
        for x in range(image.width()):
            name = image.pixelColor(x, y).name()
            counts[name] = counts.get(name, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def flags(**values):
    return SimpleNamespace(**dict.fromkeys(("input_underflow", "output_underflow",
                                           "input_overflow", "output_overflow"), False) | values)


class MonitorFakeStream:
    """Output-only stream double for the monitor player (channels as int)."""

    def __init__(self, backend, **kwargs):
        self.backend = backend
        self.kwargs = kwargs
        self.active = False
        self.closed = False
        channels = kwargs["channels"]
        self.outdata = np.full((kwargs["blocksize"], channels), np.nan, dtype=np.float32)

    def start(self):
        if self.backend.fail_start:
            raise RuntimeError("Device unavailable")
        self.active = True
        self.tick()

    def tick(self, status=None):
        self.kwargs["callback"](self.outdata, len(self.outdata), None, status or flags())

    def abort(self, **kwargs):
        self.active = False
        self.kwargs["finished_callback"]()

    def close(self, **kwargs):
        self.closed = True
        self.active = False


class FakeStream:
    def __init__(self, backend, **kwargs):
        self.backend = backend
        self.kwargs = kwargs
        self.active = False
        self.closed = False
        self.latency = (0.01, 0.01)
        self.indata = np.full((kwargs["blocksize"], kwargs["channels"][0]), 0.1, dtype=np.float32)
        self.outdata = np.full((kwargs["blocksize"], kwargs["channels"][1]), np.nan, dtype=np.float32)
    def start(self):
        if self.backend.fail_start:
            raise RuntimeError("Device unavailable")
        self.active = True
        self.tick()

    def tick(self, status=None):
        self.kwargs["callback"](self.indata, self.outdata, len(self.indata), None, status or flags())

    def abort(self, **kwargs):
        self.active = False
        self.kwargs["finished_callback"]()
        if self.backend.fail_abort:
            raise RuntimeError("Abort error")

    def close(self, **kwargs):
        if self.backend.fail_close:
            raise RuntimeError("Close error")
        self.closed = True
        self.active = False


class FakeBackend:
    class CallbackAbort(Exception):
        pass

    def __init__(self):
        self.streams = []
        self.fail_start = self.fail_abort = self.fail_close = False
        self.stereo_input_only = False
        self.mono_output_only = False
        self.removed = False

    def query_hostapis(self, index=None):
        api = dict(name="Windows WASAPI", default_input_device=0, default_output_device=1)
        return (api,) if index is None else api

    def query_devices(self, index=None):
        devices = [dict(name=INPUT.name, hostapi=0, max_input_channels=2, max_output_channels=0),
                   dict(name=OUTPUT.name, hostapi=0, max_input_channels=0, max_output_channels=2)]
        if self.removed:
            devices[0]["name"] = "Different Microphone"
        return devices if index is None else devices[index]

    def check_input_settings(self, **kwargs):
        if self.stereo_input_only and kwargs["channels"] == 1:
            raise ValueError("Only stereo supported")

    def check_output_settings(self, **kwargs):
        if self.mono_output_only and kwargs["channels"] == 2:
            raise ValueError("Only mono supported")

    def WasapiSettings(self, **kwargs):
        return kwargs

    def Stream(self, **kwargs):
        channels = kwargs["channels"]
        if isinstance(channels, int):
            # sounddevice hands a bare int to BOTH directions, so a duplex Stream
            # on an output-only device is refused by PortAudio (-9998) however
            # many speakers it has. Anything that plays to one has to ask for an
            # output stream instead.
            device = self.query_devices(kwargs["device"])
            if device["max_input_channels"] < channels:
                raise ValueError("Invalid number of channels [PaErrorCode -9998]")
            kwargs = dict(kwargs, channels=(channels, channels))
        assert not any(stream.active for stream in self.streams), "Duplicate stream"
        stream = FakeStream(self, **kwargs)
        self.streams.append(stream)
        return stream

    def OutputStream(self, **kwargs):
        assert not any(isinstance(stream, MonitorFakeStream) and stream.active
                       for stream in self.streams), "Duplicate monitor stream"
        stream = MonitorFakeStream(self, **kwargs)
        self.streams.append(stream)
        return stream
