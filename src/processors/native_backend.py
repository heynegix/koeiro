import ctypes
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


class NativeParameters(ctypes.Structure):
    _fields_ = [("pitch", ctypes.c_float), ("formant", ctypes.c_float),
                ("brightness", ctypes.c_float), ("wet", ctypes.c_float),
                ("enabled", ctypes.c_int32), ("low_cut", ctypes.c_int32), ("limiter", ctypes.c_int32)]


class NativeTiming(ctypes.Structure):
    _fields_ = [(field, ctypes.c_uint64) for field in ("count", "total_ns", "maximum_ns", "last_ns")]


def _library_candidates():
    """Platform-ordered native DSP filenames (first existing wins)."""
    # Windows x64 stays first for backward compatibility; Apple Silicon (ARM64)
    # is the tested macOS route, with an Intel-mac fallback kept second.
    if sys.platform == "win32":
        return ("female_dsp_x64.dll",)
    if sys.platform == "darwin":
        return ("female_dsp_arm64.dylib", "female_dsp_x64.dylib", "female_dsp_arm64.so")
    return ("female_dsp_x64.so", "female_dsp_arm64.so")


def _load_library():
    candidates = _library_candidates()
    missing = []
    for name in candidates:
        path = Path(__file__).parent / "native" / name
        if not path.is_file():
            missing.append(name)
            continue
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = manifest.get("dll_sha256") or manifest.get("lib_sha256") or manifest.get("sha256")
        if digest != expected:
            raise RuntimeError(f"DSP library checksum mismatch ({name}); rebuild with tools/build_native.py")
        break
    else:
        names = ", ".join(missing)
        raise RuntimeError(
            "Native Female DSP not found for this platform"
            f" (looked for {names});"
            " rebuild with tools/build_native.py")
    library = ctypes.CDLL(str(path.resolve()))
    library.avc_abi_version.restype = ctypes.c_int
    if library.avc_abi_version() != 1:
        raise RuntimeError("DSP DLL ABI mismatch")
    library.avc_create.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
    library.avc_create.restype = ctypes.c_void_p
    library.avc_destroy.argtypes = (ctypes.c_void_p,)
    library.avc_destroy.restype = None
    library.avc_reset.argtypes = (ctypes.c_void_p,)
    library.avc_reset.restype = None
    library.avc_latency.argtypes = (ctypes.c_void_p,)
    library.avc_latency.restype = ctypes.c_int
    library.avc_timing.argtypes = (ctypes.c_void_p, ctypes.POINTER(NativeTiming))
    library.avc_timing.restype = ctypes.c_int
    library.avc_process.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                   ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                   ctypes.POINTER(NativeParameters))
    library.avc_process.restype = ctypes.c_int
    return library


class NativeDSP:
    def __init__(self, sample_rate, max_frames, quality):
        # Loading/checksumming/create/warmup are strictly before stream.start().
        self.library = _load_library()
        self.handle = self.library.avc_create(sample_rate, max_frames, int(quality == "balanced"))
        if not self.handle:
            raise RuntimeError("Could not allocate Female DSP")
        self.input = np.empty((max_frames, 1), dtype=np.float32)
        self.output = np.empty((max_frames, 1), dtype=np.float32)
        self._input_pointer = self.input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._output_pointer = self.output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.parameters = NativeParameters()
        self._parameters_pointer = ctypes.pointer(self.parameters)
        self.latency_samples = self.library.avc_latency(self.handle)

    def process(self, audio, parameters):
        frames = len(audio)
        np.copyto(self.input[:frames], audio)
        p = self.parameters
        p.pitch, p.formant = parameters.pitch, parameters.formant
        p.brightness, p.wet = parameters.brightness, parameters.wet
        p.enabled = parameters.mode == "female_dsp"
        p.low_cut, p.limiter = parameters.low_cut, parameters.limiter
        result = self.library.avc_process(self.handle, self._input_pointer, self._output_pointer,
                                          frames, self._parameters_pointer)
        if result:
            raise RuntimeError(f"Native DSP failed (code {result})")
        np.copyto(audio, self.output[:frames])
        return audio

    def reset(self):
        self.library.avc_reset(self.handle)

    def timing_snapshot(self):
        result = NativeTiming()
        if self.library.avc_timing(self.handle, ctypes.byref(result)):
            raise RuntimeError("Could not read native timing")
        return dict(count=result.count, average_ms=result.total_ns/max(1, result.count)/1e6,
                    maximum_ms=result.maximum_ns/1e6, last_ms=result.last_ns/1e6)

    def close(self):
        if getattr(self, "handle", None):
            self.library.avc_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
