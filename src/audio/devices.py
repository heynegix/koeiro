from dataclasses import dataclass
import logging

import sounddevice as sd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_input: bool = False
    default_output: bool = False

    @property
    def label(self):
        return f"{self.name} [{self.host_api}]"

    @property
    def identity(self):
        # Persist names, not the indices which change after reboot/replugging.
        return {"name": self.name, "host_api": self.host_api}


# Name fragments of software loopback endpoints (lower-cased): VB-Cable on Windows,
# snd-aloop Loopback on Linux, Voicemeeter/BlackHole elsewhere. These never come from
# a microphone or a speaker; they are cables between applications. Matched against
# names only, so a host API merely named "pipewire" never misfires.
VIRTUAL_OUTPUT_HINTS = ("cable input", "loopback", "voicemeeter", "blackhole",
                        "soundflower", "null", "virtual")
VIRTUAL_INPUT_HINTS = ("cable output", "loopback", "monitor of", "voicemeeter",
                       "blackhole", "soundflower", "null", "virtual")


def is_virtual(device, kind="output"):
    """True when the device looks like a software cable, not hardware."""
    name = str(getattr(device, "name", "")).lower()
    hints = VIRTUAL_INPUT_HINTS if kind == "input" else VIRTUAL_OUTPUT_HINTS
    return any(hint in name for hint in hints)


def enumerate_devices(backend=sd):
    apis = backend.query_hostapis()
    result = []
    for index, device in enumerate(backend.query_devices()):
        api = apis[device["hostapi"]]
        result.append(AudioDevice(index, device["name"], api["name"],
                                  device["max_input_channels"], device["max_output_channels"],
                                  index == api["default_input_device"],
                                  index == api["default_output_device"]))
    log.info("Device list: %s", [device.label for device in result])
    return result


def rescan_devices(backend=sd):
    """Re-enumerate devices, picking up hardware connected after startup.

    PortAudio caches the device list at initialization, so a headset plugged
    in later (e.g. a Bluetooth hands-free profile) is invisible until the
    backend is re-initialized. sounddevice exposes this as the private
    _terminate/_initialize pair; backends without it (including test fakes)
    simply fall back to a plain enumeration. A failed re-initialization never
    breaks the refresh: it is logged and the cached list is returned.
    """
    terminate = getattr(backend, "_terminate", None)
    initialize = getattr(backend, "_initialize", None)
    if callable(terminate) and callable(initialize):
        try:
            terminate()
            initialize()
        except Exception:
            log.warning("Audio backend re-initialization failed; using cached device list")
    return enumerate_devices(backend)


def choose_device(devices, kind, saved=None, preferred_api=None):
    candidates = [d for d in devices if getattr(d, f"max_{kind}_channels") > 0]
    if saved:
        match = next((d for d in candidates if d.identity == saved), None)
        if match:
            return match
        log.warning("Saved %s device missing; selecting a safe fallback: %s", kind, saved)
    compatible = [d for d in candidates if d.host_api == preferred_api] if preferred_api else []
    pool = compatible or candidates
    if kind == "output":
        # Routing to a call app needs a software cable, not the speakers: prefer a
        # virtual endpoint (VB-Cable, snd-aloop Loopback, ...) over hardware.
        virtual = [d for d in pool if is_virtual(d, "output")]
        if virtual:
            preferred = [d for d in virtual if "wasapi" in d.host_api.lower()] or virtual
            return next((d for d in preferred if d.default_output), preferred[0])
    wasapi = [d for d in pool if "wasapi" in d.host_api.lower()]
    pool = wasapi or pool
    return next((d for d in pool if getattr(d, f"default_{kind}")), pool[0] if pool else None)
