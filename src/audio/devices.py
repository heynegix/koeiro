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
        cables = [d for d in pool if "cable input" in d.name.lower()]
        if cables:
            return next((d for d in cables if "wasapi" in d.host_api.lower()), cables[0])
    wasapi = [d for d in pool if "wasapi" in d.host_api.lower()]
    pool = wasapi or pool
    return next((d for d in pool if getattr(d, f"default_{kind}")), pool[0] if pool else None)
