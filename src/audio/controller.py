from dataclasses import dataclass
import logging
from queue import Empty, SimpleQueue
from threading import Thread

from .devices import rescan_devices
from .engine import AudioEngine
from src.utils.windows import com_apartment

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineSnapshot:
    state: str = "Stopped"
    error: str = ""
    underflows: int = 0
    overflows: int = 0
    reported_latency_ms: float | None = None
    request_id: int = 0


class AudioController:
    """Serialize all device/stream calls away from Qt and the audio callback."""

    def __init__(self, engine=None):
        self.engine = engine or AudioEngine()
        self.snapshot = EngineSnapshot()
        self.devices = ()
        self.devices_revision = 0
        self._commands = SimpleQueue()
        self._counter = 0
        self._request_id = 0
        self._thread = Thread(target=self._run, name="audio-control", daemon=True)
        self._thread.start()
        self.refresh_devices()

    def refresh_devices(self):
        return self._submit("refresh")

    def start(self, config):
        return self._submit("start", config)

    def stop(self):
        return self._submit("stop")

    def configure_ai(self, parameters):
        return self._submit('configure-ai', parameters)

    def configure_monitor(self, enabled, device=None, gain_db=0.0):
        """Apply monitor changes live; works with the stream stopped or running."""
        return self._submit('monitor', (bool(enabled), device, float(gain_db)))

    def restart_ai(self):
        return self._submit('restart-ai')

    def shutdown(self, finalize=None):
        return self._submit("shutdown", finalize)

    def _submit(self, command, config=None):
        self._counter += 1
        self._commands.put((command, config, self._counter))
        return self._counter

    @property
    def alive(self):
        return self._thread.is_alive()

    def _publish(self, state, error=""):
        self.snapshot = EngineSnapshot(state, error, self.engine.underflows,
                                       self.engine.overflows, self.engine.reported_latency_ms,
                                       self._request_id)

    def _run(self):
        try:
            with com_apartment():
                self._run_commands()
        except Exception as error:
            log.exception("Audio control thread failed")
            self._publish("Restart required", str(error))

    def _run_commands(self):
        while True:
            try:
                command, config, self._request_id = self._commands.get(timeout=0.05)
            except Empty:
                command, config = None, None
            try:
                if command == "shutdown":
                    self.engine.stop()
                    if config is not None:
                        config()
                    self._publish("Stopped")
                    return
                if command == "refresh" and not self.engine.running:
                    self._publish("Loading devices")
                    self.devices = tuple(rescan_devices(self.engine.backend))
                    self.devices_revision += 1
                    log.info('Device refresh / reconnect: %d devices',len(self.devices))
                    self._publish("Stopped")
                elif command == "start":
                    self._publish("Starting")
                    self.engine.start(config)
                    self._publish("Running")
                elif command == "stop":
                    self._publish("Stopping")
                    self.engine.stop()
                    self._publish("Stopped")
                elif command == 'monitor':
                    enabled, device, gain_db = config
                    self.engine.set_monitor_gain(gain_db)
                    if self.engine.running:
                        if enabled and device is not None:
                            self.engine.start_monitor(device)
                        else:
                            self.engine.stop_monitor()
                        self._publish('Running')
                    else:
                        self._publish('Stopped')
                elif command == 'configure-ai':
                    if self.engine.running:
                        raise ValueError('Stop before changing AI quality/CPU threads')
                    self._publish('Starting')
                    self.engine.chain.main_processor.configure_ai(config)
                    self._publish('Stopped')
                elif command == 'restart-ai':
                    self.engine.chain.main_processor.restart_ai()
                    self._publish('Running' if self.engine.running else 'Stopped')
                if self.engine.running:
                    self.engine.poll()
                    self._publish("Running")
            except Exception as error:
                log.exception("Audio control error")
                self._publish("Error", str(error))
                if command == "shutdown":
                    return
                # If closing failed, do not automatically reopen or spin/log.
                if self.engine.running:
                    try:
                        self.engine.stop()
                    except Exception:
                        self._publish("Restart required", str(error))
                        while True:
                            blocked_command, _, _ = self._commands.get()
                            if blocked_command == "shutdown":
                                return

    def set_gain(self, db):
        self.engine.chain.gain.set_gain(db)

    def set_gate(self, db):
        self.engine.chain.gate.set_threshold(db)
