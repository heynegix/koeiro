from dataclasses import dataclass
import logging
import math
import sys
import time

import numpy as np
import sounddevice as sd

from src.processors.chain import ProcessorChain
from src.settings.manager import BUFFERS, SAMPLE_RATES
from .devices import AudioDevice
from .meters import peak_level
from .monitor import MonitorPlayer, MonitorTap
from .performance import TimingStats
from .diagnostics import AudioDiagnostics

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    input_device: AudioDevice
    output_device: AudioDevice
    sample_rate: int = 48000
    buffer_size: int = 256
    monitor: bool = False
    monitor_device: AudioDevice | None = None
    monitor_gain_db: float = 0.0

    @property
    def buffer_latency_ms(self):
        return 1000 * self.buffer_size / self.sample_rate


class AudioEngineError(RuntimeError):
    pass


class AudioEngine:
    """Single duplex stream, controlled exclusively by one control thread.

    Callback fields are published without locks on standard GIL-enabled CPython
    3.11/3.12. The GUI observes approximate meter/counter snapshots only. Do not
    call start/stop/poll from multiple threads; the controller serializes them.
    """

    def __init__(self, chain=None, backend=sd):
        self.chain = chain or ProcessorChain()
        self.backend = backend
        self.monitor_tap = MonitorTap()
        self.monitor_player = None
        self.monitor_error = ""
        self.monitor_gain_db = 0.0
        self._stream = None
        self.config = None
        self.input_peak = self.output_peak = 0.0
        self.underflows = self.overflows = 0
        self._reported_counts = (0, 0)
        self._callback_error = None
        self._finished = False
        self._last_callback = 0.0
        self.reported_latency_ms = None
        self.input_channels = self.output_channels = 1
        self.performance = TimingStats()
        self.cpu_load = 0.0
        self.diagnostics = AudioDiagnostics()

    @property
    def running(self):
        return self._stream is not None

    def _check_device(self, device, direction):
        current = self.backend.query_devices(device.index)
        api = self.backend.query_hostapis(current["hostapi"])
        if (current["name"] != device.name or api["name"] != device.host_api
                or current[f"max_{direction}_channels"] < 1):
            raise AudioEngineError("Audio device disconnected / デバイスを再選択してください")

    def _supported_channels(self, device, direction, rate, extra):
        maximum = getattr(device, f"max_{direction}_channels")
        choices = (1, 2) if direction == "input" else (2, 1)
        check = getattr(self.backend, f"check_{direction}_settings")
        last_error = None
        for channels in choices:
            if channels > maximum:
                continue
            try:
                check(device=device.index, channels=channels, dtype="float32",
                      samplerate=rate, extra_settings=extra)
                return channels
            except Exception as error:
                last_error = error
        raise AudioEngineError(
            f"{direction}の{rate} Hz形式が利用できません。"
            + ("Windows側も同じHzに設定してください。" if sys.platform == "win32"
               else "OS側の音声設定も同じHzにしてください。")
            + f"\n{last_error}") from last_error

    def start(self, config: EngineConfig):
        if self.running:
            raise AudioEngineError("Audio stream is already running")
        if config.sample_rate not in SAMPLE_RATES or config.buffer_size not in BUFFERS:
            raise AudioEngineError("Unsupported sample rate or buffer")
        log.info("Selected devices: %s -> %s; sample rate=%d buffer=%d",
                 config.input_device.label, config.output_device.label,
                 config.sample_rate, config.buffer_size)
        if config.input_device.host_api != config.output_device.host_api:
            recommended = "両方 [Windows WASAPI]" if sys.platform == "win32" else "入出力で同じ方式"
            raise AudioEngineError(f"InputとOutputは同じ方式を選んでください（{recommended}を推奨）。")
        self._check_device(config.input_device, "input")
        self._check_device(config.output_device, "output")
        extra = None
        if "wasapi" in config.input_device.host_api.lower():
            extra = self.backend.WasapiSettings(exclusive=False, auto_convert=True)
        input_channels = self._supported_channels(config.input_device, "input", config.sample_rate, extra)
        output_channels = self._supported_channels(config.output_device, "output", config.sample_rate, extra)
        self.chain.prepare(config.sample_rate, config.buffer_size)
        self._work = np.empty((config.buffer_size, 1), dtype=np.float32)
        self._finite = np.empty((config.buffer_size, 1), dtype=bool)
        self.config = config
        self.input_channels, self.output_channels = input_channels, output_channels
        self.input_peak = self.output_peak = 0.0
        self.underflows = self.overflows = 0
        self._reported_counts = (0, 0)
        self._callback_error = None
        self._finished = False
        self._last_callback = time.monotonic()
        self._last_log_time = self._last_callback
        self.reported_latency_ms = None
        self.performance.reset()
        self.diagnostics.reset()
        self.diagnostics.attach()
        self.cpu_load = 0.0
        try:
            self._stream = self.backend.Stream(
                device=(config.input_device.index, config.output_device.index),
                samplerate=config.sample_rate, blocksize=config.buffer_size,
                channels=(input_channels, output_channels), dtype="float32", latency="low",
                extra_settings=(extra, extra), callback=self._callback,
                finished_callback=self._on_finished)
            self._stream.start()
            latency = self._stream.latency
            self.reported_input_latency_ms, self.reported_output_latency_ms = (1000*v for v in latency)
            self.reported_latency_ms = 1000 * sum(latency)
            log.info("Stream start: %s -> %s; sample rate=%d buffer=%d channels=%d/%d "
                     "PortAudio estimated I/O latency=%.2f ms",
                     config.input_device.label, config.output_device.label, config.sample_rate,
                     config.buffer_size, input_channels, output_channels, self.reported_latency_ms)
            self.monitor_error = ""
            self.monitor_tap.reset()
            self.set_monitor_gain(config.monitor_gain_db)
            if config.monitor and config.monitor_device is not None:
                self.start_monitor(config.monitor_device)
        except Exception as error:
            self.stop()
            raise AudioEngineError(f"開始できません。デバイス・マイク権限・形式を確認してください。\n{error}") from error

    def start_monitor(self, device):
        """Start (or restart) the best-effort monitor on a running engine.

        Monitoring must never break the main stream, so a failure is recorded in
        ``monitor_error`` and reported, not raised. The caller can change device or
        level while the duplex stream keeps running.
        """
        self.stop_monitor()
        if not self.running or self.config is None or device is None:
            return False
        try:
            player = MonitorPlayer(self.backend)
            player.start(device, self.config.sample_rate, self.config.buffer_size,
                         self.monitor_tap)
            player.set_gain_db(self.monitor_gain_db)
        except Exception as error:
            self.monitor_player = None
            self.monitor_error = str(error)
            log.warning("Monitor disabled: %s", error)
            return False
        self.monitor_player = player
        self.monitor_error = ""
        return True

    def stop_monitor(self):
        player, self.monitor_player = self.monitor_player, None
        if player is not None:
            player.stop()

    def set_monitor_gain(self, db):
        """Level only: no stream is touched, so this is safe while running."""
        try:
            value = float(db)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self.monitor_gain_db = min(12.0, max(-60.0, value))
        if self.monitor_player is not None:
            self.monitor_player.set_gain_db(self.monitor_gain_db)

    def _on_finished(self):
        self._finished = True

    def _callback(self, indata, outdata, frames, timing, status):
        started = time.perf_counter_ns()
        deadline = frames / self.config.sample_rate * 1000
        gap, host_gap, late = self.diagnostics.entered(started, timing, deadline)
        # PortAudio output is uninitialized. Silence first, including all errors.
        outdata.fill(0)
        try:
            if status.input_underflow or status.output_underflow:
                self.underflows += 1
            if status.input_overflow or status.output_overflow:
                self.overflows += 1
            if frames < 1 or frames > len(self._work):
                raise ValueError("Unexpected callback frame count")
            audio = self._work[:frames]
            if self.input_channels == 1:
                np.copyto(audio, indata)
            else:
                # Stereo capture fallback; DSP and main slot remain mono.
                np.add(indata[:, :1], indata[:, 1:2], out=audio)
                np.multiply(audio, 0.5, out=audio)
            mask = self._finite[:frames]
            np.isfinite(audio, out=mask)
            np.logical_not(mask, out=mask)
            np.copyto(audio, 0.0, where=mask)
            self.input_peak = peak_level(audio)
            audio = self.chain.process(audio, self.config.sample_rate)
            self.output_peak = peak_level(audio)
            if self.monitor_player is not None and self.monitor_player.running:
                self.monitor_tap.offer(audio[:, 0])
            # Broadcast mono to both channels for stereo-only render endpoints.
            np.copyto(outdata, audio)
            self._last_callback = time.monotonic()
        except Exception as error:
            outdata.fill(0)
            self._callback_error = error  # Format/log this on the control thread.
            raise self.backend.CallbackAbort from None
        finally:
            finished = time.perf_counter_ns()
            self.performance.record(finished - started, frames, self.config.sample_rate)
            self.diagnostics.completed(finished, gap, host_gap, late, (finished-started)/1e6,
                                       deadline, self.chain.gate)

    def poll(self):
        """Control-thread watchdog and rate-limited xrun logging."""
        if not self.running:
            return
        now = time.monotonic()
        if now - self._last_log_time >= 1.0:
            counts = self.underflows, self.overflows
            if counts != self._reported_counts:
                log.warning("Underflow=%d Overflow=%d (cumulative); increase Buffer if recurring", *counts)
                self._reported_counts = counts
            self._last_log_time = now
        try:
            self.cpu_load = float(getattr(self._stream, "cpu_load", 0.0))
            if self._callback_error is not None:
                error = self._callback_error
                raise AudioEngineError(f"音声処理エラー: {error}") from error
            if self._finished or not self._stream.active or now - self._last_callback > 2.0:
                raise AudioEngineError("Audio device disconnected / 音声デバイスが停止しました。接続を確認してください。")
        except Exception:
            log.exception("Device error / audio engine stopped")
            self.stop()
            raise

    def stop(self):
        self.diagnostics.detach()
        self.stop_monitor()
        stream = self._stream
        if stream is None:
            self.chain.stop()
            return
        # Keep the reference until close returns, preventing a second stream.
        try:
            stream.abort(ignore_errors=False)
        except Exception:
            log.exception("Device error during stream abort")
        try:
            stream.close(ignore_errors=False)
        except Exception as error:
            log.exception("Device error during stream close")
            # Keep the uncertain stream reference, but release any isolated AI
            # worker. Its callback route falls back to Original once stopped.
            try:
                self.chain.stop()
            except Exception:
                log.exception("Processor stop failed after device close error")
            # Retain the reference to prevent reopening an uncertain device.
            raise AudioEngineError("音声ストリームを閉じられません。アプリを再起動してください。") from error
        self._stream = None
        self.chain.stop()
        self.input_peak = self.output_peak = 0.0
        log.info("Stream stop: Underflow=%d Overflow=%d", self.underflows, self.overflows)
        log.info("Callback timing: %s; Main timing: %s; diagnostics: %s",
                 self.performance.snapshot(), self.chain.performance.snapshot(), self.diagnostics.snapshot())

    def take_peaks(self):
        # Observational snapshot only; audio samples never travel to the GUI.
        return self.input_peak, self.output_peak
