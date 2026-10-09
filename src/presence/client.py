"""Background presence worker: connect, throttle, reconnect, publish.

Discord's RPC server is a local endpoint that may not be running yet, and it caps how
often a client may change its activity (one update per 15 seconds, the limit the
official Rich Presence SDK documents) and how often it may reconnect (2 connections
per minute), so presence cannot simply write on every GUI frame.

One daemon worker thread owns the whole connection. It asks the transport whether a
frame is waiting before reading one -- never reading blindly -- because a read
already blocked inside a synchronous pipe cannot be cancelled on Windows. While
connected it publishes the *latest* snapshot rather than a queue of stale ones,
answers Discord's PING, reports an activity Discord rejected, and waits out its
reconnect delay before trying again.

Nothing here touches audio: the GUI thread only assigns a frozen snapshot and sets an
event, so a closed or slow Discord can never delay a callback.
"""
import logging
import os
import threading
import time

from .activity import APPLICATION_ID, PresenceState, build_activity
from .ipc import (HANDSHAKE_TIMEOUT, OP_CLOSE, DiscordIPC, PresenceProtocolError,
                  PresenceTimeout, PresenceUnavailable, open_ipc)

log = logging.getLogger(__name__)

STATUS_OFF = 'off'
STATUS_SEARCHING = 'searching'
STATUS_CONNECTED = 'connected'
STATUS_ERROR = 'error'

# One activity update per 15 seconds is the rate limit Discord's own Rich Presence
# SDK documents. Sends are coalesced below this floor, so the newest snapshot wins.
MIN_UPDATE_INTERVAL = 15.0
# Discord's RPC server allows 2 connections per minute for the same client.
RECONNECT_DELAY = 30.0
# How often the worker looks for an inbound frame (PING or a close).
POLL_INTERVAL = 0.1


class PresenceClient:
    """Publishes a :class:`PresenceState` to Discord while it stays possible."""

    def __init__(self, app_id=APPLICATION_ID, connector=None,
                 min_update_interval=MIN_UPDATE_INTERVAL,
                 reconnect_delay=RECONNECT_DELAY, poll_interval=POLL_INTERVAL,
                 handshake_timeout=HANDSHAKE_TIMEOUT, clock=time.monotonic):
        self.app_id = str(app_id)
        self.min_update_interval = float(min_update_interval)
        self.reconnect_delay = float(reconnect_delay)
        self.poll_interval = float(poll_interval)
        self.handshake_timeout = float(handshake_timeout)
        self._clock = clock
        self._connector = connector or open_ipc
        self._state = PresenceState()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread = None
        self._link = None
        self._status = STATUS_OFF
        self._error = ''
        self._sends = 0

    # -- public API (GUI thread) ------------------------------------------------

    def start(self):
        """Idempotent: connect and keep the activity current until stop()."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            self._stopping.clear()
            self._status = STATUS_SEARCHING
            self._thread = threading.Thread(target=self._run, name='discord-presence', daemon=True)
            thread = self._thread
        thread.start()
        return thread

    def stop(self, timeout=2.0):
        """Stop publishing and release the connection. Safe from the GUI thread.

        The worker does not read without data waiting, so it reaches its stop check
        within one poll interval; the join is only a ceiling, not the usual wait.
        """
        self._stopping.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            log.warning('Discord presence worker did not exit within %.1f s', timeout)
        self._status = STATUS_OFF
        self._error = ''
        return not (thread is not None and thread.is_alive())

    def update(self, state: PresenceState):
        """Offer the newest snapshot. Returns True when it differs from the last."""
        with self._lock:
            if state == self._state:
                return False
            self._state = state
        self._wake.set()
        return True

    @property
    def status(self):
        return self._status

    @property
    def error(self):
        return self._error

    @property
    def sends(self):
        """Activity updates delivered, for the settings label."""
        return self._sends

    def label(self):
        """One line for the settings page, in the reader's words."""
        if self._status == STATUS_CONNECTED:
            return ('接続中: Discordのアクティビティに表示されています'
                    + (f'（Discordが拒否: {self._error}）' if self._error else ''))
        if self._status == STATUS_SEARCHING:
            return 'Discordを待っています（起動すると自動でつながります）'
        if self._status == STATUS_ERROR:
            return f'接続できません：{self._error}' if self._error else '接続できません'
        return 'オフ'

    # -- worker ----------------------------------------------------------------

    def _run(self):
        while not self._stopping.is_set():
            if not self._connect():
                self._pause(self.reconnect_delay)
                continue
            try:
                self._pump()
            except (OSError, PresenceProtocolError) as error:
                log.info('Discord presence disconnected: %s', error)
                if not self._stopping.is_set():
                    self._set_status(STATUS_ERROR, str(error))
            finally:
                self._release()
            if not self._stopping.is_set():
                self._pause(self.reconnect_delay)

    def _connect(self):
        """Connect and hand shake. False means "try again later", never fatal."""
        link = None
        try:
            link = self._connector()
            link.handshake(self.app_id, timeout=self.handshake_timeout,
                           cancel=self._stopping.is_set)
        except PresenceUnavailable as error:
            self._set_status(STATUS_SEARCHING, str(error))
            return False
        except PresenceTimeout as error:
            # Discord delays connections over its rate limit instead of refusing
            # them, so silence right after connecting is "not yet", not "broken".
            log.info('Discord presence not answering yet: %s', error)
            self._set_status(STATUS_SEARCHING, str(error))
            if link is not None:
                link.transport.close()
            return False
        except (OSError, PresenceProtocolError) as error:
            log.info('Discord presence unavailable: %s', error)
            self._set_status(STATUS_ERROR, str(error))
            if link is not None:
                link.transport.close()
            return False
        self._link = link
        self._set_status(STATUS_CONNECTED)
        log.info('Discord presence connected; app id %s', self.app_id)
        return True

    def _pump(self):
        """Publish the newest snapshot, at most once per rate-limit window."""
        wake = self._wake
        wake.clear()
        last_sent, last_at = None, None
        while not self._stopping.is_set():
            self._poll_inbound()
            desired = self._snapshot()
            if desired != last_sent:
                elapsed = None if last_at is None else self._clock() - last_at
                if elapsed is None or elapsed >= self.min_update_interval:
                    self._link.set_activity(build_activity(desired), pid=os.getpid())
                    last_sent, last_at = desired, self._clock()
                    self._sends += 1
                    continue
            # Nothing to do: sleep a poll interval, or less if a new snapshot
            # arrives (update() sets the same event).
            wake.wait(timeout=self.poll_interval)
            wake.clear()

    def _poll_inbound(self):
        """Answer PING and notice a close without ever waiting for the peer."""
        frame = self._link.poll_frame()
        if frame is None:
            return
        opcode, payload = frame
        if opcode == OP_CLOSE:
            raise PresenceProtocolError(f'Discord closed the connection: {payload}')
        if isinstance(payload, dict) and payload.get('evt') == 'ERROR':
            # Discord accepted the connection but refused the command: keep the
            # connection, surface the reason where the settings page can show it.
            data = payload.get('data') or {}
            self._error = f"{data.get('code', '?')} {data.get('message', '')}".strip()
            log.info('Discord rejected a presence command: %s', self._error)

    def _release(self):
        link, self._link = self._link, None
        if link is not None:
            link.close()

    def _pause(self, seconds):
        self._stopping.wait(timeout=seconds)
        self._wake.clear()

    def _snapshot(self):
        with self._lock:
            return self._state

    def _set_status(self, status, error=''):
        self._status = status
        self._error = error
