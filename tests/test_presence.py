"""Discord activity tests.

Two halves, labelled separately like the audio policy asks the audio work to be:
the activity lines are pure and tested directly, and the transport is driven against
an in-process fake Discord over a socket pair. Nothing here talks to a real Discord
client -- see tests/README-presence.md for the separate live check.
"""
import ctypes
import json
import select
import socket
import sys
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from src.audio.controller import AudioController
from src.audio.engine import AudioEngine
from src.gui.main_window import MainWindow
from src.presence import (APPLICATION_ID, ICON_ENV, ICON_IMAGE, ICON_TEXT,
                          PresenceClient, PresenceState, VOICE_STANDARD, VOICE_USER,
                          build_activity, describe_voice, icon_assets)
from src.presence.ipc import (OP_CLOSE, OP_FRAME, OP_HANDSHAKE, OP_PING, OP_PONG,
                             HEADER, DiscordIPC, PresenceProtocolError,
                             PresenceUnavailable, ipc_paths, open_ipc)
from src.settings.manager import AppSettings, SettingsManager
from .fakes import FakeBackend


# --------------------------------------------------------------------------- fakes


class PairTransport:
    """The transport surface DiscordIPC needs, over one end of a socket pair.

    ``read_available`` mirrors the real Windows pipe: it only ever returns bytes that
    are already buffered, so a reassembly bug cannot hide behind a blocking read.
    """

    def __init__(self, connection):
        self.connection = connection

    def read_available(self):
        try:
            if not select.select([self.connection], [], [], 0)[0]:
                return b''
            data = self.connection.recv(65536)
        except (OSError, ValueError):
            raise PresenceProtocolError('Discord closed the connection') from None
        if not data:
            raise PresenceProtocolError('Discord closed the connection')
        return data

    def write_all(self, data):
        self.connection.sendall(data)

    def close(self):
        try:
            self.connection.close()
        except OSError:
            pass


def send_frame(connection, opcode, payload):
    data = json.dumps(payload).encode('utf-8')
    connection.sendall(HEADER.pack(opcode, len(data)) + data)


def recv_frame(connection):
    header = HEADER.unpack(recv_exactly(connection, HEADER.size))
    body = recv_exactly(connection, header[1]) if header[1] else b''
    return header[0], json.loads(body.decode('utf-8'))


def recv_exactly(connection, size):
    chunks, remaining = [], size
    while remaining > 0:
        block = connection.recv(remaining)
        if not block:
            raise PresenceProtocolError('peer closed')
        chunks.append(block)
        remaining -= len(block)
    return b''.join(chunks)


class WindowsPipe:
    """A real one-instance named pipe, for the Windows-only transport tests.

    Needed because the transport's whole point is not to block: only a real pipe can
    tell a peek-counted read apart from a one-byte drip.
    """

    def __init__(self, name):
        from ctypes import wintypes
        self._ctypes = ctypes
        self._wintypes = wintypes
        self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel32.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                                   wintypes.DWORD, wintypes.DWORD,
                                                   wintypes.DWORD, wintypes.DWORD,
                                                   wintypes.DWORD, wintypes.LPVOID]
        self.kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
        self.kernel32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
        self.kernel32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
        self.kernel32.WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                                            ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.path = f'\\\\.\\pipe\\koeiro-presence-{name}'
        self.handle = self.kernel32.CreateNamedPipeW(self.path, 0x3, 0, 1, 4096, 4096, 0, None)
        self.connected = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        self.kernel32.ConnectNamedPipe(self.handle, None)
        self.connected.set()

    def write(self, data):
        written = self._wintypes.DWORD(0)
        self.kernel32.WriteFile(self.handle, data, len(data), self._ctypes.byref(written), None)
        return written.value

    def close(self):
        self.kernel32.DisconnectNamedPipe(self.handle)
        self.kernel32.CloseHandle(self.handle)


class FakeDiscord:
    """A minimal Discord RPC peer: READY on handshake, a reply per command."""

    def __init__(self, refuse_handshake=None, error=None):
        self.server, self.client = socket.socketpair()
        self.handshake = None
        self.commands = []          # (cmd, args, nonce)
        self.pongs = 0
        self.refuse_handshake = refuse_handshake
        self.error = error           # e.g. {'code': 4000, 'message': 'Invalid Client ID'}
        self.closed = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def client_transport(self):
        return PairTransport(self.client)

    def _serve(self):
        try:
            opcode, payload = recv_frame(self.server)
            assert opcode == OP_HANDSHAKE
            self.handshake = payload
            if self.refuse_handshake is not None:
                send_frame(self.server, OP_CLOSE, self.refuse_handshake)
                return
            send_frame(self.server, OP_FRAME, {'cmd': 'DISPATCH', 'evt': 'READY',
                                               'data': {'v': 1}})
            while True:
                opcode, payload = recv_frame(self.server)
                if opcode == OP_CLOSE:
                    return
                if opcode == OP_PONG:
                    self.pongs += 1
                    continue
                if opcode != OP_FRAME:
                    continue
                command = payload.get('cmd')
                self.commands.append((command, payload.get('args'), payload.get('nonce')))
                reply = {'cmd': command, 'data': {}, 'nonce': payload.get('nonce')}
                if self.error is not None:
                    reply['evt'] = 'ERROR'
                    reply['data'] = dict(self.error)
                send_frame(self.server, OP_FRAME, reply)
        except (OSError, PresenceProtocolError, AssertionError):
            pass
        finally:
            self.closed.set()

    def close(self):
        self.client.close()
        self.server.close()


def wait_for(predicate, timeout=4.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def pump(predicate, timeout=5.0):
    """The same wait, but running Qt events so the window's timer can tick."""
    application = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def connected_client(fake, **kwargs):
    return PresenceClient(connector=lambda: DiscordIPC(fake.client_transport()),
                          poll_interval=0.01, **kwargs)


# ------------------------------------------------------------------- activity text


def test_activity_reports_running_conversion_and_the_shipped_voice():
    activity = build_activity(PresenceState(running=True, mode='ai_voice',
                                            voice='標準ボイス', voice_kind=VOICE_STANDARD,
                                            started_at=1234567890))
    assert activity['type'] == 0
    assert activity['details'] == 'ボイス変換中'
    assert activity['state'] == '声: 標準ボイス · AI Voice'
    assert activity['assets']['large_image'] == ICON_IMAGE
    assert activity['timestamps'] == {'start': 1234567890}


def test_activity_shows_the_icon_in_both_slots_the_card_and_the_call_row_use():
    # One slot alone leaves the other spot as "?": the card draws large_image and
    # its corner (and the call row) draws small_image.
    assets = build_activity(PresenceState())['assets']
    assert assets['large_image'] == assets['small_image'] == ICON_IMAGE
    assert assets['large_text'] == assets['small_text'] == ICON_TEXT


def test_a_deployed_build_can_be_pointed_at_an_uploaded_asset_key(monkeypatch):
    # The portal route needs no code change: the key replaces the URL, and Discord
    # resolves a bare key itself instead of fetching it.
    monkeypatch.setenv(ICON_ENV, 'koeiro_icon')
    assert icon_assets() == {'large_image': 'koeiro_icon', 'large_text': ICON_TEXT,
                             'small_image': 'koeiro_icon', 'small_text': ICON_TEXT}


def test_no_image_named_means_no_assets_object_at_all(monkeypatch):
    # Sending an empty assets object would be a promise Discord cannot draw.
    monkeypatch.setenv(ICON_ENV, '   ')
    assert 'assets' not in build_activity(PresenceState())


def test_activity_names_a_later_registered_voice_and_drops_the_timer_while_stopped():
    activity = build_activity(PresenceState(running=False, mode='ai_voice',
                                            voice='ゆかり', voice_kind=VOICE_USER,
                                            started_at=1234567890))
    assert activity['state'] == '声: ゆかり（追加した声） · AI Voice'
    assert activity['details'] == '待機中（変換していません）'
    # No run is happening, so counting up from the last one would be a lie.
    assert 'timestamps' not in activity


def test_activity_says_error_only_when_not_converting():
    failed = build_activity(PresenceState(failed=True, mode='ai_voice', voice='標準ボイス'))
    assert failed['details'] == 'エラー（変換停止中）'
    assert build_activity(PresenceState(running=True, failed=True,
                                       mode='ai_voice', voice='標準ボイス'))['details'] == 'ボイス変換中'


@pytest.mark.parametrize('mode,expected', [
    ('original', 'Original（無変換）'),
    ('female_dsp', 'Female DSP（軽い加工）'),
])
def test_activity_describes_the_non_ai_routes(mode, expected):
    assert build_activity(PresenceState(mode=mode))['state'] == expected


def test_activity_lines_stay_inside_discords_limit():
    # A registration may use an 80-character name; the composed line must not
    # exceed Discord's 128 bytes, and must not split a Japanese character.
    activity = build_activity(PresenceState(running=True, mode='ai_voice',
                                            voice='あ' * 80, voice_kind=VOICE_USER))
    for line in (activity['details'], activity['state']):
        assert len(line.encode('utf-8')) <= 128
        assert 'あ' * 80 not in line


def test_describe_voice_labels_kind_and_survives_an_empty_name():
    assert describe_voice('標準ボイス', VOICE_STANDARD) == '標準ボイス'
    assert describe_voice('アリス', VOICE_USER) == 'アリス（追加した声）'
    assert describe_voice('', VOICE_STANDARD) == '標準ボイス'
    assert describe_voice('', '') == ''


# ------------------------------------------------------------------------- ipc


def test_windows_ipc_paths_use_the_documented_pipe_names():
    paths = ipc_paths('win32', {})
    assert paths[0] == '\\\\.\\pipe\\discord-ipc-0'
    assert paths[-1] == '\\\\.\\pipe\\discord-ipc-9'


def test_linux_ipc_paths_prefer_xdg_then_the_snap_directory_and_tmp():
    paths = ipc_paths('linux', {'XDG_RUNTIME_DIR': '/run/user/1000'})
    assert paths[0] == '/run/user/1000/discord-ipc-0'
    assert '/run/user/1000/snap.discord/discord-ipc-0' in paths
    assert paths[-1] == '/tmp/discord-ipc-9'
    # TMPDIR is consulted when XDG_RUNTIME_DIR is not set, as Discord documents.
    assert ipc_paths('linux', {'TMPDIR': '/var/tmp'})[0] == '/var/tmp/discord-ipc-0'


def test_open_ipc_reports_unavailable_when_no_endpoint_listens(tmp_path):
    missing = [str(tmp_path / f'discord-ipc-{index}') for index in range(3)]
    with pytest.raises(PresenceUnavailable) as error:
        open_ipc('linux', {}, missing)
    assert 'Discord IPC unavailable' in str(error.value)
    assert 'discord-ipc-0' in str(error.value)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows named pipes')
def test_open_ipc_reports_a_busy_pipe_instead_of_blocking_on_it():
    """A held pipe must fail fast: Discord keeps very few instances.

    A blocking open() here is what parks the worker thread, so the pipe is held open
    for real and the second connect is watched with a join timeout: if the code ever
    waits instead of reporting, this test fails instead of hanging.
    """
    pipe = WindowsPipe('busy')
    holder = open_ipc('win32', {}, [pipe.path])
    try:
        assert pipe.connected.wait(5)
        outcome = {}

        def attempt():
            try:
                link = open_ipc('win32', {}, [pipe.path])
            except PresenceUnavailable as error:
                outcome['error'] = str(error)
            else:
                link.transport.close()
                outcome['opened'] = True

        worker = threading.Thread(target=attempt, daemon=True)
        started = time.monotonic()
        worker.start()
        worker.join(3.0)
        assert not worker.is_alive(), 'open_ipc blocked on a busy pipe'
        assert time.monotonic() - started < 2.0
        assert 'opened' not in outcome
        assert 'in use' in outcome.get('error', '')
    finally:
        holder.close()
        pipe.close()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows named pipes')
def test_a_pipe_read_takes_what_is_buffered_and_no_more():
    """The peeked count is what keeps a pipe read immediate and complete.

    Without it the transport falls back to one byte per read: still correct, but this
    is the difference between "read what already arrived" and "ask for more".
    """
    from src.presence.ipc import _FileTransport
    pipe = WindowsPipe('buffered')
    transport = _FileTransport(pipe.path)
    transport.open()
    try:
        assert pipe.connected.wait(5)
        blob = b'0123456789' * 4
        assert pipe.write(blob) == len(blob)
        assert wait_for(lambda: transport.read_available() == blob)
    finally:
        transport.close()
        pipe.close()


def test_a_frame_split_across_two_writes_is_reassembled_without_waiting():
    """Discord writes a frame's header and payload separately; both are normal.

    Reading a whole frame in one call would block inside the message Discord is still
    sending, so the split is driven right here: the first half must not produce a
    frame and must not wait, and the second half must complete it.
    """
    fake = FakeDiscord().start()
    try:
        link = DiscordIPC(fake.client_transport())
        link.handshake(APPLICATION_ID)
        body = json.dumps({'cmd': 'DISPATCH', 'evt': 'SOMETHING'}).encode('utf-8')
        wire = HEADER.pack(OP_FRAME, len(body)) + body
        fake.server.sendall(wire[:6])
        assert link.poll_frame() is None
        fake.server.sendall(wire[6:])
        assert wait_for(lambda: link.poll_frame() is not None)
    finally:
        fake.close()


def test_read_frame_times_out_instead_of_waiting_forever():
    fake = FakeDiscord().start()
    try:
        link = DiscordIPC(fake.client_transport())
        link.handshake(APPLICATION_ID)
        started = time.monotonic()
        with pytest.raises(PresenceProtocolError) as error:
            link.read_frame(timeout=0.2)
        assert 'Timed out' in str(error.value)
        assert time.monotonic() - started < 2.0
    finally:
        fake.close()


def test_handshake_and_set_activity_use_the_documented_wire_format():
    fake = FakeDiscord().start()
    try:
        link = DiscordIPC(fake.client_transport())
        link.handshake(APPLICATION_ID)
        assert link.ready
        link.set_activity(build_activity(PresenceState(running=True, mode='ai_voice',
                                                      voice='標準ボイス',
                                                      voice_kind=VOICE_STANDARD,
                                                      started_at=99)), pid=4242)
        opcode, reply = link.read_frame()
        assert opcode == OP_FRAME and reply['cmd'] == 'SET_ACTIVITY'
        assert wait_for(lambda: len(fake.commands) == 1)
        assert fake.handshake == {'v': 1, 'client_id': APPLICATION_ID}
        command, args, nonce = fake.commands[0]
        assert command == 'SET_ACTIVITY'
        assert args['pid'] == 4242
        assert args['activity']['details'] == 'ボイス変換中'
        assert nonce == reply['nonce']
        link.close()
    finally:
        fake.close()


def test_handshake_gives_up_when_discord_accepts_but_never_replies():
    """A silent peer must not park the worker: the handshake is bounded."""
    server, client = socket.socketpair()
    try:
        link = DiscordIPC(PairTransport(client))
        started = time.monotonic()
        with pytest.raises(PresenceProtocolError) as error:
            link.handshake(APPLICATION_ID, timeout=0.2)
        assert 'Timed out' in str(error.value)
        assert time.monotonic() - started < 2.0
        assert link.ready is False
    finally:
        server.close()
        client.close()


def test_handshake_reports_a_refused_client_id():
    fake = FakeDiscord(refuse_handshake={'code': 4000, 'message': 'Invalid Client ID'}).start()
    try:
        link = DiscordIPC(fake.client_transport())
        with pytest.raises(PresenceProtocolError) as error:
            link.handshake('1234')
        assert 'Invalid Client ID' in str(error.value)
    finally:
        fake.close()


def test_ping_is_answered_with_pong_and_does_not_hide_the_next_frame():
    fake = FakeDiscord().start()
    try:
        link = DiscordIPC(fake.client_transport())
        link.handshake(APPLICATION_ID)
        # A PING is not a reply: read_frame must answer it and keep waiting.
        fake.server.sendall(HEADER.pack(OP_PING, 0))
        link.set_activity(build_activity(PresenceState()), pid=1)
        opcode, reply = link.read_frame()
        assert opcode == OP_FRAME and reply['cmd'] == 'SET_ACTIVITY'
        assert wait_for(lambda: fake.pongs == 1)
    finally:
        fake.close()


# ---------------------------------------------------------------------- client


def test_client_publishes_the_newest_state_and_coalesces_bursts():
    fake = FakeDiscord().start()
    client = connected_client(fake, min_update_interval=0.2, reconnect_delay=0.05)
    try:
        # The snapshot is set before connecting, so it is what the first publish
        # carries: connecting must not leave Discord with nothing to show.
        client.update(PresenceState(running=True, mode='ai_voice', voice='標準ボイス',
                                    voice_kind=VOICE_STANDARD, started_at=1))
        client.start()
        assert wait_for(lambda: len(fake.commands) == 1)
        assert fake.commands[0][1]['activity']['details'] == 'ボイス変換中'
        # A burst inside the rate-limit window must not become three updates: only
        # the newest snapshot is sent when the window opens.
        client.update(PresenceState(running=False, mode='ai_voice', voice='ゆかり',
                                    voice_kind=VOICE_USER))
        client.update(PresenceState(running=False, mode='ai_voice', voice='アリス',
                                    voice_kind=VOICE_USER))
        assert wait_for(lambda: len(fake.commands) == 2)
        last = fake.commands[-1][1]['activity']
        assert last['state'] == '声: アリス（追加した声） · AI Voice'
        assert last['details'] == '待機中（変換していません）'
        time.sleep(0.3)
        assert len(fake.commands) == 2
        assert client.sends == 2
    finally:
        client.stop()
        fake.close()


def test_client_waits_for_discord_and_keeps_trying():
    attempts = []

    def absent():
        attempts.append(time.monotonic())
        raise PresenceUnavailable('Discord IPC not found')

    client = PresenceClient(connector=absent, reconnect_delay=0.05, poll_interval=0.01)
    try:
        client.start()
        assert wait_for(lambda: client.status == 'searching')
        assert wait_for(lambda: len(attempts) >= 3)
        assert 'Discord' in client.label() or '待っています' in client.label()
    finally:
        client.stop()


def test_client_keeps_waiting_when_discord_throttles_the_handshake():
    """Discord delays connections over its rate limit; that is "not yet".

    Silence right after connecting must read as waiting, not as a failure the reader
    cannot act on, and the worker must still be stoppable afterwards.
    """
    server, connection = socket.socketpair()
    client = PresenceClient(connector=lambda: DiscordIPC(PairTransport(connection)),
                            reconnect_delay=0.05, poll_interval=0.01,
                            handshake_timeout=0.2)
    try:
        client.start()
        assert wait_for(lambda: client.status == 'searching')
        assert '待っています' in client.label()
    finally:
        assert client.stop() is True
        server.close()
        connection.close()


def test_stopping_during_a_silent_handshake_ends_the_worker_promptly():
    """Closing the app must not wait out a handshake Discord is throttling."""
    server, connection = socket.socketpair()
    client = PresenceClient(connector=lambda: DiscordIPC(PairTransport(connection)),
                            reconnect_delay=30.0, poll_interval=0.01,
                            handshake_timeout=30.0)
    try:
        client.start()
        assert wait_for(lambda: client.status == 'searching')
        started = time.monotonic()
        assert client.stop(timeout=2.0) is True
        assert time.monotonic() - started < 1.0
        assert not any(thread.name == 'discord-presence' and thread.is_alive()
                       for thread in threading.enumerate())
    finally:
        server.close()
        connection.close()


def test_client_surfaces_an_activity_discord_rejected():
    fake = FakeDiscord(error={'code': 4000, 'message': 'Invalid Client ID'}).start()
    client = connected_client(fake, min_update_interval=0.05)
    try:
        client.start()
        client.update(PresenceState(running=True, mode='ai_voice', voice='標準ボイス'))
        assert wait_for(lambda: client.error != '')
        assert client.status == 'connected'
        assert '拒否' in client.label() and 'Invalid Client ID' in client.label()
    finally:
        client.stop()
        fake.close()


def test_client_reconnects_after_discord_closes_the_connection():
    fake = FakeDiscord().start()
    client = connected_client(fake, min_update_interval=0.05, reconnect_delay=0.05)
    try:
        client.start()
        assert wait_for(lambda: client.status == 'connected')
        # Discord dropping the pipe must not leave the app claiming it is connected.
        fake.close()
        assert wait_for(lambda: client.status in ('error', 'searching'))
    finally:
        client.stop()
    assert client.status == 'off'
    assert not any(thread.name == 'discord-presence' and thread.is_alive()
                   for thread in threading.enumerate())


def test_stop_is_idempotent_and_can_be_restarted():
    fake = FakeDiscord().start()
    client = connected_client(fake, min_update_interval=0.05, reconnect_delay=0.05)
    try:
        assert client.stop() is True
        assert client.stop() is True
        client.start()
        assert wait_for(lambda: client.status == 'connected')
        assert client.stop() is True
    finally:
        fake.close()


def test_update_reports_whether_the_snapshot_changed():
    client = PresenceClient(connector=lambda: None)
    assert client.update(PresenceState(running=True)) is True
    assert client.update(PresenceState(running=True)) is False
    assert client.update(PresenceState(running=False)) is True


# ------------------------------------------------------------------- settings


def test_discord_presence_setting_defaults_on_and_round_trips(tmp_path):
    assert AppSettings().discord_presence is True
    assert AppSettings.from_dict({'discord_presence': False}).discord_presence is False
    # Only a real boolean may turn it off; a hand-edited file cannot.
    assert AppSettings.from_dict({'discord_presence': 'no'}).discord_presence is True
    manager = SettingsManager(tmp_path / 'settings.json')
    assert manager.save(AppSettings(discord_presence=False)) is True
    assert manager.load().discord_presence is False


# ------------------------------------------------------------------------ gui


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    backend = FakeBackend()
    controller = AudioController(AudioEngine(backend=backend))
    presence = FakePresence()
    window = MainWindow(SettingsManager(tmp_path / 'settings.json'), controller,
                        presence=presence, presence_factory=lambda: presence)
    window.show()
    assert pump(lambda: window.start_button.isEnabled())
    yield window, presence
    window.close()
    assert pump(lambda: not controller.alive)


class FakePresence:
    """Stands in for the Discord worker so the GUI never talks to a real client."""

    def __init__(self):
        self.snapshots = []
        self.started = 0
        self.stopped = 0
        self.status = 'connected'

    def start(self):
        self.started += 1
        return self

    def stop(self, timeout=2.0):
        self.stopped += 1
        return True

    def update(self, state):
        self.snapshots.append(state)
        return True

    def label(self):
        return 'テスト接続中'

    @property
    def latest(self):
        return self.snapshots[-1] if self.snapshots else None


def test_gui_publishes_running_state_and_the_selected_voice(window, qapp):
    window, presence = window
    assert pump(lambda: presence.latest is not None)
    idle = presence.latest
    assert idle.running is False
    assert idle.mode == 'ai_voice'
    # The shipped voice must be labelled as the standard voice, not as a name alone.
    assert idle.voice and idle.voice_kind == VOICE_STANDARD
    window.start_button.click()
    assert pump(lambda: presence.latest is not None and presence.latest.running)
    running = presence.latest
    assert running.started_at is not None
    assert running.voice == idle.voice
    window.stop_button.click()
    assert pump(lambda: presence.latest.running is False)
    assert presence.latest.started_at is None


def test_gui_shows_what_discord_would_display(window, qapp):
    window, presence = window
    assert pump(lambda: 'テスト接続中' in window.discord_status.text())
    assert pump(lambda: 'Discordに表示される内容' in window.discord_preview.text())
    assert '待機中' in window.discord_preview.text()
    assert window.discord_toggle.isChecked() is True


def test_gui_toggle_off_disconnects_and_is_saved(window, qapp):
    window, presence = window
    window.discord_toggle.setChecked(False)
    assert presence.stopped == 1
    assert window.presence is None
    assert window.settings.discord_presence is False
    window.discord_toggle.setChecked(True)
    assert presence.started == 1
    assert window.presence is presence
