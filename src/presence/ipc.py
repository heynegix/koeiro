"""Discord's local IPC transport, without a third-party dependency.

Discord exposes a local RPC server over a named pipe on Windows and a Unix socket
on Linux, so presence needs no network, no login and no token. The wire format is
an 8-byte little-endian header (opcode, payload length) followed by a JSON payload:

    https://docs.discord.com/developers/topics/rpc

Only what presence needs is implemented: connect, handshake, send one frame and read
frames (answering the client's PING with PONG). The transports are injected through
:func:`open_ipc` so a test can drive the exact framing code over a socket pair
instead of a live Discord.

Frames are reassembled from bytes the peer has *already* sent, never from a read
that waits for more: Discord writes the header and the payload as separate writes, so
a read asking for a whole frame can wait inside a message that is still arriving. A
Windows pipe read cannot be cancelled, so that would park the worker thread until the
connection died. :meth:`DiscordIPC.poll_frame` therefore only ever consumes what is
buffered and returns None when a frame is still incomplete.
"""
import ctypes
import json
import os
import select
import socket
import struct
import sys
import time

OP_HANDSHAKE = 0
OP_FRAME = 1
OP_CLOSE = 2
OP_PING = 3
OP_PONG = 4

# Windows error codes WaitNamedPipeW reports for the two cases worth naming.
_ERROR_FILE_NOT_FOUND = 2
_ERROR_SEM_TIMEOUT = 121

HEADER = struct.Struct('<II')
RPC_VERSION = 1
PIPE_COUNT = 10
# Discord answers a handshake immediately; waiting longer than this means the
# connection is not going to be usable, so the worker should retry instead of
# sitting in a read that only the peer could end.
HANDSHAKE_TIMEOUT = 5.0
# How often a bounded wait re-asks the transport whether a frame arrived.
POLL_SECONDS = 0.02
# Nothing Discord sends is remotely this large; a longer "length" is a desync.
MAX_FRAME_BYTES = 1024 * 1024

# Discord asks a closing client for a reason string; this is what we tell it.
CLOSE_REASON = 'Disconnected'


class PresenceUnavailable(Exception):
    """No Discord IPC endpoint is listening (the desktop client is not running)."""


class PresenceProtocolError(Exception):
    """The peer broke the framing contract; the connection is not usable."""


class PresenceTimeout(PresenceProtocolError):
    """The peer took the connection but did not answer in time.

    Discord reports this while it throttles connections, so it means "not yet"
    rather than "broken" and the client keeps waiting for it.
    """


class _FileTransport:
    """A Windows named pipe, read only as far as its buffer already reaches.

    ``read_available`` reads exactly what PeekNamedPipe reports, because Microsoft
    Windows pipe reads that ask for more block until the rest arrives -- and cannot be
    cancelled from another thread once they have started.
    """

    def __init__(self, path):
        self.path = path
        self._handle = None
        self._peek = None

    def open(self):
        # A named pipe's open() blocks until an instance is free, and Discord keeps
        # very few: waiting here would park the worker for as long as another client
        # (or Discord's own connection throttling) holds them. WaitNamedPipeW with a
        # zero timeout only asks whether one is free right now, so a busy Discord is
        # reported as unavailable and retried on the next reconnect instead.
        wait = _wait_for_instance(self.path)
        if wait is not None:
            wait()
        self._handle = open(self.path, 'r+b', buffering=0)
        self._peek = _peek_available(self._handle)

    def read_available(self):
        """Bytes already buffered by the peer, possibly empty. Never waits."""
        count = self._peek() if self._peek is not None else None
        if count is None:
            # Without a peek, ask for one byte: the caller only calls this after a
            # frame was announced, so this stays cheap and still surfaces a close.
            count = 1
        if count <= 0:
            return b''
        data = self._handle.read(count)
        if not data:
            raise PresenceProtocolError('Discord closed the connection')
        return data

    def write_all(self, data):
        view = memoryview(data)
        while view:
            written = self._handle.write(view)
            if not written:
                raise PresenceProtocolError('Discord closed the connection')
            view = view[written:]

    def close(self):
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None


class _SocketTransport:
    """A Unix domain socket on Linux, where Discord keeps the same protocol."""

    def __init__(self, path):
        self.path = path
        self._socket = None

    def open(self):
        try:
            family = socket.AF_UNIX
        except AttributeError:  # a host without Unix domain sockets
            raise OSError('Unix domain sockets are unavailable on this host') from None
        connection = socket.socket(family, socket.SOCK_STREAM)
        try:
            connection.connect(self.path)
        except OSError:
            connection.close()
            raise
        self._socket = connection

    def read_available(self):
        if not select.select([self._socket], [], [], 0)[0]:
            return b''
        data = self._socket.recv(65536)
        if not data:
            raise PresenceProtocolError('Discord closed the connection')
        return data

    def write_all(self, data):
        self._socket.sendall(data)

    def close(self):
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None


def _wait_for_instance(path):
    """A zero-timeout ``WaitNamedPipeW`` probe for a Windows pipe, or None.

    Returns a callable that raises the matching ``OSError`` when the pipe is absent
    or every instance is held, and returns normally otherwise. The distinction is
    kept because "not running" and "busy" deserve different messages in the log.
    """
    if sys.platform != 'win32':
        return None
    try:
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.WaitNamedPipeW.argtypes = [ctypes.c_wchar_p, wintypes.DWORD]
        kernel32.WaitNamedPipeW.restype = wintypes.BOOL

        def wait():
            if kernel32.WaitNamedPipeW(path, 0):
                return
            code = ctypes.get_last_error()
            if code == _ERROR_FILE_NOT_FOUND:
                raise FileNotFoundError(_ERROR_FILE_NOT_FOUND, 'No such named pipe', path)
            if code == _ERROR_SEM_TIMEOUT:
                raise OSError(f'Every instance of {path} is in use')
            # Any other error (including a race we just lost) is left to open().

        return wait
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def _peek_available(handle):
    """A callable returning the bytes waiting on a Windows pipe handle, or None.

    PeekNamedPipe is the only way to ask a synchronous pipe whether data is waiting
    without entering a read. Handles are 64-bit, so the ctypes signatures are set
    explicitly: the default ``int`` return would truncate them.
    """
    if sys.platform != 'win32':
        return None
    try:
        import msvcrt
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.PeekNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                          wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD),
                                          wintypes.LPVOID]
        kernel32.PeekNamedPipe.restype = wintypes.BOOL
        # msvcrt.get_osfhandle is a built-in that already returns the full-width
        # handle as a Python int; only ctypes functions accept argtypes.
        pipe = ctypes.c_void_p(msvcrt.get_osfhandle(handle.fileno()))

        def available():
            count = wintypes.DWORD(0)
            if not kernel32.PeekNamedPipe(pipe, None, 0, None, ctypes.byref(count), None):
                return None
            # A count larger than our own read ceiling is not a real message.
            return int(count.value)

        return available
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def _posix_join(root, name):
    """Join with '/' explicitly: these are always Unix paths, whatever host builds them."""
    return f"{root.rstrip('/')}/{name}"


def ipc_paths(platform=None, environ=None):
    """Candidate IPC endpoints in the order Discord documents them."""
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    names = [f'discord-ipc-{index}' for index in range(PIPE_COUNT)]
    if platform == 'win32':
        return [f'\\\\.\\pipe\\{name}' for name in names]
    roots = []
    for key in ('XDG_RUNTIME_DIR', 'TMPDIR', 'TMP', 'TEMP'):
        value = environ.get(key)
        if value:
            roots.append(value)
    # Snap-installed Discord keeps its socket in its own runtime directory.
    for root in list(roots):
        roots.append(_posix_join(root, 'snap.discord'))
    roots.append('/tmp')
    paths = []
    for root in roots:
        paths.extend(_posix_join(root, name) for name in names)
    return paths


def open_ipc(platform=None, environ=None, paths=None):
    """Connect to the first listening Discord endpoint.

    Raises PresenceUnavailable when none is listening, which is the normal state
    while the Discord desktop client is closed.
    """
    platform = sys.platform if platform is None else platform
    candidates = ipc_paths(platform, environ) if paths is None else list(paths)
    failures = []
    for path in candidates:
        transport = _FileTransport(path) if platform == 'win32' else _SocketTransport(path)
        try:
            transport.open()
        except OSError as error:
            failures.append(f'{path}: {error}')
            continue
        return DiscordIPC(transport)
    raise PresenceUnavailable(
        'Discord IPC unavailable (is the desktop client running?)'
        + (f' — {failures[0]}' if failures else ''))


class DiscordIPC:
    """One IPC connection: framing, handshake and single-frame commands."""

    def __init__(self, transport):
        self.transport = transport
        self.ready = False
        self.closed_by_peer = None
        # Bytes received but not yet forming a complete frame. Discord writes a
        # frame's header and payload separately, so a partial frame is normal.
        self._buffer = b''

    def handshake(self, client_id, timeout=HANDSHAKE_TIMEOUT, cancel=None):
        """Opcode 0 with the app id; Discord answers with a FRAME carrying READY."""
        self.transport.write_all(self._packed(OP_HANDSHAKE, {'v': RPC_VERSION, 'client_id': str(client_id)}))
        opcode, payload = self.read_frame(timeout=timeout, cancel=cancel)
        if opcode == OP_CLOSE:
            raise PresenceProtocolError(f'Discord refused the connection: {payload}')
        if opcode != OP_FRAME:
            raise PresenceProtocolError(f'Unexpected handshake reply opcode {opcode}')
        event = payload.get('evt')
        if event not in (None, 'READY'):
            raise PresenceProtocolError(f'Unexpected handshake event {event!r}')
        self.ready = True
        return payload

    def send_command(self, command, args):
        """One FRAME command, e.g. SET_ACTIVITY. Returns the nonce used."""
        nonce = os.urandom(16).hex()
        self.transport.write_all(self._packed(OP_FRAME, {
            'cmd': command,
            'args': args,
            'nonce': nonce,
        }))
        return nonce

    def set_activity(self, activity, pid=None):
        return self.send_command('SET_ACTIVITY', {
            'pid': os.getpid() if pid is None else int(pid),
            'activity': activity,
        })

    def poll_frame(self):
        """The next complete frame, or None when one has not fully arrived yet.

        Never waits for the peer: only bytes it has already sent are consumed, so an
        incomplete frame simply stays in the buffer for the next call. A PING is not a
        reply to anything, so it is answered here and the loop continues; CLOSE is
        handed back to the caller because only it knows whether to reconnect.
        """
        while True:
            self._buffer += self.transport.read_available()
            frame = self._take_frame()
            if frame is None:
                return None
            opcode, payload = frame
            if opcode == OP_PING:
                self.transport.write_all(self._packed(OP_PONG, payload))
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                self.closed_by_peer = payload
                return opcode, payload
            if opcode != OP_FRAME:
                raise PresenceProtocolError(f'Unknown opcode {opcode}')
            return opcode, payload

    def read_frame(self, timeout=HANDSHAKE_TIMEOUT, cancel=None):
        """Wait, at most ``timeout`` seconds, for the next complete frame.

        ``cancel`` is consulted between polls so a caller that owns this connection can
        give up immediately -- the app must close without waiting out a handshake
        Discord is throttling.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            frame = self.poll_frame()
            if frame is not None:
                return frame
            if cancel is not None and cancel():
                raise PresenceTimeout('Cancelled before Discord answered')
            if time.monotonic() >= deadline:
                raise PresenceTimeout(
                    f'Timed out after {timeout:g} s waiting for Discord')
            time.sleep(POLL_SECONDS)

    def close(self):
        """Ask Discord to release the connection, then drop the transport."""
        try:
            if self.ready:
                self.transport.write_all(self._packed(OP_CLOSE, {'code': 1000, 'message': CLOSE_REASON}))
        except (OSError, PresenceProtocolError):
            pass
        self.ready = False
        self.transport.close()

    def _take_frame(self):
        if len(self._buffer) < HEADER.size:
            return None
        opcode, length = HEADER.unpack(self._buffer[:HEADER.size])
        if length > MAX_FRAME_BYTES:
            raise PresenceProtocolError(f'Frame of {length} bytes is not a Discord payload')
        end = HEADER.size + length
        if len(self._buffer) < end:
            return None
        payload = self._buffer[HEADER.size:end]
        self._buffer = self._buffer[end:]
        return opcode, _decode(payload)

    @staticmethod
    def _packed(opcode, payload):
        data = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        return HEADER.pack(opcode, len(data)) + data


def _decode(data):
    if not data:
        return {}
    try:
        value = json.loads(data.decode('utf-8'))
    except (ValueError, UnicodeDecodeError) as error:
        raise PresenceProtocolError(f'Invalid JSON from Discord: {error}') from error
    return value if isinstance(value, dict) else {'data': value}
