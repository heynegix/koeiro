"""Live check: publish one activity to the running Discord client and read its reply.

This is the only check that uses a real Discord: it opens the desktop client's own
IPC endpoint, handshakes with the app id the app ships, sends one SET_ACTIVITY and
reports Discord's response. Everything else about presence is covered by
tests/test_presence.py against an in-process fake peer.

It leaves nothing behind: closing the connection clears the activity. One update is
sent per run because Discord rate-limits activity updates to one per 15 seconds.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The activity text is Japanese; a cp932 console would otherwise refuse to print it.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from src.presence.activity import APPLICATION_ID, PresenceState, build_activity
from src.presence.ipc import (OP_CLOSE, DiscordIPC, PresenceProtocolError,
                             PresenceUnavailable, open_ipc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-id', default=APPLICATION_ID)
    parser.add_argument('--voice', default='標準ボイス')
    parser.add_argument('--kind', default='standard', choices=('standard', 'user', ''))
    parser.add_argument('--mode', default='ai_voice')
    parser.add_argument('--running', action='store_true', default=True)
    parser.add_argument('--hold', type=float, default=3.0,
                        help='Seconds to watch for an asynchronous reply before closing')
    args = parser.parse_args()
    state = PresenceState(running=args.running, mode=args.mode, voice=args.voice,
                          voice_kind=args.kind, started_at=int(time.time()) if args.running else None)
    activity = build_activity(state)
    try:
        link = open_ipc()
    except PresenceUnavailable as error:
        print(f'NO DISCORD IPC: {error}')
        return 1
    print(f'endpoint: {link.transport.path}', flush=True)
    print(f'handshake: {link.handshake(args.app_id)}', flush=True)
    nonce = link.set_activity(activity)
    print(f'sent: nonce={nonce}\nactivity={activity}', flush=True)
    accepted = False
    try:
        deadline = time.monotonic() + args.hold
        while time.monotonic() < deadline:
            frame = link.poll_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            opcode, payload = frame
            if opcode == OP_CLOSE:
                print(f'CLOSED BY DISCORD: {payload}', flush=True)
                return 1
            if payload.get('nonce') != nonce:
                continue
            print(f'reply: {payload}', flush=True)
            if payload.get('evt') == 'ERROR':
                return 1
            accepted = payload.get('cmd') == 'SET_ACTIVITY'
    except PresenceProtocolError as error:
        # Discord may drop the connection right after replying; that is not a
        # failure of the activity that was already accepted.
        print(f'connection ended: {error}', flush=True)
    finally:
        link.close()
    print('OK: Discord accepted SET_ACTIVITY' if accepted else 'no reply frame before the hold expired',
          flush=True)
    return 0 if accepted else 1


if __name__ == '__main__':
    sys.exit(main())
