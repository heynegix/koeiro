"""Discord activity (Rich Presence) integration, dependency free.

Discord's local RPC server is reached over the desktop client's own IPC endpoint, so
this package needs no network access, no login and no third-party client library:
:mod:`~src.presence.activity` decides what is shown, :mod:`~src.presence.ipc` speaks
Discord's framing and :mod:`~src.presence.client` keeps it running in the background.
"""
from .activity import (APPLICATION_ID, ICON_URL, MODE_LABELS, PresenceState,
                       VOICE_STANDARD, VOICE_USER, build_activity, describe_voice)
from .client import PresenceClient

__all__ = ['APPLICATION_ID', 'ICON_URL', 'MODE_LABELS', 'PresenceState',
           'VOICE_STANDARD', 'VOICE_USER', 'PresenceClient', 'build_activity',
           'describe_voice']
