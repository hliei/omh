"""SQLite session row codecs plus the open-session lifecycle wrapper.

Upstream keeps ``SqliteOpenSession`` in ``sqlite/session.ts`` next to the
``sqlite/session/`` modules. Python cannot have a same-named module and package,
so the thin entry lives here (the same solution the file map uses for
``runtime/drive.ts``).
"""

from omh.agent.session.facade import SessionFacade

SqliteOpenSession = SessionFacade

__all__ = ["SqliteOpenSession"]
