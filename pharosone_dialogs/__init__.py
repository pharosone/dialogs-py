"""pharosone-dialogs — zero-dependency Python client for the PharosOne dialog ingest API.

Quickstart::

    from pharosone_dialogs import PharosOne

    client = PharosOne(base_url="https://pharosone.example.com", api_key="sk-...")
    client.send_message("support-bot", "sess-1", "user", "Hi!")
"""

from ._version import __version__
from .client import PharosOne, PharosOneError
from .types import EndUser, Message, Role, ToolCall, ToolStatus

__all__ = [
    "PharosOne",
    "PharosOneError",
    "Message",
    "ToolCall",
    "EndUser",
    "Role",
    "ToolStatus",
    "__version__",
]
