"""Source-tree bootstrap shared by spawned all-axis E2E workers."""

from pathlib import Path

import mooncake


_RESHARD_PACKAGE = str(Path(__file__).resolve().parents[2] / "python" / "mooncake")
if _RESHARD_PACKAGE not in mooncake.__path__:
    mooncake.__path__.append(_RESHARD_PACKAGE)
