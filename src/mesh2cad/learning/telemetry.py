"""Small atomic status files consumed by the local operations dashboard."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def publish_telemetry(path: str | Path | None, payload: dict[str, Any]) -> bool:
    """Publish a replace-safe snapshot without making telemetry a training dependency."""
    if path is None:
        return False
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".part")
    document = {
        "schema_version": 1,
        **payload,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except OSError:
        return False
    return True
