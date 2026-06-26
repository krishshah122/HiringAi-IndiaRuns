"""Stream JSON / JSONL candidate files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected JSON array in {path}")
    return list(iter_candidates(path))


def iter_candidates(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
