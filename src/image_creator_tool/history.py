"""Output path resolution, metadata sidecars, and run history.

Manages the persistent state files that enable features like `again`
(replay last run) and `history` (browse past generations).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".config" / "image-creator-tool"
HISTORY_FILE = CONFIG_DIR / "history.jsonl"
LAST_RUN_FILE = CONFIG_DIR / "last.json"


def slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a filesystem-safe slug (lowercase, hyphens only)."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "image"


def resolve_output_path(
    output_arg: str | None,
    subject: str,
    project: str | None,
    config: dict[str, Any],
) -> Path:
    """Determine the output file path from explicit arg or auto-generation.

    Auto-generated paths follow: <output_dir>/<project?>/<timestamp>-<slug>-<hash4>.png
    The 4-char hash is derived from the subject + current time (ms precision),
    preventing collisions in parallel runs with the same prompt.
    """
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    default_out = str(Path.home() / ".local" / "share" / "image-creator-tool" / "outputs")
    base = Path(config.get("output_dir", default_out)).expanduser()
    if project:
        base = base / project
    base.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    ts = now.strftime("%Y%m%d-%H%M%S")
    # 4-char hash from subject + milliseconds to avoid collisions in parallel runs
    hash_input = f"{subject}{now.strftime('%f')}{id(subject)}"
    suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:4]
    return base / f"{ts}-{slugify(subject)}-{suffix}.png"


def write_sidecar(image_path: Path, metadata: dict[str, Any]) -> Path:
    """Write a JSON sidecar file alongside the generated image."""
    sidecar = image_path.with_suffix(".json")
    with sidecar.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return sidecar


def append_history(entry: dict[str, Any]) -> None:
    """Append a generation record to the persistent history log."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def save_last_run(entry: dict[str, Any]) -> None:
    """Save the current run state for replay via `again` command."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with LAST_RUN_FILE.open("w") as f:
        json.dump(entry, f, indent=2, default=str)


def load_last_run() -> dict[str, Any] | None:
    """Load the last run state, or None if no previous run exists."""
    if not LAST_RUN_FILE.exists():
        return None
    with LAST_RUN_FILE.open() as f:
        return json.load(f)  # type: ignore[no-any-return]
