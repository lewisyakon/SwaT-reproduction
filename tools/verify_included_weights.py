#!/usr/bin/env python3
"""Verify that Git LFS weights are materialized and match the committed manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / "weights/manifest.json").read_text(encoding="utf-8"))
    failures = 0
    for entry in manifest["entries"]:
        path = repo / entry["path"]
        if not path.is_file():
            print(f"MISSING  {entry['id']}: {path}")
            failures += 1
            continue
        actual_size = path.stat().st_size
        if actual_size != entry["file_bytes"]:
            print(
                f"SIZE     {entry['id']}: expected {entry['file_bytes']}, got {actual_size}; "
                "run `git lfs pull`"
            )
            failures += 1
            continue
        actual_hash = sha256(path)
        if actual_hash != entry["sha256"]:
            print(f"SHA256  {entry['id']}: expected {entry['sha256']}, got {actual_hash}")
            failures += 1
            continue
        print(f"OK       {entry['id']}: {actual_size} bytes")
    if failures:
        print(f"Failed entries: {failures}")
        return 1
    print(f"Verified {len(manifest['entries'])} included weight files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
