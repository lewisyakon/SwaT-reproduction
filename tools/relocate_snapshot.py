#!/usr/bin/env python3
"""Preview or apply literal root-path rewrites to tracked text snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path


TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=repo_root)
    parser.add_argument("--map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--write", action="store_true", help="Apply changes; default is a dry run.")
    args = parser.parse_args()
    mappings: list[tuple[str, str]] = []
    for item in args.map:
        if "=" not in item:
            parser.error(f"invalid --map {item!r}; expected OLD=NEW")
        old, new = item.split("=", 1)
        if not old:
            parser.error("OLD in --map cannot be empty")
        mappings.append((old, new))
    if not mappings:
        parser.error("at least one --map OLD=NEW is required")

    changed = 0
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = original
        for old, new in mappings:
            updated = updated.replace(old, new)
        if updated == original:
            continue
        print(path)
        changed += 1
        if args.write:
            path.write_text(updated, encoding="utf-8")
    print(f"{'updated' if args.write else 'would update'} {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

