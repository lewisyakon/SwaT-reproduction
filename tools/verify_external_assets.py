#!/usr/bin/env python3
"""Verify the external assets recorded by a reproduction repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "migration" / "external_assets.json",
    )
    parser.add_argument(
        "--mode",
        choices=("current", "restored"),
        default="current",
        help="Check current_path, or recommended_path below this repository.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Only check assets used by this profile; repeat for multiple profiles.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested = set(args.profile)
    checked = 0
    missing_required = 0

    for asset in manifest["assets"]:
        profiles = set(asset.get("profiles", []))
        if requested and profiles.isdisjoint(requested):
            continue
        if args.mode == "current":
            path = Path(asset["current_path"])
        else:
            path = repo_root / asset["recommended_path"]
        exists = path.exists()
        kind = asset.get("kind", "directory")
        valid_kind = (path.is_dir() if kind == "directory" else path.is_file()) if exists else False
        ok = exists and valid_kind
        required = bool(asset.get("required", True))
        status = "OK" if ok else ("MISSING" if not exists else "WRONG_KIND")
        print(f"{status:10} {asset['id']:28} {path} [{asset.get('size_hint', 'size unknown')}]")
        checked += 1
        if required and not ok:
            missing_required += 1

    if checked == 0:
        print("No assets matched the requested profile(s).")
        return 2
    if missing_required:
        print(f"Required assets missing or invalid: {missing_required}")
        return 1
    print(f"Verified {checked} asset entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

