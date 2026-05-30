"""
Run the full pipeline for one book: normalize → transcribe → rename.

Usage:
    python pipeline.py "Livre 1"
    python pipeline.py "Livre 1" --overwrite
    python pipeline.py "Livre 1" --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = [
    ("normalize",   "normalize.py"),
    ("transcribe",  "transcribe.py"),
    ("rename",      "rename.py"),
]


def run_step(name: str, script: str, book: str, extra_args: list[str]) -> bool:
    here = Path(__file__).parent
    cmd = [sys.executable, str(here / script), "--book", book, *extra_args]
    print(f"\n{'='*60}")
    print(f"  STEP: {name}  →  {script}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[pipeline] ERROR: {name} failed (exit {result.returncode}). Stopping.")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run normalize → transcribe → rename for one book folder."
    )
    parser.add_argument("book", help="Book folder name (e.g. \"Heidi jeune fille\")")
    parser.add_argument("--overwrite", action="store_true",
                        help="Pass --overwrite to each step")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pass --dry-run to rename (normalize and transcribe still run)")
    parser.add_argument("--start-from", choices=["normalize", "transcribe", "rename"],
                        default="normalize",
                        help="Skip earlier steps and start from this one (default: normalize)")
    args = parser.parse_args()

    # Build per-step extra args
    common = ["--overwrite"] if args.overwrite else []
    step_extra = {
        "normalize":  common,
        "transcribe": common,
        "rename":     common + (["--dry-run"] if args.dry_run else []),
    }

    active = False
    for name, script in STEPS:
        if name == args.start_from:
            active = True
        if not active:
            print(f"[pipeline] skipping {name}")
            continue
        if not run_step(name, script, args.book, step_extra[name]):
            return 1

    print(f"\n[pipeline] Done — \"{args.book}\" processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
