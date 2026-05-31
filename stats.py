"""
Display chapter count and total duration for each book in a folder.

Usage:
    python stats.py completed
    python stats.py renamed
    python stats.py fixed --root /other/path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mutagen import File as MutagenFile

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def book_stats(book_dir: Path) -> tuple[int, float]:
    """Return (chapter_count, total_seconds) for a book folder."""
    chapters = 0
    total_s  = 0.0
    for f in sorted(book_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXTS:
            continue
        try:
            audio = MutagenFile(str(f))
            if audio and audio.info:
                total_s += audio.info.length
                chapters += 1
        except Exception:
            chapters += 1  # count the file even if duration unreadable
    return chapters, total_s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show chapter count and duration for each book in a folder."
    )
    parser.add_argument("folder", help="Folder to scan (e.g. completed, renamed, fixed)")
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Root directory (default: current directory)")
    args = parser.parse_args()

    scan_dir = (args.root / args.folder).resolve()
    if not scan_dir.is_dir():
        sys.exit(f"Folder not found: {scan_dir}")

    book_dirs = sorted(d for d in scan_dir.iterdir() if d.is_dir())
    if not book_dirs:
        sys.exit(f"No subfolders found in {scan_dir}")

    rows: list[tuple[str, int, float]] = []
    for book_dir in book_dirs:
        chapters, total_s = book_stats(book_dir)
        if chapters > 0:
            rows.append((book_dir.name, chapters, total_s))

    if not rows:
        print("No audio files found.")
        return 0

    # Column widths
    col_name = max(len(r[0]) for r in rows)
    col_name = max(col_name, 4)  # min width for "Book" header

    header = f"{'Book':<{col_name}}   {'Chapters':>8}   {'Duration':>9}   {'Avg/chapter':>11}"
    sep    = "─" * len(header)

    print(f"\n{scan_dir.name}/\n")
    print(header)
    print(sep)

    total_chapters = 0
    total_seconds  = 0.0
    for name, chapters, secs in rows:
        avg = secs / chapters if chapters else 0.0
        print(f"{name:<{col_name}}   {chapters:>8}   {fmt_duration(secs):>9}   {fmt_duration(avg):>11}")
        total_chapters += chapters
        total_seconds  += secs

    print(sep)
    avg_total = total_seconds / total_chapters if total_chapters else 0.0
    print(
        f"{'TOTAL  (' + str(len(rows)) + ' books)':<{col_name}}"
        f"   {total_chapters:>8}"
        f"   {fmt_duration(total_seconds):>9}"
        f"   {fmt_duration(avg_total):>11}"
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
