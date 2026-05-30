"""
Normalize audio recordings to -16 LUFS using ffmpeg's two-pass loudnorm filter.

Reads from   ./input/<book>/*.mp3 (or other audio)
Writes to    ./normalized/<book>/*.mp3   (mirrors subfolder structure)

Two-pass loudnorm gives more accurate normalization than single-pass:
  pass 1: measure integrated loudness, true peak, LRA, threshold
  pass 2: apply linear normalization using the measured values
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TARGET_I = -16.0   # integrated loudness (LUFS)
TARGET_TP = -1.5   # true peak (dBTP)
TARGET_LRA = 11.0  # loudness range (LU)

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}


def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not found in PATH. Install it from https://ffmpeg.org/ and retry.")
    return exe


def measure(ffmpeg: str, src: Path) -> dict:
    """First pass: measure loudness stats. Returns the JSON block ffmpeg prints."""
    cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-i", str(src),
        "-af", f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg measure failed for {src.name}:\n{proc.stderr[-2000:]}")
    # ffmpeg writes the JSON to stderr, usually as the final {...} block
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr, re.DOTALL)
    if not match:
        raise RuntimeError(f"could not parse loudnorm JSON for {src.name}:\n{proc.stderr[-2000:]}")
    return json.loads(match.group(0))


def apply(ffmpeg: str, src: Path, dst: Path, stats: dict) -> None:
    """Second pass: apply linear normalization using measured stats."""
    af = (
        f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}"
        f":measured_I={stats['input_i']}"
        f":measured_TP={stats['input_tp']}"
        f":measured_LRA={stats['input_lra']}"
        f":measured_thresh={stats['input_thresh']}"
        f":offset={stats['target_offset']}"
        f":linear=true:print_format=summary"
    )
    cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-y", "-i", str(src),
        "-af", af,
        "-ar", "48000",   # loudnorm resamples to 192k internally; pin output rate
        "-c:a", "libmp3lame", "-q:a", "2",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg apply failed for {src.name}:\n{proc.stderr[-2000:]}")


def normalize_file(ffmpeg: str, src: Path, dst: Path, overwrite: bool) -> tuple[Path, str]:
    if dst.exists() and not overwrite:
        return src, "skipped (exists)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    stats = measure(ffmpeg, src)
    apply(ffmpeg, src, dst, stats)
    return src, f"ok (input_i={stats['input_i']} LUFS -> {TARGET_I})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize audio to -16 LUFS with ffmpeg loudnorm.")
    parser.add_argument("--input", type=Path, default=Path("input"), help="input root folder")
    parser.add_argument("--output", type=Path, default=Path("normalized"), help="output root folder")
    parser.add_argument("--workers", type=int, default=4, help="parallel ffmpeg jobs")
    parser.add_argument("--overwrite", action="store_true", help="re-process files that already exist in output")
    parser.add_argument("--book", type=str, default=None, help="process only this book (folder name)")
    args = parser.parse_args()

    ffmpeg = check_ffmpeg()
    in_root: Path = args.input.resolve()
    out_root: Path = args.output.resolve()

    if not in_root.is_dir():
        sys.exit(f"input folder not found: {in_root}")

    if args.book:
        book_dir = in_root / args.book
        if not book_dir.is_dir():
            sys.exit(f"book folder not found: {book_dir}")
        sources = [p for p in book_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    else:
        sources = [p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if not sources:
        sys.exit(f"no audio files found under {in_root}")

    print(f"Found {len(sources)} audio file(s). Target: {TARGET_I} LUFS, {args.workers} workers.\n")

    tasks = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for src in sources:
            rel = src.relative_to(in_root)
            dst = (out_root / rel).with_suffix(".mp3")
            tasks.append(pool.submit(normalize_file, ffmpeg, src, dst, args.overwrite))

        done = 0
        failed = 0
        for fut in as_completed(tasks):
            done += 1
            try:
                src, status = fut.result()
                print(f"[{done}/{len(tasks)}] {src.relative_to(in_root)} -> {status}")
            except Exception as e:
                failed += 1
                print(f"[{done}/{len(tasks)}] ERROR: {e}")

    print(f"\nDone. {len(tasks) - failed} succeeded, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
