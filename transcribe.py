"""
Transcribe the first and last 20 s of each normalized audio file using Deepgram nova-2.

Reads from   ./normalized/<book>/*.mp3  (or other audio)
Writes to    ./transcribed/<book>.json  (one JSON per book)

Each entry contains:
  transcript      – first 20 s  (used by rename.py for chapter detection)
  transcript_end  – last 20 s   (informational: verify completeness, spot continuations)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mutagen.mp3 import MP3
import os

load_dotenv(Path(__file__).parent / ".env", override=True, encoding="utf-8-sig")

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}
SNIPPET_SECONDS = 20


def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not found. Install it from https://ffmpeg.org/ and add to PATH.")
    return exe


def extract_snippet(ffmpeg: str, src: Path, duration: int = SNIPPET_SECONDS) -> bytes:
    """Return first N seconds of audio as 16 kHz mono WAV bytes."""
    cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-y",
        "-i", str(src),
        "-t", str(duration),
        "-ar", "16000", "-ac", "1",
        "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-300:].decode(errors='replace')}")
    return proc.stdout


def extract_tail(ffmpeg: str, src: Path, duration: int = SNIPPET_SECONDS) -> bytes:
    """Return last N seconds of audio as 16 kHz mono WAV bytes."""
    total = MP3(str(src)).info.length
    start = max(0.0, total - duration)
    cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-y",
        "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{min(duration, total):.3f}",
        "-ar", "16000", "-ac", "1",
        "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg tail failed: {proc.stderr[-300:].decode(errors='replace')}")
    return proc.stdout


def transcribe_deepgram(api_key: str, audio_bytes: bytes) -> str:
    resp = httpx.post(
        "https://api.deepgram.com/v1/listen",
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
        params={"model": "nova-2", "language": "fr", "smart_format": "true"},
        content=audio_bytes,
        timeout=30,
    )
    resp.raise_for_status()
    try:
        return resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError):
        return ""


def transcribe_file(api_key: str, ffmpeg: str, src: Path) -> dict:
    transcript     = transcribe_deepgram(api_key, extract_snippet(ffmpeg, src))
    transcript_end = transcribe_deepgram(api_key, extract_tail(ffmpeg, src))
    return {"filename": src.name, "transcript": transcript, "transcript_end": transcript_end}


def process_book(
    api_key: str,
    ffmpeg: str,
    book_dir: Path,
    out_file: Path,
    overwrite: bool,
    workers: int,
) -> None:
    if out_file.exists() and not overwrite:
        print(f"  skipped (already exists: {out_file.name})")
        return

    files = sorted(
        p for p in book_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not files:
        print("  no audio files found")
        return

    # Use index map so results stay sorted despite as_completed ordering
    results: list[dict | None] = [None] * len(files)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        idx_map = {
            pool.submit(transcribe_file, api_key, ffmpeg, src): i
            for i, src in enumerate(files)
        }
        done = 0
        for fut in as_completed(idx_map):
            i = idx_map[fut]
            done += 1
            try:
                results[i] = fut.result()
                preview = results[i]["transcript"]
                if len(preview) > 70:
                    preview = preview[:70] + "…"
                print(f"  [{done}/{len(files)}] {files[i].name}: {preview!r}")
            except Exception as exc:
                results[i] = {"filename": files[i].name, "transcript": "", "error": str(exc)}
                print(f"  [{done}/{len(files)}] {files[i].name}: ERROR {exc}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "book": book_dir.name,
        "snippet_seconds": SNIPPET_SECONDS,
        "transcriptions": results,
    }
    out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved → {out_file.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe first 20 s of each normalized audio file using Deepgram."
    )
    parser.add_argument("--input", type=Path, default=Path("normalized"), help="normalized root (default: normalized)")
    parser.add_argument("--output", type=Path, default=Path("transcribed"), help="output folder for JSON files (default: transcribed)")
    parser.add_argument("--overwrite", action="store_true", help="re-transcribe books that already have a JSON")
    parser.add_argument("--workers", type=int, default=4, help="parallel Deepgram requests per book (default: 4)")
    parser.add_argument("--book", type=str, default=None, help="process only this book (folder name)")
    args = parser.parse_args()

    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        sys.exit("DEEPGRAM_API_KEY not set in .env")

    ffmpeg = check_ffmpeg()
    in_root = args.input.resolve()
    if not in_root.is_dir():
        sys.exit(f"Input folder not found: {in_root}")

    if args.book:
        book_dir = in_root / args.book
        if not book_dir.is_dir():
            sys.exit(f"Book folder not found: {book_dir}")
        book_dirs = [book_dir]
    else:
        book_dirs = sorted(d for d in in_root.iterdir() if d.is_dir())
        if not book_dirs:
            sys.exit(f"No book subfolders found under {in_root}")

    for book_dir in book_dirs:
        print(f"\n=== {book_dir.name} ===")
        out_file = args.output.resolve() / f"{book_dir.name}.json"
        process_book(api_key, ffmpeg, book_dir, out_file, args.overwrite, args.workers)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
