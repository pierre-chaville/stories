"""
Merge audiobook chapter files with their _continued counterparts,
detecting and removing any re-read overlap.

For each chapter that has _continued files:
  1. Transcribe last 60 s of the running file and first 60 s of the next part
     (Deepgram nova-2, with word-level timestamps)
  2. Ask the LLM whether the end of part N overlaps the beginning of part N+1
  3. If overlap: locate the exact cut time using word timestamps
  4. Concatenate with ffmpeg (trimming the overlap from the continuation)

Reads from   ./renamed/<book>/
Writes to    ./fixed/<book>/

Usage:
  python fixer.py "Heidi jeune fille"
  python fixer.py "Heidi jeune fille" --input renamed --output fixed
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
import os
from mutagen.mp3 import MP3
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(Path(__file__).parent / ".env", override=True, encoding="utf-8-sig")

OPENAI_MODEL = "gpt-5.4"
TAIL_SECONDS = 60
HEAD_SECONDS = 60


# ── Structured output schema ──────────────────────────────────────────────────

class OverlapAnalysis(BaseModel):
    has_overlap: bool = Field(
        description=(
            "True if the end of transcript 1 and the beginning of transcript 2 "
            "contain the same text read twice (overlap / re-read)."
        )
    )
    overlap_start_in_file1: Optional[str] = Field(
        None,
        description=(
            "The first 8-10 words in transcript 1 where the overlap with transcript 2 begins "
            "(i.e., the point where file 1 should be cut). "
            "Copy them verbatim from transcript 1. None if no overlap."
        ),
    )


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not found. Install it from https://ffmpeg.org/ and add to PATH.")
    return exe


def audio_duration(src: Path) -> float:
    return MP3(str(src)).info.length


def extract_tail(ffmpeg: str, src: Path, duration: float = TAIL_SECONDS) -> bytes:
    """Return last N seconds of src as 16 kHz mono WAV bytes."""
    total = audio_duration(src)
    start = max(0.0, total - duration)
    cmd = [
        ffmpeg, "-hide_banner", "-nostats",
        "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{min(duration, total):.3f}",
        "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg tail failed: {proc.stderr[-300:].decode(errors='replace')}")
    return proc.stdout


def extract_head(ffmpeg: str, src: Path, duration: float = HEAD_SECONDS) -> bytes:
    """Return first N seconds of src as 16 kHz mono WAV bytes."""
    cmd = [
        ffmpeg, "-hide_banner", "-nostats",
        "-i", str(src), "-t", f"{duration:.3f}",
        "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg head failed: {proc.stderr[-300:].decode(errors='replace')}")
    return proc.stdout


def ffmpeg_concat(ffmpeg: str, file1: Path, file2: Path, cut_end: float | None, dst: Path) -> None:
    """Concatenate file1 (optionally trimmed at cut_end seconds) with all of file2."""
    if cut_end is not None:
        filter_str = (
            f"[0:a]atrim=end={cut_end:.3f}[t];"
            "[t][1:a]concat=n=2:v=0:a=1[out]"
        )
    else:
        filter_str = "[0:a][1:a]concat=n=2:v=0:a=1[out]"
    cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-y",
        "-i", str(file1), "-i", str(file2),
        "-filter_complex", filter_str,
        "-map", "[out]", "-map_metadata", "0",
        "-c:a", "libmp3lame", "-q:a", "2", "-ar", "48000",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {proc.stderr[-600:]}")


# ── Deepgram ──────────────────────────────────────────────────────────────────

def transcribe_with_words(api_key: str, audio_bytes: bytes) -> tuple[str, list[dict]]:
    """Return (transcript, words) where each word has 'word', 'start', 'end'."""
    resp = httpx.post(
        "https://api.deepgram.com/v1/listen",
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
        params={"model": "nova-2", "language": "fr", "smart_format": "true", "words": "true"},
        content=audio_bytes,
        timeout=60,
    )
    resp.raise_for_status()
    try:
        alt = resp.json()["results"]["channels"][0]["alternatives"][0]
        return alt.get("transcript", ""), alt.get("words", [])
    except (KeyError, IndexError):
        return "", []


# ── LLM overlap detection ─────────────────────────────────────────────────────

def detect_overlap(llm_key: str, end_transcript: str, start_transcript: str) -> OverlapAnalysis:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=llm_key)
    structured = llm.with_structured_output(OverlapAnalysis)

    system = (
        "Tu analyses deux transcriptions consécutives d'un livre audio en français.\n"
        "Transcription 1 = FIN du fichier précédent (dernière minute).\n"
        "Transcription 2 = DÉBUT du fichier suivant (première minute).\n\n"
        "Détermine s'il y a un chevauchement : le narrateur a relu au début du fichier 2 "
        "des passages déjà présents à la fin du fichier 1.\n"
        "Si oui, indique verbatim (mot pour mot depuis la transcription 1) "
        "les premiers mots dans la transcription 1 où commence le chevauchement "
        "(c'est-à-dire le point où il faut couper le fichier 1).\n"
        "Si non, overlap_start_in_file1 = null."
    )

    user = (
        f"Transcription 1 (fin du fichier précédent) :\n"
        f"{end_transcript or '(silence)'}\n\n"
        f"Transcription 2 (début du fichier suivant) :\n"
        f"{start_transcript or '(silence)'}"
    )

    return structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])


# ── Cut time resolution ───────────────────────────────────────────────────────

def find_cut_time(words: list[dict], first_new_text: str) -> float:
    """Find start time (seconds) of first_new_text in the word list."""
    if not first_new_text or not words:
        return 0.0

    def norm(w: str) -> str:
        return re.sub(r"[^\w]", "", w.lower())

    search = [norm(w) for w in first_new_text.split()[:8] if norm(w)]
    if not search:
        return 0.0

    word_norms = [norm(w["word"]) for w in words]

    # Try matching the full phrase first, then fall back to fewer words
    for length in range(len(search), 0, -1):
        seq = search[:length]
        for i in range(len(word_norms) - length + 1):
            if word_norms[i : i + length] == seq:
                return words[i]["start"]

    return 0.0


# ── Chapter grouping ──────────────────────────────────────────────────────────

def _continued_sort_key(path: Path) -> tuple[int, int]:
    """Sort continued files: _continued first (0), then _continued(1), _continued(2)…"""
    m = re.search(r"_continued\((\d+)\)$", path.stem)
    return (0, int(m.group(1)) if m else 0)


def group_chapters(book_dir: Path) -> dict[str, tuple[list[Path], list[Path]]]:
    """
    Returns {ChapNN: (base_files, continued_files)}.
    base_files   = files without _continued (sorted alphabetically)
    continued_files = files with _continued (sorted: plain first, then (1), (2)…)
    """
    groups: dict[str, tuple[list[Path], list[Path]]] = {}

    for f in sorted(book_dir.glob("*.mp3")):
        m = re.match(r"(Chap\d+)", f.stem)
        key = m.group(1) if m else f.stem
        if key not in groups:
            groups[key] = ([], [])
        if "_continued" in f.stem:
            groups[key][1].append(f)
        else:
            groups[key][0].append(f)

    for key in groups:
        groups[key][1].sort(key=_continued_sort_key)

    return groups


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_chapter(
    ffmpeg: str,
    deepgram_key: str,
    llm_key: str,
    files: list[Path],  # [base, cont1, cont2, …]
    dst: Path,
    tmpdir: str,
) -> None:
    """Iteratively merge files into dst."""
    current = files[0]

    for idx, next_file in enumerate(files[1:], 1):
        print(f"    [{idx}/{len(files)-1}] + {next_file.name}")

        # Transcribe boundary regions
        print("      transcribing boundary…", end="", flush=True)
        tail_bytes = extract_tail(ffmpeg, current)
        head_bytes = extract_head(ffmpeg, next_file)
        end_tx, end_words = transcribe_with_words(deepgram_key, tail_bytes)
        start_tx, _       = transcribe_with_words(deepgram_key, head_bytes)
        print(" done")

        # Detect overlap
        print("      detecting overlap…", end="", flush=True)
        analysis = detect_overlap(llm_key, end_tx, start_tx)
        print(f" has_overlap={analysis.has_overlap}")

        # Determine where to cut the END of file1
        cut_end: float | None = None

        if analysis.has_overlap and analysis.overlap_start_in_file1:
            # Timestamps in end_words are relative to the tail snippet start
            cur_dur    = audio_duration(current)
            tail_start = max(0.0, cur_dur - TAIL_SECONDS)
            time_in_tail = find_cut_time(end_words, analysis.overlap_start_in_file1)
            cut_end = tail_start + time_in_tail

            if cut_end <= 0 or cut_end >= cur_dur:
                print(f"      WARNING: cut_end {cut_end:.1f}s out of range — simple concatenation")
                cut_end = None
            else:
                print(f"      cut file1 at {cut_end:.2f}s  ({analysis.overlap_start_in_file1!r})")
        else:
            print("      no overlap — simple concatenation")

        merged = Path(tmpdir) / f"merged_{idx}.mp3"
        ffmpeg_concat(ffmpeg, current, next_file, cut_end, merged)
        current = merged

    shutil.copy2(current, dst)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge chapter files with their continuations, removing overlaps."
    )
    parser.add_argument("book", help="Book folder name (e.g. \"Heidi jeune fille\")")
    parser.add_argument("--input",  type=Path, default=Path("renamed"), help="input root  (default: renamed)")
    parser.add_argument("--output", type=Path, default=Path("fixed"),   help="output root (default: fixed)")
    args = parser.parse_args()

    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
    llm_key      = os.getenv("OPENAI_API_KEY", "")
    if not deepgram_key:
        sys.exit("DEEPGRAM_API_KEY not set in .env")
    if not llm_key:
        sys.exit("OPENAI_API_KEY not set in .env")

    ffmpeg    = check_ffmpeg()
    book_dir  = args.input.resolve()  / args.book
    out_dir   = args.output.resolve() / args.book

    if not book_dir.is_dir():
        sys.exit(f"Book folder not found: {book_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    groups = group_chapters(book_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        for chapter_key, (base_files, cont_files) in groups.items():

            if not cont_files:
                # No continuations — copy all files unchanged
                for f in base_files:
                    shutil.copy2(f, out_dir / f.name)
                    print(f"  {f.name}  →  copied")
                continue

            if not base_files:
                print(f"  WARNING: no base file for {chapter_key}, copying continuations as-is")
                for f in cont_files:
                    shutil.copy2(f, out_dir / f.name)
                continue

            # Merge base + all continuations into one file
            to_merge = [base_files[0]] + cont_files
            dst = out_dir / base_files[0].name
            print(f"\n  {chapter_key}: merging {len(to_merge)} file(s)")
            for f in to_merge:
                print(f"    - {f.name}")

            try:
                merge_chapter(ffmpeg, deepgram_key, llm_key, to_merge, dst, tmpdir)
                print(f"  → {dst.name}")
            except Exception as exc:
                print(f"  ERROR merging {chapter_key}: {exc}")
                # Fall back: copy original files unchanged
                for f in to_merge:
                    shutil.copy2(f, out_dir / f.name)

            # Copy any extra base-file duplicates as-is (Chap04_title(1).mp3 etc.)
            for dup in base_files[1:]:
                shutil.copy2(dup, out_dir / dup.name)
                print(f"  {dup.name}  →  copied (duplicate take)")

    print(f"\nDone. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
