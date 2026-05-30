"""
Rename normalized audiobook recordings by chapter using LLM structured output.

Reads transcriptions from  ./transcribed/<book>.json   (produced by transcribe.py)
Reads source audio from    ./normalized/<book>/*.mp3
Writes to                  ./renamed/<book>/<chapter_name>.mp3

The LLM receives all transcripts for a book at once and returns:
  - whether the book uses numbered chapters
  - for each file: chapter number, title, and continuation flag
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(Path(__file__).parent / ".env", override=True, encoding="utf-8-sig")

OPENAI_MODEL = "gpt-5.4"


# ── Structured output schema ──────────────────────────────────────────────────

class FileLabel(BaseModel):
    filename: str = Field(description="Original filename exactly as provided")
    chapter_number: Optional[int] = Field(
        None,
        description=(
            "Chapter number this file belongs to. "
            "Fill in even for continuations by inferring from surrounding files. "
            "None only if truly indeterminate."
        ),
    )
    chapter_title: Optional[str] = Field(
        None,
        description="Chapter title in lowercase French as spoken. None if no title exists.",
    )
    is_continuation: bool = Field(
        False,
        description="True if the recording starts mid-chapter without a new chapter announcement.",
    )


class BookAnalysis(BaseModel):
    uses_chapter_numbers: bool = Field(
        description="True if the book uses numbered chapters (e.g. 'Chapitre 1', 'Chapter 4')."
    )
    files: List[FileLabel] = Field(
        description="One entry per input file, in the same order as the input list."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    """Remove characters invalid in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*]', "", text).strip()


def build_filename(label: FileLabel, nn: int) -> str:
    title = sanitize(label.chapter_title.lower().strip()) if label.chapter_title else ""
    base = f"Chap{nn:02d}_{title}" if title else f"Chap{nn:02d}"
    return f"{base}_continued" if label.is_continuation else base


def unique_name(stem: str, used: set[str]) -> str:
    """Return stem if unused, otherwise stem(1), stem(2) …"""
    if stem not in used:
        return stem
    i = 1
    while f"{stem}({i})" in used:
        i += 1
    return f"{stem}({i})"


def write_id3_tags(dst: Path, book_name: str, chapter_title: str | None, nn: int, is_continuation: bool) -> None:
    from mutagen.id3 import ID3, TIT2, TALB, TRCK, ID3NoHeaderError

    try:
        tags = ID3(str(dst))
    except ID3NoHeaderError:
        tags = ID3()

    display_title = chapter_title or f"Chapitre {nn:02d}"
    if is_continuation:
        display_title += " (suite)"

    tags.add(TALB(encoding=3, text=book_name))
    tags.add(TRCK(encoding=3, text=str(nn)))
    tags.add(TIT2(encoding=3, text=display_title))
    tags.save(str(dst))


# ── LLM analysis ─────────────────────────────────────────────────────────────

def analyse_book(openai_key: str, book_name: str, transcriptions: list[dict]) -> BookAnalysis:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=openai_key)
    structured_llm = llm.with_structured_output(BookAnalysis)

    file_list = "\n".join(
        f"{i + 1}. {t['filename']}: {t['transcript']!r}"
        for i, t in enumerate(transcriptions)
    )

    system = (
        "Tu analyses des enregistrements audio d'un livre audio en français.\n"
        "On te donne la transcription des premières secondes de chaque fichier.\n"
        "Analyse TOUS les fichiers ensemble pour comprendre la structure du livre.\n\n"
        "Pour chaque fichier, détermine :\n"
        "• Le numéro du chapitre auquel appartient ce fichier "
        "(déduis-le du contexte même pour les continuations)\n"
        "• Le titre du chapitre en minuscules tel qu'il est énoncé "
        "(None si le livre n'a pas de titres)\n"
        "• Si l'enregistrement démarre au milieu d'un chapitre sans annonce "
        "(is_continuation=true)\n\n"
        "Indique aussi si le livre utilise des numéros de chapitres."
    )

    user = f'Livre : "{book_name}"\n\nFichiers dans l\'ordre :\n{file_list}'

    return structured_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])


# ── Book processing ───────────────────────────────────────────────────────────

def process_book(
    openai_key: str,
    json_file: Path,
    normalized_root: Path,
    out_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int]:
    data = json.loads(json_file.read_text(encoding="utf-8"))
    book_name: str = data["book"]
    transcriptions: list[dict] = data["transcriptions"]

    if not transcriptions:
        print("  no transcriptions found, skipping")
        return 0, 0

    print(f"  Analysing {len(transcriptions)} files with LLM…", flush=True)
    analysis = analyse_book(openai_key, book_name, transcriptions)
    print(f"  uses_chapter_numbers = {analysis.uses_chapter_numbers}\n")

    labels = {lbl.filename: lbl for lbl in analysis.files}
    book_src = normalized_root / book_name
    book_dst = out_root / book_name

    if not dry_run:
        book_dst.mkdir(parents=True, exist_ok=True)

    # Pre-compute NN for every file.
    # Uses chapter_number from LLM when the book has numbered chapters,
    # otherwise builds a sequential index from title occurrence order.
    chapter_seq: dict[str, int] = {}
    seq_counter = 0
    current_nn = 0
    file_nn: dict[str, int] = {}

    for t in transcriptions:
        lbl = labels.get(t["filename"])
        if lbl is None:
            file_nn[t["filename"]] = 0
            continue
        if analysis.uses_chapter_numbers and lbl.chapter_number is not None:
            current_nn = lbl.chapter_number
        elif not lbl.is_continuation:
            key = lbl.chapter_title or t["filename"]
            if key not in chapter_seq:
                seq_counter += 1
                chapter_seq[key] = seq_counter
            current_nn = chapter_seq[key]
        file_nn[t["filename"]] = current_nn

    used_names: set[str] = set()
    ok = failed = 0

    for t in transcriptions:
        filename = t["filename"]
        src = book_src / filename
        label = labels.get(filename)

        try:
            if label is None:
                raise ValueError(f"LLM returned no label for {filename}")
            if not src.exists():
                raise FileNotFoundError(f"Source not found: {src}")

            nn = file_nn.get(filename, 0)
            stem = build_filename(label, nn)
            final = unique_name(stem, used_names)
            used_names.add(final)

            dup_note = f"  [was: {stem}]" if final != stem else ""
            print(f"  {filename}  →  {final}{dup_note}")

            if not dry_run:
                dst = book_dst / (final + ".mp3")
                if dst.exists() and not overwrite:
                    print(f"    skipped (file already exists)")
                else:
                    shutil.copy2(src, dst)
                    write_id3_tags(dst, book_name, label.chapter_title, nn, label.is_continuation)

            ok += 1

        except Exception as exc:
            print(f"  {filename}  ERROR: {exc}")
            failed += 1

    return ok, failed


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename normalized audio files by chapter using LLM structured output."
    )
    parser.add_argument("--transcribed", type=Path, default=Path("transcribed"),
                        help="folder containing JSON files from transcribe.py (default: transcribed)")
    parser.add_argument("--normalized", type=Path, default=Path("normalized"),
                        help="source audio root (default: normalized)")
    parser.add_argument("--output", type=Path, default=Path("renamed"),
                        help="output root (default: renamed)")
    parser.add_argument("--book", type=str, default=None,
                        help="process only this book (folder name, e.g. \"Heidi jeune fille\")")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite output files that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the renaming plan without copying any files")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        sys.exit("OPENAI_API_KEY not set in .env")

    transcribed_root = args.transcribed.resolve()
    if not transcribed_root.is_dir():
        sys.exit(f"Transcribed folder not found: {transcribed_root}. Run transcribe.py first.")

    if args.book:
        json_file = transcribed_root / f"{args.book}.json"
        if not json_file.exists():
            sys.exit(f"No transcription found for \"{args.book}\" (expected: {json_file})")
        json_files = [json_file]
    else:
        json_files = sorted(transcribed_root.glob("*.json"))
        if not json_files:
            sys.exit(f"No JSON files found in {transcribed_root}. Run transcribe.py first.")

    total_ok = total_failed = 0
    for json_file in json_files:
        print(f"\n=== {json_file.stem} ===")
        ok, failed = process_book(
            openai_key,
            json_file,
            args.normalized.resolve(),
            args.output.resolve(),
            args.overwrite,
            args.dry_run,
        )
        total_ok += ok
        total_failed += failed

    print(f"\nDone. {total_ok} renamed, {total_failed} failed.")
    if args.dry_run:
        print("(dry-run: no files were copied)")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
