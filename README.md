# Audiobook Recording Pipeline

A set of Python scripts to process home audiobook recordings captured on a Zoom recorder. The pipeline normalises loudness, transcribes the first seconds of each file with Deepgram, uses an LLM to identify chapters and rename the files, then optionally merges split recordings back into single chapter files.

## Pipeline overview

```
input/<book>/ZOOM00xx.MP3
        │
        ▼
   normalize.py  ──► normalized/<book>/ZOOM00xx.mp3   (–16 LUFS, two-pass loudnorm)
        │
        ▼
  transcribe.py  ──► transcribed/<book>.json           (first + last 20 s, Deepgram nova-2)
        │
        ▼
    rename.py    ──► renamed/<book>/ChapNN_title.mp3   (LLM chapter detection + ID3 tags)
        │
        ▼  (optional)
    fixer.py     ──► fixed/<book>/ChapNN_title.mp3     (merge continued recordings, remove overlap)
```

Run all three main steps at once with **`pipeline.py`**.  
Inspect library statistics (chapter counts, durations) with **`stats.py`**.

---

## Prerequisites

| Dependency | Install |
|---|---|
| Python 3.11+ | [python.org](https://www.python.org/) |
| ffmpeg | `winget install Gyan.FFmpeg` (Windows) or `brew install ffmpeg` (macOS) |
| Deepgram account | [deepgram.com](https://deepgram.com) — free tier available |
| OpenAI account | [platform.openai.com](https://platform.openai.com) |

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd stories

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install Python dependencies
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root (never commit this file):

```env
DEEPGRAM_API_KEY=your_deepgram_key_here
OPENAI_API_KEY=your_openai_key_here
```

The model used for chapter detection is set at the top of [rename.py](rename.py) and [fixer.py](fixer.py):

```python
OPENAI_MODEL = "gpt-5.4"   # change to any OpenAI model you have access to
```

---

## Folder structure

```
stories/
├── input/                  # place your raw recordings here
│   └── <book title>/
│       └── ZOOM00xx.MP3
├── normalized/             # output of normalize.py
├── transcribed/            # output of transcribe.py  (JSON files)
├── renamed/                # output of rename.py
├── fixed/                  # output of fixer.py
│
├── normalize.py
├── transcribe.py
├── rename.py
├── fixer.py
├── pipeline.py
├── stats.py
├── requirements.txt
└── .env                    # not committed
```

---

## Scripts

### `normalize.py` — Loudness normalisation

Normalises every recording to **–16 LUFS** using ffmpeg's two-pass `loudnorm` filter. Output files are re-encoded as MP3 (VBR quality 2, 48 kHz).

```bash
python normalize.py                          # all books
python normalize.py --book "Heidi grandit"   # one book only
python normalize.py --overwrite              # re-process existing files
python normalize.py --workers 8              # more parallel ffmpeg jobs
```

| Argument | Default | Description |
|---|---|---|
| `--book` | *(all)* | Process only this subfolder |
| `--input` | `input` | Root folder for raw recordings |
| `--output` | `normalized` | Root folder for output |
| `--workers` | `4` | Number of parallel ffmpeg jobs |
| `--overwrite` | off | Re-process files that already exist |

---

### `transcribe.py` — Deepgram transcription

Transcribes the **first and last 20 seconds** of each normalised file using [Deepgram](https://deepgram.com) nova-2 (French). Results are saved as one JSON file per book in `transcribed/`.

- `transcript` — beginning of the file, used by `rename.py` for chapter detection.
- `transcript_end` — end of the file, informational: verify the recording is complete, or confirm it feeds into a `_continued` file.

After `rename.py` runs, a `renamed_file` field is written back into each entry as an audit trail.

```bash
python transcribe.py                          # all books
python transcribe.py --book "Heidi grandit"   # one book only
python transcribe.py --overwrite              # re-transcribe existing JSONs
```

| Argument | Default | Description |
|---|---|---|
| `--book` | *(all)* | Process only this subfolder |
| `--input` | `normalized` | Root folder for source audio |
| `--output` | `transcribed` | Root folder for JSON output |
| `--workers` | `4` | Parallel Deepgram requests per book |
| `--overwrite` | off | Re-transcribe even if JSON exists |

**Output format** (`transcribed/<book>.json`):
```json
{
  "book": "Heidi grandit",
  "snippet_seconds": 20,
  "transcriptions": [
    {
      "filename": "ZOOM0006.mp3",
      "transcript":     "Chapitre 4, heureuse nouvelle. Tony a rencontré le docteur.",
      "transcript_end": "…et le docteur repartit vers la vallée.",
      "renamed_file":   "Chap04_heureuse nouvelle.mp3"
    },
    …
  ]
}
```

> `renamed_file` is `null` for files that failed during renaming, and absent until `rename.py` has been run.

---

### `rename.py` — LLM-based chapter renaming

Sends all transcripts for a book to an LLM in a single call. The LLM returns structured output (via LangChain) identifying:
- whether the book uses numbered chapters
- for each file: chapter number, chapter title, and whether it is a continuation of the previous chapter

Files are then copied to `renamed/<book>/` with names like `Chap04_heureuse nouvelle.mp3`, and **ID3 tags** are written (album, title, track).

**Naming convention:**

| Case | Filename |
|---|---|
| New chapter with title | `Chap04_heureuse nouvelle.mp3` |
| New chapter, no title | `Chap04.mp3` |
| Continuation | `Chap04_heureuse nouvelle_continued.mp3` |
| Duplicate take | `Chap04_heureuse nouvelle(1).mp3` |
| Book without chapter numbers | `Chap01_titre.mp3` (sequential index) |

```bash
python rename.py --dry-run                          # preview without copying
python rename.py                                    # rename all books
python rename.py --book "Heidi grandit"             # one book only
python rename.py --book "Heidi grandit" --dry-run   # preview one book
python rename.py --overwrite                        # overwrite existing output
```

| Argument | Default | Description |
|---|---|---|
| `--book` | *(all)* | Process only this book |
| `--transcribed` | `transcribed` | Folder containing JSON files |
| `--normalized` | `normalized` | Source audio root |
| `--output` | `renamed` | Output root |
| `--dry-run` | off | Show plan without copying |
| `--overwrite` | off | Overwrite existing output files |

---

### `fixer.py` — Merge split chapters

When a recording session was interrupted mid-chapter, the recorder creates two files: the base chapter and a `_continued` file. `fixer.py` merges them back into one, detecting and removing any overlapping text that was re-read at the start of the continuation.

**Algorithm per chapter pair:**
1. Extract the last 60 s of the current file and the first 60 s of the continuation.
2. Transcribe both with Deepgram (word-level timestamps).
3. Ask the LLM if there is overlap; if so, identify where in the current file the overlap begins.
4. Cut the current file at that point and concatenate with the full continuation.

```bash
python fixer.py "Heidi grandit"
python fixer.py "Heidi grandit" --input renamed --output fixed
```

| Argument | Default | Description |
|---|---|---|
| `book` | *(required)* | Book folder name |
| `--input` | `renamed` | Root folder containing renamed chapters |
| `--output` | `fixed` | Root folder for merged output |

---

### `stats.py` — Library statistics

Scans all book subfolders in a given folder and displays chapter count, total duration, and average chapter duration per book.

```bash
python stats.py completed
python stats.py renamed
python stats.py fixed
```

Example output:

```
completed/

Book                                      Chapters    Duration   Avg/chapter
────────────────────────────────────────────────────────────────────────────
Book 1                                          18    4:23:15       0:14:38
Book 2                                          17    3:58:42       0:14:03
Book 3 - by John Smith                          22    5:12:43       0:14:11
────────────────────────────────────────────────────────────────────────────
TOTAL  (3 books)                                57   13:34:40       0:14:17
```

| Argument | Default | Description |
|---|---|---|
| `folder` | *(required)* | Subfolder to scan (e.g. `completed`, `renamed`) |
| `--root` | `.` | Root directory if different from project folder |

---

### `pipeline.py` — Full pipeline

Runs `normalize → transcribe → rename` for a single book in one command.

```bash
python pipeline.py "Heidi grandit"

# Skip steps already done
python pipeline.py "Heidi grandit" --start-from transcribe
python pipeline.py "Heidi grandit" --start-from rename

# Preview rename without copying
python pipeline.py "Heidi grandit" --dry-run

# Re-run everything from scratch
python pipeline.py "Heidi grandit" --overwrite
```

| Argument | Default | Description |
|---|---|---|
| `book` | *(required)* | Book folder name |
| `--start-from` | `normalize` | Skip to `transcribe` or `rename` |
| `--overwrite` | off | Pass `--overwrite` to each step |
| `--dry-run` | off | Pass `--dry-run` to the rename step |

---

## Typical workflow

```bash
# 1. Drop raw recordings into input/<book>/
#    (ZOOM recorder files named ZOOM0001.MP3, ZOOM0002.MP3, …)

# 2. Run the full pipeline
python pipeline.py "Heidi grandit"

# 3. Check the rename plan before committing
python rename.py --book "Heidi grandit" --dry-run

# 4. If some chapters were split across recordings, merge them
python fixer.py "Heidi grandit"

# 5. Check statistics for the final output
python stats.py completed
```

---

## Notes

- **Language**: transcription and chapter detection are configured for **French**. Change `"language": "fr"` in `transcribe.py` and the system prompts in `rename.py` / `fixer.py` to adapt to another language.
- **Models**: the Deepgram model is `nova-2`. The OpenAI model is set via `OPENAI_MODEL` in `rename.py` and `fixer.py`.
- **Transcription cache**: the JSON files in `transcribed/` persist between runs — `rename.py` reads them without re-calling Deepgram, and writes the `renamed_file` audit trail back into them.
- **ID3 tags**: `rename.py` writes `TALB` (album = book folder name), `TIT2` (title = chapter title), and `TRCK` (track = chapter number) to every output file.
