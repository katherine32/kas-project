"""
Parse all .srt files in data/srt/ into:
  - data/processed/<basename>.txt        (clean, de-timestamped text)
  - data/processed/segments.jsonl        (one JSON object per subtitle segment)
Designed for: UTF-8 SRTs with standard numbering + timestamps.
"""

import os
import json
import srt
from datetime import timedelta
from pathlib import Path
from typing import List
from tqdm import tqdm

RAW_DIR = Path("data/srt")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENTS_JSONL = OUT_DIR / "segments.jsonl"

def read_text(path: Path) -> str:
    # Try utf-8; if BOM or other quirks exist, fall back gracefully.
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # Last resort: read bytes and replace errors
    return path.read_text(encoding="utf-8", errors="replace")

def normalize_lines(text: str) -> str:
    """
    Minimal cleanup for typical SRT artifacts:
    - Collapse awkward line breaks inside a cue into spaces
    - Normalize multiple spaces/newlines
    - Keep punctuation spacing sensible
    """
    # Replace hard line breaks that happen *inside* cues with spaces later (we do this per-cue).
    # Here we just make sure the final doc doesn't have triple+ newlines.
    return "\n".join(line.rstrip() for line in text.splitlines())

def cue_text_to_sentence(cue_text: str) -> str:
    """
    Convert a single cue's text block to a single clean line:
    - Join internal lines with spaces
    - Trim whitespace
    - Keep quotes and punctuation as-is
    """
    # Lines within a cue are typically broken for timing, not sentence boundaries.
    parts = [ln.strip() for ln in cue_text.splitlines() if ln.strip()]
    joined = " ".join(parts)
    # Normalize accidental double spaces
    while "  " in joined:
        joined = joined.replace("  ", " ")
    return joined.strip()

def needs_reparse(raw: Path, cleaned: Path) -> bool:
    """
    Returns True if the raw SRT file is newer than its cleaned TXT,
    or if the cleaned TXT doesn't exist yet.
    """
    return (not cleaned.exists()) or (raw.stat().st_mtime > cleaned.stat().st_mtime)

def duration_to_seconds(td: timedelta) -> float:
    return td.total_seconds()

def parse_single_srt(path: Path) -> dict:
    """
    Returns:
      {
        "filename": "...",
        "text": "<full cleaned text>",
        "segments": [
            {
              "filename": "...",
              "index": 1,
              "start": 0.0,
              "end": 3.2,
              "start_ts": "00:00:00,000",
              "end_ts": "00:00:03,200",
              "text": "My name is ...",
            },
            ...
        ]
      }
    """
    raw = read_text(path)
    raw = normalize_lines(raw)

    # Parse SRT into cues
    subs = list(srt.parse(raw))
    segments = []
    clean_lines: List[str] = []

    for cue in subs:
        clean = cue_text_to_sentence(cue.content)
        if not clean:
            continue
        start_sec = duration_to_seconds(cue.start)
        end_sec = duration_to_seconds(cue.end)
        segments.append({
            "filename": path.name,
            "index": cue.index,
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "start_ts": srt.timedelta_to_srt_timestamp(cue.start),
            "end_ts": srt.timedelta_to_srt_timestamp(cue.end),
            "text": clean,
        })
        clean_lines.append(clean)

    full_text = "\n".join(clean_lines)
    return {"filename": path.name, "text": full_text, "segments": segments}

def write_outputs(parsed: dict):
    # 1) Write cleaned full text per file
    base = Path(parsed["filename"]).stem
    txt_out = OUT_DIR / f"{base}.txt"
    txt_out.write_text(parsed["text"] + "\n", encoding="utf-8")

    # 2) Append segments to a global JSONL (one object per line)
    with SEGMENTS_JSONL.open("a", encoding="utf-8") as f:
        for seg in parsed["segments"]:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

def write_clean_txt(parsed: dict):
    base = Path(parsed["filename"]).stem
    txt_out = OUT_DIR / f"{base}.txt"
    txt_out.write_text(parsed["text"] + "\n", encoding="utf-8")

def append_segments(segments: list):
    with SEGMENTS_JSONL.open("a", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")

def main():
    # Always rebuild the global segments file fresh
    if SEGMENTS_JSONL.exists():
        SEGMENTS_JSONL.unlink()

    srt_files = sorted(p for p in RAW_DIR.glob("*.srt"))
    if not srt_files:
        print(f"No .srt files found in {RAW_DIR.resolve()}")
        return

    print(f"Parsing {len(srt_files)} SRT file(s) from {RAW_DIR} …")
    for path in tqdm(srt_files, ncols=80):
        base = path.stem
        txt_out = OUT_DIR / f"{base}.txt"

        # Parse every file once (cheap) so we can ALWAYS rebuild segments.jsonl
        parsed = parse_single_srt(path)

        # 1) Always append segments
        append_segments(parsed["segments"])

        # 2) Only (re)write the cleaned TXT if needed
        if not needs_reparse(path, txt_out):
            continue
        write_clean_txt(parsed)

    print(f"\nDone. Outputs:")
    print(f"  - Clean text files: {OUT_DIR}/*.txt")
    print(f"  - Segments JSONL:  {SEGMENTS_JSONL}")

if __name__ == "__main__":
    main()