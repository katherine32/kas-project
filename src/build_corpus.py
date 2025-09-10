# src/build_corpus.py
from pathlib import Path
import json, hashlib, os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.app import (
    SEGMENTS_JSONL, EMBED_MODEL,
    load_all_segments_grouped, chunk_all_recordings,
    embed_texts, build_index_np,
)

# Index artifact paths (import from app if present; otherwise default here)
try:
    from src.app import INDEX_DIR, CHUNKS_PARQUET, EMBS_NPY, NORMED_NPY
except Exception:
    INDEX_DIR = Path("data/index")
    CHUNKS_PARQUET = INDEX_DIR / "chunks.parquet"
    EMBS_NPY = INDEX_DIR / "embeddings.npy"
    NORMED_NPY = INDEX_DIR / "embeddings_normed.npy"

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_openai_client_cli():
    """Minimal, non-Streamlit client for CLI use."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment or .env")
    from openai import OpenAI as _OpenAI
    return _OpenAI(api_key=api_key, base_url=base_url) if base_url else _OpenAI(api_key=api_key)

def main():
    if not SEGMENTS_JSONL.exists():
        raise SystemExit(f"Missing {SEGMENTS_JSONL}. Put transcripts in place first.")

    client = ensure_openai_client_cli()

    grouped = load_all_segments_grouped()
    chunks  = chunk_all_recordings(grouped)
    texts   = [c["text"] for c in chunks]

    embs    = embed_texts(client, texts).astype("float32")
    np_idx  = build_index_np(embs.copy())

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(chunks).to_parquet(CHUNKS_PARQUET)
    np.save(EMBS_NPY, embs)
    np.save(NORMED_NPY, np_idx)

    seg_sha = _sha256_file(SEGMENTS_JSONL)
    manifest = {
        "embed_model": EMBED_MODEL,
        "segments_sha256": seg_sha,
        "n_chunks": len(chunks),
        "dim": int(embs.shape[1]),
    }
    (INDEX_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Built index for {len(chunks)} chunks → {INDEX_DIR}")

if __name__ == "__main__":
    main()