# src/app.py
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
import io, zipfile, srt
from datetime import timedelta


# --- OpenAI SDK v1 ---
try:
    from openai import OpenAI
    _HAS_OPENAI_V1 = True
except Exception:
    _HAS_OPENAI_V1 = False

# ----------------------------
# Config
# ----------------------------
DATA_DIR = Path("data/processed")
SEGMENTS_JSONL = DATA_DIR / "segments.jsonl"
EMBED_MODEL = "text-embedding-3-small"   # fast + inexpensive
CHAT_MODEL = "gpt-4o-mini"               # good quality for demos
TOP_K = 5
CHUNK_TARGET_CHARS = 900
CHUNK_OVERLAP_CHARS = 180

def ensure_segments_on_disk():
    """
    If segments.jsonl is missing, let the user upload either:
    - segments.jsonl, or
    - a single .srt, or
    - a .zip containing many .srt files.
    Then build segments.jsonl here on the server.
    """
    if SEGMENTS_JSONL.exists():
        return

    st.info("No transcript data found. Upload your **segments.jsonl**, a single **.srt**, or a **.zip** of SRTs to begin.")
    uploaded = st.file_uploader("Upload segments.jsonl / .srt / .zip", type=["jsonl", "srt", "zip"])
    if uploaded is None:
        st.stop()

    name = uploaded.name.lower()

    if name.endswith(".jsonl"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "segments.jsonl").write_bytes(uploaded.getvalue())
        st.success("segments.jsonl saved. Click **Rerun**.")
        st.stop()

    elif name.endswith(".srt"):
        recs = _parse_srt_bytes(uploaded.name, uploaded.getvalue())
        if not recs:
            st.error("That .srt file contained no records.")
            st.stop()
        _write_segments_jsonl(recs)
        st.success(f"Parsed {len(recs)} segments from {uploaded.name}. Click **Rerun**.")
        st.stop()

    elif name.endswith(".zip"):
        # Parse all .srt files in the zip
        buf = io.BytesIO(uploaded.getvalue())
        with zipfile.ZipFile(buf) as zf:
            recs_all = []
            srt_names = [n for n in zf.namelist() if n.lower().endswith(".srt")]
            if not srt_names:
                st.error("No .srt files found in the .zip.")
                st.stop()
            for n in srt_names:
                recs_all.extend(_parse_srt_bytes(Path(n).name, zf.read(n)))
        if not recs_all:
            st.error("Zip parsed but no segments found.")
            st.stop()
        _write_segments_jsonl(recs_all)
        st.success(f"Parsed {len(recs_all)} segments from {len(srt_names)} SRT file(s). Click **Rerun**.")
        st.stop()

    else:
        st.error("Unsupported file type. Upload .jsonl, .srt, or .zip.")
        st.stop()


# ----------------------------
# Utilities
# ----------------------------
import re
from pathlib import Path

def extract_interviewee(filename: str) -> str:
    """Pulls the name from filenames like LP_20151015_Haeng Soon Park_ENG.srt"""
    base = Path(filename).stem
    m = re.search(r'_(\d{8})_(.+?)_ENG', base)
    if m:
        return m.group(2)
    parts = base.split("_")
    return parts[2] if len(parts) >= 3 else base


def ensure_openai_client():
    load_dotenv()
    # Prefer Streamlit Secrets in the cloud; fall back to .env locally.
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Set it in Streamlit secrets or a local .env file.")
        st.stop()

    # Try the modern client first. If the environment injects unsupported kwargs
    # (e.g., 'proxies' on older builds), fall back to the module-level API.
    try:
        from openai import OpenAI as _OpenAI
        return _OpenAI(api_key=api_key)
    except TypeError as e:
        # Fallback: module-level API works across many versions
        import openai as _openai
        _openai.api_key = api_key
        st.warning("Using OpenAI module-level client for compatibility.")
        return _openai


def load_segments_for_file(filename: str) -> List[Dict]:
    """
    Read segments.jsonl and filter for the selected filename.
    Each line has: {filename, index, start, end, start_ts, end_ts, text}
    """
    if not SEGMENTS_JSONL.exists():
        st.error(f"Missing {SEGMENTS_JSONL}. Run your parser first.")
        st.stop()

    out = []
    with SEGMENTS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("filename") == filename:
                out.append(rec)
    out.sort(key=lambda r: r.get("index", 0))
    return out


def merge_segments_to_chunks(
    segments: List[Dict],
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS
) -> List[Dict]:
    """
    Merge consecutive subtitle segments into ~paragraph-sized chunks.
    Keeps first/last timestamps for citation, preserves filename.
    """
    chunks = []
    buf, buf_len = [], 0
    start_ts = end_ts = None
    start_sec = end_sec = None
    filename = segments[0]["filename"] if segments else None

    def flush():
        if not buf:
            return
        text = " ".join(buf).strip()
        chunks.append({
            "filename": filename,
            "text": text,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "start": start_sec,
            "end": end_sec,
        })

    i = 0
    while i < len(segments):
        seg = segments[i]
        seg_text = seg["text"]
        if not buf:
            start_ts = seg["start_ts"]
            start_sec = seg["start"]
        end_ts = seg["end_ts"]
        end_sec = seg["end"]

        if buf_len + len(seg_text) + 1 <= target_chars:
            buf.append(seg_text)
            buf_len += len(seg_text) + 1
            i += 1
            continue

        # Buffer full -> flush, then overlap for smoothness
        flush()
        tail = (" ".join(buf))[-overlap_chars:] if overlap_chars > 0 else ""
        buf = [tail] if tail else []
        buf_len = len(tail)
        start_ts = seg["start_ts"]
        start_sec = seg["start"]
        end_ts = seg["end_ts"]
        end_sec = seg["end"]

    flush()
    return chunks


def embed_texts(client, texts: List[str]) -> np.ndarray:
    """
    Returns a 2D numpy array (n x d) of embeddings (float32).
    """
    if not texts:
        return np.zeros((0, 1536), dtype="float32")
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    return vecs


# -------- NumPy-only cosine similarity "index" --------
def build_index_np(embeddings: np.ndarray):
    """Return L2-normalized embeddings for cosine similarity search."""
    if embeddings.shape[0] == 0:
        return None
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms


def search_topk_np(normed_embeddings: np.ndarray, query_vec: np.ndarray, k: int):
    """Return indices of top-k cosine-similar rows to query_vec (shape: 1 x d)."""
    q = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    scores = normed_embeddings @ q.T  # (n, 1)
    idx = np.argsort(-scores.squeeze())[:k]
    return idx
# ------------------------------------------------------


def format_citations(chunks: List[Dict]) -> str:
    """
    Create a short, citation-rich context with timestamps.
    """
    lines = []
    for ch in chunks:
        ts = f"[{ch['start_ts']}–{ch['end_ts']}]"
        snippet = ch["text"]
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "…"
        lines.append(f"{ts} {snippet}")
    return "\n".join(lines)


@st.cache_resource(show_spinner=False)
def prepare_recording(_client, filename: str):
    """Load segments → chunk → embed → build NumPy index (cached per filename)."""
    segments = load_segments_for_file(filename)
    chunks = merge_segments_to_chunks(segments)
    texts  = [c["text"] for c in chunks]
    embs   = embed_texts(_client, texts).astype("float32")
    np_idx = build_index_np(embs.copy())
    return chunks, embs, np_idx


def answer_question(client, question: str, chunks: List[Dict], embs: np.ndarray, np_index) -> Dict:
    q_vec = embed_texts(client, [question]).astype("float32")  # (1, d)
    if np_index is None or embs.shape[0] == 0:
        return {"answer": "No content indexed for this recording.", "used": []}

    k = min(TOP_K, embs.shape[0])
    idx = search_topk_np(np_index, q_vec, k=k)
    top_chunks = [chunks[int(i)] for i in idx]

    context = format_citations(top_chunks)
    system = (
        "You are a helpful oral-history assistant. "
        "Answer using ONLY the provided transcript context. "
        "Cite timestamps like [HH:MM:SS,mmm–HH:MM:SS,mmm]. "
        "If the answer isn’t in the context, say you don’t know."
    )
    user = f"Question: {question}\n\nContext:\n{context}"

    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    ans = chat.choices[0].message.content.strip()
    return {"answer": ans, "used": top_chunks}


def _td_to_ts(td: timedelta) -> str:
    # HH:MM:SS,mmm
    total_ms = int(td.total_seconds() * 1000)
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60000) % 60
    h = total_ms // 3600000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _parse_srt_bytes(filename: str, data: bytes) -> list[dict]:
    """Return list of segment dicts matching your JSONL schema."""
    text = data.decode("utf-8", errors="ignore")
    records = []
    for idx, sub in enumerate(srt.parse(text), start=1):
        content = " ".join(line.strip() for line in str(sub.content).splitlines()).strip()
        records.append({
            "filename": filename,
            "index": idx,
            "start": sub.start.total_seconds(),
            "end": sub.end.total_seconds(),
            "start_ts": _td_to_ts(sub.start),
            "end_ts": _td_to_ts(sub.end),
            "text": content,
        })
    return records

def _write_segments_jsonl(records: list[dict]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = SEGMENTS_JSONL
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# UI
# ----------------------------
def main():
    st.set_page_config(page_title="KAS Chatbot (Demo)", page_icon="🎧", layout="wide")
    st.title("Korean American Story — Chatbot Demo 🎧")
    st.caption("Ask questions about a selected recording. Answers include timestamp citations.")

    client = ensure_openai_client()
    ensure_segments_on_disk()

    if not SEGMENTS_JSONL.exists():
        st.error(f"Missing {SEGMENTS_JSONL}. Run src/parse_srt.py first.")
        st.stop()

    # Collect available filenames from segments.jsonl
    filenames = []
    with SEGMENTS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            fn = rec.get("filename")
            if fn:
                filenames.append(fn)
    filenames = sorted(set(filenames))
    if not filenames:
        st.error("No recordings found in segments.jsonl")
        st.stop()

    sel = st.sidebar.selectbox("Select a recording", filenames, index=0)
    st.sidebar.write("Model:", CHAT_MODEL)
    st.sidebar.write("Embed:", EMBED_MODEL)

    # --- announce selection changes in the chat ---
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "current_file" not in st.session_state:
        st.session_state.current_file = sel
        st.session_state.chat.append({
            "role": "system",
            "text": f"Switched recording to {extract_interviewee(sel)} (`{sel}`)"
        })
    elif sel != st.session_state.current_file:
        st.session_state.current_file = sel
        st.session_state.chat.append({
            "role": "system",
            "text": f"Switched recording to {extract_interviewee(sel)} (`{sel}`)"
        })

    with st.spinner("Preparing recording (chunking + embeddings)…"):
        chunks, embs, np_index = prepare_recording(client, sel)

    # -------- Chat history (handle system + regular turns) --------
    for turn in st.session_state.chat:
        # System announcements
        if turn.get("role") == "system":
            with st.chat_message("system"):
                st.markdown(f"**{turn.get('text','')}**")
            continue

        # Regular Q/A turns (be tolerant of missing keys)
        if "q" in turn:
            with st.chat_message("user"):
                st.markdown(turn["q"])
        if "a" in turn:
            with st.chat_message("assistant"):
                st.markdown(turn["a"])
                used = turn.get("used") or []
                if used:
                    with st.expander("Cited segments"):
                        for i, c in enumerate(used, 1):
                            st.markdown(
                                f"**{i}. {c['start_ts']}–{c['end_ts']}**  \n"
                                f"{c['text'][:500]}{'…' if len(c['text'])>500 else ''}"
                            )

    # -------- Chat input --------
    q = st.chat_input("Ask about this recording…")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                out = answer_question(client, q, chunks, embs, np_index)
            st.markdown(out["answer"])
            with st.expander("Cited segments"):
                for i, c in enumerate(out["used"], 1):
                    st.markdown(
                        f"**{i}. {c['start_ts']}–{c['end_ts']}**  \n"
                        f"{c['text'][:500]}{'…' if len(c['text'])>500 else ''}"
                    )

        # Persist turn
        st.session_state.chat.append({"q": q, "a": out["answer"], "used": out["used"]})


if __name__ == "__main__":
    main()