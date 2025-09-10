# src/app.py
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import io, zipfile, srt
from datetime import timedelta
import re

# --- OpenAI SDK v1 ---
try:
    from openai import OpenAI
    _HAS_OPENAI_V1 = True
except Exception:
    _HAS_OPENAI_V1 = False

# ----------------------------
# Config (transcripts)
# ----------------------------
DATA_DIR = Path("data/processed")
SEGMENTS_JSONL = DATA_DIR / "segments.jsonl"
EMBED_MODEL = "text-embedding-3-small"   # fast + inexpensive
CHAT_MODEL = "gpt-4o-mini"               # good quality for demos
TOP_K = 5
CHUNK_TARGET_CHARS = 900
CHUNK_OVERLAP_CHARS = 180

# Prebuilt / bootstrap
INDEX_DIR = Path("data/index")
CHUNKS_PARQUET = INDEX_DIR / "chunks.parquet"
EMBS_NPY = INDEX_DIR / "embeddings.npy"
NORMED_NPY = INDEX_DIR / "embeddings_normed.npy"
MANIFEST_JSON = INDEX_DIR / "manifest.json"
BOOTSTRAP_SEGMENTS = Path("data/bootstrap/segments.jsonl")

# ----------------------------
# Config (census)
# ----------------------------
CENSUS_PARQUET = Path("data/census/census.parquet")
# Prefer variables that look like counts of "Asian alone", then generic "Asian population"
PREF_VARS = [
    r"asian\s+alone",
    r"asian.*(population|estimate|count)",
    r"\basian\b"
]
# Variables that look like total population (for % share)
TOTAL_VARS = [r"total.*population", r"population.*total", r"overall.*population"]

STATE_NAMES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut",
    "delaware","florida","georgia","hawaii","idaho","illinois","indiana","iowa","kansas",
    "kentucky","louisiana","maine","maryland","massachusetts","michigan","minnesota",
    "mississippi","missouri","montana","nebraska","nevada","new hampshire","new jersey",
    "new mexico","new york","north carolina","north dakota","ohio","oklahoma","oregon",
    "pennsylvania","rhode island","south carolina","south dakota","tennessee","texas",
    "utah","vermont","virginia","washington","west virginia","wisconsin","wyoming",
    "district of columbia","washington, dc","dc"
}

# ----------------------------
# Secrets / Env helpers
# ----------------------------
def _maybe_secret(key: str):
    """Return st.secrets[key] if secrets.toml exists; otherwise None (no exception)."""
    try:
        return st.secrets[key]
    except Exception:
        return None

def ensure_openai_client():
    load_dotenv()  # load .env from project root
    api_key = os.getenv("OPENAI_API_KEY") or _maybe_secret("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or _maybe_secret("OPENAI_BASE_URL")

    if not api_key:
        st.error(
            "Missing OPENAI_API_KEY. Put it in a local `.env` or in `.streamlit/secrets.toml`.\n\n"
            "Example `.env`:\nOPENAI_API_KEY=sk-...\n# (optional) OPENAI_BASE_URL=https://api.openai.com/v1"
        )
        st.stop()

    try:
        # Preferred: class client
        from openai import OpenAI as _OpenAI
        client = _OpenAI(api_key=api_key, base_url=base_url) if base_url else _OpenAI(api_key=api_key)
        _ = client.chat  # light sanity check attr
        return client
    except TypeError:
        # Fallback: module-level client
        import openai as _openai
        _openai.api_key = api_key
        if base_url:
            _openai.base_url = base_url
        st.warning("Using OpenAI module-level client for compatibility.")
        return _openai

# ----------------------------
# File bootstrap (transcripts)
# ----------------------------
def ensure_segments_on_disk():
    """
    Ensure data/processed/segments.jsonl exists.
    Priority:
      (1) If it's already there, use it.
      (2) Else, if a bootstrap copy exists in data/bootstrap/, copy it.
      (3) Else, prompt for upload (.jsonl / .srt / .zip).
    """
    if SEGMENTS_JSONL.exists():
        return

    # (2) Bootstrap path for no-upload cold starts
    if BOOTSTRAP_SEGMENTS.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SEGMENTS_JSONL.write_bytes(BOOTSTRAP_SEGMENTS.read_bytes())
        st.success("Loaded transcripts from data/bootstrap/segments.jsonl")
        return

    # (3) Fallback to uploader
    st.info("No transcript data found. Upload your **segments.jsonl**, a single **.srt**, or a **.zip** of SRTs to begin.")
    uploaded = st.file_uploader("Upload segments.jsonl / .srt / .zip", type=["jsonl", "srt", "zip"])
    if uploaded is None:
        st.stop()

    name = uploaded.name.lower()
    if name.endswith(".jsonl"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SEGMENTS_JSONL.write_bytes(uploaded.getvalue())
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
# Utilities (transcripts)
# ----------------------------
def extract_interviewee(filename: str) -> str:
    """Pulls the name from filenames like LP_20151015_Haeng Soon Park_ENG.srt"""
    base = Path(filename).stem
    m = re.search(r'_(\d{8})_(.+?)_ENG', base)
    if m:
        return m.group(2)
    parts = base.split("_")
    return parts[2] if len(parts) >= 3 else base

def merge_segments_to_chunks(
    segments: List[Dict],
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS
) -> List[Dict]:
    """Merge consecutive subtitle segments into ~paragraph-sized chunks."""
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
            start_ts = seg["start_ts"]; start_sec = seg["start"]
        end_ts = seg["end_ts"]; end_sec = seg["end"]

        if buf_len + len(seg_text) + 1 <= target_chars:
            buf.append(seg_text); buf_len += len(seg_text) + 1
            i += 1; continue

        flush()
        tail = (" ".join(buf))[-overlap_chars:] if overlap_chars > 0 else ""
        buf = [tail] if tail else []; buf_len = len(tail)
        start_ts = seg["start_ts"]; start_sec = seg["start"]
        end_ts = seg["end_ts"]; end_sec = seg["end"]

    flush()
    return chunks

# ---------- Embeddings (BATCHED to avoid token caps) ----------
def embed_texts(client, texts: List[str], batch_size: int = 128) -> np.ndarray:
    """
    Return a 2D numpy array (n x d) of embeddings.
    Batches requests to stay under provider token caps.
    """
    if not texts:
        return np.zeros((0, 1536), dtype="float32")

    vecs_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs_all.extend([d.embedding for d in resp.data])
    return np.array(vecs_all, dtype="float32")

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
    """Create short, citation-rich context with interviewee + timestamps."""
    lines = []
    for ch in chunks:
        who = ch.get("interviewee") or extract_interviewee(ch["filename"])
        ts = f"[{ch['start_ts']}–{ch['end_ts']}]"
        snippet = ch["text"]
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "…"
        lines.append(f"{who} — {Path(ch['filename']).stem} {ts}\n{snippet}")
    return "\n\n".join(lines)

def answer_question(client, question: str, chunks: List[Dict], embs: np.ndarray, np_index) -> Dict:
    q_vec = embed_texts(client, [question]).astype("float32")  # shape (1, d)
    if np_index is None or embs.shape[0] == 0:
        return {"answer": "No content indexed.", "used": []}

    n = embs.shape[0]
    k = min(TOP_K, n)
    candidate_k = min(max(5 * k, k), n)
    raw_idx = search_topk_np(np_index, q_vec, k=candidate_k)
    sel_idx = diversify_indices(raw_idx, chunks, k=k, per_file_limit=3)

    top_chunks = [chunks[int(i)] for i in sel_idx]
    context = format_citations(top_chunks)

    system = (
        "You are a helpful oral-history assistant. "
        "Synthesize across multiple interviews, but answer using ONLY the provided transcript context. "
        "Cite timestamps like [HH:MM:SS,mmm–HH:MM:SS,mmm] and include the interviewee name. "
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

def _parse_srt_bytes(filename: str, data: bytes) -> List[Dict]:
    """Return list of segment dicts matching your JSONL schema."""
    text = data.decode("utf-8", errors="ignore")
    records: List[Dict] = []
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

def _write_segments_jsonl(records: List[Dict]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = SEGMENTS_JSONL
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------
# Global-corpus helpers (transcripts)
# ----------------------------
def load_all_segments_grouped() -> Dict[str, List[Dict]]:
    """Read segments.jsonl and group records by filename."""
    if not SEGMENTS_JSONL.exists():
        st.error(f"Missing {SEGMENTS_JSONL}. Run your parser first.")
        st.stop()

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    with SEGMENTS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fn = rec.get("filename")
            if fn:
                grouped[fn].append(rec)

    for fn in grouped:
        grouped[fn].sort(key=lambda r: r.get("index", 0))
    return grouped

def chunk_all_recordings(grouped: Dict[str, List[Dict]]) -> List[Dict]:
    """Chunk each recording separately, then flatten; attach interviewee."""
    all_chunks: List[Dict] = []
    for fn, segs in grouped.items():
        chs = merge_segments_to_chunks(segs)
        for ch in chs:
            ch["interviewee"] = extract_interviewee(fn)
        all_chunks.extend(chs)
    return all_chunks

import hashlib, json as _json

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _save_index_artifacts(chunks: List[Dict], embs: np.ndarray, np_idx: np.ndarray, seg_sha: str):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Save chunks as a parquet for quick reload
    import pandas as pd
    pd.DataFrame(chunks).to_parquet(CHUNKS_PARQUET)
    np.save(EMBS_NPY, embs.astype("float32"))
    np.save(NORMED_NPY, np_idx.astype("float32"))
    MANIFEST_JSON.write_text(_json.dumps({
        "embed_model": EMBED_MODEL,
        "segments_sha256": seg_sha,
        "n_chunks": len(chunks),
        "dim": int(embs.shape[1]),
    }, indent=2))

def _load_index_artifacts():
    if not (MANIFEST_JSON.exists() and CHUNKS_PARQUET.exists() and EMBS_NPY.exists() and NORMED_NPY.exists()):
        return None
    manifest = _json.loads(MANIFEST_JSON.read_text())
    if manifest.get("embed_model") != EMBED_MODEL:
        return None
    # If segments.jsonl exists, verify it matches the manifest hash
    if SEGMENTS_JSONL.exists():
        cur_sha = _sha256_file(SEGMENTS_JSONL)
        if cur_sha != manifest.get("segments_sha256"):
            return None
    import pandas as pd
    chunks_df = pd.read_parquet(CHUNKS_PARQUET)
    chunks = chunks_df.to_dict(orient="records")
    embs = np.load(EMBS_NPY)
    np_idx = np.load(NORMED_NPY)
    return chunks, embs, np_idx

@st.cache_resource(show_spinner=False)
def prepare_corpus(_client, _mtime_key: float):
    """
    Prefer loading a prebuilt index from data/index/.
    If missing or stale, build it, save artifacts, then return.
    """
    pre = _load_index_artifacts()
    if pre is not None:
        return pre

    # Build fresh
    grouped = load_all_segments_grouped()
    chunks  = chunk_all_recordings(grouped)
    texts   = [c["text"] for c in chunks]
    embs    = embed_texts(_client, texts).astype("float32")
    np_idx  = build_index_np(embs.copy())

    seg_sha = _sha256_file(SEGMENTS_JSONL) if SEGMENTS_JSONL.exists() else ""
    _save_index_artifacts(chunks, embs, np_idx, seg_sha)
    return chunks, embs, np_idx

# ----------------------------
# Census loader + query helpers
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_census_df():
    if not CENSUS_PARQUET.exists():
        return None
    df = pd.read_parquet(CENSUS_PARQUET)
    # ensure helper cols exist (forward compatibility)
    for c in ["geo_norm", "var_norm", "state_norm"]:
        if c not in df.columns:
            if c == "geo_norm":
                df[c] = df["geo"].astype(str).str.lower()
            elif c == "var_norm":
                df[c] = df["variable"].astype(str).str.lower()
            else:
                df[c] = df["state"].astype(str).str.lower()
    return df

def is_census_query(q: str) -> bool:
    ql = q.lower()
    keywords = ("population","percent","share","how many","number","estimate","census")
    asianish = ("asian","aapi","korean","chinese","filipino","vietnamese","indian","pakistani","bangladeshi","japanese")
    return any(k in ql for k in keywords) and any(a in ql for a in asianish)

def _pick_regex(candidates: list[str], patterns: list[str]) -> list[str]:
    out = []
    for pat in patterns:
        rx = re.compile(pat, re.I)
        hits = [c for c in candidates if rx.search(str(c))]
        if hits:
            out.extend(hits)
            break
    return out

def find_geo_rows(df: pd.DataFrame, q: str) -> pd.DataFrame:
    ql = q.lower()
    # 1) if a state name appears, filter by it
    state_hits = [s for s in STATE_NAMES if s in ql]
    if state_hits:
        df = df[df["state_norm"].isin(state_hits)]
    # 2) try exact/substring match on 'geo' (county/place/etc.); prefer longest hit
    geos = df["geo"].dropna().astype(str).unique().tolist()
    subs = [g for g in geos if g.lower() in ql or ql in g.lower()]
    if subs:
        subs = sorted(subs, key=lambda x: len(x), reverse=True)
        df = df[df["geo"] == subs[0]]
    return df

def pick_variable_rows(df: pd.DataFrame, q: str) -> pd.DataFrame:
    vars_unique = df["var_norm"].dropna().unique().tolist()
    picks = _pick_regex(vars_unique, PREF_VARS)
    if not picks:
        return df  # fallback: let downstream choose first available numeric
    return df[df["var_norm"].isin([p.lower() for p in picks])]

def answer_from_census(question: str, census_df: pd.DataFrame) -> Dict:
    if census_df is None or census_df.empty:
        return {"answer": "Census dataset not loaded. Please run the census ingest step.", "used_census": []}

    narrowed = find_geo_rows(census_df, question)
    if narrowed.empty:
        return {"answer": "I couldn’t match a geography in your question. Try including a state, county, or city name.", "used_census": []}

    narrowed = pick_variable_rows(narrowed, question)
    if narrowed.empty:
        return {"answer": "I couldn’t identify an ‘Asian’ population column in your spreadsheets.", "used_census": []}

    # Prefer most recent year with data
    if "year" in narrowed.columns and narrowed["year"].notna().any():
        yr = pd.to_numeric(narrowed["year"], errors="coerce")
        narrowed = narrowed.assign(_yr=yr)
        narrowed = narrowed.dropna(subset=["_yr"]).sort_values("_yr")
        latest_year = int(narrowed["_yr"].max())
        narrowed = narrowed[narrowed["_yr"] == latest_year]
    else:
        latest_year = None

    # If we have a total population var, compute % share
    vars_unique = narrowed["var_norm"].dropna().unique().tolist()
    total_candidates = _pick_regex(vars_unique, TOTAL_VARS)
    share_line = ""
    if total_candidates:
        tot_df = narrowed[narrowed["var_norm"].isin([total_candidates[0].lower()])]
        asian_df = narrowed[~narrowed["var_norm"].isin([total_candidates[0].lower()])]
        pop_total = pd.to_numeric(tot_df["value"], errors="coerce").dropna().max() if not tot_df.empty else None
        asian_val = pd.to_numeric(asian_df["value"], errors="coerce").dropna().max() if not asian_df.empty else None
        if pop_total and asian_val is not None and pop_total > 0:
            share_val = 100.0 * float(asian_val) / float(pop_total)
            share_line = f" (≈ {share_val:.1f}% of total population)"

    # Build a compact answer
    narrowed = narrowed.sort_values(["variable","year"])
    sample = narrowed.iloc[0]
    geo = str(sample["geo"])
    asian_var = str(sample["variable"])
    val = int(float(sample["value"]))
    year_str = f" in {latest_year}" if latest_year else ""
    answer = f"{val:,} Asian (from “{asian_var}”){share_line} in **{geo}**{year_str}."

    # Citations: show all contributing rows with file/sheet/variable/year
    used = narrowed[["geo","variable","value","year","source_file","sheet"]].copy()
    used = used.sort_values(["variable","year"]).to_dict(orient="records")
    return {"answer": answer, "used_census": used}

# ----------------------------
# UI
# ----------------------------
def main():
    st.set_page_config(page_title="Korean American Story — Chatbot Demo", page_icon="🎧", layout="wide")
    st.title("Korean American Story — Chatbot Demo 🎧")
    st.caption("Ask questions about the full archive. Answers include interviewee + timestamp citations.")

    client = ensure_openai_client()
    ensure_segments_on_disk()

    if not SEGMENTS_JSONL.exists():
        st.error(f"Missing {SEGMENTS_JSONL}. Run src/parse_srt.py first.")
        st.stop()

    # --- Global corpus: build one index over ALL recordings ---
    with st.spinner("Preparing corpus (chunking + embeddings)…"):
        mtime = SEGMENTS_JSONL.stat().st_mtime  # cache key
        chunks, embs, np_index = prepare_corpus(client, mtime)

    if not chunks:
        st.error("No recordings found in segments.jsonl")
        st.stop()

    # Sidebar info (read-only)
    st.sidebar.write("Model:", CHAT_MODEL)
    st.sidebar.write("Embed:", EMBED_MODEL)
    st.sidebar.write("OpenAI key:", "✓ set")

    # Load census dataset (if present)
    census_df = load_census_df()
    if census_df is None:
        st.sidebar.warning("Census: no data loaded (run `python -m src.ingest_census`).")
    else:
        st.sidebar.write(f"Census rows: {len(census_df):,}")

    # Quick corpus summary
    num_files = len({c["filename"] for c in chunks})
    st.info(
        f"Indexed **{num_files} recordings** into **{len(chunks)} chunks**. "
        "Ask a question to synthesize across interviews; answers will cite interviewee + timestamps."
    )

    # -------- Chat history --------
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for turn in st.session_state.chat:
        if turn.get("role") == "system":
            with st.chat_message("system"):
                st.markdown(f"**{turn.get('text','')}**")
            continue

        if "q" in turn:
            with st.chat_message("user"):
                st.markdown(turn["q"])
        if "a" in turn:
            with st.chat_message("assistant"):
                st.markdown(turn["a"])

                # Transcript citations (if present)
                used = turn.get("used") or []
                if used:
                    with st.expander("Cited segments"):
                        for i, c in enumerate(used, 1):
                            who = c.get("interviewee") or extract_interviewee(c["filename"])
                            st.markdown(
                                f"**{i}. {who} — {Path(c['filename']).stem} "
                                f"{c['start_ts']}–{c['end_ts']}**  \n"
                                f"{c['text'][:500]}{'…' if len(c['text'])>500 else ''}"
                            )
                # Census citations (if present)
                used_census = turn.get("used_census") or []
                if used_census:
                    with st.expander("Census sources"):
                        for i, r in enumerate(used_census, 1):
                            yr = r.get("year", "")
                            val = r.get("value", "")
                            st.markdown(
                                f"**{i}. {r['geo']} — {r['variable']}** "
                                f"{'('+str(yr)+')' if pd.notna(yr) and yr!='' else ''}  \n"
                                f"Value: {int(float(val)) if pd.notna(val) and str(val)!='' else val:,}  \n"
                                f"Source: {r['source_file']} / {r['sheet']}"
                            )

    # -------- Chat input --------
    q = st.chat_input("Ask about the archive (e.g., themes of immigration, identity, foodways)…")
    if q:
        with st.chat_message("user"):
            st.markdown(q)

        # Route: census questions use deterministic answers from parquet; else transcript RAG
        if is_census_query(q) and census_df is not None:
            with st.chat_message("assistant"):
                out = answer_from_census(q, census_df)
                st.markdown(out["answer"])
                if out.get("used_census"):
                    with st.expander("Census sources"):
                        for i, r in enumerate(out["used_census"], 1):
                            yr = r.get("year", "")
                            val = r.get("value", "")
                            st.markdown(
                                f"**{i}. {r['geo']} — {r['variable']}** "
                                f"{'('+str(yr)+')' if pd.notna(yr) and yr!='' else ''}  \n"
                                f"Value: {int(float(val)) if pd.notna(val) and str(val)!='' else val:,}  \n"
                                f"Source: {r['source_file']} / {r['sheet']}"
                            )
            # Persist turn
            st.session_state.chat.append({"q": q, "a": out["answer"], "used_census": out.get("used_census", [])})

        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    out = answer_question(client, q, chunks, embs, np_index)
                st.markdown(out["answer"])
                if out.get("used"):
                    with st.expander("Cited segments"):
                        for i, c in enumerate(out["used"], 1):
                            who = c.get("interviewee") or extract_interviewee(c["filename"])
                            st.markdown(
                                f"**{i}. {who} — {Path(c['filename']).stem} "
                                f"{c['start_ts']}–{c['end_ts']}**  \n"
                                f"{c['text'][:500]}{'…' if len(c['text'])>500 else ''}"
                            )
            # Persist turn
            st.session_state.chat.append({"q": q, "a": out["answer"], "used": out.get("used", [])})

if __name__ == "__main__":
    main()