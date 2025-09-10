# src/ingest_census.py
from pathlib import Path
import pandas as pd
import numpy as np
import re

RAW_DIR = Path("data/census/raw")
OUT_PARQUET = Path("data/census/census.parquet")

def _clean_cols(cols):
    return [re.sub(r"\s+", " ", c.strip()).strip() for c in cols]

TEXT_LIKE = re.compile(r"(name|geograph|area.*name|county|place|tract|district)", re.I)

def _detect_geo_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if TEXT_LIKE.search(str(c)):
            return c
    # fallback: first object dtype column
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    return None

def _to_numeric(s: pd.Series) -> pd.Series:
    if s.dtype != "object":
        return s
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("\u00a0", " ", regex=False)
         .str.replace("%", "", regex=False)
         .str.strip(),
        errors="coerce"
    )

def _possible_year(text: str) -> int | None:
    m = re.search(r"(20\d{2}|19\d{2})", text or "")
    return int(m.group(1)) if m else None

def ingest():
    frames = []
    files = sorted(RAW_DIR.glob("*.xls"))
    if not files:
        raise SystemExit(f"No .xls files in {RAW_DIR}. Put your spreadsheets there.")

    for f in files:
        try:
            # engine="xlrd" for .xls
            xls = pd.ExcelFile(f, engine="xlrd")
        except Exception as e:
            raise SystemExit(
                f"Failed to open {f.name}. Make sure 'xlrd' is installed. Error: {e}"
            )

        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet_name=sheet, dtype=object)
            except Exception:
                continue  # skip unreadable sheets

            if df.empty:
                continue

            df.columns = _clean_cols(df.columns)

            # heuristic: keep only fully/mostly data rows
            df = df.dropna(how="all").reset_index(drop=True)

            # find geography/name column
            geo_col = _detect_geo_col(df)
            if not geo_col:
                # cannot do much without a label; synthesize one from index
                df["Geography"] = [f"Row {i+1}" for i in range(len(df))]
                geo_col = "Geography"

            # numeric-ize everything else
            for c in df.columns:
                if c == geo_col:
                    continue
                df[c] = _to_numeric(df[c])

            # keep at least one numeric column
            num_cols = [c for c in df.columns if c != geo_col and pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                continue

            # melt to long format
            long = df[[geo_col] + num_cols].melt(
                id_vars=[geo_col], var_name="variable", value_name="value"
            ).dropna(subset=["value"])

            long = long.rename(columns={geo_col: "geo"})
            long["source_file"] = f.name
            long["sheet"] = sheet

            # try to infer state from filename and year from sheet/variable
            stem = f.stem.replace("_", " ")
            state_guess = stem  # fine for files like "Arkansas.xls"
            long["state"] = state_guess

            long["year"] = None
            year_from_sheet = _possible_year(sheet)
            if year_from_sheet:
                long["year"] = year_from_sheet
            else:
                # fallback: look for a year inside variable name
                years_in_vars = long["variable"].astype(str).map(_possible_year)
                if years_in_vars.notna().any():
                    long["year"] = years_in_vars

            frames.append(long)

    if not frames:
        raise SystemExit("No usable data found. Check your spreadsheets' structure.")

    out = pd.concat(frames, ignore_index=True)

    # Sanitize dtypes before writing to Parquet
    for col in ["geo", "variable", "state", "source_file", "sheet"]:
        out[col] = out[col].astype("string")          # keep place names + codes as strings
    out["year"]  = pd.to_numeric(out["year"], errors="coerce").astype("Int64")  # nullable integer
    out["value"] = pd.to_numeric(out["value"], errors="coerce")                 # float

    # (Re)build normalized keys from the now-clean columns
    out["geo_norm"]   = out["geo"].str.strip().str.lower()
    out["var_norm"]   = out["variable"].str.strip().str.lower()
    out["state_norm"] = out["state"].str.strip().str.lower()

    # normalize helper columns for matching
    out["geo_norm"] = out["geo"].astype(str).str.strip().str.lower()
    out["var_norm"] = out["variable"].astype(str).str.strip().str.lower()
    out["state_norm"] = out["state"].astype(str).str.strip().str.lower()

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET)
    print(f"Wrote {len(out):,} rows to {OUT_PARQUET}")

if __name__ == "__main__":
    ingest()