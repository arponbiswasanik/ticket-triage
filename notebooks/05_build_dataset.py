"""05 (v2) - Build working dataset: canonical headers, filter, dedupe, temporal split.

v2: normalizes hyphens/spaces in headers, prints a column manifest,
and HARD-FAILS if any required column is missing (no silent skips).
"""
import pandas as pd

RAW, OUT = "data/raw/complaints_mirror.csv", "data/processed"

WANT = ["complaint_id", "date_received", "product", "sub_product", "issue",
        "sub_issue", "consumer_complaint_narrative", "company", "state",
        "submitted_via", "company_response_to_consumer", "consumer_disputed"]

def canon(h):
    return h.strip().lower().replace(" ", "_").replace("-", "_").replace("?", "")

header = pd.read_csv(RAW, nrows=0).columns
norm = {canon(h): h for h in header}

print("Raw header -> canonical name:")
for h in header:
    print(f"  {h!r:42} -> {canon(h)!r}")

inv, missing = {}, []
for w in WANT:
    if w in norm:
        inv[norm[w]] = w
    else:
        missing.append(w)

if missing:
    raise SystemExit(f"\nMISSING required columns: {missing}\n"
                     f"Available: {sorted(norm)}")

chunks = []
for chunk in pd.read_csv(RAW, usecols=list(inv), chunksize=200_000, low_memory=False):
    chunk = chunk.rename(columns=inv)
    disputed = chunk["consumer_disputed"].astype(str).str.strip().str.upper()
    mask = chunk["consumer_complaint_narrative"].notna() & disputed.isin(["YES", "NO"])
    c = chunk.loc[mask].copy()
    c["consumer_disputed"] = disputed.loc[mask]
    chunks.append(c)

df = pd.concat(chunks, ignore_index=True)
before = len(df)
df = df.drop_duplicates(subset="consumer_complaint_narrative")
print(f"\nRows (narrative + disputed label) : {len(df):,}")
print(f"Exact duplicates removed          : {before - len(df):,}")

print(f"\nClass balance:\n{df['consumer_disputed'].value_counts().to_string()}")

try:
    df["date"] = pd.to_datetime(df["date_received"], errors="coerce", format="mixed")
except (TypeError, ValueError):
    df["date"] = pd.to_datetime(df["date_received"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

n_test = int(len(df) * 0.15)
train, test = df.iloc[:-n_test], df.iloc[-n_test:]
print(f"Temporal split: train={len(train):,} | test={len(test):,}")

train.to_parquet(f"{OUT}/train.parquet", index=False)
test.to_parquet(f"{OUT}/test.parquet", index=False)
print(f"\nSaved. Columns: {list(train.columns)}")