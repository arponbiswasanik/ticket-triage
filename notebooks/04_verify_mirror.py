import pandas as pd

PATH = "data/raw/complaints_mirror.csv"

# --- 1. Normalize whatever column names this mirror uses ---
raw_cols = pd.read_csv(PATH, nrows=0).columns
norm = {h.strip().lower().replace(" ", "_").replace("?", ""): h for h in raw_cols}

required = ["consumer_complaint_narrative", "product", "issue", "company_response_to_consumer"]
optional = ["consumer_disputed", "timely_response", "date_received"]

print("Columns found (normalized):")
print(sorted(norm))

print("\n--- CHECKLIST ---")
for r in required:
    print(f"  [{'PASS' if r in norm else 'MISSING'}] {r}")
for o in optional:
    print(f"  [{'present' if o in norm else 'absent '}] {o}")

missing = [r for r in required if r not in norm]
if missing:
    raise SystemExit(f"\nMirror FAILS - missing {missing}. Find another mirror.")

# --- 2. Stream the stats we gate on ---
cols = [norm[r] for r in required + optional if r in norm]
total = narr = disputed = 0
shown = 0

for chunk in pd.read_csv(PATH, usecols=cols, chunksize=200_000, low_memory=False):
    total += len(chunk)
    n = chunk[norm["consumer_complaint_narrative"]]
    narr += int(n.notna().sum())
    if shown < 3:
        for t in n.dropna().head(3 - shown):
            print(f"\n--- sample narrative ---\n{t[:500]}")
            shown += 1
    if "consumer_disputed" in norm:
        disputed += int(chunk[norm["consumer_disputed"]].notna().sum())

print(f"\nTotal rows:             {total:>12,}")
print(f"Rows with narrative:    {narr:>12,}  ({narr / max(total, 1):.1%})")
if "consumer_disputed" in norm:
    print(f"Rows with disputed set: {disputed:>12,}  ({disputed / max(total, 1):.1%})")