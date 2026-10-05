import pandas as pd
from collections import Counter

PATH = "data/raw/complaints_raw.csv"

header = pd.read_csv(PATH, nrows=0).columns.tolist()
print("Columns found in file:")
for c in header:
    print(f"  - {c}")

cols = [c for c in ["Date received", "Product", "Issue",
                    "Consumer complaint narrative",
                    "Company response to consumer",
                    "Consumer disputed?"] if c in header]

total = 0
has_narrative = 0
products = Counter()
responses = Counter()
disputed = Counter()
sampled = 0

for chunk in pd.read_csv(PATH, usecols=cols, chunksize=200_000, low_memory=False):
    total += len(chunk)
    if "Consumer complaint narrative" in chunk:
        has_narrative += chunk["Consumer complaint narrative"].notna().sum()
        if sampled < 3:
            for t in chunk["Consumer complaint narrative"].dropna().head(3):
                print("\n--- sample narrative ---")
                print(t[:600])
                sampled += 1
    if "Product" in chunk:
        products.update(chunk["Product"].value_counts().to_dict())
    if "Company response to consumer" in chunk:
        responses.update(chunk["Company response to consumer"].value_counts().to_dict())
    if "Consumer disputed?" in chunk:
        disputed.update(chunk["Consumer disputed?"].value_counts(dropna=False).to_dict())

print(f"\nTotal rows: {total:,}")
print(f"Rows with narrative: {has_narrative:,} ({has_narrative/max(total,1):.1%})")
print("\nTop 15 products:")
for k, v in products.most_common(15):
    print(f"  {v:>10,}  {k}")
print("\nCompany response to consumer:")
for k, v in responses.most_common():
    print(f"  {v:>10,}  {k}")
print("\nConsumer disputed?:")
for k, v in disputed.most_common():
    print(f"  {v:>10,}  {k}")