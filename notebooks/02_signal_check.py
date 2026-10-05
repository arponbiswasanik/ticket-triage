import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline

df = pd.read_csv("data/raw/tickets_raw.csv")

# 1. See FULL descriptions (are placeholders really unfilled?)
print("=" * 60, "\nFULL SAMPLE DESCRIPTIONS\n", "=" * 60)
for i, txt in enumerate(df["Ticket Description"].dropna().sample(5, random_state=42)):
    print(f"\n--- {i+1} ---\n{txt}")

# 2. Placeholder bug + duplication check
ph = df["Ticket Description"].str.contains("{product_purchased}", regex=False).mean()
print(f"\nRows with literal '{{product_purchased}}' placeholder: {ph:.1%}")
print(f"Unique descriptions: {df['Ticket Description'].nunique()} of {len(df)}")

# 3. Is Priority correlated with anything real?
print("\nPriority distribution within each Ticket Type:")
print(pd.crosstab(df["Ticket Type"], df["Ticket Priority"], normalize="index").round(2))

# 4. Learnability test: text-only, deduped to avoid duplicate contamination
data = (df.dropna(subset=["Ticket Description", "Ticket Priority"])
          .drop_duplicates(subset=["Ticket Description"]))
print(f"\nLearnability test on {len(data)} deduplicated rows (chance = majority class share)")

for target in ["Ticket Priority", "Ticket Type"]:
    pipe = make_pipeline(
        TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20000),
        LogisticRegression(max_iter=1000),
    )
    scores = cross_val_score(
        pipe, data["Ticket Description"], data[target],
        cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy",
    )
    chance = data[target].value_counts(normalize=True).max()
    print(f"{target}: accuracy = {scores.mean():.3f} +/- {scores.std():.3f}  (chance = {chance:.3f})")