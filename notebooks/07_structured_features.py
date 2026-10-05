"""07 - Feature engineering: text + structured (company, product, channel, ...).

Measures the lift of structured features over the text-only baseline,
holding the model fixed (LogisticRegression).
"""
import json
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

CAT = ["product", "sub_product", "issue", "sub_issue", "submitted_via", "state", "company"]
NUM = ["n_chars", "n_words"]
for df in (train, test):
    df["n_chars"] = df["consumer_complaint_narrative"].str.len()
    df["n_words"] = df["consumer_complaint_narrative"].str.split().str.len()
    for c in CAT:
        df[c] = df[c].fillna("missing").astype(str)

y_train = (train["consumer_disputed"] == "YES").astype(int)
y_test = (test["consumer_disputed"] == "YES").astype(int)

pre = ColumnTransformer([
    ("text", TfidfVectorizer(min_df=5, ngram_range=(1, 2),
                             max_features=50_000, sublinear_tf=True),
     "consumer_complaint_narrative"),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT),
    ("num", StandardScaler(), NUM),
])

model = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                               solver="liblinear", random_state=42)),
])

print("Training text + structured model...")
t0 = time.time()
model.fit(train, y_train)
print(f"done in {time.time()-t0:.0f}s | total features: {len(model.named_steps['pre'].get_feature_names_out()):,}")

proba = model.predict_proba(test)[:, 1]
results = {
    "model": "tfidf_plus_structured_logreg",
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
    "pr_auc": round(average_precision_score(y_test, proba), 4),
}

baseline = json.load(open("evaluation/baseline_results.json"))
print("\n--- COMPARISON (temporal test: 2017 holdout) ---")
for k in ["roc_auc", "pr_auc"]:
    b, n = baseline[k], results[k]
    print(f"{k:>8}: {b:.4f} (text only)  ->  {n:.4f} (text+structured)  [{(n-b)/b:+.1%}]")

names = model.named_steps["pre"].get_feature_names_out()
coefs = model.named_steps["clf"].coef_.ravel()
cat_idx = [i for i, n in enumerate(names) if n.startswith("cat__")]
print("\nTop structured DISPUTE signals:")
for i in sorted(cat_idx, key=lambda i: coefs[i])[-10:][::-1]:
    print(f"  {coefs[i]:+6.2f}  {names[i]}")
print("\nTop structured NO-DISPUTE signals:")
for i in sorted(cat_idx, key=lambda i: coefs[i])[:10]:
    print(f"  {coefs[i]:+6.2f}  {names[i]}")

joblib.dump(model, "models/logreg_full.joblib")
results["baseline_roc_auc"] = baseline["roc_auc"]
results["lift_vs_baseline"] = round(results["roc_auc"] - baseline["roc_auc"], 4)
with open("evaluation/step6_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: models/logreg_full.joblib, evaluation/step6_results.json")