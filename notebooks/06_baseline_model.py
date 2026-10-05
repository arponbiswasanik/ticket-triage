"""06 - Baseline: TF-IDF + LogisticRegression, text only.

The honest floor for dispute-risk prediction.
Train: 2015-2016 | Test: 2017 holdout (temporal split).
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

y_train = (train["consumer_disputed"] == "YES").astype(int)
y_test = (test["consumer_disputed"] == "YES").astype(int)
print(f"Train: {len(train):,} (dispute rate {y_train.mean():.1%})")
print(f"Test : {len(test):,} (dispute rate {y_test.mean():.1%})")

print("\n[1/3] Vectorizing text...")
t0 = time.time()
vec = TfidfVectorizer(min_df=5, ngram_range=(1, 2),
                      max_features=50_000, sublinear_tf=True)
X_train = vec.fit_transform(train["consumer_complaint_narrative"].fillna(""))
X_test = vec.transform(test["consumer_complaint_narrative"].fillna(""))
print(f"      {X_train.shape[0]:,} docs x {X_train.shape[1]:,} features ({time.time()-t0:.0f}s)")

print("[2/3] Training LogisticRegression...")
t0 = time.time()
clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                         solver="liblinear", random_state=42)
clf.fit(X_train, y_train)
print(f"      done ({time.time()-t0:.0f}s)")

print("[3/3] Evaluating on 2017 holdout...")
proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print(classification_report(y_test, pred, target_names=["no dispute", "dispute"]))
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"Confusion matrix   TP={tp:>6,}  FP={fp:>6,}  FN={fn:>6,}  TN={tn:>6,}")

metrics = {
    "model": "tfidf_logreg_text_only",
    "pr_auc": round(average_precision_score(y_test, proba), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
    "f1_dispute": round(f1_score(y_test, pred), 4),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
}
print(json.dumps(metrics, indent=2))

names = vec.get_feature_names_out()
coefs = clf.coef_.ravel()
print("\nStrongest DISPUTE signals:")
for i in np.argsort(coefs)[-15:][::-1]:
    print(f"  +{coefs[i]:6.2f}  {names[i]}")
print("\nStrongest NO-DISPUTE signals:")
for i in np.argsort(coefs)[:15]:
    print(f"  {coefs[i]:6.2f}  {names[i]}")

joblib.dump({"vectorizer": vec, "model": clf}, "models/baseline_text.joblib")
with open("evaluation/baseline_results.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved: models/baseline_text.joblib, evaluation/baseline_results.json")