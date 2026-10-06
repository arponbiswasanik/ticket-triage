"""08 - Model selection + isotonic calibration.

Discipline: features FROZEN after step 07. Candidates compared on a held-out
temporal calibration slice; winner calibrated there; 2017 holdout used once.
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

n_cal = int(len(train) * 0.15)
fit, calib = train.iloc[:-n_cal], train.iloc[-n_cal:]
print(f"fit   : {len(fit):,} rows ({fit['date'].min().date()} -> {fit['date'].max().date()})")
print(f"calib : {len(calib):,} rows ({calib['date'].min().date()} -> {calib['date'].max().date()})")
print(f"test  : {len(test):,} rows ({test['date'].min().date()} -> {test['date'].max().date()})")

CAT = ["product", "sub_product", "issue", "sub_issue", "submitted_via", "state", "company"]
NUM = ["n_chars", "n_words"]

def prep(df):
    df = df.copy()
    df["n_chars"] = df["consumer_complaint_narrative"].str.len()
    df["n_words"] = df["consumer_complaint_narrative"].str.split().str.len()
    for c in CAT:
        df[c] = df[c].fillna("missing").astype(str)
    return df

fit, calib, test = prep(fit), prep(calib), prep(test)
y = lambda d: (d["consumer_disputed"] == "YES").astype(int)
y_fit, y_calib, y_test = y(fit), y(calib), y(test)
print(f"\nDispute rate: fit {y_fit.mean():.1%} | calib {y_calib.mean():.1%} | test {y_test.mean():.1%}")

def make_pre():
    return ColumnTransformer([
        ("text", TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=50_000,
                                 sublinear_tf=True), "consumer_complaint_narrative"),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT),
        ("num", StandardScaler(), NUM),
    ])

candidates = {}

t0 = time.time()
lr = Pipeline([("pre", make_pre()),
               ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          solver="liblinear", random_state=42))])
lr.fit(fit, y_fit)
candidates["logreg"] = (lr, time.time() - t0)

t0 = time.time()
lgbm = Pipeline([("pre", make_pre()),
                 ("clf", LGBMClassifier(n_estimators=250, learning_rate=0.07,
                                        num_leaves=63, min_child_samples=100,
                                        colsample_bytree=0.7, subsample=0.8,
                                        subsample_freq=1, random_state=42,
                                        n_jobs=-1, verbosity=-1))])
lgbm.fit(fit, y_fit)
candidates["lightgbm"] = (lgbm, time.time() - t0)

print("\n--- candidate comparison on CALIBRATION slice ---")
scores = {}
for name, (m, secs) in candidates.items():
    p = m.predict_proba(calib)[:, 1]
    scores[name] = {"roc_auc": round(roc_auc_score(y_calib, p), 4),
                    "pr_auc": round(average_precision_score(y_calib, p), 4),
                    "fit_seconds": round(secs)}
    print(f"{name:>10}: ROC-AUC {scores[name]['roc_auc']:.4f} | "
          f"PR-AUC {scores[name]['pr_auc']:.4f} | fit {secs:.0f}s")

winner = max(scores, key=lambda k: scores[k]["roc_auc"])
print(f"\nWinner: {winner}")
model = candidates[winner][0]

# --- isotonic calibration, fitted on the calibration slice ---
p_calib_raw = model.predict_proba(calib)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
iso.fit(p_calib_raw, y_calib)

p_test_raw = model.predict_proba(test)[:, 1]
p_test_cal = iso.predict(p_test_raw)

def ece(y_true, p, n_bins=10):
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return sum((bins == b).mean() * abs(y_true[bins == b].mean() - p[bins == b].mean())
               for b in range(n_bins) if (bins == b).any())

results = {
    "winner": winner,
    "calib_slice_scores": scores,
    "test_roc_auc_raw": round(roc_auc_score(y_test, p_test_raw), 4),
    "test_roc_auc_calibrated": round(roc_auc_score(y_test, p_test_cal), 4),
    "test_pr_auc_calibrated": round(average_precision_score(y_test, p_test_cal), 4),
    "test_brier_raw": round(brier_score_loss(y_test, p_test_raw), 4),
    "test_brier_calibrated": round(brier_score_loss(y_test, p_test_cal), 4),
    "test_ece_raw": round(ece(y_test.values, p_test_raw), 4),
    "test_ece_calibrated": round(ece(y_test.values, p_test_cal), 4),
}
print("\n--- FINAL evaluation on 2017 holdout (reported once) ---")
print(json.dumps(results, indent=2))

def reliability(y_true, p, n_bins=10):
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            xs.append(p[m].mean()); ys.append(y_true[m].mean())
    return xs, ys

fig, ax = plt.subplots(figsize=(6, 6))
xs, ys = reliability(y_test.values, p_test_raw)
ax.plot(xs, ys, "o--", label=f"uncalibrated (ECE {results['test_ece_raw']:.3f})")
xs, ys = reliability(y_test.values, p_test_cal)
ax.plot(xs, ys, "s-", label=f"isotonic (ECE {results['test_ece_calibrated']:.3f})")
ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
ax.set_xlabel("predicted probability"); ax.set_ylabel("observed dispute rate")
ax.set_title(f"Reliability on 2017 holdout - {winner}")
ax.legend(); ax.grid(alpha=0.3)
fig.savefig("evaluation/reliability_curve.png", dpi=150, bbox_inches="tight")

joblib.dump({"model": model, "calibrator": iso, "winner": winner},
            "models/final_calibrated.joblib")
with open("evaluation/step7_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: evaluation/reliability_curve.png, models/final_calibrated.joblib, evaluation/step7_results.json")