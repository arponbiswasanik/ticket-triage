"""09 - Routing policy: cost-derived auto-draft threshold + risk-tiered queues.

- auto_draft:  calibrated P(dispute) < t* = C_human / C_dispute_missed
- priority_human: top decile of risk (percentile threshold -> drift-robust)
- standard_human: everything else
Thresholds chosen WITHOUT the 2017 test set; test is reported once.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- illustrative business costs (assumptions, stated in README) ---
C_HUMAN = 15.0    # $ agent handling cost per complaint
C_DISPUTE = 75.0  # $ cost when an auto-drafted complaint ends in dispute
C_AUTO = 1.0      # $ LLM drafting cost per auto-drafted complaint
T_STAR = C_HUMAN / C_DISPUTE

art = joblib.load("models/final_calibrated.joblib")
model, iso = art["model"], art["calibrator"]

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

CAT = ["product", "sub_product", "issue", "sub_issue", "submitted_via", "state", "company"]

def prep(df):
    df = df.copy()
    df["n_chars"] = df["consumer_complaint_narrative"].str.len()
    df["n_words"] = df["consumer_complaint_narrative"].str.split().str.len()
    for c in CAT:
        df[c] = df[c].fillna("missing").astype(str)
    return df

n_cal = int(len(train) * 0.15)          # same temporal slice as step 08
calib = prep(train.iloc[-n_cal:])
test = prep(test)

p_calib = iso.predict(model.predict_proba(calib)[:, 1])
p_test = iso.predict(model.predict_proba(test)[:, 1])
y_test = (test["consumer_disputed"] == "YES").astype(int).values

T_PRIORITY = float(np.quantile(p_calib, 0.90))

print(f"auto-draft threshold t* = C_human/C_dispute = {T_STAR:.3f}")
print(f"priority threshold = 90th pct of calib risk = {T_PRIORITY:.3f}")
print(f"mean predicted risk: calib {p_calib.mean():.3f} | test {p_test.mean():.3f} "
      f"(actual test {y_test.mean():.3f})")

def route(p):
    return "auto_draft" if p < T_STAR else ("priority_human" if p >= T_PRIORITY else "standard_human")

routes = np.array([route(p) for p in p_test])
test["p_dispute"] = p_test
test["route"] = routes
n = len(test)

print("\n--- ROUTING OUTCOME on 2017 holdout (reported once) ---")
for tier in ["auto_draft", "standard_human", "priority_human"]:
    m = routes == tier
    print(f"{tier:>16}: {m.sum():>6,} ({m.mean():5.1%}) | "
          f"actual dispute rate {y_test[m].mean():5.1%}")

m_auto = routes == "auto_draft"
m_prio = routes == "priority_human"
disputes_total = y_test.sum()
print(f"\npriority queue captures {y_test[m_prio].sum():,} of {disputes_total:,} "
      f"disputes ({y_test[m_prio].sum()/disputes_total:.1%}) while reviewing {m_prio.mean():.1%} of volume")

missed = int(y_test[m_auto].sum())
cost_policy = m_auto.sum() * C_AUTO + (~m_auto).sum() * C_HUMAN + missed * C_DISPUTE
cost_baseline = n * C_HUMAN
print(f"\ncost per complaint: policy ${cost_policy/n:.2f} vs all-human ${cost_baseline/n:.2f} "
      f"({(1 - cost_policy/cost_baseline):.1%} saved)")

# --- threshold sweep (exhibit only; operating point came from cost arithmetic) ---
sweep = []
for t in np.arange(0.05, 0.61, 0.05):
    auto = p_test < t
    na = int(auto.sum())
    miss = int(y_test[auto].sum())
    cost = na * C_AUTO + (n - na) * C_HUMAN + miss * C_DISPUTE
    sweep.append({"t": round(float(t), 2), "automation": round(na / n, 3),
                  "dispute_rate_auto": round(miss / na, 3) if na else None,
                  "cost_per_complaint": round(cost / n, 2)})

sweep_df = pd.DataFrame(sweep)
print("\n--- threshold sweep (test exhibit) ---")
print(sweep_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(sweep_df["automation"], sweep_df["cost_per_complaint"], "o-")
ours = m_auto.mean()
ax.axvline(ours, color="r", ls="--", label=f"chosen t*={T_STAR:.2f}")
ax.set_xlabel("automation rate (share auto-drafted)")
ax.set_ylabel("cost per complaint ($)")
ax.set_title("Routing cost curve - 2017 holdout")
ax.legend(); ax.grid(alpha=0.3)
fig.savefig("evaluation/routing_cost_curve.png", dpi=150, bbox_inches="tight")

test.to_parquet("data/processed/test_routed.parquet", index=False)
with open("evaluation/routing_config.json", "w") as f:
    json.dump({"t_auto_draft": round(T_STAR, 4), "t_priority": round(T_PRIORITY, 4),
               "costs": {"C_human": C_HUMAN, "C_dispute": C_DISPUTE, "C_auto": C_AUTO},
               "automation_rate": round(float(m_auto.mean()), 4),
               "dispute_rate_auto_draft": round(float(y_test[m_auto].mean()), 4),
               "priority_dispute_capture": round(float(y_test[m_prio].sum() / disputes_total), 4),
               "cost_saved_pct": round(float(1 - cost_policy / cost_baseline), 4)}, f, indent=2)
print("\nSaved: evaluation/routing_cost_curve.png, evaluation/routing_config.json, data/processed/test_routed.parquet")