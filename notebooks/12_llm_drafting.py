"""12 - LLM drafting: grounded replies with citations + low-similarity guardrail.

Flow: retrieve top-k similar past cases (hybrid)
   -> guardrail: top-1 cosine sim must clear threshold (set on CALIB slice, p10)
   -> LLM drafts reply citing [C-<id>]
   -> citation check: cited ids must be within the retrieved set
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd
from chromadb import PersistentClient
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise SystemExit("GROQ_API_KEY missing -> create at console.groq.com, put in .env")

TOP_K, CAND, RRF_K = 5, 20, 60

# Hosted LLM providers deprecate/rename models constantly -> resolve at runtime
PREFERRED_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

corpus = pd.read_parquet("data/processed/retrieval_corpus.parquet")
texts = corpus["consumer_complaint_narrative"].fillna("").tolist()
bm25 = BM25Okapi([t.lower().split() for t in texts])
col = PersistentClient(path="data/vectorstore").get_collection("complaints")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
id2row = {f"c{int(cid)}": i for i, cid in enumerate(corpus["complaint_id"])}

def retrieve(query, k=TOP_K, exclude_id=None):
    qv = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    res = col.query(query_embeddings=[qv], n_results=CAND + 5)
    sim_by_id = {i: 1.0 - d for i, d in zip(res["ids"][0], res["distances"][0])}
    dense_ids = [i for i in res["ids"][0] if i != exclude_id][:CAND]
    rrf = {}
    for rank, i in enumerate(dense_ids):
        rrf[id2row[i]] = rrf.get(id2row[i], 0) + 1 / (RRF_K + rank + 1)
    for rank, r in enumerate(np.argsort(bm25.get_scores(query.lower().split()))[::-1][:CAND]):
        rrf[r] = rrf.get(r, 0) + 1 / (RRF_K + rank + 1)
    rows = sorted(rrf, key=rrf.get, reverse=True)[:k]
    row2cid = {id2row[i]: i for i in sim_by_id}
    sims = [sim_by_id.get(row2cid.get(r), 0.0) for r in rows]
    return rows, sims

# --- guardrail threshold: p10 of top-1 similarity on CALIB queries (cached) ---
CACHE = "evaluation/guardrail_threshold.json"
if os.path.exists(CACHE):
    T_SIM = json.load(open(CACHE))["threshold"]
    print(f"Guardrail threshold (cached): {T_SIM:.3f}")
else:
    train = pd.read_parquet("data/processed/train.parquet")
    n_cal = int(len(train) * 0.15)
    calib_q = train.iloc[-n_cal:].sample(200, random_state=42)
    print("Guardrail: computing top-1 similarity on 200 calibration queries (~2 min)...")
    top_sims = []
    for _, q in calib_q.iterrows():
        _, sims = retrieve(q["consumer_complaint_narrative"], k=3,
                           exclude_id=f"c{int(q['complaint_id'])}")
        if sims:
            top_sims.append(max(sims))
    T_SIM = float(np.quantile(top_sims, 0.10))
    print(f"top-1 similarity: median {np.median(top_sims):.3f} | "
          f"guardrail threshold (p10) = {T_SIM:.3f}")
    with open(CACHE, "w") as f:
        json.dump({"threshold": round(T_SIM, 4),
                   "n_calib_queries": len(top_sims)}, f, indent=2)

SYSTEM = """You draft customer-support replies for a financial-services complaint desk.

Rules:
1. Ground factual statements in the SIMILAR PAST CASES provided, citing them as [C-<case id>].
2. Never invent names, amounts, dates, account numbers, or outcomes. Use only what the cases state.
3. You may describe how cases like this are typically resolved (see "Typical outcome"), with citations.
4. If the cases do not support a concrete next step, acknowledge the complaint, apologize, and say a specialist will follow up shortly. Do not fabricate specifics.
5. Empathetic, professional tone. Under 180 words. Output only the reply text."""

def build_prompt(narrative, rows):
    outcomes = corpus.iloc[rows]["company_response_to_consumer"].value_counts()
    majority = outcomes.index[0] if len(outcomes) else "Closed with explanation"
    cases = []
    for r in rows:
        d = corpus.iloc[r]
        excerpt = " ".join(str(d["consumer_complaint_narrative"]).split())[:400]
        cases.append(f"[C-{int(d['complaint_id'])}] (Product: {d['product']} | "
                     f"Issue: {d['issue']} | Outcome: {d['company_response_to_consumer']})\n{excerpt}")
    user = (f"NEW COMPLAINT:\n{narrative}\n\nSIMILAR PAST CASES:\n"
            + "\n---\n".join(cases)
            + f"\n\nTypical outcome for these cases: {majority}.\nDraft the support reply.")
    return user

client = Groq()

available = {m.id for m in client.models.list().data}
MODEL_LLM = next((m for m in PREFERRED_MODELS if m in available), None)
if MODEL_LLM is None:
    raise SystemExit(f"No preferred model available.\nAvailable models: {sorted(available)}")
print(f"LLM resolved: {MODEL_LLM}")

def draft_reply(narrative, exclude_id=None):
    rows, sims = retrieve(narrative, exclude_id=exclude_id)
    top = max(sims) if sims else 0.0
    if top < T_SIM:
        return {"status": "guardrail", "top_sim": round(top, 3),
                "threshold": round(T_SIM, 3)}
    t0 = time.time()
    kwargs = dict(
        model=MODEL_LLM,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": build_prompt(narrative, rows)}],
        temperature=0.2, max_tokens=700,
    )
    if "gpt-oss" in MODEL_LLM:
        kwargs["reasoning_effort"] = "low"   # reasoning tokens share the budget
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content.strip()
    cited = set(re.findall(r"\[C-(\d+)\]", text))
    valid = {str(int(corpus.iloc[r]["complaint_id"])) for r in rows}
    return {"status": "drafted", "top_sim": round(top, 3), "threshold": round(T_SIM, 3),
            "draft": text, "cited": sorted(cited),
            "citations_valid": bool(cited) and cited <= valid,
            "llm_seconds": round(time.time() - t0, 1)}

test = pd.read_parquet("data/processed/test_routed.parquet")
samples = test[test["route"] == "auto_draft"].sample(3, random_state=7)

outputs = []
for _, q in samples.iterrows():
    out = draft_reply(q["consumer_complaint_narrative"])
    print("=" * 70)
    print(f"COMPLAINT ({q['product']} | {q['issue']}):\n"
          f"{q['consumer_complaint_narrative'][:300]}...")
    if out["status"] == "guardrail":
        print(f"\n>> GUARDRAIL FIRED: top similarity {out['top_sim']} < "
              f"{out['threshold']} -> escalate to human, no draft")
    else:
        print(f"\n>> similarity {out['top_sim']} | citations valid: "
              f"{out['citations_valid']} | {out['llm_seconds']}s\n")
        print(out["draft"])
    outputs.append({"complaint_id": int(q["complaint_id"]), **out})

with open("evaluation/draft_samples.json", "w") as f:
    json.dump(outputs, f, indent=2)
print("\nSaved: evaluation/draft_samples.json")