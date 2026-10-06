"""13 - Eval harness: LLM-as-judge faithfulness + relevance. Ablation: RAG vs no-RAG.

Faithfulness = share of factual claims traceable to provided context.
Deliberate: accurate general knowledge (laws, typical timelines) counts as
UNSUPPORTED - regulated support desks need verifiable grounding.
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
    raise SystemExit("GROQ_API_KEY missing in .env")

N_SAMPLE = 20                     # lower to 10 if rate limits bite
TOP_K, CAND, RRF_K = 5, 20, 60
PREFERRED_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

corpus = pd.read_parquet("data/processed/retrieval_corpus.parquet")
texts = corpus["consumer_complaint_narrative"].fillna("").tolist()
bm25 = BM25Okapi([t.lower().split() for t in texts])
col = PersistentClient(path="data/vectorstore").get_collection("complaints")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
id2row = {f"c{int(cid)}": i for i, cid in enumerate(corpus["complaint_id"])}
T_SIM = json.load(open("evaluation/guardrail_threshold.json"))["threshold"]

client = Groq()
available = {m.id for m in client.models.list().data}
MODEL_LLM = next((m for m in PREFERRED_MODELS if m in available), None)
if MODEL_LLM is None:
    raise SystemExit(f"No preferred model. Available: {sorted(available)}")
print(f"Model: {MODEL_LLM} | guardrail threshold: {T_SIM}")

def call_llm(system, user, temperature, max_tokens):
    for attempt in range(5):
        try:
            kwargs = dict(model=MODEL_LLM,
                          messages=[{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                          temperature=temperature, max_tokens=max_tokens)
            if "gpt-oss" in MODEL_LLM:
                kwargs["reasoning_effort"] = "low"
            return client.chat.completions.create(**kwargs).choices[0].message.content.strip()
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(f"  API error ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("API failed after 5 retries")

def retrieve(query, k=TOP_K):
    qv = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    res = col.query(query_embeddings=[qv], n_results=CAND + 5)
    sim_by_id = {i: 1.0 - d for i, d in zip(res["ids"][0], res["distances"][0])}
    rrf = {}
    for rank, i in enumerate(res["ids"][0][:CAND]):
        rrf[id2row[i]] = rrf.get(id2row[i], 0) + 1 / (RRF_K + rank + 1)
    for rank, r in enumerate(np.argsort(bm25.get_scores(query.lower().split()))[::-1][:CAND]):
        rrf[r] = rrf.get(r, 0) + 1 / (RRF_K + rank + 1)
    rows = sorted(rrf, key=rrf.get, reverse=True)[:k]
    row2cid = {id2row[i]: i for i in sim_by_id}
    sims = [sim_by_id.get(row2cid.get(r), 0.0) for r in rows]
    return rows, sims

SYSTEM_RAG = """You draft customer-support replies for a financial-services complaint desk.

Rules:
1. Ground factual statements in the SIMILAR PAST CASES provided, citing them as [C-<case id>].
2. Never invent names, amounts, dates, account numbers, or outcomes. Use only what the cases state.
3. You may describe how cases like this are typically resolved (see "Typical outcome"), with citations.
4. If the cases do not support a concrete next step, acknowledge the complaint, apologize, and say a specialist will follow up shortly. Do not fabricate specifics.
5. Empathetic, professional tone. Under 180 words. Output only the reply text."""

SYSTEM_NORAG = """You draft customer-support replies for a financial-services complaint desk.
You have NO access to similar past cases.

Rules:
1. Never invent names, amounts, dates, account numbers, or outcomes about this customer's situation.
2. You may use general knowledge where helpful.
3. If you cannot be concrete, acknowledge the complaint, apologize, and say a specialist will follow up.
4. Empathetic, professional tone. Under 180 words. Output only the reply text."""

SYSTEM_JUDGE = """You audit AI-drafted support replies for grounding. Given a CONTEXT (customer complaint, plus similar past cases when available) and a DRAFT, judge every factual claim.

Rules:
- Restating the customer's own words: SUPPORTED.
- Generic empathy or generic next steps without specifics (e.g. "a specialist will review"): SUPPORTED.
- Specifics (names, amounts, dates, timelines/SLAs, legal rules, past-case outcomes): SUPPORTED only if stated in or directly implied by the CONTEXT.
- Enumerate every factual claim; skip pure pleasantries.

Output ONLY valid JSON:
{"claims": [{"text": "<claim>", "supported": true}],
 "relevance": <1-5, how well the draft addresses the customer's actual issue>,
 "notes": "<one line>"}"""

def cases_block(rows):
    parts = []
    for r in rows:
        d = corpus.iloc[r]
        excerpt = " ".join(str(d["consumer_complaint_narrative"]).split())[:300]
        parts.append(f"[C-{int(d['complaint_id'])}] (Product: {d['product']} | "
                     f"Issue: {d['issue']} | Outcome: {d['company_response_to_consumer']})\n{excerpt}")
    return "\n---\n".join(parts)

def judge(context, draft):
    raw = call_llm(SYSTEM_JUDGE, f"CONTEXT:\n{context}\n\nDRAFT:\n{draft}", 0.0, 900)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group()) if m else None

def score(j):
    if not j or not j.get("claims"):
        return None, None, None
    claims = j["claims"]
    faith = sum(c["supported"] for c in claims) / len(claims)
    return round(faith, 3), len(claims), j.get("relevance")

test = pd.read_parquet("data/processed/test_routed.parquet")
samples = test[test["route"] == "auto_draft"].sample(N_SAMPLE, random_state=123)
print(f"Evaluating {len(samples)} complaints x 2 conditions (~10 min)...\n")

records = []
for i, (_, q) in enumerate(samples.iterrows(), 1):
    narrative = q["consumer_complaint_narrative"]
    rec = {"complaint_id": int(q["complaint_id"]), "product": q["product"]}
    print(f"[{i:>2}/{N_SAMPLE}] {rec['complaint_id']} ({q['product'][:20]})")

    # --- A: RAG ---
    rows, sims = retrieve(narrative)
    top = max(sims) if sims else 0.0
    if top < T_SIM:
        rec["rag_status"], rec["top_sim"] = "guardrail", round(top, 3)
        print(f"    RAG: guardrail (sim {top:.3f})")
    else:
        draft = call_llm(SYSTEM_RAG,
                         f"NEW COMPLAINT:\n{narrative}\n\nSIMILAR PAST CASES:\n{cases_block(rows)}"
                         f"\n\nTypical outcome: {corpus.iloc[rows]['company_response_to_consumer'].mode()[0]}.\nDraft the support reply.",
                         0.2, 700)
        cited = set(re.findall(r"\[C-(\d+)\]", draft))
        valid = {str(int(corpus.iloc[r]["complaint_id"])) for r in rows}
        faith, nclaims, rel = score(judge(f"COMPLAINT:\n{narrative}\n\nPAST CASES:\n{cases_block(rows)}", draft))
        rec.update(rag_status="drafted", top_sim=round(top, 3), rag_draft=draft,
                   rag_citations_valid=bool(cited) and cited <= valid,
                   rag_faith=faith, rag_claims=nclaims, rag_relevance=rel,
                   rag_words=len(draft.split()))
        print(f"    RAG: faith={faith} rel={rel} cite_ok={rec['rag_citations_valid']}")
    time.sleep(2)

    # --- B: no-RAG (same complaint, no retrieval) ---
    draft = call_llm(SYSTEM_NORAG, narrative, 0.2, 700)
    faith, nclaims, rel = score(judge(f"COMPLAINT:\n{narrative}\n\nPAST CASES: none available", draft))
    rec.update(norag_draft=draft, norag_faith=faith, norag_claims=nclaims,
               norag_relevance=rel, norag_words=len(draft.split()))
    print(f"    no-RAG: faith={faith} rel={rel}")
    records.append(rec)
    time.sleep(2)

# --- summary ---
rag = [r for r in records if r["rag_status"] == "drafted"]
def m(rs, k): 
    v = [r[k] for r in rs if r.get(k) is not None]
    return round(float(np.mean(v)), 3) if v else None

summary = {
    "n_samples": len(records),
    "rag": {"n_drafts": len(rag),
            "guardrail_rate": round(1 - len(rag) / len(records), 3),
            "faithfulness": m(rag, "rag_faith"),
            "any_unsupported": round(float(np.mean([r["rag_faith"] < 1.0 for r in rag if r.get("rag_faith") is not None])), 3),
            "relevance": m(rag, "rag_relevance"),
            "citation_validity": m(rag, "rag_citations_valid"),
            "mean_words": m(rag, "rag_words"),
            "over_word_limit": round(float(np.mean([r["rag_words"] > 180 for r in rag])), 3)},
    "norag": {"faithfulness": m(records, "norag_faith"),
              "any_unsupported": round(float(np.mean([r["norag_faith"] < 1.0 for r in records if r.get("norag_faith") is not None])), 3),
              "relevance": m(records, "norag_relevance"),
              "mean_words": m(records, "norag_words")},
    "llm": MODEL_LLM,
}
print("\n" + "=" * 60)
print(json.dumps(summary, indent=2))

with open("evaluation/harness_results.json", "w") as f:
    json.dump({"summary": summary, "records": records}, f, indent=2)
print("\nSaved: evaluation/harness_results.json")