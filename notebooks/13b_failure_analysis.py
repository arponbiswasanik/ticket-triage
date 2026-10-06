"""13b - Failure-mode analysis: what did the judge actually flag?

1. Claim-density check (is the faithfulness gap a denominator artifact?)
2. Re-judge the 5 worst RAG drafts, print every unsupported claim verbatim
3. Diagnose the 3 citation-validation failures (empty vs hallucinated ids)
"""
import json
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
res = json.load(open("evaluation/harness_results.json"))
records, MODEL = res["records"], res["summary"]["llm"]

rag = [r for r in records if r["rag_status"] == "drafted" and r.get("rag_claims")]
nor = [r for r in records if r.get("norag_claims")]
print("--- 1. claim density (denominator check) ---")
print(f"mean claims/draft: RAG {np.mean([r['rag_claims'] for r in rag]):.1f} | "
      f"no-RAG {np.mean([r['norag_claims'] for r in nor]):.1f}")
print(f"mean words:        RAG {np.mean([r['rag_words'] for r in rag]):.0f} | "
      f"no-RAG {np.mean([r['norag_words'] for r in nor]):.0f}")

corpus = pd.read_parquet("data/processed/retrieval_corpus.parquet")
texts = corpus["consumer_complaint_narrative"].fillna("").tolist()
bm25 = BM25Okapi([t.lower().split() for t in texts])
col = PersistentClient(path="data/vectorstore").get_collection("complaints")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
id2row = {f"c{int(cid)}": i for i, cid in enumerate(corpus["complaint_id"])}
CAND, RRF_K, TOP_K = 20, 60, 5

def retrieve(query):
    qv = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    r = col.query(query_embeddings=[qv], n_results=CAND + 5)
    rrf = {}
    for rank, i in enumerate(r["ids"][0][:CAND]):
        rrf[id2row[i]] = rrf.get(id2row[i], 0) + 1 / (RRF_K + rank + 1)
    for rank, x in enumerate(np.argsort(bm25.get_scores(query.lower().split()))[::-1][:CAND]):
        rrf[x] = rrf.get(x, 0) + 1 / (RRF_K + rank + 1)
    return sorted(rrf, key=rrf.get, reverse=True)[:TOP_K]

def cases_block(rows):
    parts = []
    for x in rows:
        d = corpus.iloc[x]
        excerpt = " ".join(str(d["consumer_complaint_narrative"]).split())[:300]
        parts.append(f"[C-{int(d['complaint_id'])}] (Product: {d['product']} | "
                     f"Issue: {d['issue']} | Outcome: {d['company_response_to_consumer']})\n{excerpt}")
    return "\n---\n".join(parts)

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

client = Groq()

def call(system, user):
    for attempt in range(5):
        try:
            kwargs = dict(model=MODEL,
                          messages=[{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                          temperature=0.0, max_tokens=900)
            if "gpt-oss" in MODEL:
                kwargs["reasoning_effort"] = "low"
            return client.chat.completions.create(**kwargs).choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error ({type(e).__name__}), retry in {20*(attempt+1)}s")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("API failed")

def judge(context, draft):
    raw = call(SYSTEM_JUDGE, f"CONTEXT:\n{context}\n\nDRAFT:\n{draft}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group()) if m else None

test = pd.read_parquet("data/processed/test_routed.parquet")
by_id = {int(r["complaint_id"]): r for _, r in test.iterrows()}

print("\n--- 2. five worst RAG drafts: every claim, verdict ---")
worst = sorted(rag, key=lambda r: r["rag_faith"])[:5]
analysis = []
for w in worst:
    q = by_id[w["complaint_id"]]
    rows = retrieve(q["consumer_complaint_narrative"])
    ctx = f"COMPLAINT:\n{q['consumer_complaint_narrative']}\n\nPAST CASES:\n{cases_block(rows)}"
    j = judge(ctx, w["rag_draft"])
    print("=" * 70)
    print(f"id {w['complaint_id']} | stored faith {w['rag_faith']} | "
          f"({q['product']} | {q['issue']})")
    if j and j.get("claims"):
        for c in j["claims"]:
            print(f"  [{'OK' if c['supported'] else 'XX'}] {c['text'][:150]}")
    else:
        print("  (judge parse failed)")
    analysis.append({"complaint_id": w["complaint_id"], "stored_faith": w["rag_faith"],
                     "claims": j.get("claims") if j else None})
    time.sleep(2)

print("\n--- 3. citation failures ---")
for r in records:
    if r["rag_status"] == "drafted" and not r["rag_citations_valid"]:
        q = by_id[r["complaint_id"]]
        valid = {str(int(corpus.iloc[x]["complaint_id"])) for x in retrieve(q["consumer_complaint_narrative"])}
        cited = set(re.findall(r"\[C-(\d+)\]", r["rag_draft"]))
        print(f"id {r['complaint_id']}: cited={sorted(cited) or 'NONE'} | "
              f"retrieved={sorted(valid)}")
        if not cited:
            loose = re.findall(r"C-?\s?\d{5,}", r["rag_draft"])
            print(f"   near-miss citation patterns in draft: {loose[:5]}")

with open("evaluation/failure_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
print("\nSaved: evaluation/failure_analysis.json")