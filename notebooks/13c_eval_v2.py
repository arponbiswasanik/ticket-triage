"""13c - Prompt v2: SLA/legal/commitment bans, deterministic footer,
citation normalization + retry. Same 20 samples as v1 for comparison.
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

N_SAMPLE, TOP_K, CAND, RRF_K = 20, 5, 20, 60
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
print(f"Model: {MODEL_LLM} | guardrail: {T_SIM}")

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
            print(f"  API error ({type(e).__name__}), retry in {20*(attempt+1)}s")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("API failed after 5 retries")

def retrieve(query):
    qv = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    res = col.query(query_embeddings=[qv], n_results=CAND + 5)
    rrf = {}
    for rank, i in enumerate(res["ids"][0][:CAND]):
        rrf[id2row[i]] = rrf.get(id2row[i], 0) + 1 / (RRF_K + rank + 1)
    for rank, r in enumerate(np.argsort(bm25.get_scores(query.lower().split()))[::-1][:CAND]):
        rrf[r] = rrf.get(r, 0) + 1 / (RRF_K + rank + 1)
    rows = sorted(rrf, key=rrf.get, reverse=True)[:TOP_K]
    return rows

# --- citation handling: normalize BEFORE validating (lesson from 13b) ---
def normalize_citations(text):
    t = text
    for ch in "\u2010\u2011\u2012\u2013\u2014\u2212":      # unicode hyphens/dashes
        t = t.replace(ch, "-")
    t = t.replace("\uff3b", "[").replace("\uff3d", "]")   # fullwidth brackets
    t = re.sub(r"(?<!\[)\bC-(\d{6,})\b", r"[C-\1]", t)    # bare C-123456 -> [C-123456]
    return t

def extract_citations(text):
    return set(re.findall(r"\[C-(\d+)\]", normalize_citations(text)))

SYSTEM_RAG_V2 = """You draft the BODY of a customer-support reply for a financial-services complaint desk. A fixed footer with process information is added automatically - do not add process statements yourself.

STRICT RULES - violations make the draft unusable:
1. NO time commitments. Never mention business days, hours, or deadlines.
2. NO legal or regulatory statements (no FCRA, FDCPA, "required by law"). If legality is relevant, say only that the appropriate team will review it.
3. NO commitments to specific company actions (investigations, fraud alerts, written confirmations, account changes, refunds). Those decisions belong to specialists.
4. You MAY: acknowledge the issue, empathize, summarize what the customer described, and note how similar cases were typically closed (use the "Typical outcome" label).
5. Every statement about similar past cases MUST cite case ids in EXACTLY this format: [C-123456] - square brackets, capital C, hyphen, digits.
6. Use ONLY facts from the NEW COMPLAINT and the SIMILAR PAST CASES. No outside knowledge.
7. 120 words or fewer. Empathetic, professional. Output only the reply body."""

SYSTEM_NORAG_V2 = """You draft the BODY of a customer-support reply for a financial-services complaint desk. A fixed footer is added automatically. You have NO access to similar past cases.

STRICT RULES:
1. NO time commitments. Never mention business days, hours, or deadlines.
2. NO legal or regulatory statements.
3. NO commitments to specific company actions. Those decisions belong to specialists.
4. You MAY: acknowledge the issue, empathize, summarize what the customer described, and say the appropriate team will review it.
5. Use only what the customer told you. No outside knowledge about their situation.
6. 120 words or fewer. Empathetic, professional. Output only the reply body."""

FOOTER = "\n\nYour complaint has been logged. A specialist will review the details you provided and follow up with you."

SYSTEM_JUDGE = """You audit AI-drafted support replies for grounding. Given a CONTEXT (customer complaint, plus similar past cases when available) and a DRAFT, judge every factual claim.

Rules:
- Restating the customer's own words: SUPPORTED.
- Generic empathy or generic next steps without specifics (e.g. "a specialist will review"): SUPPORTED.
- Specifics (names, amounts, dates, timelines, legal rules, past-case outcomes): SUPPORTED only if stated in or directly implied by the CONTEXT.
- Enumerate every factual claim; skip pure pleasantries.

Output ONLY valid JSON, no markdown fences:
{"claims": [{"text": "<claim>", "supported": true}],
 "relevance": <1-5, how well the draft addresses the customer's actual issue>,
 "notes": "<one line>"}"""

def cases_block(rows):
    parts = []
    for x in rows:
        d = corpus.iloc[x]
        excerpt = " ".join(str(d["consumer_complaint_narrative"]).split())[:300]
        parts.append(f"[C-{int(d['complaint_id'])}] (Product: {d['product']} | "
                     f"Issue: {d['issue']} | Outcome: {d['company_response_to_consumer']})\n{excerpt}")
    return "\n---\n".join(parts)

def parse_judge(raw):
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group()) if m else None

def judge(context, draft_body):
    j = parse_judge(call_llm(SYSTEM_JUDGE, f"CONTEXT:\n{context}\n\nDRAFT:\n{draft_body}", 0.0, 900))
    if not j or not j.get("claims"):
        return None, None, None
    claims = j["claims"]
    return round(sum(c["supported"] for c in claims) / len(claims), 3), len(claims), j.get("relevance")

def draft_rag(narrative, rows):
    typical = corpus.iloc[rows]["company_response_to_consumer"].mode()[0]
    user = (f"NEW COMPLAINT:\n{narrative}\n\nSIMILAR PAST CASES:\n{cases_block(rows)}"
            f"\n\nTypical outcome for these cases: {typical}.\nDraft the support reply body.")
    body = call_llm(SYSTEM_RAG_V2, user, 0.2, 700)
    valid = {str(int(corpus.iloc[x]["complaint_id"])) for x in rows}
    raw_cited = set(re.findall(r"\[C-(\d+)\]", body))
    cited = extract_citations(body)
    retried = False
    if not (cited and cited <= valid and raw_cited == cited):
        if not cited or not (cited & valid):
            body = call_llm(SYSTEM_RAG_V2, user + "\n\nREMINDER: cite at least one case id in the exact format [C-123456].", 0.2, 700)
            cited = extract_citations(body)
            retried = True
    return body, cited, valid, retried

test = pd.read_parquet("data/processed/test_routed.parquet")
samples = test[test["route"] == "auto_draft"].sample(N_SAMPLE, random_state=123)
print(f"v2 run: {len(samples)} complaints x 2 conditions (~15 min)...\n")

records = []
for i, (_, q) in enumerate(samples.iterrows(), 1):
    narrative = q["consumer_complaint_narrative"]
    rec = {"complaint_id": int(q["complaint_id"]), "product": q["product"]}
    print(f"[{i:>2}/{N_SAMPLE}] {rec['complaint_id']} ({q['product'][:20]})")

    rows = retrieve(narrative)
    sim_map = {}
    qv = embedder.encode([narrative], normalize_embeddings=True)[0].tolist()
    res = col.query(query_embeddings=[qv], n_results=1)
    top = 1.0 - res["distances"][0][0]
    if top < T_SIM:
        rec["rag_status"] = "guardrail"
        print(f"    RAG: guardrail (sim {top:.3f})")
    else:
        body, cited, valid, retried = draft_rag(narrative, rows)
        faith, nclaims, rel = judge(f"COMPLAINT:\n{narrative}\n\nPAST CASES:\n{cases_block(rows)}", body)
        rec.update(rag_status="drafted", top_sim=round(top, 3), rag_body=body,
                   rag_final_draft=body + FOOTER, rag_cited=sorted(cited),
                   rag_citations_valid=bool(cited) and cited <= valid,
                   rag_retry=retried, rag_faith=faith, rag_claims=nclaims,
                   rag_relevance=rel, rag_words=len(body.split()))
        print(f"    RAG: faith={faith} rel={rel} cite_ok={rec['rag_citations_valid']} retry={retried}")
    time.sleep(2)

    body = call_llm(SYSTEM_NORAG_V2, narrative, 0.2, 700)
    faith, nclaims, rel = judge(f"COMPLAINT:\n{narrative}\n\nPAST CASES: none available", body)
    rec.update(norag_body=body, norag_final_draft=body + FOOTER,
               norag_faith=faith, norag_claims=nclaims, norag_relevance=rel,
               norag_words=len(body.split()))
    print(f"    no-RAG: faith={faith} rel={rel}")
    records.append(rec)
    time.sleep(2)

rag = [r for r in records if r["rag_status"] == "drafted"]
def m(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return round(float(np.mean(v)), 3) if v else None

summary = {
    "n_samples": len(records),
    "rag": {"n_drafts": len(rag), "guardrail_rate": round(1 - len(rag)/len(records), 3),
            "faithfulness": m(rag, "rag_faith"),
            "any_unsupported": round(float(np.mean([r["rag_faith"] < 1.0 for r in rag if r.get("rag_faith") is not None])), 3),
            "relevance": m(rag, "rag_relevance"), "claims_per_draft": m(rag, "rag_claims"),
            "citation_validity": m(rag, "rag_citations_valid"),
            "retry_rate": round(float(np.mean([r["rag_retry"] for r in rag])), 3),
            "mean_words": m(rag, "rag_words")},
    "norag": {"faithfulness": m(records, "norag_faith"),
              "any_unsupported": round(float(np.mean([r["norag_faith"] < 1.0 for r in records if r.get("norag_faith") is not None])), 3),
              "relevance": m(records, "norag_relevance"), "claims_per_draft": m(records, "norag_claims"),
              "mean_words": m(records, "norag_words")},
    "llm": MODEL_LLM,
}

v1 = json.load(open("evaluation/harness_results.json"))["summary"]
print("\n" + "=" * 62)
print(f"{'metric':<26}{'v1 RAG':>10}{'v2 RAG':>10}{'v1 noRAG':>10}{'v2 noRAG':>10}")
rows = [("faithfulness", v1["rag"]["faithfulness"], summary["rag"]["faithfulness"],
         v1["norag"]["faithfulness"], summary["norag"]["faithfulness"]),
        ("relevance", v1["rag"]["relevance"], summary["rag"]["relevance"],
         v1["norag"]["relevance"], summary["norag"]["relevance"]),
        ("claims/draft", None, summary["rag"]["claims_per_draft"],
         None, summary["norag"]["claims_per_draft"]),
        ("citation validity", v1["rag"]["citation_validity"], summary["rag"]["citation_validity"], None, None),
        ("mean words", v1["rag"]["mean_words"], summary["rag"]["mean_words"],
         v1["norag"]["mean_words"], summary["norag"]["mean_words"])]
for name, a, b, c, d in rows:
    print(f"{name:<26}{str(a):>10}{str(b):>10}{str(c):>10}{str(d):>10}")

with open("evaluation/harness_v2_results.json", "w") as f:
    json.dump({"summary": summary, "records": records}, f, indent=2)
print("\nSaved: evaluation/harness_v2_results.json")