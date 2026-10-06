"""Production pipeline: risk scoring, routing, retrieval, grounded drafting.

Loads all artifacts produced by the notebook experiments and exposes them
as one callable pipeline. Used by the FastAPI app (src.api).
"""
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
from chromadb import PersistentClient
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

CAT = ["product", "sub_product", "issue", "sub_issue", "submitted_via", "state", "company"]
TOP_K, CAND, RRF_K = 5, 20, 60
PREFERRED_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

SYSTEM_RAG_V2 = """You draft the BODY of a customer-support reply for a financial-services complaint desk. A fixed footer with process information is added automatically - do not add process statements yourself.

STRICT RULES - violations make the draft unusable:
1. NO time commitments. Never mention business days, hours, or deadlines.
2. NO legal or regulatory statements (no FCRA, FDCPA, "required by law"). If legality is relevant, say only that the appropriate team will review it.
3. NO commitments to specific company actions (investigations, fraud alerts, written confirmations, account changes, refunds). Those decisions belong to specialists.
4. You MAY: acknowledge the issue, empathize, summarize what the customer described, and note how similar cases were typically closed (use the "Typical outcome" label).
5. Every statement about similar past cases MUST cite case ids in EXACTLY this format: [C-123456] - square brackets, capital C, hyphen, digits.
6. Use ONLY facts from the NEW COMPLAINT and the SIMILAR PAST CASES. No outside knowledge.
7. 120 words or fewer. Empathetic, professional. Output only the reply body."""

FOOTER = "\n\nYour complaint has been logged. A specialist will review the details you provided and follow up with you."


def normalize_citations(text):
    t = text
    for ch in "\u2010\u2011\u2012\u2013\u2014\u2212":
        t = t.replace(ch, "-")
    t = t.replace("\uff3b", "[").replace("\uff3d", "]")
    t = re.sub(r"(?<!\[)\bC-(\d{6,})\b", r"[C-\1]", t)
    return t


def extract_citations(text):
    return set(re.findall(r"\[C-(\d+)\]", normalize_citations(text)))


class TriagePipeline:
    def __init__(self):
        art = joblib.load("models/final_calibrated.joblib")
        self.model, self.iso = art["model"], art["calibrator"]
        rc = json.load(open("evaluation/routing_config.json"))
        self.t_auto, self.t_priority = rc["t_auto_draft"], rc["t_priority"]
        self.t_sim = json.load(open("evaluation/guardrail_threshold.json"))["threshold"]

        self.corpus = pd.read_parquet("data/processed/retrieval_corpus.parquet")
        texts = self.corpus["consumer_complaint_narrative"].fillna("").tolist()
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])
        self.col = PersistentClient(path="data/vectorstore").get_collection("complaints")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.id2row = {f"c{int(cid)}": i for i, cid in enumerate(self.corpus["complaint_id"])}

        self.client = Groq()
        available = {m.id for m in self.client.models.list().data}
        self.llm = next((m for m in PREFERRED_MODELS if m in available), None)
        if self.llm is None:
            raise RuntimeError(f"No preferred LLM. Available: {sorted(available)}")

    # --- risk + routing ---
    def risk(self, complaint: dict) -> float:
        df = pd.DataFrame([{**{c: complaint.get(c, "missing") for c in CAT},
                            "consumer_complaint_narrative": complaint["narrative"]}])
        df["n_chars"] = df["consumer_complaint_narrative"].str.len()
        df["n_words"] = df["consumer_complaint_narrative"].str.split().str.len()
        raw = self.model.predict_proba(df)[:, 1]
        return float(self.iso.predict(raw)[0])

    def route(self, p: float) -> str:
        if p < self.t_auto:
            return "auto_draft"
        return "priority_human" if p >= self.t_priority else "standard_human"

    # --- retrieval ---
    def retrieve(self, query):
        qv = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        res = self.col.query(query_embeddings=[qv], n_results=CAND + 5)
        sim_by_id = {i: 1.0 - d for i, d in zip(res["ids"][0], res["distances"][0])}
        rrf = {}
        for rank, i in enumerate(res["ids"][0][:CAND]):
            rrf[self.id2row[i]] = rrf.get(self.id2row[i], 0) + 1 / (RRF_K + rank + 1)
        for rank, r in enumerate(np.argsort(self.bm25.get_scores(query.lower().split()))[::-1][:CAND]):
            rrf[r] = rrf.get(r, 0) + 1 / (RRF_K + rank + 1)
        rows = sorted(rrf, key=rrf.get, reverse=True)[:TOP_K]
        row2cid = {self.id2row[i]: i for i in sim_by_id}
        sims = [sim_by_id.get(row2cid.get(r), 0.0) for r in rows]
        return rows, sims

    def cases_block(self, rows):
        parts = []
        for x in rows:
            d = self.corpus.iloc[x]
            excerpt = " ".join(str(d["consumer_complaint_narrative"]).split())[:300]
            parts.append(f"[C-{int(d['complaint_id'])}] (Product: {d['product']} | "
                         f"Issue: {d['issue']} | Outcome: {d['company_response_to_consumer']})\n{excerpt}")
        return "\n---\n".join(parts)

    # --- drafting ---
    def draft(self, narrative):
        rows, sims = self.retrieve(narrative)
        top = max(sims) if sims else 0.0
        if top < self.t_sim:
            return {"draft_status": "guardrail", "top_sim": round(top, 3),
                    "guardrail_threshold": round(self.t_sim, 3)}
        typical = self.corpus.iloc[rows]["company_response_to_consumer"].mode()[0]
        user = (f"NEW COMPLAINT:\n{narrative}\n\nSIMILAR PAST CASES:\n{self.cases_block(rows)}"
                f"\n\nTypical outcome for these cases: {typical}.\nDraft the support reply body.")
        kwargs = dict(model=self.llm,
                      messages=[{"role": "system", "content": SYSTEM_RAG_V2},
                                {"role": "user", "content": user}],
                      temperature=0.2, max_tokens=700)
        if "gpt-oss" in self.llm:
            kwargs["reasoning_effort"] = "low"
        body = self.client.chat.completions.create(**kwargs).choices[0].message.content.strip()
        cited = extract_citations(body)
        valid = {str(int(self.corpus.iloc[r]["complaint_id"])) for r in rows}
        return {"draft_status": "drafted", "top_sim": round(top, 3),
                "draft": body + FOOTER,
                "citations": sorted(cited),
                "citations_valid": bool(cited) and cited <= valid}

    # --- full flow ---
    def triage(self, complaint: dict, want_draft: bool = True) -> dict:
        t0 = time.perf_counter()
        p = self.risk(complaint)
        out = {"risk": round(p, 4), "route": self.route(p),
               "risk_ms": int((time.perf_counter() - t0) * 1000)}
        if want_draft:
            out.update(self.draft(complaint["narrative"]))
        return out