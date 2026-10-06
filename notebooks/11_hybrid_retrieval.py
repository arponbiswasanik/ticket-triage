"""11 - Hybrid retrieval (BM25 + dense, RRF fusion) + sanity benchmark.

Proxy metric: product-match of retrieved docs (real RAG eval comes later
with the golden set + LLM answers).
"""
import time

import numpy as np
import pandas as pd
from chromadb import PersistentClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

TOP_K, RRF_K, CAND = 5, 60, 20

corpus = pd.read_parquet("data/processed/retrieval_corpus.parquet")
texts = corpus["consumer_complaint_narrative"].fillna("").tolist()
bm25 = BM25Okapi([t.lower().split() for t in texts])
print(f"BM25 index built over {len(texts):,} docs")

col = PersistentClient(path="data/vectorstore").get_collection("complaints")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
id2row = {f"c{int(cid)}": i for i, cid in enumerate(corpus["complaint_id"])}

def dense_search(query, k=CAND):
    qv = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    return [id2row[i] for i in col.query(query_embeddings=[qv], n_results=k)["ids"][0]]

def bm25_search(query, k=CAND):
    return list(np.argsort(bm25.get_scores(query.lower().split()))[::-1][:k])

def hybrid(query, k=TOP_K):
    rrf = {}
    for rank, i in enumerate(dense_search(query)):
        rrf[i] = rrf.get(i, 0) + 1 / (RRF_K + rank + 1)
    for rank, i in enumerate(bm25_search(query)):
        rrf[i] = rrf.get(i, 0) + 1 / (RRF_K + rank + 1)
    return sorted(rrf, key=rrf.get, reverse=True)[:k]

# --- qualitative: 3 auto-draft-tier complaints ---
test = pd.read_parquet("data/processed/test_routed.parquet")
for _, q in test[test["route"] == "auto_draft"].sample(3, random_state=7).iterrows():
    print("=" * 70)
    print(f"QUERY ({q['product']} | {q['issue']}):\n{q['consumer_complaint_narrative'][:280]}...")
    for r in hybrid(q["consumer_complaint_narrative"]):
        d = corpus.iloc[r]
        print(f"  -> [{d['product']} | {d['issue']} | {d['company_response_to_consumer']}] "
              f"{d['consumer_complaint_narrative'][:110]}...")

# --- quantitative proxy: product match@5 on 200 test queries ---
sample = test[test["route"] == "auto_draft"].sample(200, random_state=42)
queries = sample["consumer_complaint_narrative"].fillna("").tolist()
q_prod = sample["product"].tolist()

def match_at_k(rows_fn, k=TOP_K):
    hits = [np.mean([corpus.iloc[r]["product"] == p for r in rows_fn(q)]) for q, p in zip(queries, q_prod)]
    return float(np.mean(hits))

print("\n--- product-match@5 proxy (200 auto-draft queries) ---")
print(f"dense  : {match_at_k(dense_search):.3f}")
print(f"bm25   : {match_at_k(bm25_search):.3f}")
print(f"hybrid : {match_at_k(hybrid):.3f}")

t0 = time.time()
for q in queries[:50]:
    hybrid(q)
print(f"hybrid latency: {(time.time()-t0)/50*1000:.0f} ms/query")