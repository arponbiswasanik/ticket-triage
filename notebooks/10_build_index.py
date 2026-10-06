"""10 - Build retrieval corpus + dense vector index (Chroma, cosine).

Corpus: train-period complaints ONLY (test period never indexed -> no leakage).
"""
import json
import time

import pandas as pd
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

CORPUS_SIZE = 20_000
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

train = pd.read_parquet("data/processed/train.parquet")
corpus = train.sample(n=CORPUS_SIZE, random_state=42).reset_index(drop=True)
print(f"Corpus: {len(corpus):,} train-period complaints")
print(corpus["product"].value_counts().head(6).to_string())

texts = corpus["consumer_complaint_narrative"].fillna("").tolist()

print(f"\nEmbedding {len(texts):,} docs with {MODEL} (CPU, be patient)...")
t0 = time.time()
embedder = SentenceTransformer(MODEL)
vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True)
print(f"done in {time.time()-t0:.0f}s | dim={vectors.shape[1]}")

client = PersistentClient(path="data/vectorstore")
try:
    client.delete_collection("complaints")
except Exception:
    pass
col = client.create_collection("complaints", metadata={"hnsw:space": "cosine"})

ids = [f"c{int(cid)}" for cid in corpus["complaint_id"]]
metas = corpus[["complaint_id", "product", "sub_product", "issue", "sub_issue",
                "company", "company_response_to_consumer", "consumer_disputed",
                "date"]].astype(str).to_dict("records")

for i in range(0, len(vectors), 5000):
    col.add(ids=ids[i:i+5000], embeddings=vectors[i:i+5000].tolist(),
            documents=texts[i:i+5000], metadatas=metas[i:i+5000])
print(f"Chroma index: {col.count():,} docs -> data/vectorstore")

corpus.to_parquet("data/processed/retrieval_corpus.parquet", index=False)
with open("evaluation/retrieval_config.json", "w") as f:
    json.dump({"corpus_size": CORPUS_SIZE, "embed_model": MODEL,
               "dim": int(vectors.shape[1]), "source": "train period only"}, f, indent=2)
print("Saved: retrieval_corpus.parquet, retrieval_config.json")