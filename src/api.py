"""FastAPI service for the complaint triage copilot."""
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.pipeline import TriagePipeline

app = FastAPI(title="ticket-triage",
              description="ML triage + RAG response drafting for consumer-finance complaints",
              version="0.1.0")
pipe = TriagePipeline()


class ComplaintIn(BaseModel):
    narrative: str = Field(..., min_length=20, description="Consumer complaint text")
    product: str = "missing"
    sub_product: str = "missing"
    issue: str = "missing"
    sub_issue: str = "missing"
    submitted_via: str = "missing"
    state: str = "missing"
    company: str = "missing"
    draft: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "llm": pipe.llm, "corpus_docs": int(len(pipe.corpus)),
            "thresholds": {"auto_draft": pipe.t_auto, "priority": pipe.t_priority,
                           "guardrail_sim": pipe.t_sim}}


@app.post("/v1/triage")
def triage(c: ComplaintIn):
    d = c.model_dump()
    narrative, want_draft = d.pop("narrative"), d.pop("draft")
    t0 = time.perf_counter()
    result = pipe.triage({"narrative": narrative, **d}, want_draft=want_draft)
    result["total_ms"] = int((time.perf_counter() - t0) * 1000)
    return result