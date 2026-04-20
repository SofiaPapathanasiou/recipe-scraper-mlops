import os
from fastapi import APIRouter, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(prefix="/ml", tags=["ml-feedback"])

MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

feedback_positive = Counter("model_feedback_positive_total", "Thumbs up", ["model_version"])
feedback_negative = Counter("model_feedback_negative_total", "Thumbs down", ["model_version"])

@router.post("/feedback")
def feedback(slug: str, rating: str):
    if rating == "accept":
        feedback_positive.labels(model_version=MODEL_VERSION).inc()
    else:
        feedback_negative.labels(model_version=MODEL_VERSION).inc()
    return {"ok": True}

@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
