import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import inference, model_registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analyze"])


class AnalyzeRequest(BaseModel):
    model_id: str
    prompt: str = Field(min_length=1)


@router.post("/analyze")
def analyze(body: AnalyzeRequest):
    try:
        model_registry.local_model_path(body.model_id)  # validates the id format
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not model_registry.is_downloaded(body.model_id):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model_id}' is not downloaded yet. "
            "Download it from the Find Model page first.",
        )

    try:
        return inference.analyze(body.model_id, body.prompt)
    except inference.ModelLoadError as exc:
        logger.exception("Failed to analyze model_id=%s", body.model_id)
        raise HTTPException(status_code=500, detail=f"Failed to load model '{body.model_id}'.") from exc
