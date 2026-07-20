from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services import model_registry

router = APIRouter(prefix="/models", tags=["models"])


class DownloadRequest(BaseModel):
    model_id: str


@router.get("/search")
def search_models(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50)):
    try:
        return {"results": model_registry.search_hub_models(q, limit=limit)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face Hub search failed: {exc}")


@router.post("/download")
def download_model(body: DownloadRequest):
    try:
        status = model_registry.start_download(body.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"model_id": body.model_id, **status}


@router.get("/download/status")
def download_status(model_id: str = Query(...)):
    status = model_registry.get_download_status(model_id)
    return {"model_id": model_id, **status}


@router.get("/local")
def local_models():
    return {"models": model_registry.list_local_models()}
