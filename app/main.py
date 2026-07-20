from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers.analyze import router as analyze_router
from app.routers.models import router as models_router

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="ModelReviewer")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

app.include_router(models_router)
app.include_router(analyze_router)


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "try_model.html", {"active_page": "try"})


@app.get("/find-model")
def find_model(request: Request):
    return templates.TemplateResponse(request, "find_model.html", {"active_page": "find"})
