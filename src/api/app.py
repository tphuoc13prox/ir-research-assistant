from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="IR Research Assistant")

    @app.middleware("http")
    async def add_no_cache_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    app.include_router(router)
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
