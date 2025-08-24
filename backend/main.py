from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import routers
from .models import Base
from sqlalchemy import create_engine

SQLALCHEMY_DATABASE_URL = "sqlite:///./chat_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

def create_app() -> FastAPI:
    app = FastAPI(title="RAG Chatbot Demo", version="0.1.0")

    # CORS: allow local Streamlit
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten for prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ensure tables
    Base.metadata.create_all(bind=engine)

    # include routes
    app.include_router(routers.router, prefix="/api")

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    return app

app = create_app()
# uvicorn backend.main:app --reload --port 8000