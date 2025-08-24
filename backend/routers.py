from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from . import crud, schemas, models
from .db import get_connection
from .models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ---------- DB Setup ----------
SQLALCHEMY_DATABASE_URL = "sqlite:///./chat_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables if not exist
Base.metadata.create_all(bind=engine)

# ---------- Router ----------
router = APIRouter()

# ---------- Users ----------
@router.post("/login", response_model=schemas.User)
def login(email: schemas.UserCreate, db: Session = Depends(get_db)):
    """Get or create user by email (no password/auth)."""
    return crud.get_or_create_user(db, email.email)

# ---------- Chat Sessions ----------
@router.post("/sessions", response_model=schemas.ChatSession)
def create_session(session: schemas.ChatSessionCreate, user_id: int, db: Session = Depends(get_db)):
    """Create a new chat session for a user."""
    return crud.create_chat_session(db, user_id, session.title)


@router.get("/sessions", response_model=List[schemas.ChatSession])
def list_sessions(user_id: int, db: Session = Depends(get_db)):
    """Get all chat sessions for a user."""
    return crud.get_chat_sessions(db, user_id)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, user_id: int, db: Session = Depends(get_db)):
    """Delete a chat session."""
    deleted = crud.delete_chat_session(db, session_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


# ---------- Messages ----------
@router.get("/sessions/{session_id}/messages", response_model=List[schemas.Message])
def get_messages(session_id: int, db: Session = Depends(get_db)):
    """Get all messages for a session."""
    return crud.get_messages(db, session_id)


@router.post("/sessions/{session_id}/message", response_model=schemas.Message)
def send_message(session_id: int, text: str, db: Session = Depends(get_db)):
    """Send a message in a session and get AI response."""
    return crud.send_message(db, session_id, text)


# ---------- Documents (optional RAG) ----------
@router.post("/documents", response_model=schemas.Document)
def upload_document(doc: schemas.DocumentCreate, user_id: int, db: Session = Depends(get_db)):
    """Upload a document for RAG."""
    return crud.add_document(db, user_id, doc.title or "Untitled", doc.content)


@router.get("/documents", response_model=List[schemas.Document])
def list_documents(user_id: int, db: Session = Depends(get_db)):
    """List user's uploaded documents."""
    return crud.get_documents(db, user_id)
