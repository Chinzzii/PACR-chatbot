import io
import os
import numpy as np
from typing import Dict, Any, List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from .db import SessionLocal, engine
from .models import Base, User, ChatSession, ChatMessage, Document

# OpenAI (modern SDK)
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
client = OpenAI(api_key=OPENAI_API_KEY)

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="LLM Chat Backend")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- User Endpoints ---

@app.post("/login")
def login_user(user: Dict[str, Any], db: Session = Depends(get_db)):
    """Create or get user by email."""
    email = (user or {}).get("email")
    if not email:
        raise HTTPException(400, "Email is required")
    db_user = db.query(User).filter(User.email == email).first()
    if not db_user:
        db_user = User(email=email)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    return {"id": db_user.id, "email": db_user.email}


# --- Session Endpoints ---

@app.post("/sessions")
def create_session(data: Dict[str, Any], db: Session = Depends(get_db)):
    """Create a new chat session for a user."""
    user_id = (data or {}).get("user_id")
    if not user_id:
        raise HTTPException(400, "user_id is required")
    if not db.get(User, user_id):
        raise HTTPException(404, "User not found")
    new_session = ChatSession(user_id=user_id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {"id": new_session.id, "user_id": new_session.user_id, "created_at": str(new_session.created_at)}


@app.get("/users/{user_id}/sessions")
def get_sessions(user_id: int, db: Session = Depends(get_db)):
    """List all chat sessions for a user."""
    sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
    return [{"id": s.id, "user_id": s.user_id, "created_at": str(s.created_at)} for s in sessions]


@app.delete("/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    """Delete a session (and its messages/docs)."""
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    db.delete(session)
    db.commit()
    return {"ok": True}


# --- Message Endpoints ---

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: int, db: Session = Depends(get_db)):
    """Retrieve all messages for a session."""
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
        .all()
    )
    return [{"id": m.id, "role": m.role, "content": m.content, "created_at": str(m.created_at)} for m in msgs]


# --- Document Upload ---

@app.post("/sessions/{session_id}/documents")
def upload_document(session_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload PDF or DOCX to a session for RAG."""
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    filename = file.filename or "upload.bin"
    lowered = filename.lower()

    try:
        if lowered.endswith(".pdf"):
            import fitz  # PyMuPDF
            raw = file.file.read()
            pdf = fitz.open(stream=raw, filetype="pdf")
            content = ""
            for page in pdf:
                content += page.get_text() or ""
        elif lowered.endswith(".docx"):
            from docx import Document as Docx
            raw = file.file.read()
            doc = Docx(io.BytesIO(raw))
            content = "\n".join(para.text for para in doc.paragraphs)
        else:
            raise HTTPException(400, "Unsupported file type")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"File processing error: {e}")

    if not content.strip():
        raise HTTPException(400, "No extractable text found in document")

    doc = Document(session_id=session_id, filename=filename, content=content)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {"doc_id": doc.id, "filename": doc.filename}


# --- Chat Endpoint (with Streaming) ---

def _safe_embed(text: str, model: str) -> List[float]:
    """Embed text; return [] on failure."""
    try:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding  # type: ignore[return-value]
    except Exception:
        return []


@app.post("/chat")
def chat_endpoint(request: Dict[str, Any], db: Session = Depends(get_db)):
    """
    Chat completion: save user message, perform naive RAG (if docs exist), and return assistant reply.
    Supports streaming via StreamingResponse.
    """
    model = (request or {}).get("model") or "gpt-4o-mini"
    messages = (request or {}).get("messages", [])
    stream = bool((request or {}).get("stream", False))
    session_id = (request or {}).get("session_id")
    embed_model = (request or {}).get("embed_model") or "text-embedding-3-small"

    if not session_id:
        raise HTTPException(400, "session_id is required")
    if not messages:
        raise HTTPException(400, "No messages provided")

    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Save latest user message to DB (basic reliability)
    user_msg = messages[-1].get("content", "")
    if not isinstance(user_msg, str) or not user_msg.strip():
        raise HTTPException(400, "Last user message is empty")
    user_msg_db = ChatMessage(session_id=session_id, role="user", content=user_msg)
    db.add(user_msg_db)
    db.commit()

    # --- Naive RAG Context Injection (computed on-the-fly; consider pre-chunking for production) ---
    docs = db.query(Document).filter(Document.session_id == session_id).all()
    if docs:
        q_embed = _safe_embed(user_msg, embed_model)
        if q_embed:
            sims: List[tuple[float, str]] = []
            qv = np.array(q_embed, dtype=np.float32)
            qn = np.linalg.norm(qv) or 1.0
            # NOTE: expensive — in production, store chunked embeddings
            for doc in docs:
                # Limit doc text to reduce cost/latency (simple heuristic)
                doc_text = doc.content[:100_000]
                d_embed = _safe_embed(doc_text, embed_model)
                if not d_embed:
                    continue
                dv = np.array(d_embed, dtype=np.float32)
                dn = np.linalg.norm(dv) or 1.0
                sim = float(np.dot(qv, dv) / (qn * dn))
                sims.append((sim, doc_text))
            sims.sort(key=lambda x: x[0], reverse=True)
            top_context = "\n\n".join([content for _, content in sims[:3]])
            if top_context:
                messages = messages + [{"role": "system", "content": f"Relevant context:\n{top_context}"}]

    # --- Chat call ---
    if stream:
        def generator():
            full_text = ""
            try:
                with client.chat.completions.stream(
                    model=model,
                    messages=messages,
                ) as s:
                    for event in s:
                        if event.type == "content.delta":
                            delta = event.delta or ""
                            if delta:
                                full_text += delta
                                # stream plain text to client
                                yield delta
                        elif event.type == "error":
                            # surface error to client stream
                            yield f"\n[error] {event.error}\n"
                    # finished
            finally:
                # Save assistant reply (even if empty)
                assistant_msg = ChatMessage(session_id=session_id, role="assistant", content=full_text)
                db.add(assistant_msg)
                db.commit()

        return StreamingResponse(generator(), media_type="text/plain")
    else:
        try:
            res = client.chat.completions.create(model=model, messages=messages)
            answer = res.choices[0].message.content or ""  # type: ignore[assignment]
        except Exception as e:
            raise HTTPException(500, f"LLM error: {e}")
        assistant_msg = ChatMessage(session_id=session_id, role="assistant", content=answer)
        db.add(assistant_msg)
        db.commit()
        return {"response": answer}
