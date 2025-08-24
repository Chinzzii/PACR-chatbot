from sqlalchemy.orm import Session
from typing import List, Optional
from . import models, schemas, llm_service, rag

# ---------- User ----------
def get_or_create_user(db: Session, email: str) -> models.User:
    user = db.query(models.User).filter(models.User.email == email).first()
    if user:
        return user
    new_user = models.User(email=email)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# ---------- Chat Sessions ----------
def create_chat_session(db: Session, user_id: int, title: str) -> models.ChatSession:
    session = models.ChatSession(user_id=user_id, title=title or "New Chat")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def get_chat_sessions(db: Session, user_id: int) -> List[models.ChatSession]:
    return db.query(models.ChatSession)\
        .filter(models.ChatSession.user_id == user_id)\
        .order_by(models.ChatSession.created_at.desc())\
        .all()

def delete_chat_session(db: Session, session_id: int, user_id: int) -> bool:
    s = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == user_id
    ).first()
    if not s:
        return False
    # cascade delete messages
    db.query(models.Message).filter(models.Message.session_id == session_id).delete()
    db.delete(s)
    db.commit()
    return True

# ---------- Messages ----------
def get_messages(db: Session, session_id: int) -> List[models.Message]:
    return db.query(models.Message)\
        .filter(models.Message.session_id == session_id)\
        .order_by(models.Message.created_at)\
        .all()

def _session_user_id(db: Session, session_id: int) -> Optional[int]:
    s = db.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
    return s.user_id if s else None

def send_message(db: Session, session_id: int, user_text: str) -> models.Message:
    """
    Stores user message, performs RAG retrieval for the owner of the session,
    calls LLM with context+short history, then stores AI response.
    """
    # Save user message
    user_msg = models.Message(session_id=session_id, role="user", content=user_text)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # Auto-title first time
    sess = db.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
    if sess and (sess.title == "New Chat" or not sess.title):
        sess.title = (user_text[:60] + "…") if len(user_text) > 60 else user_text
        db.add(sess)
        db.commit()

    # Retrieve context via FAISS
    user_id = _session_user_id(db, session_id)
    context = ""
    if user_id is not None:
        context, _ = rag.retrieve_context(db, user_id, user_text, k=4)

    # Short rolling history (last 6 messages)
    hist_rows = get_messages(db, session_id)[-6:]
    history = [{"role": ("assistant" if m.role == "ai" else "user"), "content": m.content} for m in hist_rows]

    # Call LLM
    ai_response_text = llm_service.generate_response(user_text, context=context, history=history)

    # Save AI response
    ai_msg = models.Message(session_id=session_id, role="ai", content=ai_response_text)
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)
    return ai_msg

# ---------- Documents ----------
def add_document(db: Session, user_id: int, title: str, content: str) -> models.Document:
    doc = models.Document(user_id=user_id, title=title, content=content)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Embed + add to FAISS
    rag.upsert_document_vector(db, doc)
    return doc

def get_documents(db: Session, user_id: int) -> List[models.Document]:
    return db.query(models.Document).filter(models.Document.user_id == user_id).all()
