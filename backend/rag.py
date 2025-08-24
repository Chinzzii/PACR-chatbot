import os
import faiss
import numpy as np
from typing import Tuple, List
from sqlalchemy.orm import Session
from . import models
from .llm_service import embed_text

# Simple in-memory user indexes; rebuild on startup or on demand
_user_indexes = {}  # user_id -> (faiss.IndexFlatIP, id_map: List[int])

def _normalize(v: np.ndarray) -> np.ndarray:
    # cosine via inner product on normalized vectors
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def _ensure_index(db: Session, user_id: int):
    if user_id in _user_indexes:
        return

    # load all docs with embeddings
    docs = db.query(models.Document).filter(models.Document.user_id == user_id).all()
    vectors = []
    ids = []
    for d in docs:
        if d.embedding:
            vec = np.frombuffer(d.embedding, dtype=np.float32)
            vectors.append(vec)
            ids.append(d.id)

    dim = 1536  # embedding size for text-embedding-3-small
    index = faiss.IndexFlatIP(dim)
    if vectors:
        mat = np.vstack(vectors).astype("float32")
        mat = _normalize(mat)
        index.add(mat)

    _user_indexes[user_id] = (index, ids)

def upsert_document_vector(db: Session, doc: models.Document):
    """Embed and store vector in DB; then add to in-memory FAISS index."""
    vec = embed_text(doc.content)  # np.array shape (1536,)
    db_vec = vec.astype("float32").tobytes()
    doc.embedding = db_vec
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # update memory index
    _ensure_index(db, doc.user_id)
    index, ids = _user_indexes[doc.user_id]
    vec = _normalize(vec.reshape(1, -1).astype("float32"))
    index.add(vec)
    ids.append(doc.id)

def retrieve_context(db: Session, user_id: int, query: str, k: int = 4) -> Tuple[str, List[models.Document]]:
    """
    Returns a concatenated context string and the retrieved Document rows.
    """
    _ensure_index(db, user_id)
    index, ids = _user_indexes[user_id]

    if index.ntotal == 0:
        return "", []

    qvec = embed_text(query).reshape(1, -1).astype("float32")
    qvec = _normalize(qvec)
    scores, idxs = index.search(qvec, min(k, index.ntotal))  # cosine sim via IP on normalized vecs

    doc_ids = [ids[i] for i in idxs[0] if i != -1]
    docs = db.query(models.Document).filter(models.Document.id.in_(doc_ids)).all()

    # sort by search order
    id_to_doc = {d.id: d for d in docs}
    docs_sorted = [id_to_doc[i] for i in doc_ids if i in id_to_doc]

    context_chunks = []
    for d in docs_sorted:
        context_chunks.append(f"[{d.title or 'Document'}]\n{d.content.strip()}\n")

    context = "\n---\n".join(context_chunks)
    return context, docs_sorted
