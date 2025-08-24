from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr


# ---------- User ----------
class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    pass  # same as UserBase for now, since it's just email


class User(UserBase):
    id: int

    class Config:
        orm_mode = True


# ---------- Message ----------
class MessageBase(BaseModel):
    role: str  # "user" or "ai"
    content: str


class MessageCreate(MessageBase):
    pass


class Message(MessageBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True


# ---------- Chat Session ----------
class ChatSessionBase(BaseModel):
    title: str


class ChatSessionCreate(ChatSessionBase):
    pass


class ChatSession(ChatSessionBase):
    id: int
    created_at: datetime
    messages: List[Message] = []

    class Config:
        orm_mode = True


# ---------- Document ----------
class DocumentBase(BaseModel):
    title: Optional[str] = None
    content: str


class DocumentCreate(DocumentBase):
    pass


class Document(DocumentBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True


# ---------- Response Shapes ----------
class SessionWithMessages(ChatSession):
    """For returning a full session with messages"""
    messages: List[Message] = []


class UserWithSessions(User):
    """For returning user with sessions and docs"""
    sessions: List[ChatSession] = []
    documents: List[Document] = []
