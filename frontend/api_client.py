# frontend/api_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")

def login(email: str):
    r = requests.post(f"{API_BASE}/login", json={"email": email})
    r.raise_for_status()
    return r.json()

def list_sessions(user_id: int):
    r = requests.get(f"{API_BASE}/sessions", params={"user_id": user_id})
    r.raise_for_status()
    return r.json()

def create_session(user_id: int, title: str = "New Chat"):
    r = requests.post(f"{API_BASE}/sessions", params={"user_id": user_id}, json={"title": title})
    r.raise_for_status()
    return r.json()

def delete_session(user_id: int, session_id: int):
    r = requests.delete(f"{API_BASE}/sessions/{session_id}", params={"user_id": user_id})
    r.raise_for_status()
    return r.json()

def get_messages(session_id: int):
    r = requests.get(f"{API_BASE}/sessions/{session_id}/messages")
    r.raise_for_status()
    return r.json()

def send_message(session_id: int, text: str):
    r = requests.post(f"{API_BASE}/sessions/{session_id}/message", params={"text": text})
    r.raise_for_status()
    return r.json()

def upload_document(user_id: int, title: str, content: str):
    r = requests.post(f"{API_BASE}/documents", params={"user_id": user_id}, json={"title": title, "content": content})
    r.raise_for_status()
    return r.json()

def list_documents(user_id: int):
    r = requests.get(f"{API_BASE}/documents", params={"user_id": user_id})
    r.raise_for_status()
    return r.json()
