# frontend/app.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

REQUEST_TIMEOUT = (5, 120)  # (connect, read) seconds


def initialize_session():
    # Auth/session
    st.session_state.setdefault("user_id", None)
    st.session_state.setdefault("user_email", None)
    st.session_state.setdefault("session_id", None)

    # Data
    st.session_state.setdefault("sessions", [])
    st.session_state.setdefault("messages", [])

    # Settings
    st.session_state.setdefault("model", "gpt-4o-mini")
    st.session_state.setdefault("embed_model", "text-embedding-3-small")
    st.session_state.setdefault("streaming", True)


def login():
    st.title("Login")
    email = st.text_input("Enter your email", key="email_input")
    if st.button("Login"):
        if not email:
            st.error("Please enter an email.")
            return
        try:
            resp = requests.post(
                f"{BACKEND_URL}/login",
                json={"email": email},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            user = resp.json()
            st.session_state.user_id = user["id"]
            st.session_state.user_email = user["email"]
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")


def select_or_create_session():
    st.sidebar.header("Chat Sessions")
    user_id = st.session_state.user_id

    # Load sessions
    try:
        resp = requests.get(
            f"{BACKEND_URL}/users/{user_id}/sessions",
            timeout=REQUEST_TIMEOUT,
        )
        sessions = resp.json() if resp.ok else []
    except Exception:
        sessions = []

    # Keep in state
    st.session_state.sessions = sessions

    # Build options list
    options = ["New session"] + [str(s["id"]) for s in sessions]
    selection = st.sidebar.selectbox("Select Session", options, key="session_select")

    # Create
    if selection == "New session":
        if st.sidebar.button("Create Session"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/sessions",
                    json={"user_id": user_id},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                new_s = resp.json()
                st.session_state.session_id = new_s["id"]
                st.session_state.messages = []
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Create failed: {e}")
    else:
        # Switch session: load messages only when changed
        session_id = int(selection)
        if st.session_state.session_id != session_id:
            st.session_state.session_id = session_id
            try:
                msg_resp = requests.get(
                    f"{BACKEND_URL}/sessions/{session_id}/messages",
                    timeout=REQUEST_TIMEOUT,
                )
                if msg_resp.ok:
                    msgs = msg_resp.json()
                    st.session_state.messages = [
                        {"role": m["role"], "content": m["content"]} for m in msgs
                    ]
                else:
                    st.session_state.messages = []
            except Exception as e:
                st.sidebar.error(f"Load messages failed: {e}")
                st.session_state.messages = []

    # Delete
    if st.sidebar.button("Delete Session"):
        if st.session_state.session_id:
            try:
                del_resp = requests.delete(
                    f"{BACKEND_URL}/sessions/{st.session_state.session_id}",
                    timeout=REQUEST_TIMEOUT,
                )
                if del_resp.ok:
                    st.session_state.session_id = None
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.sidebar.error(f"Delete failed: {del_resp.text}")
            except Exception as e:
                st.sidebar.error(f"Delete failed: {e}")
        else:
            st.sidebar.info("No session selected.")


def upload_documents():
    st.sidebar.markdown("---")
    files = st.sidebar.file_uploader(
        "Upload Documents (PDF, DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="uploader",
    )
    if files:
        if not st.session_state.session_id:
            st.error("Select a session first.")
            return
        for file in files:
            try:
                # requests expects: ('fieldname', (filename, bytes, content_type))
                files_payload = {
                    "file": (file.name, file.getvalue(), file.type or "application/octet-stream")
                }
                resp = requests.post(
                    f"{BACKEND_URL}/sessions/{st.session_state.session_id}/documents",
                    files=files_payload,
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.ok:
                    st.sidebar.success(f"Uploaded {file.name}")
                else:
                    st.sidebar.error(f"{file.name}: {resp.text}")
            except Exception as e:
                st.sidebar.error(f"{file.name}: {e}")


def display_message(role, content):
    with st.chat_message(role):
        st.markdown(content)


def chat_completion(messages, stream=False):
    payload = {
        "model": st.session_state.model,
        "messages": messages,
        "stream": stream,
        "session_id": st.session_state.session_id,
        "embed_model": st.session_state.embed_model,
    }
    if stream:
        return requests.post(f"{BACKEND_URL}/chat", json=payload, stream=True, timeout=REQUEST_TIMEOUT)
    else:
        return requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=REQUEST_TIMEOUT)


def main():
    st.set_page_config(page_title="AI Assistant", page_icon="🤖")
    initialize_session()

    if not st.session_state.user_id:
        login()
        return

    st.title("AI Assistant 🤖")
    select_or_create_session()
    if st.session_state.session_id:
        upload_documents()

    with st.sidebar:
        st.header("Settings")
        model_opts = ["gpt-4o-mini", "Perplexity-Sonar", "Perplexity-Reasoning", "Perplexity-DeepResearch"]
        st.session_state.model = st.selectbox("Select Chat Model", model_opts, index=0)
        embed_opts = ["text-embedding-3-small"]
        st.session_state.embed_model = st.selectbox("Embedding Model", embed_opts, index=0)
        st.session_state.streaming = st.checkbox("Enable Streaming", value=True)
        st.markdown("---")
        st.markdown(f"**Signed in:** `{st.session_state.user_email}`")
        st.markdown(f"**Chat Model:** `{st.session_state.model}`")
        st.markdown(f"**Embed Model:** `{st.session_state.embed_model}`")

    # Display past messages
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])

    # Chat input
    if prompt := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)

        with st.chat_message("assistant"):
            try:
                if st.session_state.streaming:
                    res = chat_completion(st.session_state.messages, stream=True)
                    res.raise_for_status()
                    resp_container = st.empty()
                    full_text = ""
                    # Stream plain-text chunks
                    for chunk in res.iter_content(chunk_size=1024):
                        if not chunk:
                            continue
                        text = chunk.decode("utf-8", errors="ignore")
                        if not text.strip():
                            continue
                        full_text += text
                        resp_container.markdown(full_text)
                    if full_text.strip():
                        st.session_state.messages.append({"role": "assistant", "content": full_text})
                else:
                    res = chat_completion(st.session_state.messages, stream=False)
                    res.raise_for_status()
                    answer = res.json().get("response", "")
                    if answer:
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
