# frontend/chat.py
import streamlit as st
from frontend import api_client as api

def _bubble(role: str, text: str):
    is_user = role == "user"
    align = "flex-end" if is_user else "flex-start"
    bg = "#DCF8C6" if is_user else "#FFFFFF"
    color = "#000000"  # black text
    st.markdown(
        f"""
        <div style="display:flex; justify-content:{align}; margin:4px 0;">
          <div style="max-width:80%; padding:10px 14px; background:{bg}; color:{color}; border-radius:14px; box-shadow:0 1px 2px rgba(0,0,0,0.05); white-space:pre-wrap;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_chat():
    user = st.session_state.get("user")
    session_id = st.session_state.get("current_session_id")

    if not user:
        st.info("Please log in with your email on the landing page.")
        return

    if not session_id:
        st.info("Start a new chat from the left sidebar.")
        return

    st.markdown("### Chat")

    # History
    msgs = api.get_messages(session_id)
    for m in msgs:
        _bubble(m["role"], m["content"])

    # Input row
    with st.form("send_form", clear_on_submit=True):
        text = st.text_area("Your message", height=100, label_visibility="collapsed", placeholder="Type your message…")
        submitted = st.form_submit_button("Send")
    if submitted and text.strip():
        api.send_message(session_id, text.strip())
        st.rerun()

    # Upload (simple: paste text content)
    st.markdown("---")
    st.subheader("Upload text for RAG")
    title = st.text_input("Title", value="")
    content = st.text_area("Paste text content", height=120)
    if st.button("Upload"):
        if content.strip():
            api.upload_document(user["id"], title or "Untitled", content.strip())
            st.success("Document uploaded and indexed.")
        else:
            st.warning("Please paste some text content.")
