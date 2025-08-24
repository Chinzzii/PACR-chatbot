# frontend/sidebar.py
import streamlit as st
from frontend import api_client as api

def render_sidebar():
    st.sidebar.title("Chats")

    user = st.session_state.get("user")
    if not user:
        return

    if st.sidebar.button("➕ New chat", use_container_width=True):
        sess = api.create_session(user["id"], "New Chat")
        st.session_state["current_session_id"] = sess["id"]
        st.rerun()

    sessions = api.list_sessions(user["id"])
    if not sessions:
        st.sidebar.info("No chats yet.")
        return

    for s in sessions:
        cols = st.sidebar.columns([0.8, 0.2])
        label = s["title"]
        if cols[0].button(label, key=f"open_{s['id']}", use_container_width=True):
            st.session_state["current_session_id"] = s["id"]
            st.rerun()
        if cols[1].button("🗑️", key=f"del_{s['id']}"):
            api.delete_session(user["id"], s["id"])
            if st.session_state.get("current_session_id") == s["id"]:
                st.session_state["current_session_id"] = None
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", type="primary", use_container_width=True):
        for k in ["user", "current_session_id"]:
            st.session_state.pop(k, None)
        st.rerun()
