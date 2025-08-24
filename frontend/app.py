# frontend/app.py
import streamlit as st
from frontend.sidebar import render_sidebar
from frontend.chat import render_chat
from frontend import api_client as api

st.set_page_config(page_title="RAG Chatbot", layout="wide")

def landing():
    st.title("RAG Chatbot Demo")
    st.caption("Enter your email to view your chats.")
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="you@example.com")
        submitted = st.form_submit_button("Continue")
    if submitted and email:
        try:
            user = api.login(email)
            st.session_state["user"] = user
            # load or create a session
            sessions = api.list_sessions(user["id"])
            st.session_state["current_session_id"] = sessions[0]["id"] if sessions else None
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")

def main():
    user = st.session_state.get("user")

    # left column is reserved for Streamlit sidebar
    render_sidebar()

    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col2:
        if not user:
            landing()
        else:
            render_chat()

if __name__ == "__main__":
    main()
# streamlit run app.py