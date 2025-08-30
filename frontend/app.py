# frontend/app.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer 
import sys
import tempfile
import json

# Add the parent directory to the path to access summarizer
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the summarizer modules with proper error handling
try:
    from summarizer.pipelines.pytesseract import process_pdf_with_tesseract
    SUMMARIZER_AVAILABLE = True
    print("✓ Summarizer modules loaded successfully")
except ImportError as e:
    print(f"✗ Summarizer modules not available: {e}")
    SUMMARIZER_AVAILABLE = False

load_dotenv()
BACKEND_URL = os.getenv("API_BASE", "http://localhost:8000")

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

    # Summary state
    st.session_state.setdefault("show_summary_popup", False)
    st.session_state.setdefault("current_summary", "")
    st.session_state.setdefault("current_analysis", {})


def login():
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
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
    st.sidebar.subheader("📁 Document Upload")
    files = st.sidebar.file_uploader(
        "Upload Documents (PDF, DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="uploader",
    )
    if files:
        if not st.session_state.session_id:
            st.sidebar.error("Select a session first.")
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
                    st.sidebar.success(f"✅ {file.name}")
                else:
                    st.sidebar.error(f"❌ {file.name}: {resp.text}")
            except Exception as e:
                st.sidebar.error(f"❌ {file.name}: {e}")


def get_session_documents(session_id):
    """Fetch documents tied to a session."""
    try:
        resp = requests.get(
            f"{BACKEND_URL}/sessions/{session_id}/documents",
            timeout=REQUEST_TIMEOUT,
        )
        if resp.ok:
            return resp.json()
    except Exception as e:
        st.error(f"Load documents failed: {e}")
    return []


def generate_summary():
    """Generate summary for the first PDF document in the session using process_pdf_with_tesseract."""
    if not SUMMARIZER_AVAILABLE:
        st.sidebar.error("❌ Summarizer functionality is not available.")
        return
        
    if not st.session_state.session_id:
        st.sidebar.error("❌ Please select a session first.")
        return
    
    # Get documents for the session
    docs = get_session_documents(st.session_state.session_id)
    pdf_docs = [d for d in docs if d["filename"].lower().endswith(".pdf")]
    
    if not pdf_docs:
        st.sidebar.error("❌ No PDF documents found in this session.")
        return
    
    first_pdf = pdf_docs[0]
    st.sidebar.info(f"🔄 Processing: {first_pdf['filename']}")
    
    try:
        # Download the PDF file
        resp = requests.get(f"{BACKEND_URL}/files/{first_pdf['id']}", timeout=REQUEST_TIMEOUT)
        if not resp.ok:
            st.sidebar.error(f"❌ Failed to download PDF: {resp.status_code}")
            return
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(resp.content)
            tmp_file_path = tmp_file.name
        
        try:
            # Verify the PDF file is valid
            file_size = os.path.getsize(tmp_file_path)
            if file_size == 0:
                st.sidebar.error("❌ Downloaded PDF file is empty.")
                return
            
            # Process PDF using the main function
            with st.spinner("🔍 Analyzing PDF with Tesseract OCR..."):
                analysis_result = process_pdf_with_tesseract(tmp_file_path)
            
            if not analysis_result:
                st.sidebar.error("❌ Failed to process PDF.")
                return
            
            # Store the analysis result in session state
            st.session_state.current_analysis = analysis_result
            st.session_state.current_summary = analysis_result.get('overall_summary', 'No summary available')
            st.session_state.show_summary_popup = True
            
            st.sidebar.success(f"✅ Analysis complete! ({analysis_result.get('processing_time', 'unknown time')})")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)[:100]}...")
        print(f"Full error: {e}")  # For debugging


def show_sidebar_summary_section():
    """Show the document analysis section in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Document Analysis")
    
    if not SUMMARIZER_AVAILABLE:
        st.sidebar.error("❌ Summarizer not available")
        st.sidebar.caption("Check installation and dependencies")
        return
    
    if not st.session_state.session_id:
        st.sidebar.info("📋 Select a session to analyze documents")
        return
    
    # Check if there are PDF documents in the current session
    docs = get_session_documents(st.session_state.session_id)
    pdf_docs = [d for d in docs if d["filename"].lower().endswith(".pdf")]
    
    if pdf_docs:
        st.sidebar.success(f"📄 Found {len(pdf_docs)} PDF document(s)")
        
        # Show the first PDF that will be analyzed
        first_pdf = pdf_docs[0]
        st.sidebar.caption(f"📄 **Target:** {first_pdf['filename'][:30]}{'...' if len(first_pdf['filename']) > 30 else ''}")
        
        # Analyze button
        if st.sidebar.button("🔍 Analyze PDF", 
                           help="Generate comprehensive analysis with:\n• Overall summary\n• Section-wise summaries\n• Key contributions\n• Methodology details\n• Key findings",
                           use_container_width=True):
            generate_summary()
        
        # Show analysis status
        if st.session_state.current_analysis:
            st.sidebar.success("✅ Analysis Ready")
            
            # Quick preview with summary
            analysis = st.session_state.current_analysis
            with st.sidebar.expander("📋 Quick Preview", expanded=True):
                # Show brief summary (first 2 sentences)
                overall_summary = analysis.get('overall_summary', 'No summary available')
                if overall_summary and overall_summary != 'No summary available':
                    # Extract first 2 sentences or first 200 characters
                    sentences = overall_summary.split('. ')
                    if len(sentences) >= 2:
                        brief_summary = '. '.join(sentences[:2]) + '.'
                    else:
                        brief_summary = overall_summary[:200] + '...' if len(overall_summary) > 200 else overall_summary
                    
                    st.caption("**Summary:**")
                    st.caption(brief_summary)
                    st.caption("")  # Add some spacing
                
                # Show sections found
                st.caption("**Sections Found:**")
                structure = analysis.get('structure', [])
                if structure:
                    for section in structure[:4]:  # Show first 4 sections
                        st.caption(f"• {section}")
                    if len(structure) > 4:
                        st.caption(f"... and {len(structure) - 4} more")
                else:
                    st.caption("No structured sections found")
                
                # Show key stats
                st.caption("")  # Add spacing
                st.caption("**Analysis Stats:**")
                section_count = len(analysis.get('section_summaries', {}))
                st.caption(f"• {section_count} sections analyzed")
                
                # Show processing time if available
                processing_time = analysis.get('processing_time', '')
                if processing_time:
                    st.caption(f"• Processed in {processing_time}")
            
            if st.sidebar.button("📊 View Full Analysis", 
                               use_container_width=True,
                               help="Open detailed analysis with all sections"):
                st.session_state.show_summary_popup = True
                st.rerun()
    else:
        st.sidebar.info("📤 Upload a PDF to enable analysis")
        st.sidebar.caption("Supported: Academic papers, research documents, reports")


def format_text_with_bullets(text):
    """Convert bullet points to proper markdown format and handle regular text."""
    if not text:
        return "No information available"
    
    # Check if text contains bullet points
    if '•' in text or '●' in text or text.strip().startswith('-'):
        # Split by bullet points and format each one
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('●'):
                # Remove the bullet and add markdown bullet
                clean_line = line[1:].strip()
                if clean_line:
                    formatted_lines.append(f"- {clean_line}")
            elif line.startswith('-'):
                # Already a markdown bullet
                formatted_lines.append(line)
            elif line:
                # Regular text line
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    else:
        # Regular text, just return as is
        return text


def show_summary_popup():
    """Display the analysis results in a popup-style container."""
    if st.session_state.show_summary_popup and st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        
        # Create a modal-like container
        with st.container():
            st.markdown("### 📄 Document Analysis Results")
            st.markdown(f"**File:** {analysis.get('filename', 'Unknown')}")
            st.markdown(f"**Processed:** {analysis.get('timestamp', 'Unknown time')}")
            st.markdown(f"**Processing Time:** {analysis.get('processing_time', 'Unknown')}")
            
            # Close button
            col1, col2 = st.columns([6, 1])
            with col2:
                if st.button("✕ Close", key="close_summary"):
                    st.session_state.show_summary_popup = False
                    st.rerun()
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📝 Overall Summary", 
                "📑 Sections", 
                "🔬 Contributions", 
                "⚙️ Methodology", 
                "🎯 Key Findings"
            ])
            
            with tab1:
                st.markdown("#### Overall Summary")
                summary_text = analysis.get('overall_summary', 'No summary available')
                st.markdown(summary_text)
            
            with tab2:
                st.markdown("#### Section Summaries")
                section_summaries = analysis.get('section_summaries', {})
                if section_summaries:
                    for section, summary in section_summaries.items():
                        with st.expander(f"📖 {section}", expanded=False):
                            formatted_summary = format_text_with_bullets(summary)
                            st.markdown(formatted_summary)
                else:
                    st.info("No section summaries available")
            
            with tab3:
                st.markdown("#### Contributions")
                contributions_text = analysis.get('contributions', 'No contributions identified')
                formatted_contributions = format_text_with_bullets(contributions_text)
                st.markdown(formatted_contributions)
            
            with tab4:
                st.markdown("#### Methodology")
                methodology_text = analysis.get('methodology', 'No methodology information available')
                formatted_methodology = format_text_with_bullets(methodology_text)
                st.markdown(formatted_methodology)
            
            with tab5:
                st.markdown("#### Key Findings")
                findings_text = analysis.get('key_findings', 'No key findings identified')
                formatted_findings = format_text_with_bullets(findings_text)
                st.markdown(formatted_findings)
            
            # Additional metadata section
            st.markdown("---")
            with st.expander("📊 Analysis Metadata"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Document Structure:**")
                    structure = analysis.get('structure', [])
                    if structure:
                        for i, section in enumerate(structure, 1):
                            st.markdown(f"{i}. {section}")
                    else:
                        st.markdown("No structure information available")
                
                with col2:
                    st.markdown("**Processing Details:**")
                    st.markdown(f"- **Sections Found:** {len(analysis.get('section_summaries', {}))}")
                    st.markdown(f"- **File Size:** Available in backend")
                    st.markdown(f"- **OCR Method:** Tesseract")
                    st.markdown(f"- **AI Model:** {analysis.get('ai_model', 'Groq/DeepSeek')}")
            
            # Options to add to chat
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💬 Add Summary to Chat", use_container_width=True):
                    if st.session_state.session_id:
                        summary_message = f"**Document Summary:**\n\n{analysis.get('overall_summary', '')}"
                        st.session_state.messages.append({"role": "assistant", "content": summary_message})
                        st.session_state.show_summary_popup = False
                        st.rerun()
            
            with col2:
                if st.button("📋 Add Full Analysis to Chat", use_container_width=True):
                    if st.session_state.session_id:
                        # Format the full analysis with proper bullet points
                        full_analysis = f"""**Complete Document Analysis:**

**Overall Summary:**
{analysis.get('overall_summary', '')}

**Contributions:**
{format_text_with_bullets(analysis.get('contributions', ''))}

**Methodology:**
{format_text_with_bullets(analysis.get('methodology', ''))}

**Key Findings:**
{format_text_with_bullets(analysis.get('key_findings', ''))}"""
                        st.session_state.messages.append({"role": "assistant", "content": full_analysis})
                        st.session_state.show_summary_popup = False
                        st.rerun()
            
            with col3:
                # Download JSON button
                json_str = json.dumps(analysis, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"{analysis.get('filename', 'document')}_analysis.json",
                    mime="application/json",
                    use_container_width=True
                )


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


def get_session_documents(session_id):
    """Fetch documents tied to a session."""
    try:
        resp = requests.get(
            f"{BACKEND_URL}/sessions/{session_id}/documents",
            timeout=REQUEST_TIMEOUT,
        )
        if resp.ok:
            return resp.json()
    except Exception as e:
        st.error(f"Load documents failed: {e}")
    return []


def main():
    st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
    initialize_session()

    if not st.session_state.user_id:
        login()
        return

    st.title("🤖 AI Assistant")
    
    # Sidebar sections
    select_or_create_session()
    
    if st.session_state.session_id:
        upload_documents()
        show_sidebar_summary_section()  # Add the summary section to sidebar

    # Settings section in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ Settings")
        model_opts = ["gpt-4o-mini", "Perplexity-Sonar", "Perplexity-Reasoning", "Perplexity-DeepResearch"]
        st.session_state.model = st.selectbox("Chat Model", model_opts, index=0)
        embed_opts = ["text-embedding-3-small"]
        st.session_state.embed_model = st.selectbox("Embedding Model", embed_opts, index=0)
        st.session_state.streaming = st.checkbox("Enable Streaming", value=True)
        
        st.markdown("---")
        st.markdown(f"**👤 User:** `{st.session_state.user_email}`")
        st.markdown(f"**🤖 Model:** `{st.session_state.model}`")
        st.markdown(f"**🔗 Embed:** `{st.session_state.embed_model}`")

    # Show summary popup if needed
    if st.session_state.show_summary_popup:
        show_summary_popup()

    # Main chat area
    if st.session_state.session_id:
        docs = get_session_documents(st.session_state.session_id)
    else:
        docs = []

    if docs:  # If this session has documents - show chat and PDF preview
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("💬 Chat")
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
                        
        with col2:
            st.subheader("📄 Document Preview")
            first_doc = next((d for d in docs if d["filename"].lower().endswith(".pdf")), None)
            if first_doc:
                try:
                    resp = requests.get(f"{BACKEND_URL}/files/{first_doc['id']}", timeout=REQUEST_TIMEOUT)
                    if resp.ok:
                        pdf_bytes = resp.content
                        pdf_viewer(pdf_bytes, width="100%", height=600)
                    else:
                        st.error("Could not load PDF preview.")
                except Exception as e:
                    st.error(f"PDF preview error: {e}")
            else:
                st.info("No PDF available for preview.")
    else:
        # No docs → full-width chat
        st.subheader("💬 Chat")
        for msg in st.session_state.messages:
            display_message(msg["role"], msg["content"])

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