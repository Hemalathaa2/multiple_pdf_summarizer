"""
app.py – Final Clean Version (Summary + Chat History in Sidebar)
"""

import streamlit as st
from rag_engine import RAGEngine

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide",
)

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "files_hash" not in st.session_state:
    st.session_state.files_hash = None

rag: RAGEngine = st.session_state.rag

# ──────────────────────────────────────────────
# SIDEBAR → CHAT + HISTORY
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("💬 Chat & History")

    # Show full history (chat + summary)
    for msg in rag.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask about documents...")

    if query:
        rag.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer = ""

            for token in rag.stream_answer(query):
                answer += token
                placeholder.markdown(answer + "▌")

            placeholder.markdown(answer)

    st.divider()

    if st.button("🗑️ Clear History"):
        rag.chat_history = []
        st.rerun()

# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────
st.title("📚 Multi-Document AI Summarizer")

# Input choice
option = st.radio(
    "Choose Input Method:",
    ["📂 Upload Files", "✍️ Paste Text"],
    horizontal=True
)

# ──────────────────────────────────────────────
# FILE UPLOAD
# ──────────────────────────────────────────────
if option == "📂 Upload Files":
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md", "csv", "pptx", "xlsx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        new_hash = tuple((f.name, f.size) for f in uploaded_files)

        if new_hash != st.session_state.files_hash:
            st.session_state.files_hash = new_hash
            rag.clear()

            with st.spinner("🔍 Processing files..."):
                rag.load_files(uploaded_files)
                st.success("Files indexed successfully!")

# ──────────────────────────────────────────────
# TEXT INPUT
# ──────────────────────────────────────────────
else:
    user_text = st.text_area(
        "Paste your text here:",
        height=200
    )

# ──────────────────────────────────────────────
# GENERATE SUMMARY
# ──────────────────────────────────────────────
if st.button("📝 Generate Summary"):

    placeholder = st.empty()
    result = ""

    # Handle text input
    if option == "✍️ Paste Text":
        if not user_text.strip():
            st.warning("Please enter text.")
            st.stop()

        rag.clear()
        rag.chunks = [{
            "text": user_text,
            "source": "User Input",
            "page": 1
        }]

    else:
        if not rag.chunks:
            st.warning("Upload files first.")
            st.stop()

    # Add user message (like ChatGPT)
    rag.chat_history.append({
        "role": "user",
        "content": "Generate summary"
    })

    # STREAM SUMMARY (only once)
    for token in rag.stream_summary():
        result += token
        placeholder.markdown(result + "▌")

    placeholder.markdown(result)

    # Store summary in history (sidebar)
    rag.chat_history.append({
        "role": "assistant",
        "content": result
    })

    # Download button
    st.download_button(
        label="⬇️ Download Summary",
        data=result,
        file_name="summary.txt",
        mime="text/plain",
    )
