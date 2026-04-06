"""
app.py – Clean UI (No duplicate output + Input choice + Upload in main)
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

if "summary_md" not in st.session_state:
    st.session_state.summary_md = ""

rag: RAGEngine = st.session_state.rag

# ──────────────────────────────────────────────
# SIDEBAR → CHAT ONLY
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("💬 Chat Assistant")

    for msg in rag.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about documents...")

    if query:
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

    if st.button("🗑️ Clear Chat"):
        rag.chat_history = []
        st.rerun()

# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────
st.title("📚 Multi-Document AI Summarizer")

# 🔹 USER CHOICE
option = st.radio(
    "Choose Input Method:",
    ["📂 Upload Files", "✍️ Paste Text"],
    horizontal=True
)

# ──────────────────────────────────────────────
# OPTION 1 → FILE UPLOAD
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
            st.session_state.summary_md = ""
            rag.clear()

            with st.spinner("🔍 Processing files..."):
                rag.load_files(uploaded_files)
                st.success("Files indexed successfully!")

# ──────────────────────────────────────────────
# OPTION 2 → TEXT INPUT
# ──────────────────────────────────────────────
else:
    user_text = st.text_area(
        "Paste your paragraph / content here:",
        height=200
    )

# ──────────────────────────────────────────────
# GENERATE SUMMARY
# ──────────────────────────────────────────────
if st.button("📝 Generate Summary"):

    st.session_state.summary_md = ""
    placeholder = st.empty()
    result = ""

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

    # 🔹 STREAM SUMMARY (FIXED → NO DOUBLE PRINT)
    for token in rag.stream_summary():
        result += token
        placeholder.markdown(result + "▌")

    st.session_state.summary_md = result

# ──────────────────────────────────────────────
# DISPLAY SUMMARY (ONLY ONCE ✅)
# ──────────────────────────────────────────────

    # ✅ Download button
    st.download_button(
        label="⬇️ Download Summary",
        data=st.session_state.summary_md,
        file_name="summary.txt",
        mime="text/plain",
    )
