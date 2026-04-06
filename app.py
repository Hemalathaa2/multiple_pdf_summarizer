
"""
app.py – Updated UI (Text input + Summary in main + Chat in sidebar)
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
# SIDEBAR → CHAT + FILE UPLOAD
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("💬 Chat Assistant")

    # Chat history
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

    st.header("📂 Upload Files")

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

    st.divider()

    if st.button("🗑️ Clear Chat"):
        rag.chat_history = []
        st.rerun()

# ──────────────────────────────────────────────
# MAIN PAGE → SUMMARY + TEXT INPUT
# ──────────────────────────────────────────────
st.title("📚 Multi-Document AI Summarizer")

# 🔹 TEXT INPUT OPTION
st.subheader("✍️ Or Enter Text Manually")

user_text = st.text_area(
    "Paste your paragraph / content here:",
    height=200
)

if st.button("📝 Generate Summary"):
    if not rag.chunks and not user_text.strip():
        st.warning("Upload files OR enter text.")
    else:
        st.session_state.summary_md = ""
        placeholder = st.empty()
        result = ""

        # 🔹 If user entered text → temporarily treat as document
        if user_text.strip():
            rag.clear()
            rag.chunks = [{
                "text": user_text,
                "source": "User Input",
                "page": 1
            }]

        # 🔹 Generate summary
        for token in rag.stream_summary():
            result += token
            placeholder.markdown(result + "▌")

        placeholder.markdown(result)
        st.session_state.summary_md = result

# 🔹 DISPLAY SUMMARY IN CENTER
if st.session_state.summary_md:
    st.divider()
    st.subheader("📄 Summary Output")
    st.markdown(st.session_state.summary_md)
    st.download_button(
            label="⬇️ Download Summary",
            data=st.session_state.summary_md,
            file_name="summary.txt",
            mime="text/plain",
        )
