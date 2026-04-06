"""
app.py  –  Production Streamlit front-end
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
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Minimal CSS polish
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        .stChatMessage { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────
if "rag"        not in st.session_state:
    st.session_state.rag        = RAGEngine()
if "files_hash" not in st.session_state:
    st.session_state.files_hash = None
if "summary_md" not in st.session_state:
    st.session_state.summary_md = ""

rag: RAGEngine = st.session_state.rag

# ──────────────────────────────────────────────
# Sidebar  –  upload + controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, CSV, PPTX, XLSX)",
        type=["pdf", "docx", "txt", "md", "csv", "pptx", "xlsx"],
        accept_multiple_files=True,
        help="All processing happens on your server — no data is sent externally.",
    )

    if uploaded_files:
        new_hash = tuple((f.name, f.size) for f in uploaded_files)

        if new_hash != st.session_state.files_hash:
            st.session_state.files_hash = new_hash
            st.session_state.summary_md = ""
            rag.clear()

            with st.spinner("🔍 Parsing & indexing PDFs…"):
                try:
                    rag.load_files(uploaded_files)
                    st.success(
                        f"✅ Indexed **{len(rag.chunks)}** chunks "
                        f"from **{len(rag.source_list)}** file(s)."
                    )
                except ValueError as e:
                    st.error(str(e))

    # PDF selector
    pdf_list = rag.source_list
    selected_pdf = None
    if pdf_list:
        selected_pdf = st.selectbox(
            "🔎 Query scope",
            ["All documents"] + pdf_list,
        )
        if selected_pdf == "All documents":
            selected_pdf = None

    st.divider()

    # Summary button
    if st.button("📝 Generate Summary", use_container_width=True):
        if not rag.chunks:
            st.warning("Upload PDFs first.")
        else:
            st.session_state.summary_md = ""
            buf = ""
            placeholder = st.empty()
            for token in rag.stream_summary():
                buf += token
                placeholder.markdown(buf + "▌")
            placeholder.markdown(buf)
            st.session_state.summary_md = buf

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        rag.chat_history = []
        st.rerun()

    st.divider()
    st.caption(
        "Model: `llama-3.1-8b-instant` via Groq  \n"
        "Embeddings: `all-MiniLM-L6-v2`  \n"
        "Reranker: `ms-marco-MiniLM-L-6-v2`"
    )

# ──────────────────────────────────────────────
# Main panel  –  chat
# ──────────────────────────────────────────────
st.title("📚 Multi-PDF AI Assistant")

if not rag.chunks:
    st.info("👈  Upload PDFs in the sidebar to get started.")
    st.stop()

# Render existing conversation
for msg in rag.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (native Streamlit widget — persists properly)
query = st.chat_input("Ask a question about your documents…")

if query:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # Stream assistant reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        for token in rag.stream_answer(query, source_filter=selected_pdf):
            answer += token
            placeholder.markdown(answer + "▌")
        placeholder.markdown(answer)
