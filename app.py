"""
app.py – ChatGPT-style UI (Threads + Center Chat)
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

if "threads" not in st.session_state:
    st.session_state.threads = {}

if "current_thread" not in st.session_state:
    st.session_state.current_thread = None

if "files_hash" not in st.session_state:
    st.session_state.files_hash = None

rag: RAGEngine = st.session_state.rag

# ──────────────────────────────────────────────
# SIDEBAR → THREAD LIST (LIKE CHATGPT)
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("💬 Conversations")

    # New chat button
    if st.button("➕ New Chat"):
        thread_id = f"Chat {len(st.session_state.threads) + 1}"
        st.session_state.threads[thread_id] = []
        st.session_state.current_thread = thread_id
        rag.chat_history = []
        st.rerun()

    st.divider()

    # Show threads
    for thread in st.session_state.threads:
        if st.button(thread):
            st.session_state.current_thread = thread
            rag.chat_history = st.session_state.threads[thread]
            st.rerun()

# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────
st.title("📚 Multi-Document AI Assistant")

# Input choice
option = st.radio(
    "Choose Input Method:",
    ["📂 Upload Files", "✍️ Paste Text"],
    horizontal=True
)

# Upload
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

            with st.spinner("Processing..."):
                rag.load_files(uploaded_files)
                st.success("Files loaded!")

# Text input
else:
    user_text = st.text_area("Paste your text here:", height=200)

# ──────────────────────────────────────────────
# GENERATE SUMMARY
# ──────────────────────────────────────────────
if st.button("📝 Generate Summary"):

    placeholder = st.empty()
    result = ""

    # Create new thread automatically
    thread_id = f"Summary {len(st.session_state.threads) + 1}"
    st.session_state.current_thread = thread_id
    st.session_state.threads[thread_id] = []
    rag.chat_history = st.session_state.threads[thread_id]

    if option == "✍️ Paste Text":
        if not user_text.strip():
            st.warning("Enter text")
            st.stop()

        rag.clear()
        rag.chunks = [{
            "text": user_text,
            "source": "User Input",
            "page": 1
        }]

    else:
        if not rag.chunks:
            st.warning("Upload files first")
            st.stop()

    # Store user message
    rag.chat_history.append({
        "role": "user",
        "content": "Generate summary"
    })

    # Stream summary
    for token in rag.stream_summary():
        result += token
        placeholder.markdown(result + "▌")

    placeholder.markdown(result)

    # Store assistant response
    rag.chat_history.append({
        "role": "assistant",
        "content": result
    })

   

    # Download
    st.download_button(
        "⬇️ Download Summary",
        data=result,
        file_name="summary.txt",
    )

# ──────────────────────────────────────────────
# DISPLAY CURRENT THREAD
# ──────────────────────────────────────────────
if st.session_state.current_thread:
    st.divider()

    for msg in rag.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ──────────────────────────────────────────────
# CHAT INPUT (BOTTOM CENTER ✅)
# ──────────────────────────────────────────────
query = st.chat_input("Ask questions about your document...")

if query and st.session_state.current_thread:

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

    # Save thread
    st.session_state.threads[st.session_state.current_thread] = rag.chat_history
