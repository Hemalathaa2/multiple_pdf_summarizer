import streamlit as st
from rag_engine import RAGEngine

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
)

# ──────────────────────────────────────────────
# CUSTOM UI STYLES
# ──────────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #4f46e5;
}
.sub-title {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 20px;
}
.stButton>button {
    border-radius: 12px;
    height: 55px;
    font-weight: 600;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
}
.block {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 SmartDoc AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Summarise & Chat with Documents</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "threads" not in st.session_state:
    st.session_state.threads = {}

if "current_thread" not in st.session_state:
    st.session_state.current_thread = None

if "files_hash" not in st.session_state:
    st.session_state.files_hash = None

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "upload"

rag: RAGEngine = st.session_state.rag

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💬 Workspace")
    st.info("Manage conversations")

    if st.button("➕ New Chat"):
        thread_id = f"Chat {len(st.session_state.threads) + 1}"
        st.session_state.threads[thread_id] = []
        st.session_state.current_thread = thread_id
        rag.chat_history = []
        st.rerun()

    st.divider()

    for thread in st.session_state.threads:
        if st.button(thread):
            st.session_state.current_thread = thread
            rag.chat_history = st.session_state.threads[thread]
            st.rerun()

# ──────────────────────────────────────────────
# INPUT MODE BUTTONS
# ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    if st.button("📂 Upload Files", use_container_width=True):
        st.session_state.input_mode = "upload"

with col2:
    if st.button("✍️ Paste Text", use_container_width=True):
        st.session_state.input_mode = "text"

option = st.session_state.input_mode

# ──────────────────────────────────────────────
# FILE UPLOAD
# ──────────────────────────────────────────────
if option == "upload":
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

# ──────────────────────────────────────────────
# TEXT INPUT
# ──────────────────────────────────────────────
else:
    user_text = st.text_area("Paste your text here:", height=200)

# ──────────────────────────────────────────────
# GENERATE SUMMARY
# ──────────────────────────────────────────────
if st.button("📝 Generate Summary"):

    placeholder = st.empty()
    result = ""

    thread_id = f"Summary {len(st.session_state.threads) + 1}"
    st.session_state.current_thread = thread_id
    st.session_state.threads[thread_id] = []
    rag.chat_history = st.session_state.threads[thread_id]

    if option == "text":
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

    rag.chat_history.append({"role": "user", "content": "Generate summary"})

    for token in rag.stream_summary():
        result += token
        placeholder.markdown(f'<div class="block" style="background:#eef2ff">{result}</div>', unsafe_allow_html=True)

    rag.chat_history.append({"role": "assistant", "content": result})

    st.download_button("⬇️ Download Summary", data=result, file_name="summary.txt")

# ──────────────────────────────────────────────
# DISPLAY CHAT
# ──────────────────────────────────────────────
if st.session_state.current_thread:
    st.divider()

    for msg in rag.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="block" style="background:#dbeafe">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="block" style="background:#ede9fe">{msg["content"]}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────────────
query = st.chat_input("Ask something...")

if query and st.session_state.current_thread:

    rag.chat_history.append({"role": "user", "content": query})

    answer = ""
    placeholder = st.empty()

    for token in rag.stream_answer(query):
        answer += token
        placeholder.markdown(f'<div class="block" style="background:#ede9fe">{answer}</div>', unsafe_allow_html=True)

    st.session_state.threads[st.session_state.current_thread] = rag.chat_history

# FOOTER
st.markdown("---")
st.markdown("🚀 Built with Streamlit | SmartDoc AI")
