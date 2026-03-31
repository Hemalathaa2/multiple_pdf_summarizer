import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="PDF AI Assistant")

st.title("📚 Multi-PDF AI Knowledge Assistant")

# -------------------------
# SESSION STATE
# -------------------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# -------------------------
# FILE UPLOAD
# -------------------------
files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if files and not st.session_state.indexed:
    with st.spinner("Indexing PDFs..."):
        st.session_state.rag.load_pdfs(files)
        st.session_state.indexed = True
    st.success("PDFs Ready!")

# -------------------------
# CHAT DISPLAY
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# USER INPUT
# -------------------------
query = st.chat_input("Ask something about PDFs...")

if query:

    rag = st.session_state.rag
    query = rag.reformulate_query(query)

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        for token, contexts in rag.stream_answer(query):
            full_response += token
            placeholder.markdown(full_response + "▌")

        citation_text = "\n\n---\n**Sources:**\n"
        for c in contexts:
            citation_text += f"- 📄 {c['source']} (Page {c['page']})\n"

        placeholder.markdown(full_response + citation_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
