import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="PDF AI Assistant")

st.title("📚 Multi-PDF AI Knowledge Assistant")

# SESSION
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- Upload ----------------
files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if files:
    with st.spinner("Processing PDFs..."):
        for f in files:
            st.session_state.rag.load_pdf(f)

    st.success("Knowledge Base Updated ✅")

# ---------------- Summary ----------------
if st.button("Generate Smart Summary"):
    with st.spinner("Generating..."):
        summary = st.session_state.rag.generate_summary()
        st.write(summary)

# ---------------- Chatbot ----------------
st.subheader("Chat with PDFs")

question = st.text_input("Ask a question")

if question:
    answer = st.session_state.rag.ask_question(question)

    st.session_state.chat.append((question, answer))

# Display history
for q, a in reversed(st.session_state.chat):
    st.markdown(f"**🧑 {q}**")
    st.markdown(f"**🤖 {a}**")
