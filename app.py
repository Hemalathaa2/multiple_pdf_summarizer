import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="PDF AI Assistant")

st.title("📚 Multi-PDF AI Knowledge Assistant")

# ---------------- SESSION ----------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

if "summary" not in st.session_state:
    st.session_state.summary = ""

# ---------------- UPLOAD MULTIPLE PDFs ----------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        for file in uploaded_files:
            st.session_state.rag.load_pdf(file)

    st.success("Knowledge Base Updated ✅")

# ---------------- SUMMARY ----------------
if st.button("Generate Paragraph Summary"):
    with st.spinner("Generating summary..."):
        st.session_state.summary = (
            st.session_state.rag.generate_summary()
        )

st.subheader("Summary")

if st.session_state.summary:
    st.write(st.session_state.summary)
else:
    st.write("No summary generated.")

# ---------------- QA ----------------
st.subheader("Chat with PDFs")

question = st.text_input("Ask something")

if question:
    with st.spinner("Thinking..."):
        answer = st.session_state.rag.ask_question(question)

    st.write("### 🤖 Answer")
    st.write(answer)
