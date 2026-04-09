import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

st.title("📄 SmartDoc AI")

if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

rag = st.session_state.rag

# USER CHOICE
option = st.radio("Choose Mode", ["📄 Document Summarization", "✍️ Text Summarization"])

uploaded_files = None
user_text = ""

# DOCUMENT MODE
if option == "📄 Document Summarization":
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "pptx"],
        accept_multiple_files=True
    )

# TEXT MODE
else:
    user_text = st.text_area("Paste your text here", height=200)

# TEXT LIMIT
if user_text and len(user_text) > 15000:
    st.warning("Text too long. Please limit to 15000 characters.")

# GENERATE SUMMARY
if st.button("Generate Summary"):

    result = ""
    placeholder = st.empty()

    if option == "✍️ Text Summarization":
        if not user_text.strip():
            st.warning("Please enter text")
            st.stop()

        rag.chunks = [{
            "text": user_text,
            "source": "User Input",
            "page": 1
        }]

    else:
        if not uploaded_files:
            st.warning("Upload at least one document")
            st.stop()

        rag.load_files(uploaded_files)

    for token in rag.stream_summary():
        result += token
        placeholder.markdown(result + "▌")

    placeholder.empty()

    st.subheader("Final Summary")
    st.write(result)

    st.download_button("⬇️ Download Summary", result, "summary.txt")

st.markdown("---")
st.markdown("✨ SmartDoc AI")
