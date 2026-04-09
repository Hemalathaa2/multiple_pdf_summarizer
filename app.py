# =========================
# app.py (AUTO DETECT + TEXT LIMIT + DISPLAY FIX)
# =========================

import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

st.title("📄 SmartDoc AI")

if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

rag = st.session_state.rag

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
user_text = st.text_area("Or paste text here", height=200)

# 🔥 AUTO DETECT
use_text = user_text.strip() != ""

# TEXT LIMIT
if len(user_text) > 15000:
    st.warning("Text too long. Please limit to 15000 characters.")

if st.button("Generate Summary"):

    result = ""
    placeholder = st.empty()

    if use_text:
        rag.chunks = [{"text": user_text, "source": "User Input", "page": 1}]
    else:
        if not uploaded_files:
            st.warning("Upload file or paste text")
            st.stop()
        rag.load_files(uploaded_files)

    for token in rag.stream_summary():
        result += token
        placeholder.markdown(result + "▌")

    placeholder.empty()

    st.subheader("Final Summary")
    st.write(result)

    st.download_button("Download Summary", result, "summary.txt")

st.markdown("---")
st.markdown("✨ SmartDoc AI")
