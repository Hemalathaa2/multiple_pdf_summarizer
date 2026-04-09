
# =========================
# app.py (MINIMAL CHANGE - NO HISTORY STORAGE)
# =========================

import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

st.markdown('<div class="main-title">📄 SmartDoc AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Summarise & Chat with Documents</div>', unsafe_allow_html=True)

if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

rag = st.session_state.rag

uploaded_files = st.file_uploader("Upload documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing..."):
        rag.load_files(uploaded_files)
        st.success("Files loaded!")

if st.button("📝 Generate Summary"):
    result = ""
    placeholder.empty()

# ✅ DISPLAY SUMMARY PROPERLY
    st.subheader("Final Summary")
    st.write(result)
    
    # Save final summary
    rag.chat_history.append({
        "role": "assistant",
        "content": result
    })
    
    # ✅ DOWNLOAD OPTION
    st.download_button("⬇️ Download Summary", data=result, file_name="summary.txt")

query = st.chat_input("Ask something about your document...")

if query:
    answer = ""
    placeholder = st.empty()

    for token in rag.stream_answer(query):
        answer += token
        placeholder.markdown(answer + "▌")

    placeholder.empty()
    st.markdown(answer)

st.markdown("---")
st.markdown("✨ Built with SmartDoc AI")
