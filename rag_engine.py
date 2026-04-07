import io
import re
import csv
import time
import threading
import numpy as np
import torch
import faiss
import fitz
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from typing import Generator

# ──────────────────────────────────────────────
# GLOBAL LOCKS (MULTI-USER SAFE)
# ──────────────────────────────────────────────
api_lock = threading.Lock()
index_lock = threading.Lock()

# ──────────────────────────────────────────────
# CONSTANTS (OPTIMIZED)
# ──────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3-8b-8192"

CHUNK_SIZE = 1200
OVERLAP = 200
TOP_K = 5

# 🔥 SAFE LIMITS (NO API CRASH)
SUMMARY_BATCH_CHARS = 8000
MAX_INPUT_SIZE = 25000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# CACHE MODELS
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return embedder, reranker, client

# ──────────────────────────────────────────────
# CLEAN TEXT
# ──────────────────────────────────────────────
def _clean(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,()\- ]", " ", text)
    return text.strip()

# ──────────────────────────────────────────────
# PDF EXTRACT (FAST + SAFE)
# ──────────────────────────────────────────────
def extract_pdf(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")

    pages = []
    for i, page in enumerate(doc):
        text = _clean(page.get_text())

        if len(text) > 40:
            pages.append({
                "text": text,
                "source": file.name,
                "page": i + 1
            })

    return pages

# ──────────────────────────────────────────────
# RAG ENGINE
# ──────────────────────────────────────────────
class RAGEngine:

    def __init__(self):
        self.embedder, self.reranker, self.client = load_models()
        self.chunks = []
        self.index = None
        self.chat_history = []

    # ──────────────────────────────────────────
    # SPLIT TEXT
    # ──────────────────────────────────────────
    def _split(self, text):
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i:i+CHUNK_SIZE])
            i += CHUNK_SIZE - OVERLAP
        return chunks

    # ──────────────────────────────────────────
    # LOAD FILES
    # ──────────────────────────────────────────
    def load_files(self, files):
        self.chunks = []

        for file in files:
            pages = extract_pdf(file)

            for p in pages:
                for c in self._split(p["text"]):
                    self.chunks.append({
                        "text": c,
                        "source": p["source"],
                        "page": p["page"]
                    })

        with index_lock:
            self._build_index()

    # ──────────────────────────────────────────
    # BUILD INDEX
    # ──────────────────────────────────────────
    def _build_index(self):
        texts = [c["text"] for c in self.chunks]

        vecs = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self.index = index

    # ──────────────────────────────────────────
    # RETRIEVE
    # ──────────────────────────────────────────
    def retrieve(self, query):
        if self.index is None:
            return []

        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        _, idx = self.index.search(q, TOP_K)

        return [self.chunks[i] for i in idx[0]]

    # ──────────────────────────────────────────
    # SAFE API CALL
    # ──────────────────────────────────────────
    def _safe_llm(self, prompt, max_tokens=500):

        retries = 3
        delay = 2

        for attempt in range(retries):
            try:
                time.sleep(1)

                with api_lock:
                    response = self.client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=0.3,
                        stream=False
                    )

                return response.choices[0].message.content

            except Exception:
                time.sleep(delay)
                delay *= 2

        return "⚠️ Server busy. Try again in a few seconds."

    # ──────────────────────────────────────────
    # STREAM ANSWER
    # ──────────────────────────────────────────
    def stream_answer(self, query) -> Generator[str, None, None]:

        contexts = self.retrieve(query)

        if not contexts:
            yield "No relevant info found."
            return

        context = "\n".join([c["text"] for c in contexts])[:5000]

        prompt = f"Answer using this:\n{context}\n\nQ: {query}"

        answer = self._safe_llm(prompt, 600)

        for word in answer.split():
            yield word + " "

    # ──────────────────────────────────────────
    # MAP REDUCE SUMMARY (🔥 STABLE)
    # ──────────────────────────────────────────
    def stream_summary(self):

        if not self.chunks:
            yield "No document."
            return

        full_text = " ".join([c["text"] for c in self.chunks])[:MAX_INPUT_SIZE]

        # 🔹 STEP 1: SPLIT INTO SAFE BATCHES
        batches = [
            full_text[i:i+SUMMARY_BATCH_CHARS]
            for i in range(0, len(full_text), SUMMARY_BATCH_CHARS)
        ]

        partials = []

        # 🔹 STEP 2: MAP
        for i, batch in enumerate(batches, 1):
            yield f"🔹 Processing part {i}/{len(batches)}...\n"

            prompt = f"Summarize in 5 bullet points:\n{batch}"
            summary = self._safe_llm(prompt, 300)

            partials.append(summary)

        # 🔹 STEP 3: REDUCE
        yield "\n🔹 Final summary...\n"

        combined = "\n".join(partials)[:10000]

        final_prompt = f"Combine into final summary (8–12 bullets):\n{combined}"
        final = self._safe_llm(final_prompt, 600)

        yield final

    # ──────────────────────────────────────────
    def clear(self):
        self.chunks = []
        self.index = None
        self.chat_history = []
