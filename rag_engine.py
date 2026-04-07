"""
rag_engine.py - Production-grade RAG engine
"""

import io
import re
import csv
import time
import numpy as np
import torch
import faiss
import fitz
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from typing import Generator, Optional


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 1500
OVERLAP = 300
TOP_K_FETCH = 12
TOP_K_FINAL = 5
BATCH_SIZE = 64

SUMMARY_BATCH_CHARS = 20000
SUMMARY_MAX_BATCHES = 60
SUMMARY_REDUCE_LIMIT = 18000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
# CLEAN TEXT
# ──────────────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,()\- ]", " ", text)
    return text.strip()


# ──────────────────────────────────────────────
# FILE EXTRACTORS
# ──────────────────────────────────────────────
def _extract_pdf(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")

    pages = []
    for i, page in enumerate(doc):
        text = _clean(page.get_text())

        if len(text) >= 40:
            pages.append({
                "text": text,
                "source": file.name,
                "page": i + 1
            })

    return pages


def extract_pages(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        return _extract_pdf(file)

    raise ValueError(f"Unsupported file type: {file.name}")


# ──────────────────────────────────────────────
# RAG ENGINE
# ──────────────────────────────────────────────
class RAGEngine:

    def __init__(self):
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)

        self.chunks = []
        self._faiss_index = None
        self.chat_history = []

    # ──────────────────────────────
    def _split_text(self, text):
        chunks = []
        i = 0

        while i < len(text):
            chunks.append(text[i:i + CHUNK_SIZE])
            i += CHUNK_SIZE - OVERLAP

        return chunks

    # ──────────────────────────────
    def load_files(self, files):
        self.chunks = []

        for file in files:
            pages = extract_pages(file)

            for p in pages:
                for c in self._split_text(p["text"]):
                    self.chunks.append({
                        "text": c,
                        "source": p["source"],
                        "page": p["page"]
                    })

        self._build_index()

    # ──────────────────────────────
    def _build_index(self):
        texts = [c["text"] for c in self.chunks]

        vecs = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

        self._faiss_index = index

    # ──────────────────────────────
    def retrieve(self, query):
        if self._faiss_index is None:
            return []

        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        _, idx = self._faiss_index.search(q, TOP_K_FINAL)

        return [self.chunks[i] for i in idx[0]]

    # ──────────────────────────────
    def _call_llm(self, prompt, max_tokens=700):
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt[:12000]}],
                max_tokens=max_tokens,
                temperature=0.2,
            )

            return response.choices[0].message.content or ""

        except Exception:
            return "⚠️ API error"

    # ──────────────────────────────
    def stream_answer(self, query) -> Generator[str, None, None]:

        contexts = self.retrieve(query)

        if not contexts:
            yield "No relevant info found."
            return

        context = "\n".join([c["text"] for c in contexts])[:5000]

        prompt = f"Answer using this:\n{context}\n\nQ: {query}"

        answer = self._call_llm(prompt, 600)

        for word in answer.split():
            yield word + " "

    # ──────────────────────────────
    def stream_summary(self):

        if not self.chunks:
            yield "No document."
            return

        full_text = " ".join([c["text"] for c in self.chunks])[:20000]

        batches = [
            full_text[i:i + 6000]
            for i in range(0, len(full_text), 6000)
        ]

        summaries = []

        for i, batch in enumerate(batches[:3]):
            yield f"🔹 Processing part {i+1}...\n"

            prompt = "Summarize in 5 bullet points:\n" + batch
            summaries.append(self._call_llm(prompt, 250))

        yield "\n🔹 Generating final summary...\n"

        combined = "\n".join(summaries)

        final = self._call_llm(
            "Combine into final summary:\n" + combined,
            400
        )

        yield final

    # ──────────────────────────────
    def clear(self):
        self.chunks = []
        self._faiss_index = None
        self.chat_history = []
