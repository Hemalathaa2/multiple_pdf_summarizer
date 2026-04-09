# =========================
# rag_engine.py (FINAL FIXED MULTI-FILE + FORMAT)
# =========================

import re
import numpy as np
import torch
import faiss
import fitz
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from typing import Generator

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 1500
OVERLAP = 300
TOP_K_FINAL = 5
SUMMARY_BATCH_CHARS = 20000
SUMMARY_REDUCE_LIMIT = 18000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_pdf(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")

    pages = []
    for i, page in enumerate(doc):
        text = _clean(page.get_text())
        if len(text) > 40:
            pages.append({"text": text, "source": file.name, "page": i + 1})
    return pages


def extract_pages(file):
    if file.name.lower().endswith(".pdf"):
        return _extract_pdf(file)
    raise ValueError("Unsupported file")


class RAGEngine:

    def __init__(self):
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)
        self.chunks = []
        self._faiss_index = None

    def _split_text(self, text):
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i:i + CHUNK_SIZE])
            i += CHUNK_SIZE - OVERLAP
        return chunks

    def load_files(self, files):
        self.chunks = []  # 🔥 FIX: always reset
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

    def _build_index(self):
        texts = [c["text"] for c in self.chunks]
        vecs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self._faiss_index = index

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
            return "API Error"

    def stream_summary(self):

        if not self.chunks:
            yield "No document"
            return

        # 🔥 GROUP BY FILE (CRITICAL FIX)
        by_source = {}
        for c in self.chunks:
            by_source.setdefault(c["source"], []).append(c["text"])

        final_output = ""

        for source, texts in by_source.items():
            yield f"\n🔹 Processing {source}...\n"

            full_text = " ".join(texts)[:SUMMARY_BATCH_CHARS]

            batches = [full_text[i:i+6000] for i in range(0, len(full_text), 6000)]

            partials = []
            for batch in batches:
                prompt = "Give a well-structured summary with headings and bullet points:\n" + batch
                partials.append(self._call_llm(prompt, 250))

            combined = "\n".join(partials)

            file_summary = self._call_llm(
                "Create final structured summary with headings and bullet points:\n" + combined,
                400
            )

            final_output += f"\n\n📄 {source}\n{file_summary}\n"

        yield "\n🔹 Final Combined Summary Ready\n"
        yield final_output

    def stream_answer(self, query):
        yield "Ask after summary"


