"""
rag_engine.py  –  Production-grade RAG engine
Optimised for H100 deployment (CUDA-aware, batched encoding, FAISS IVF index)
"""

import re
import numpy as np
import torch
import faiss
import fitz                          # pymupdf
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from typing import Generator, Optional

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"          # bi-encoder  (fast retrieval)
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # cross-encoder (precision)
LLM_MODEL     = "llama-3.1-8b-instant"

CHUNK_SIZE    = 800    # characters  (larger = more context per chunk)
OVERLAP       = 200    # character overlap between consecutive chunks
TOP_K_FETCH   = 12     # candidates retrieved before reranking
TOP_K_FINAL   = 5      # chunks sent to LLM after reranking
BATCH_SIZE    = 64     # embedding batch size (tune for GPU VRAM)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
# Helper – clean raw PDF text
# ──────────────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)          # collapse whitespace
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line-breaks
    return text.strip()


# ──────────────────────────────────────────────
# RAGEngine
# ──────────────────────────────────────────────
class RAGEngine:

    def __init__(self):
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        # Bi-encoder for fast ANN retrieval
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.embedder.max_seq_length = 512

        # Cross-encoder reranker for precision
        self.reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)

        self.chunks: list[dict]       = []   # [{text, source, page}, ...]
        self._faiss_index: faiss.Index | None = None
        self.chat_history: list[dict] = []

    # ────────────────────────────────────────
    # Text splitting  (sliding-window)
    # ────────────────────────────────────────
    def _split_text(self, text: str) -> list[str]:
        pieces, start = [], 0
        while start < len(text):
            end   = min(start + CHUNK_SIZE, len(text))
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            start += CHUNK_SIZE - OVERLAP
        return pieces

    # ────────────────────────────────────────
    # Load PDFs
    # ────────────────────────────────────────
    def load_pdfs(self, files) -> None:
        self.chunks = []

        for file in files:
            file.seek(0)
            doc = fitz.open(stream=file.read(), filetype="pdf")

            for page_num, page in enumerate(doc):
                raw = page.get_text()
                text = _clean(raw)
                if len(text) < 40:          # skip near-empty pages
                    continue

                for chunk in self._split_text(text):
                    self.chunks.append({
                        "text":   chunk,
                        "source": file.name,
                        "page":   page_num + 1,
                    })

        if not self.chunks:
            raise ValueError("No readable text found in the uploaded PDFs.")

        self._build_index()

    # ────────────────────────────────────────
    # Build FAISS index (IVF for large corpora,
    # FlatIP for small ones)
    # ────────────────────────────────────────
    def _build_index(self) -> None:
        texts = [c["text"] for c in self.chunks]
        vecs  = self.embedder.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

        dim = vecs.shape[1]
        n   = len(vecs)

        if n >= 256:
            # IVF index — good for 100 k+ chunks on H100
            nlist = min(int(4 * np.sqrt(n)), n // 4)
            quantiser = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vecs)
            index.nprobe = max(8, nlist // 8)
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(vecs)

        # Move to GPU if available (faiss-gpu)
        if DEVICE == "cuda":
            try:
                res   = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                pass  # faiss-cpu only — stay on CPU

        self._faiss_index = index
        self._embeddings  = vecs   # keep for reranker input shape

    # ────────────────────────────────────────
    # Retrieve  (ANN  →  cross-encoder rerank)
    # ────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        source_filter: Optional[str] = None,
        top_k: int = TOP_K_FINAL,
    ) -> list[dict]:

        if self._faiss_index is None or not self.chunks:
            return []

        # 1. Encode query
        q_vec = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # 2. ANN search – fetch more candidates when filtering by source
        fetch_k = TOP_K_FETCH * 3 if source_filter else TOP_K_FETCH
        fetch_k = min(fetch_k, len(self.chunks))

        _, idxs = self._faiss_index.search(q_vec, fetch_k)
        candidates = [self.chunks[i] for i in idxs[0] if i < len(self.chunks)]

        # 3. Source filter
        if source_filter:
            candidates = [c for c in candidates if c["source"] == source_filter]
            if not candidates:
                # fallback: search entire corpus
                candidates = [self.chunks[i] for i in idxs[0] if i < len(self.chunks)]

        if not candidates:
            return []

        # 4. Cross-encoder rerank
        pairs  = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]

    # ────────────────────────────────────────
    # Prompt builder
    # ────────────────────────────────────────
    @staticmethod
    def _build_prompt(query: str, contexts: list[dict], history: list[dict]) -> list[dict]:
        context_block = "\n\n---\n\n".join(
            f"[Source: {c['source']}  |  Page {c['page']}]\n{c['text']}"
            for c in contexts
        )

        system = (
            "You are a precise document assistant. "
            "Answer the user's question using ONLY the provided document excerpts. "
            "If the answer is partially available, share what is found and note what is missing. "
            "Never fabricate information. "
            "Cite the source file and page number when relevant. "
            "Be concise but complete."
        )

        # Build message list for multi-turn awareness
        messages: list[dict] = [{"role": "system", "content": system}]

        # Inject last 4 turns of history for continuity (keeps token budget low)
        for turn in history[-8:]:
            messages.append(turn)

        user_msg = (
            f"Document excerpts:\n{context_block}\n\n"
            f"Question: {query}"
        )
        messages.append({"role": "user", "content": user_msg})
        return messages

    # ────────────────────────────────────────
    # Streaming Q&A
    # ────────────────────────────────────────
    def stream_answer(
        self,
        query: str,
        source_filter: Optional[str] = None,
    ) -> Generator[str, None, None]:

        contexts = self.retrieve(query, source_filter=source_filter)

        if not contexts:
            yield "⚠️ Could not retrieve relevant sections. Try rephrasing the question."
            return

        messages = self._build_prompt(query, contexts, self.chat_history)

        stream = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,        # low = factual, consistent
            max_tokens=1024,
            stream=True,
        )

        full_text = ""
        for chunk in stream:
            token = getattr(chunk.choices[0].delta, "content", "") or ""
            if token:
                full_text += token
                yield token

        # Persist history (without system / context to save tokens)
        self.chat_history.append({"role": "user",      "content": query})
        self.chat_history.append({"role": "assistant", "content": full_text})

    # ────────────────────────────────────────
    # Streaming summary  (per document)
    # ────────────────────────────────────────
    def stream_summary(self) -> Generator[str, None, None]:
        if not self.chunks:
            yield "⚠️ No documents loaded."
            return

        # Group chunks by source
        docs: dict[str, list[str]] = {}
        for c in self.chunks:
            docs.setdefault(c["source"], []).append(c["text"])

        for filename, texts in docs.items():
            yield f"\n\n### 📄 {filename}\n\n"

            # Use first ~4 000 chars for summarisation
            collected = ""
            for t in texts:
                if len(collected) > 4000:
                    break
                collected += t + "\n"

            prompt = (
                "Produce a concise bullet-point summary of the following text.\n"
                "Rules:\n"
                "- Use '-' bullets only\n"
                "- 6–10 points\n"
                "- Each point: one clear, informative sentence\n"
                "- No headers, no paragraphs\n\n"
                f"TEXT:\n{collected}"
            )

            stream = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=700,
                stream=True,
            )

            raw = ""
            for chunk in stream:
                token = getattr(chunk.choices[0].delta, "content", "") or ""
                raw += token

            # Normalise bullets
            lines = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r"^[\*\•\–\—\d+\.\)]+\s*", "", line)
                lines.append(f"- {line}")

            yield "\n".join(lines)

    # ────────────────────────────────────────
    # Utility
    # ────────────────────────────────────────
    def clear(self) -> None:
        self.chunks        = []
        self._faiss_index  = None
        self.chat_history  = []

    @property
    def source_list(self) -> list[str]:
        return sorted({c["source"] for c in self.chunks})
