import io
import re
import csv
import time
import numpy as np
import torch
import faiss
import fitz
import streamlit as st
import threading

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from typing import Generator, Optional

# ──────────────────────────────────────────────
# GLOBAL LOCK (MULTI-USER SAFETY)
# ──────────────────────────────────────────────
lock = threading.Lock()

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL    = "llama-3.1-8b-instant"

CHUNK_SIZE  = 1500
OVERLAP     = 300
TOP_K_FETCH = 12
TOP_K_FINAL = 5
BATCH_SIZE  = 64

SUMMARY_BATCH_CHARS  = 20000
SUMMARY_MAX_BATCHES  = 60
SUMMARY_REDUCE_LIMIT = 18000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# MODEL CACHING (🔥 IMPORTANT)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    embedder.max_seq_length = 512
    reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return embedder, reranker, client

# ──────────────────────────────────────────────
# TEXT CLEANING
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
    import pytesseract
    from PIL import Image

    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()

        # 🔥 OCR OPTIMIZED
        if len(text.strip()) < 20 and page_num < 10:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)

        text = _clean(text)

        if len(text) >= 40:
            pages.append({
                "text": text,
                "source": file.name,
                "page": page_num + 1
            })

    return pages


def _extract_docx(file):
    from docx import Document
    file.seek(0)
    doc = Document(file)

    pages, buf, page_num = [], [], 1

    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        buf.append(t)

        if len(buf) >= 40:
            pages.append({"text": _clean(" ".join(buf)),
                          "source": file.name, "page": page_num})
            page_num += 1
            buf = []

    if buf:
        pages.append({"text": _clean(" ".join(buf)),
                      "source": file.name, "page": page_num})

    return pages


def _extract_txt(file):
    file.seek(0)
    raw = file.read().decode("utf-8", errors="replace")
    lines = raw.splitlines()

    pages, buf, page_num = [], [], 1

    for line in lines:
        buf.append(line)

        if len(buf) >= 60:
            pages.append({"text": _clean(" ".join(buf)),
                          "source": file.name, "page": page_num})
            page_num += 1
            buf = []

    if buf:
        pages.append({"text": _clean(" ".join(buf)),
                      "source": file.name, "page": page_num})

    return pages


def _extract_csv(file):
    file.seek(0)
    text = file.read().decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))

    rows = [" | ".join(r) for r in reader if any(c.strip() for c in r)]
    pages = []

    for i in range(0, len(rows), 30):
        batch = rows[i:i+30]
        pages.append({"text": _clean("\n".join(batch)),
                      "source": file.name, "page": i // 30 + 1})

    return pages


def _extract_pptx(file):
    from pptx import Presentation
    file.seek(0)
    prs = Presentation(file)

    pages = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        texts = []

        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        texts.append(t)

        if texts:
            pages.append({"text": _clean(" ".join(texts)),
                          "source": file.name, "page": slide_num})

    return pages


def _extract_xlsx(file):
    import openpyxl
    file.seek(0)
    wb = openpyxl.load_workbook(file, read_only=True, data_only=True)

    pages = []

    for sheet in wb.worksheets:
        rows = []

        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None and str(c).strip()]
            if cells:
                rows.append(" | ".join(cells))

        for i in range(0, len(rows), 30):
            batch = rows[i:i+30]
            pages.append({"text": _clean("\n".join(batch)),
                          "source": file.name, "page": i // 30 + 1})

    return pages


def extract_pages(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        return _extract_pdf(file)
    if name.endswith(".docx"):
        return _extract_docx(file)
    if name.endswith((".txt", ".md", ".rst")):
        return _extract_txt(file)
    if name.endswith(".csv"):
        return _extract_csv(file)
    if name.endswith(".pptx"):
        return _extract_pptx(file)
    if name.endswith(".xlsx"):
        return _extract_xlsx(file)

    raise ValueError(f"Unsupported file type: {file.name}")

# ──────────────────────────────────────────────
# RAG ENGINE
# ──────────────────────────────────────────────
class RAGEngine:

    def __init__(self):
        self.embedder, self.reranker, self.client = load_models()
        self.chunks = []
        self._faiss_index = None
        self.chat_history = []

    def _split_text(self, text):
        pieces, start = [], 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            piece = text[start:end].strip()

            if piece:
                pieces.append(piece)

            start += CHUNK_SIZE - OVERLAP

        return pieces

    def load_files(self, files):
        self.chunks = []

        for file in files:
            pages = extract_pages(file)

            for page_info in pages:
                for chunk_text in self._split_text(page_info["text"]):
                    self.chunks.append({
                        "text": chunk_text,
                        "source": page_info["source"],
                        "page": page_info["page"],
                    })

        with lock:
            self._build_index()

    def _build_index(self):
        texts = [c["text"] for c in self.chunks]

        vecs = self.embedder.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

        self._faiss_index = index

    def retrieve(self, query):
        if self._faiss_index is None:
            return []

        q_vec = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        _, idxs = self._faiss_index.search(q_vec, TOP_K_FETCH)
        return [self.chunks[i] for i in idxs[0]]

    def stream_answer(self, query):
        contexts = self.retrieve(query)

        if not contexts:
            yield "No relevant data found."
            return

        context_text = "\n".join([c["text"] for c in contexts])

        prompt = f"Answer based on context:\n{context_text}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in response:
            token = getattr(chunk.choices[0].delta, "content", "") or ""
            yield token

    def stream_summary(self):
        if not self.chunks:
            yield "No documents loaded."
            return
    
        full_text = " ".join([c["text"] for c in self.chunks])[:50000]
    
        prompt = f"Summarize the following document clearly:\n{full_text}"
    
        retries = 3
        delay = 2
    
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    max_tokens=800,
                    temperature=0.3,
                )
    
                for chunk in response:
                    token = getattr(chunk.choices[0].delta, "content", "") or ""
                    if token:
                        yield token
    
                return  # success → exit
    
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    yield "⚠️ Retrying...\n"
                else:
                    yield "❌ Groq API error. Please try again after a few seconds."

    def clear(self):
        self.chunks = []
        self._faiss_index = None
        self.chat_history = []
