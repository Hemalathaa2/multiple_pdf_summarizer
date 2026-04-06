"""
rag_engine.py  -  Production-grade RAG engine
Optimised for H100 deployment (CUDA-aware, batched encoding, FAISS IVF index)

Supported input formats:
  .pdf              - PyMuPDF
  .docx             - python-docx
  .txt / .md / .rst - plain text
  .csv              - stdlib csv
  .pptx             - python-pptx (slide text)
  .xlsx             - openpyxl (cell text)
"""

import io
import re
import csv
import time
import numpy as np
import torch
import faiss
import fitz                           # pymupdf
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from typing import Generator, Optional

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL    = "llama-3.1-8b-instant"

CHUNK_SIZE  = 1500    # characters per RAG chunk
OVERLAP     = 300    # overlap between adjacent chunks
TOP_K_FETCH = 12     # ANN candidates before reranking
TOP_K_FINAL = 5      # chunks sent to LLM after reranking
BATCH_SIZE  = 64     # embedding batch size (set to 256 on H100)

# Map-reduce summarisation
SUMMARY_BATCH_CHARS  = 20_000   # characters per MAP batch (~3 000 tokens)
SUMMARY_MAX_BATCHES  = 60       # cap: 720 k chars ~ 400+ pages per file
SUMMARY_REDUCE_LIMIT = 18_000   # max chars fed into the final REDUCE call

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".rst", ".csv", ".pptx", ".xlsx"
}


# ──────────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # REMOVE weird characters (IMPORTANT)
    text = re.sub(r"[^a-zA-Z0-9.,()\- ]", " ", text)

    return text.strip()


# ──────────────────────────────────────────────────────────────────
# Per-format page extractors
# Each returns list[dict]  {text, source, page}
# ──────────────────────────────────────────────────────────────────

def _extract_pdf(file) -> list:
    import pytesseract
    from PIL import Image

    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")

    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()

        # 🔴 If normal extraction fails → use OCR
        if len(text) < 50:
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


def _extract_docx(file) -> list:
    from docx import Document
    file.seek(0)
    doc = Document(file)
    pages, buf, page_num = [], [], 1
    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        buf.append(t)
        # Treat every 40 paragraphs as a logical page for citation purposes
        if len(buf) >= 40:
            pages.append({"text": _clean(" ".join(buf)),
                          "source": file.name, "page": page_num})
            page_num += 1
            buf = []
    if buf:
        pages.append({"text": _clean(" ".join(buf)),
                      "source": file.name, "page": page_num})
    return pages


def _extract_txt(file) -> list:
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


def _extract_csv(file) -> list:
    file.seek(0)
    text = file.read().decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = [" | ".join(r) for r in reader if any(c.strip() for c in r)]
    pages = []
    for i in range(0, len(rows), 30):
        batch = rows[i: i + 30]
        pages.append({"text": _clean("\n".join(batch)),
                      "source": file.name, "page": i // 30 + 1})
    return pages


def _extract_pptx(file) -> list:
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


def _extract_xlsx(file) -> list:
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
            batch = rows[i: i + 30]
            pages.append({"text": _clean("\n".join(batch)),
                          "source": file.name, "page": i // 30 + 1})
    return pages


def extract_pages(file) -> list:
    """Route each uploaded file to the correct extractor by extension."""
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


# ──────────────────────────────────────────────────────────────────
# RAGEngine
# ──────────────────────────────────────────────────────────────────
class RAGEngine:

    def __init__(self):
        self.client   = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.embedder.max_seq_length = 512
        self.reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)

        self.chunks: list       = []
        self._faiss_index       = None
        self.chat_history: list = []

    # ── Text splitting  (sliding-window) ─────────────────────────────
    def _split_text(self, text: str) -> list:
        pieces, start = [], 0
        while start < len(text):
            end   = min(start + CHUNK_SIZE, len(text))
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            start += CHUNK_SIZE - OVERLAP
        return pieces

    # ── Universal file loader ─────────────────────────────────────────
    def load_files(self, files) -> None:
        """
        Accept any mix of .pdf .docx .txt .md .csv .pptx .xlsx files
        and build the FAISS RAG index.
        """
        self.chunks = []
        errors = []

        for file in files:
            try:
                pages = extract_pages(file)
            except ValueError as e:
                errors.append(str(e))
                continue
            for page_info in pages:
                for chunk_text in self._split_text(page_info["text"]):
                    self.chunks.append({
                        "text":   chunk_text,
                        "source": page_info["source"],
                        "page":   page_info["page"],
                    })

        if errors and not self.chunks:
            raise ValueError("\n".join(errors))
        if not self.chunks:
            raise ValueError("No readable text found in the uploaded files.")

        self._build_index()

    # Backward-compat alias so existing app.py code keeps working
    def load_pdfs(self, files) -> None:
        self.load_files(files)

    # ── Build FAISS index ─────────────────────────────────────────────
    def _build_index(self) -> None:
        texts = [c["text"] for c in self.chunks]
        vecs = self.embedder.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

        dim, n = vecs.shape[1], len(vecs)

        if n >= 256:
            nlist     = min(int(4 * np.sqrt(n)), n // 4)
            quantiser = faiss.IndexFlatIP(dim)
            index     = faiss.IndexIVFFlat(
                quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.train(vecs)
            index.nprobe = max(8, nlist // 8)
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(vecs)

        # Move to GPU when available (requires faiss-gpu on H100)
        if DEVICE == "cuda":
            try:
                res   = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                pass

        self._faiss_index = index

    # ── Retrieve  (ANN -> cross-encoder rerank) ───────────────────────
    def retrieve(
        self,
        query: str,
        source_filter: Optional[str] = None,
        top_k: int = TOP_K_FINAL,
    ) -> list:

        if self._faiss_index is None or not self.chunks:
            return []

        q_vec = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        fetch_k = TOP_K_FETCH * 3 if source_filter else TOP_K_FETCH
        fetch_k = min(fetch_k, len(self.chunks))

        _, idxs    = self._faiss_index.search(q_vec, fetch_k)
        candidates = [self.chunks[i] for i in idxs[0] if i < len(self.chunks)]

        if source_filter:
            filtered   = [c for c in candidates if c["source"] == source_filter]
            candidates = filtered if filtered else candidates

        if not candidates:
            return []

        pairs  = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]

    # ── Prompt builder ────────────────────────────────────────────────
    @staticmethod
    def _build_prompt(query: str, contexts: list, history: list) -> list:
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
        messages = [{"role": "system", "content": system}]
        for turn in history[-8:]:
            messages.append(turn)
        messages.append({
            "role": "user",
            "content": f"Document excerpts:\n{context_block}\n\nQuestion: {query}",
        })
        return messages

    # ── Streaming Q&A ─────────────────────────────────────────────────
    def stream_answer(
        self,
        query: str,
        source_filter: Optional[str] = None,
    ) -> Generator[str, None, None]:

        contexts = self.retrieve(query, source_filter=source_filter)
        if not contexts:
            yield "No relevant sections found. Try rephrasing your question."
            return

        messages = self._build_prompt(query, contexts, self.chat_history)
        stream   = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )

        full_text = ""
        for chunk in stream:
            token = getattr(chunk.choices[0].delta, "content", "") or ""
            if token:
                full_text += token
                yield token

        self.chat_history.append({"role": "user",      "content": query})
        self.chat_history.append({"role": "assistant", "content": full_text})

    # ── Map-reduce summary  (reads the ENTIRE document) ───────────────

    

    def _call_llm(self, prompt: str, max_tokens: int = 700) -> str:
      retries = 3
      delay = 2  # seconds
  
      for attempt in range(retries):
          try:
              resp = self.client.chat.completions.create(
                  model=LLM_MODEL,
                  messages=[{"role": "user", "content": prompt}],
                  temperature=0.2,
                  max_tokens=max_tokens,
                  stream=False,
              )
              return resp.choices[0].message.content or ""
  
          except Exception:
              if attempt < retries - 1:
                  time.sleep(delay)
                  delay *= 2
              else:
                  return "⚠️ Rate limit reached. Please try again after a few seconds."

    @staticmethod
    def _normalise_bullets(raw: str) -> str:
        lines = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[*\d+.)]+\s*", "", line).lstrip("-").strip()
            if line:
                lines.append(f"- {line}")
        return "\n".join(lines)

    def _map_batch(self, batch_text: str, batch_num: int, total: int) -> str:
        """MAP step: summarise one 12 000-char section into bullet points."""
       prompt = (
    f"You are summarizing handwritten notes.\n\n"
    "IMPORTANT RULES:\n"
    "- ONLY use information clearly present in the text\n"
    "- DO NOT guess or infer missing content\n"
    "- DO NOT add topics not explicitly written\n"
    "- If text is unclear, ignore it\n\n"
    "Summarise into 5–7 bullet points.\n\n"
    f"TEXT:\n{batch_text}"
)
        return self._call_llm(prompt, max_tokens=500)

    def _reduce_summaries(self, partial_summaries: str, filename: str) -> str:
        """REDUCE step: merge all section bullets into one final overview."""
        prompt = (
            f"Below are section-by-section summaries of the document '{filename}'.\n"
            "Write a single cohesive overall summary covering the ENTIRE document.\n"
            "Rules:\n"
            "- 8 to 14 bullets that together give a complete overview\n"
            "- Each bullet: one clear, informative sentence\n"
            "- Merge duplicate points; keep unique insights from every section\n"
            "- Preserve key facts, numbers, dates, and conclusions\n"
            "- No headers, no paragraphs, no preamble\n\n"
            f"SECTION SUMMARIES:\n{partial_summaries}"
        )
        return self._call_llm(prompt, max_tokens=900)

    def stream_summary(self) -> Generator[str, None, None]:
        """
        Map-reduce summarisation pipeline.
        Reads the ENTIRE document regardless of size.

        Pipeline per uploaded file:
          1. Reassemble all chunk text in document order
          2. Split into SUMMARY_BATCH_CHARS batches (no LLM token-limit issue)
          3. MAP    : call LLM once per batch -> partial bullet summary
          4. REDUCE : call LLM once to merge all partials -> final overview
        """
        if not self.chunks:
            yield "No documents loaded."
            return

        # Group chunks in document order by source file
        docs = {}
        for c in self.chunks:
            docs.setdefault(c["source"], []).append(c["text"])

        for filename, texts in docs.items():
            yield f"\n\n### {filename}\n\n"

            full_text = " ".join(texts)[:100000]
            total_chars = len(full_text)

            # Build batches
            batches, pos = [], 0
            while pos < total_chars and len(batches) < SUMMARY_MAX_BATCHES:
                end = min(pos + SUMMARY_BATCH_CHARS, total_chars)
                batches.append(full_text[pos:end])
                pos = end

            total_batches = len(batches)

            yield (
                f"*Analysing {total_chars:,} characters "
                f"across {total_batches} section(s)...*\n\n"
            )

            # MAP step
            partial_summaries = []
            for i, batch in enumerate(batches, start=1):
                
                time.sleep(1)
                yield f"  Summarising section {i} / {total_batches}...\n"
                section_summary = self._map_batch(batch, i, total_batches)
                partial_summaries.append(
                    f"[Section {i}/{total_batches}]\n{section_summary}"
                )

            # REDUCE step
            yield "\n*Generating overall summary...*\n\n"

            reduce_input = "\n\n".join(partial_summaries)
            if len(reduce_input) > SUMMARY_REDUCE_LIMIT:
                reduce_input = reduce_input[:SUMMARY_REDUCE_LIMIT]

            if total_batches == 1:
                final = self._normalise_bullets(partial_summaries[0])
            else:
                raw_final = self._reduce_summaries(reduce_input, filename)
                final     = self._normalise_bullets(raw_final)

            yield "#### Overall Summary\n\n"
            yield final

    # ── Utility ───────────────────────────────────────────────────────
    def clear(self) -> None:
        self.chunks       = []
        self._faiss_index = None
        self.chat_history = []

    @property
    def source_list(self) -> list:
        return sorted({c["source"] for c in self.chunks})
