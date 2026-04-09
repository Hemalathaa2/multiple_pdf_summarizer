"""
rag_engine.py - Production-grade RAG engine
"""

import re
import torch
import faiss
import fitz
import streamlit as st

import docx
from pptx import Presentation

from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from typing import Generator

# Constants
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 1500
OVERLAP = 300
SUMMARY_BATCH_CHARS = 20000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# CLEAN TEXT
def _clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# FILE EXTRACTORS
def _extract_pdf(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")

    pages = []
    for i, page in enumerate(doc):
        text = _clean(page.get_text())
        if len(text) > 40:
            pages.append({"text": text, "source": file.name, "page": i + 1})
    return pages


def _extract_txt(file):
    file.seek(0)
    text = file.read().decode("utf-8", errors="ignore")
    return [{"text": _clean(text), "source": file.name, "page": 1}]


def _extract_docx(file):
    file.seek(0)
    doc = docx.Document(file)
    text = " ".join([para.text for para in doc.paragraphs])
    return [{"text": _clean(text), "source": file.name, "page": 1}]


def _extract_pptx(file):
    file.seek(0)
    prs = Presentation(file)
    slides = []

    for i, slide in enumerate(prs.slides):
        content = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                content.append(shape.text)

        text = _clean(" ".join(content))
        if text:
            slides.append({
                "text": text,
                "source": file.name,
                "page": i + 1
            })

    return slides


def extract_pages(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        return _extract_pdf(file)
    elif name.endswith(".txt"):
        return _extract_txt(file)
    elif name.endswith(".docx"):
        return _extract_docx(file)
    elif name.endswith(".pptx"):
        return _extract_pptx(file)
    else:
        raise ValueError(f"Unsupported file type: {file.name}")


# RAG ENGINE
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
        self.chunks = []  # reset

        for file in files:
            pages = extract_pages(file)

            for p in pages:
                for c in self._split_text(p["text"]):
                    self.chunks.append({
                        "text": c,
                        "source": p["source"],
                        "page": p["page"]
                    })

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

    def stream_summary(self):
    
        if not self.chunks:
            yield "No document."
            return
    
        by_source = {}
        for c in self.chunks:
            by_source.setdefault(c["source"], []).append(c["text"])
    
        final_output = ""
    
        for source, texts in by_source.items():
            yield f"\n🔹 Processing {source}...\n"
    
            # STEP 1: STRICT CHUNK SUMMARIES
            chunk_summaries = []
    
            for text in texts:
                prompt = f"""
    You are a strict summarizer.
    
    Summarize the below content in MAXIMUM 5 bullet points.
    Do NOT rewrite or explain.
    Do NOT copy sentences.
    Compress aggressively.
    
    Content:
    {text}
    """
                summary = self._call_llm(prompt, 200)
                chunk_summaries.append(summary)
    
            # STEP 2: COMBINE ALL CHUNK SUMMARIES
            combined_text = "\n".join(chunk_summaries)
    
            # STEP 3: FINAL GLOBAL SUMMARY
            final_prompt = f"""
    You are an expert summarizer.
    
    Below are multiple partial summaries of a document.
    
    Generate ONE FINAL summary:
    - Use clear headings
    - Use bullet points
    - Keep it concise
    - Capture ONLY key insights
    - Avoid repetition
    
    Content:
    {combined_text}
    """
    
            file_summary = self._call_llm(final_prompt, 400)
    
            final_output += f"\n\n📄 {source}\n{file_summary}\n"
    
        yield "\n🔹 Final Summary Ready\n"
        yield final_output
