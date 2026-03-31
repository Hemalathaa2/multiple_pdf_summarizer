import requests
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "gemma3:1b"


class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

        self.chunks = []
        self.embeddings = None
        self.summary_cache = ""

    # -----------------------------
    # Load PDF (APPEND MODE)
    # -----------------------------
    def load_pdf(self, file):

        doc = fitz.open(stream=file.read(), filetype="pdf")

        text = ""
        for page in doc:
            text += page.get_text()

        new_chunks = self.split_text(text)
        self.add_embeddings(new_chunks)

    # -----------------------------
    # Split text
    # -----------------------------
    def split_text(self, text, chunk_size=500):

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))

        return chunks

    # -----------------------------
    # Incremental embeddings
    # -----------------------------
    def add_embeddings(self, new_chunks):

        if not new_chunks:
            return

        new_embeddings = self.embedder.encode(new_chunks)

        if self.embeddings is None:
            self.embeddings = new_embeddings
            self.chunks = new_chunks
        else:
            self.embeddings = np.vstack(
                [self.embeddings, new_embeddings]
            )
            self.chunks.extend(new_chunks)

        # reset cached summary when new data added
        self.summary_cache = ""

    # -----------------------------
    # Retrieval (cosine similarity)
    # -----------------------------
    def retrieve(self, query, top_k=3):

        query_embedding = self.embedder.encode([query])[0]

        scores = np.dot(
            self.embeddings,
            query_embedding
        ) / (
            np.linalg.norm(self.embeddings, axis=1)
            * np.linalg.norm(query_embedding) + 1e-10
        )

        top_indices = np.argsort(scores)[-top_k:]

        return "\n".join([self.chunks[i] for i in top_indices])

    # -----------------------------
    # Call Ollama
    # -----------------------------
    def call_llm(self, prompt):

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=600
        )

        result = response.json()
        return result.get("response", "").strip()

    # -----------------------------
    # Multi-PDF Summary
    # -----------------------------
    def generate_summary(self):

        if not self.chunks:
            return "No PDF loaded."

        # reuse cached summary
        if self.summary_cache:
            return self.summary_cache

        step = max(len(self.chunks) // 5, 1)
        context = "\n".join(self.chunks[::step][:5])

        prompt = f"""
Create ONE clear paragraph summary combining all documents:

{context}
"""

        summary = self.call_llm(prompt)
        self.summary_cache = summary

        return summary

    # -----------------------------
    # Question Answering
    # -----------------------------
    def ask_question(self, question):

        context = self.retrieve(question)

        prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        return self.call_llm(prompt)
