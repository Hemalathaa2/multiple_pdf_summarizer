import requests
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "gemma3:1b"


class RAGEngine:

    def __init__(self):
        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

        # ---- multi document storage ----
        self.documents = {}   # doc_id -> data
        self.global_chunks = []
        self.global_embeddings = None

        self.chat_history = []
        self.summary_cache = ""

    # ------------------------------------------------
    # HASH (avoid duplicate uploads)
    # ------------------------------------------------
    def file_hash(self, file):
        return hashlib.md5(file.getvalue()).hexdigest()

    # ------------------------------------------------
    # LOAD PDF
    # ------------------------------------------------
    def load_pdf(self, file):

        doc_id = self.file_hash(file)

        if doc_id in self.documents:
            return

        pdf = fitz.open(stream=file.read(), filetype="pdf")

        text = ""
        for page in pdf:
            text += page.get_text()

        chunks = self.split_text(text)
        embeddings = self.embedder.encode(chunks)

        # topic vector = mean embedding
        topic_vector = np.mean(embeddings, axis=0)

        self.documents[doc_id] = {
            "chunks": chunks,
            "embeddings": embeddings,
            "topic": topic_vector
        }

        self._update_global_index()
        self.summary_cache = ""

    # ------------------------------------------------
    def split_text(self, text, chunk_size=500):
        words = text.split()
        return [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

    # ------------------------------------------------
    # GLOBAL SEARCH INDEX
    # ------------------------------------------------
    def _update_global_index(self):

        all_chunks = []
        all_embeddings = []

        for doc in self.documents.values():
            all_chunks.extend(doc["chunks"])
            all_embeddings.append(doc["embeddings"])

        self.global_chunks = all_chunks

        if all_embeddings:
            self.global_embeddings = np.vstack(all_embeddings)

    # ------------------------------------------------
    # COSINE SIM
    # ------------------------------------------------
    def cosine(self, A, B):
        return np.dot(A, B) / (
            np.linalg.norm(A) *
            np.linalg.norm(B) + 1e-10
        )

    # ------------------------------------------------
    # TOPIC SIMILARITY CHECK
    # ------------------------------------------------
    def documents_are_similar(self, threshold=0.75):

        topics = [d["topic"] for d in self.documents.values()]

        if len(topics) < 2:
            return True

        sims = []
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                sims.append(self.cosine(topics[i], topics[j]))

        return np.mean(sims) > threshold

    # ------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------
    def retrieve(self, query, top_k=4):

        query_emb = self.embedder.encode([query])[0]

        scores = np.dot(
            self.global_embeddings,
            query_emb
        ) / (
            np.linalg.norm(self.global_embeddings, axis=1)
            * np.linalg.norm(query_emb) + 1e-10
        )

        idx = np.argsort(scores)[-top_k:]

        return "\n".join(self.global_chunks[i] for i in idx)

    # ------------------------------------------------
    # LLM CALL
    # ------------------------------------------------
    def call_llm(self, prompt):

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        return r.json().get("response", "").strip()

    # ------------------------------------------------
    # SMART SUMMARY
    # ------------------------------------------------
    def generate_summary(self):

        if not self.documents:
            return "No PDFs loaded."

        if self.summary_cache:
            return self.summary_cache

        # ---------- SAME TOPIC ----------
        if self.documents_are_similar():

            context = "\n".join(
                self.global_chunks[:8]
            )

            prompt = f"""
Create ONE unified summary combining all documents.

{context}
"""

        # ---------- DIFFERENT TOPICS ----------
        else:

            context = ""
            for i, doc in enumerate(self.documents.values(), 1):
                context += f"\nDocument {i}:\n"
                context += "\n".join(doc["chunks"][:3])

            prompt = f"""
Create separate short summaries for each document.

{context}
"""

        summary = self.call_llm(prompt)
        self.summary_cache = summary
        return summary

    # ------------------------------------------------
    # CHATBOT WITH MEMORY
    # ------------------------------------------------
    def ask_question(self, question):

        context = self.retrieve(question)

        history_text = "\n".join(
            [f"Q:{q}\nA:{a}" for q, a in self.chat_history[-3:]]
        )

        prompt = f"""
You are a helpful AI assistant.

Conversation history:
{history_text}

Context:
{context}

Question: {question}
Answer:
"""

        answer = self.call_llm(prompt)

        self.chat_history.append((question, answer))

        return answer
