import requests
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import json

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "gemma3:1b"


class RAGEngine:

    def __init__(self):

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

        # ---- multi document storage ----
        self.documents = {}
        self.global_chunks = []
        self.global_embeddings = None

        self.chat_history = []
        self.summary_cache = ""

    # ------------------------------------------------
    # HASH
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

        filename = file.name
        pdf = fitz.open(stream=file.read(), filetype="pdf")

        text = ""
        for page in pdf:
            text += page.get_text()

        chunks = self.split_text(text)
        embeddings = self.embedder.encode(
            chunks,
            normalize_embeddings=True
        )

        topic_vector = np.mean(embeddings, axis=0)

        self.documents[doc_id] = {
            "filename": filename,
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
    # GLOBAL INDEX
    # ------------------------------------------------
    def _update_global_index(self):

        all_chunks = []
        all_embeddings = []

        for doc in self.documents.values():

            for chunk in doc["chunks"]:
                all_chunks.append({
                    "text": chunk,
                    "source": doc["filename"]
                })

            all_embeddings.append(doc["embeddings"])

        self.global_chunks = all_chunks

        if all_embeddings:
            self.global_embeddings = np.vstack(all_embeddings)

    # ------------------------------------------------
    def cosine(self, A, B):
        return np.dot(A, B) / (
            np.linalg.norm(A) *
            np.linalg.norm(B) + 1e-10
        )

    # ------------------------------------------------
    # TOPIC SIMILARITY
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
    # RETRIEVE + CITATIONS
    # ------------------------------------------------
    def retrieve(self, query, top_k=6):

        query_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )[0]

        scores = np.dot(self.global_embeddings, query_emb)

        idx = np.argsort(scores)[-top_k:][::-1]

        selected = [self.global_chunks[i] for i in idx]

        context = "\n".join([c["text"] for c in selected])
        sources = list(set([c["source"] for c in selected]))

        return context, sources

    # ------------------------------------------------
    # NORMAL LLM CALL
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
    # STREAMING LLM
    # ------------------------------------------------
    def stream_llm(self, prompt):

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True
        }

        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True
        ) as r:

            for line in r.iter_lines():
                if line:
                    try:
                        token = json.loads(
                            line.decode()
                        )["response"]
                        yield token
                    except:
                        pass

    # ------------------------------------------------
    # SMART FOLLOWUPS
    # ------------------------------------------------
    def resolve_followup(self, question):

        followups = [
            "explain more",
            "tell more",
            "summarize that",
            "elaborate",
            "details"
        ]

        if self.chat_history and any(
            f in question.lower() for f in followups
        ):
            last_q = self.chat_history[-1][0]
            return last_q + " " + question

        return question

    # ------------------------------------------------
    # SUMMARY
    # ------------------------------------------------
    def generate_summary(self):

        if not self.documents:
            return "No PDFs loaded."

        if self.summary_cache:
            return self.summary_cache

        if self.documents_are_similar():

            context = "\n".join(
                [c["text"] for c in self.global_chunks[:8]]
            )

            prompt = f"""
Create ONE unified concise summary combining all documents.

{context}
"""

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
    # STREAMING CHAT (SPECIFIC ANSWERS)
    # ------------------------------------------------
    def ask_question_stream(self, question):

        question = self.resolve_followup(question)

        context, sources = self.retrieve(question)

        prompt = f"""
You are a precise document QA assistant.

Rules:
- Answer ONLY using context.
- Be specific.
- Include numbers, definitions, names when present.
- No assumptions.
- If missing say: Not found in documents.

Context:
{context}

Question: {question}

Answer:
"""

        answer = ""

        for token in self.stream_llm(prompt):
            answer += token
            yield token, None

        self.chat_history.append((question, answer))

        yield "", sources
