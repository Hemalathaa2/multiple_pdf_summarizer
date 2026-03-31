import fitz
import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer

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
        self.chat_history = []

    # --------------------------------
    # PDF LOADING
    # --------------------------------
    def load_pdfs(self, files):

        self.chunks = []

        for file in files:
            doc = fitz.open(stream=file.read(), filetype="pdf")

            for page_num, page in enumerate(doc):
                text = page.get_text().strip()

                if not text:
                    continue

                for chunk in self.split_text(text):
                    self.chunks.append({
                        "text": chunk,
                        "source": file.name,
                        "page": page_num + 1
                    })

        if not self.chunks:
            raise ValueError("No readable text found in PDFs.")

        texts = [c["text"] for c in self.chunks]

        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True
        )

        # ✅ normalize once (fast cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / norms

    # --------------------------------
    # TEXT SPLITTER
    # --------------------------------
    def split_text(self, text, size=400, overlap=80):

        words = text.split()
        chunks = []

        step = size - overlap

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + size])
            if chunk:
                chunks.append(chunk)

        return chunks

    # --------------------------------
    # RETRIEVAL
    # --------------------------------
    def retrieve(self, query, k=3):

        q_embed = self.embedder.encode([query])[0]
        q_embed = q_embed / np.linalg.norm(q_embed)

        scores = np.dot(self.embeddings, q_embed)

        top_idx = np.argsort(scores)[-k:][::-1]

        return [self.chunks[i] for i in top_idx]

    # --------------------------------
    # PROMPT BUILDER
    # --------------------------------
    def build_prompt(self, query, contexts):

        context_text = "\n\n".join([
            f"[Source: {c['source']} | Page {c['page']}]\n{c['text']}"
            for c in contexts
        ])

        history = "\n".join(self.chat_history[-6:])

        return f"""
You are a helpful PDF assistant.

Use ONLY the provided context.
If answer not present, say you don't know.

Conversation History:
{history}

Context:
{context_text}

Question:
{query}

Answer clearly and cite sources.
"""

    # --------------------------------
    # STREAMING ANSWER
    # --------------------------------
    def stream_answer(self, query):

        contexts = self.retrieve(query)
        prompt = self.build_prompt(query, contexts)

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True
        )

        full_text = ""

        for line in response.iter_lines():

            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            token = data.get("response", "")
            full_text += token

            yield token, contexts

        # ✅ store conversation memory
        self.chat_history.append(f"User: {query}")
        self.chat_history.append(f"Assistant: {full_text}")

    # --------------------------------
    # FOLLOW-UP UNDERSTANDING
    # --------------------------------
    def reformulate_query(self, query):

        follow_words = [
            "explain more",
            "tell more",
            "summarize that",
            "why",
            "how"
        ]

        if any(w in query.lower() for w in follow_words):

            # find last user question
            for msg in reversed(self.chat_history):
                if msg.startswith("User:"):
                    last_question = msg.replace("User:", "").strip()
                    return f"{query} (regarding: {last_question})"

        return query
