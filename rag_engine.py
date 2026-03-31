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
    # LOAD PDFs
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

        # normalize embeddings
    # STREAM CHAT ANSWER
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

        # save conversation memory
        self.chat_history.append(f"User: {query}")
        self.chat_history.append(f"Assistant: {full_text}")

    # --------------------------------
    # FOLLOW-UP QUERY UNDERSTANDING
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

            for msg in reversed(self.chat_history):
                if msg.startswith("User:"):
                    last_question = msg.replace("User:", "").strip()
                    return f"{query} (regarding: {last_question})"

        return query

    # --------------------------------
    # STREAM DOCUMENT SUMMARY
    # --------------------------------
    def stream_summary(self):

        if not self.chunks:
            yield "No PDFs loaded.", []
            return

        texts = [c["text"] for c in self.chunks[:15]]
        context_text = "\n\n".join(texts)

        prompt = f"""
You are an expert document analyst.

Provide a structured summary including:
- Main topics
- Key insights
- Important findings
- Conclusion

Document Content:
{context_text}
"""

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
            except:
                continue

            token = data.get("response", "")
            full_text += token

            yield token, self.chunks[:3]

