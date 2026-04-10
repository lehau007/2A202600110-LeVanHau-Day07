from __future__ import annotations

import hashlib
import math
import os

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "my_embedding_server"
EMBEDDING_PROVIDER_LINK = os.getenv("EMBEDDING_PROVIDER_LINK", "https://default-placeholder-link.dev/")

"""
Embeddings provider is hosted in this link. Based on this implementation, you can understand its usage

from fastapi import FastAPI

app = FastAPI()

@app.get("/{text}")
def model_response(text: str):
    global model
    return model.encode(text)

from pyngrok import ngrok
import uvicorn
import threading

# Ensure the ngrok authtoken is set for this session
# The NGROK_TOKEN variable is available from `google.colab.userdata`
ngrok.set_auth_token(NGROK_TOKEN)

# Open a tunnel on port 8000
public_url = ngrok.connect(8000, domain="")
print("Public URL:", public_url)

# Run uvicorn in a separate thread
def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run, daemon=True)
thread.start()
"""

class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


_mock_embed = MockEmbedder()
class EmbeddingProvider:
    def __init__(self, provider_link: str) -> None:
        self.provider_link = provider_link
        self._backend_name = provider_link
    
    def __call__(self, text: str) -> list[float]:
        embedding = self._fetch_embedding(text)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]

    def _fetch_embedding(self, text: str) -> list[float]:
        import requests

        try:
            # Text need to be URL-encoded to handle spaces and special characters
            text = requests.utils.quote(text)
        
            response = requests.get(f"{self.provider_link}/{text}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching embedding from provider: {e}")
            return _mock_embed(text)
