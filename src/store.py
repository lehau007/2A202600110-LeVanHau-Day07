from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb
            from chromadb.config import Settings

            # Initialize ephemeral chromadb client
            self._client = chromadb.Client(Settings(allow_reset=True, is_persistent=False))
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """
        Build a normalized stored record for one document.
        """
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": doc.metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """
        Run in-memory similarity search over provided records.
        """
        query_embedding = self._embedding_fn(query)
        similarities = [
            {
                "record": record,
                "similarity": _dot(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return [sim["record"] for sim in similarities[:top_k]]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.
        """
        if self._use_chroma:
            ids = [doc.id for doc in docs]
            contents = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            self._collection.add(ids=ids, documents=contents, embeddings=embeddings)
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.
        """
        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
            return [
                {"id": doc_id, "content": doc, "score": score}
                for doc_id, doc, score in zip(results["ids"], results["documents"], results["distances"])
            ]
        else:
            query_embedding = self._embedding_fn(query)
            similarities = [
                {
                    "record": record,
                    "score": _dot(query_embedding, record["embedding"]),
                }
                for record in self._store
            ]
            similarities.sort(key=lambda x: x["score"], reverse=True)
            return [
                {
                    "id": sim["record"]["id"],
                    "content": sim["record"]["content"],
                    "score": sim["score"],
                    "metadata": sim["record"].get("metadata", {}),
                }
                for sim in similarities[:top_k]
            ]

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.
        """
        if metadata_filter:
            filtered_records = [
                record for record in self._store if all(
                    record["metadata"].get(key) == value for key, value in metadata_filter.items()
                )
            ]
        else:
            filtered_records = self._store

        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.
        """
        initial_size = len(self._store)
        self._store = [record for record in self._store if record["id"] != doc_id]
        return len(self._store) < initial_size
