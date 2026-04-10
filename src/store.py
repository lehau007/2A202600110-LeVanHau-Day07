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
            try:
                self._client.delete_collection(name=self._collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _normalize_metadata_for_chroma(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Chroma metadata values must be primitive scalar types."""
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return normalized

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
            if self._collection is None:
                return
            ids = [f"{doc.id}::{self._next_index + idx}" for idx, doc in enumerate(docs)]
            contents = [doc.content for doc in docs]
            embeddings: list[Any] = [self._embedding_fn(doc.content) for doc in docs]
            metadatas: list[Any] = []
            for doc in docs:
                metadata_with_doc_id = {"doc_id": doc.id, **doc.metadata}
                normalized = self._normalize_metadata_for_chroma(metadata_with_doc_id)
                metadatas.append(normalized if normalized else None)

            self._collection.add(ids=ids, documents=contents, embeddings=embeddings, metadatas=metadatas)
            self._next_index += len(docs)
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.
        """
        if self._use_chroma:
            if self._collection is None:
                return []
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
            ids = (results.get("ids") or [[]])[0]
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            return [
                {
                    "id": (metadata or {}).get("doc_id", doc_id),
                    "content": doc,
                    "score": 1.0 / (1.0 + float(distance)),
                    "metadata": metadata or {},
                }
                for doc_id, doc, metadata, distance in zip(ids, documents, metadatas, distances)
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
            if self._collection is None:
                return 0
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.
        """
        if self._use_chroma:
            if self._collection is None:
                return []
            query_embedding = self._embedding_fn(query)
            where = self._normalize_metadata_for_chroma(metadata_filter or {})
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
            }
            if where:
                query_kwargs["where"] = where

            results = self._collection.query(**query_kwargs)
            ids = (results.get("ids") or [[]])[0]
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            return [
                {
                    "id": (metadata or {}).get("doc_id", doc_id),
                    "content": doc,
                    "score": 1.0 / (1.0 + float(distance)),
                    "metadata": metadata or {},
                }
                for doc_id, doc, metadata, distance in zip(ids, documents, metadatas, distances)
            ]

        if metadata_filter:
            filtered_records = [
                record for record in self._store if all(
                    record["metadata"].get(key) == value for key, value in metadata_filter.items()
                )
            ]
        else:
            filtered_records = self._store

        matched_records = self._search_records(query, filtered_records, top_k)
        query_embedding = self._embedding_fn(query)
        return [
            {
                "id": record["id"],
                "content": record["content"],
                "score": _dot(query_embedding, record["embedding"]),
                "metadata": record.get("metadata", {}),
            }
            for record in matched_records
        ]

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.
        """
        if self._use_chroma:
            if self._collection is None:
                return False
            ids_to_delete: set[str] = set()

            try:
                by_doc_id = self._collection.get(where={"doc_id": doc_id})
                ids_to_delete.update(by_doc_id.get("ids", []))
            except Exception:
                pass

            try:
                direct = self._collection.get(ids=[doc_id])
                ids_to_delete.update(direct.get("ids", []))
            except Exception:
                pass

            if not ids_to_delete:
                return False

            self._collection.delete(ids=list(ids_to_delete))
            return True

        initial_size = len(self._store)
        self._store = [
            record
            for record in self._store
            if record["id"] != doc_id and record.get("metadata", {}).get("doc_id") != doc_id
        ]
        return len(self._store) < initial_size
