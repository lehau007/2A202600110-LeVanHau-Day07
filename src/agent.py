from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        """
        Initialize the KnowledgeBaseAgent with a store and an LLM function.
        """
        self.store = store
        self.llm_fn = llm_fn
        # self.conversation_history: list[tuple[str, str]] = []  # List of (question, answer) pairs

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Retrieve chunks, build a prompt, and call the LLM function to generate an answer.
        """
        # Step 1: Retrieve top-k relevant chunks from the store
        relevant_chunks = self.store.search(query=question, top_k=top_k)

        # Step 2: Build a prompt with the chunks as context
        context = "\n".join(chunk["content"] for chunk in relevant_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        # Step 3: Call the LLM to generate an answer
        return self.llm_fn(prompt)
