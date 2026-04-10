from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3, overlap_size: int = 1) -> None:
        if max_sentences_per_chunk <= overlap_size:
            raise ValueError("max_sentences_per_chunk must be greater than overlap_size")
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)
        self.overlap_size = max(0, overlap_size)

    def chunk(self, text: str) -> list[str]:
        texts = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks = [] 
        for ri in range(0, len(texts), self.max_sentences_per_chunk - self.overlap_size): 
            chunk = " ".join(texts[ri:min(ri + self.max_sentences_per_chunk, len(texts))])
            chunks.append(chunk)

        return chunks
        


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:

        ["\n\n", "\n", ". ", " ", ""]

    """
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500, overlap: int = 50) -> None:
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Phase 1: Recursively break text down into the smallest coherent units."""
        final_chunks = []
        
        # Get the highest priority separator
        separator = separators[0] if separators else ""
        new_separators = separators[1:]
        
        # Split the text
        if separator:
            splits = text.split(separator)
        else:
            # If no separators left, force-split by characters (safety net)
            splits = list(text)

        for s in splits:
            if len(s) <= self.chunk_size:
                final_chunks.append(s)
            elif new_separators:
                # Recurse with the next separator
                final_chunks.extend(self._split_text(s, new_separators))
            else:
                # Last resort: Hard cut if still too long
                final_chunks.append(s)
                
        return final_chunks

    def chunk(self, text: str) -> list[str]:
        """Phase 2: Merge small units into chunks with a sliding window."""
        if not text: return []
        
        # 1. Get the "atoms" (A, B, C, D from your diagram)
        units = self._split_text(text, self.separators)
        
        # 2. Merge units into chunks using the sliding window logic
        merged_chunks = []
        current_doc = []
        current_len = 0
        
        for unit in units:
            # If adding this unit exceeds chunk_size, finalize current_doc
            if current_len + len(unit) > self.chunk_size and current_doc:
                merged_chunks.append("".join(current_doc))
                
                # Start new chunk with overlap (the "Sliding Window")
                # We keep units from the end until we hit the overlap size
                overlap_docs = []
                overlap_len = 0
                for d in reversed(current_doc):
                    if overlap_len + len(d) <= self.overlap:
                        overlap_docs.insert(0, d)
                        overlap_len += len(d)
                    else:
                        break
                current_doc = overlap_docs
                current_len = overlap_len
            
            current_doc.append(unit)
            current_len += len(unit)
            
        if current_doc:
            merged_chunks.append("".join(current_doc))
            
        return merged_chunks

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b or math.sqrt(_dot(vec_a, vec_a)) == 0 or math.sqrt(_dot(vec_b, vec_b)) == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (math.sqrt(_dot(vec_a, vec_a)) * math.sqrt(_dot(vec_b, vec_b)))

class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=20),
            "sentence": SentenceChunker(max_sentences_per_chunk=2, overlap_size=1),
            "recursive": RecursiveChunker(chunk_size=chunk_size * 2, overlap=40),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=2, overlap_size=1),
        }
        results = {}
        for name, strategy in strategies.items():
            chunks = strategy.chunk(text)
            results[name] = {
                "num_chunks": len(chunks),
                "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "count": len(chunks),
                "chunks": chunks,
            }
        return results
