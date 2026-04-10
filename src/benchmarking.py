from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from .agent import KnowledgeBaseAgent
from .embeddings import _mock_embed
from .models import Document
from .store import EmbeddingStore


BENCHMARK_FILES = [
    "data/chi_pheo.txt",
    "data/1_bua_no.txt",
    "data/quet_nha.txt",
    "data/trang_sang.txt",
    "data/tu_cach_mo.txt",
]


@dataclass
class BenchmarkCase:
    query: str
    expected_source_suffix: str
    gold_keywords: list[str]


BENCHMARK_CASES = [
    BenchmarkCase(
        query="Trong Chí Phèo, nhân vật Chí Phèo mở đầu truyện bằng hành động gì?",
        expected_source_suffix="chi_pheo.txt",
        gold_keywords=["chửi", "rượu", "vũ đại"],
    ),
    BenchmarkCase(
        query="Trong truyện Một bữa no, bà lão sống bằng cách nào khi tuổi già sức yếu?",
        expected_source_suffix="1_bua_no.txt",
        gold_keywords=["đói", "xin", "bánh đúc", "bà phó"],
    ),
    BenchmarkCase(
        query="Trong truyện Quét nhà, vì sao cha mẹ của Hồng hay cáu gắt trong giai đoạn khó khăn?",
        expected_source_suffix="quet_nha.txt",
        gold_keywords=["túng", "nợ", "khó khăn", "mắng"],
    ),
    BenchmarkCase(
        query="Trong truyện Trăng sáng, Điền thay đổi quan niệm nghệ thuật như thế nào?",
        expected_source_suffix="trang_sang.txt",
        gold_keywords=["nghệ thuật", "đau khổ", "ánh trăng", "sự thật"],
    ),
    BenchmarkCase(
        query="Trong truyện Tư cách mõ, Lộ bị biến đổi tính cách do những tác động xã hội nào?",
        expected_source_suffix="tu_cach_mo.txt",
        gold_keywords=["mõ", "khinh", "nhục", "đê tiện"],
    ),
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    docs: list[Document] = []
    for raw_path in file_paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        docs.append(
            Document(
                id=path.stem,
                content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )
    return docs


class GeminiFlashLiteLLM:
    def __init__(self, model: str = "gemini-3.1-flash-lite-preview") -> None:
        load_dotenv(override=False)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment")

        try:
            from google import genai
        except Exception as exc:  # pragma: no cover - import failure depends on runtime env
            raise RuntimeError("google-genai package is required for Gemini benchmark") from exc

        self.model = model
        self._backend_name = model
        self._client = genai.Client(api_key=api_key)

    def __call__(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return (response.text or "").strip()


def run_benchmark(
    llm_fn: Callable[[str], str],
    file_paths: list[str] | None = None,
    top_k: int = 3,
    embedding_fn: Callable[[str], list[float]] | None = None,
) -> dict:
    files = file_paths or BENCHMARK_FILES
    docs = load_documents_from_files(files)

    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=embedding_fn or _mock_embed)
    store.add_documents(docs)

    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)

    details = []
    retrieval_hits = 0
    keyword_scores: list[float] = []

    for case in BENCHMARK_CASES:
        retrieved = store.search(case.query, top_k=top_k)
        sources = [str(item.get("metadata", {}).get("source", "")) for item in retrieved]
        retrieval_hit = any(src.endswith(case.expected_source_suffix) for src in sources)
        retrieval_hits += int(retrieval_hit)

        answer = agent.answer(case.query, top_k=top_k)
        lowered_answer = answer.lower()
        hit_keywords = sum(1 for kw in case.gold_keywords if kw.lower() in lowered_answer)
        keyword_score = hit_keywords / max(1, len(case.gold_keywords))
        keyword_scores.append(keyword_score)

        details.append(
            {
                "query": case.query,
                "retrieval_hit": retrieval_hit,
                "retrieved_sources": sources,
                "keyword_score": keyword_score,
                "answer_preview": answer[:220],
            }
        )

    return {
        "num_docs_loaded": len(docs),
        "store_size": store.get_collection_size(),
        "num_cases": len(BENCHMARK_CASES),
        "retrieval_hit_rate": retrieval_hits / max(1, len(BENCHMARK_CASES)),
        "avg_keyword_score": sum(keyword_scores) / max(1, len(keyword_scores)),
        "details": details,
    }
