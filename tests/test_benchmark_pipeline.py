from __future__ import annotations

from src.benchmarking import BENCHMARK_CASES, BENCHMARK_FILES, compare_retrieval_strategies, run_benchmark


def test_benchmark_runs_with_five_docs() -> None:
    def fake_llm(prompt: str) -> str:
        return "This answer mentions recursive strategy and metadata filter for retrieval."

    result = run_benchmark(llm_fn=fake_llm, file_paths=BENCHMARK_FILES)

    assert result["num_docs_loaded"] == 5
    assert result["num_chunks_loaded"] >= 5
    assert result["store_size"] == result["num_chunks_loaded"]
    assert result["num_cases"] == len(BENCHMARK_CASES)
    assert result["chunking_strategy"] == "RecursiveChunker"
    assert 0.0 <= result["retrieval_hit_rate"] <= 1.0
    assert 0.0 <= result["avg_keyword_score"] <= 1.0
    assert len(result["details"]) == len(BENCHMARK_CASES)
    assert any(detail["used_metadata_filter"] for detail in result["details"])


def test_compare_retrieval_strategies_returns_three_strategies() -> None:
    def fake_llm(prompt: str) -> str:
        return "This answer mentions retrieval and chunking strategy."

    result = compare_retrieval_strategies(llm_fn=fake_llm, file_paths=BENCHMARK_FILES)

    assert set(result["strategies"].keys()) == {"fixed_size", "sentence", "recursive"}
    for stats in result["strategies"].values():
        assert stats["num_chunks_loaded"] >= 5
        assert 0.0 <= stats["retrieval_hit_rate"] <= 1.0
        assert 0.0 <= stats["avg_keyword_score"] <= 1.0

    assert result["best_by_retrieval_hit_rate"] in result["strategies"]
    assert result["best_by_avg_keyword_score"] in result["strategies"]
