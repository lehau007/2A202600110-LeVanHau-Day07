from __future__ import annotations

from src.benchmarking import BENCHMARK_CASES, BENCHMARK_FILES, run_benchmark


def test_benchmark_runs_with_five_docs() -> None:
    def fake_llm(prompt: str) -> str:
        return "This answer mentions recursive strategy and metadata filter for retrieval."

    result = run_benchmark(llm_fn=fake_llm, file_paths=BENCHMARK_FILES)

    assert result["num_docs_loaded"] == 5
    assert result["store_size"] == 5
    assert result["num_cases"] == len(BENCHMARK_CASES)
    assert 0.0 <= result["retrieval_hit_rate"] <= 1.0
    assert 0.0 <= result["avg_keyword_score"] <= 1.0
    assert len(result["details"]) == len(BENCHMARK_CASES)
