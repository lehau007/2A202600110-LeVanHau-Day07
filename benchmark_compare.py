from __future__ import annotations

from src.benchmarking import GeminiFlashLiteLLM, compare_retrieval_strategies


def _fallback_llm(prompt: str) -> str:
    return "Fallback response because Gemini configuration is missing."


def main() -> int:
    try:
        llm = GeminiFlashLiteLLM(model="gemini-3.1-flash-lite-preview")
        backend = llm._backend_name
    except Exception as exc:
        print(f"[WARN] Could not initialize Gemini model: {exc}")
        llm = _fallback_llm
        backend = "fallback"

    result = compare_retrieval_strategies(llm_fn=llm)

    print(f"Comparison LLM backend: {backend}")
    print("\n=== Retrieval Strategy Comparison ===")
    print("strategy      chunks   hit_rate   avg_keyword")
    for name, stats in result["strategies"].items():
        print(
            f"{name:<12} {stats['num_chunks_loaded']:<8} "
            f"{stats['retrieval_hit_rate']:<10.3f} {stats['avg_keyword_score']:<10.3f}"
        )

    print("\nBest by retrieval_hit_rate:", result["best_by_retrieval_hit_rate"])
    print("Best by avg_keyword_score:", result["best_by_avg_keyword_score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
