from __future__ import annotations

import json

from src.benchmarking import GeminiFlashLiteLLM, run_benchmark


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

    result = run_benchmark(llm_fn=llm)
    print(f"Benchmark LLM backend: {backend}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
