# Lab 7 Summary

## Project Scope
- Domain: Vietnamese literature (Nam Cao short stories).
- Benchmark corpus: 5 files in data folder.
- LLM backend used: gemini-3.1-flash-lite-preview.

## Implemented Improvements
- Added metadata filtering path in agent answer flow.
- Added metadata-aware benchmark cases (including one ambiguous query requiring filter).
- Integrated chunking into benchmark ingestion (no longer indexing full file as one document).
- Added richer metadata schema: source, extension, title, author, category, year.
- Added strategy-comparison pipeline across FixedSize, Sentence, Recursive chunkers.

## Key Files Updated
- src/agent.py
- src/benchmarking.py
- benchmark.py
- benchmark_compare.py
- tests/test_benchmark_pipeline.py
- report/REPORT.md

## Latest Validated Results
- Tests: 44 passed.
- Benchmark (recursive default):
  - num_docs_loaded: 5
  - num_chunks_loaded: 144
  - retrieval_hit_rate: 0.8
  - avg_keyword_score: 0.4667
- Strategy comparison:
  - fixed_size: chunks=134, hit_rate=0.200, avg_keyword=0.517
  - sentence: chunks=700, hit_rate=0.600, avg_keyword=0.567
  - recursive: chunks=144, hit_rate=0.800, avg_keyword=0.467
  - Best by hit_rate: recursive
  - Best by avg_keyword_score: sentence

## Notes For Report/Defense
- Requirement Exercise 3.2 is satisfied: at least one benchmark query requires metadata filtering.
- Requirement Exercise 3.1/3.4 is satisfied: benchmark now uses chunkers and supports strategy comparison.
- A remaining failure pattern appears on semantically similar social-topic passages across stories; mitigation is stronger metadata filtering plus reranking.

## Run Commands
- Run tests:
  - c:/Users/msilaptop/Desktop/VinUni/lab7/2A202600110-LeVanHau-Day07/.venv/Scripts/python.exe -m pytest -q
- Run benchmark:
  - c:/Users/msilaptop/Desktop/VinUni/lab7/2A202600110-LeVanHau-Day07/.venv/Scripts/python.exe benchmark.py
- Run strategy comparison:
  - c:/Users/msilaptop/Desktop/VinUni/lab7/2A202600110-LeVanHau-Day07/.venv/Scripts/python.exe benchmark_compare.py
