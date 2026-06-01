# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Analytics Copilot — a Streamlit app that answers natural-language business questions over uploaded CSVs. The core design principle is **decoupling deterministic calculation from LLM reasoning**: the LLM only generates code and prose, while all math runs locally in a sandboxed Python executor. This keeps numeric answers 100% reproducible regardless of LLM hallucination.

## Commands

```bash
pip install -r requirements.txt          # install deps
cp .env.example .env                      # then add API keys (see note below)
streamlit run app.py                      # run the app

# Tests are standalone scripts, NOT a pytest suite. Run individually from repo root:
python test_pipeline.py                   # end-to-end headless pipeline smoke test
python test_ml.py                         # ml_engine.train_and_predict
python test_models.py                     # lists available Gemini models for your key
python scratch_verify.py                  # validator/golden-store/stats-engine checks
```

There is no linter, formatter, or test runner configured. Tests must be run from the repo root so `src.*` imports resolve.

## Environment / API Keys

`.env.example` is incomplete: the app's default model tier is **Gemini**, so `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is effectively required even though it's missing from the example file. `config.py` reads `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`/`GOOGLE_API_KEY`. Provider availability is auto-detected; the picker order is openai → gemini → anthropic.

Per-provider models are hardcoded in `src/llm/client.py`: OpenAI uses `gpt-4o-mini`, Anthropic uses `claude-3-haiku-20240307`, Gemini uses the tier constants `MODEL_FAST = gemini-2.5-flash` and `MODEL_PRO = gemini-2.5-pro`. Note that `model_override` (the FAST/PRO tier routing) is **only wired through the Gemini path** — OpenAI/Anthropic ignore it.

## Architecture

### Request flow
`app.py` (UI only) → `src/llm/pipeline.run_headless_pipeline()` (generator that yields intermediate UI states) → routed to one of three execution paths.

`pipeline.py` first calls `classify_query_complexity()` to route, then runs a `LEARN_RULE` check via `agent.evaluate_agent_state()`. If the user is teaching a rule rather than asking a question, it's saved to semantic memory and the pipeline returns early. Otherwise it dispatches:

- **SIMPLE** (`_run_simple_path`): single-shot code gen (Flash) → execute → one self-correction retry → single-shot interpretation. Target latency 3–5s.
- **COMPLEX** (`_run_complex_path`): a 5-agent DAG — Schema → (Stats ∥ Viz in parallel via `ThreadPoolExecutor`) → Insight → Recommender, all on Pro.
- **PREDICTIVE** (`_run_predictive_path`): ML agent generates code calling `ml_engine.train_and_predict` (RandomForest), then Insight agent synthesizes.

Each agent is a thin wrapper in `src/llm/*_agent.py` that calls `llm_client.generate_structured_output()` with a Pydantic schema from `src/llm/schemas.py`.

### Deterministic execution sandbox (the security-critical core)
LLM-generated pandas code is run by `src/analysis/engine.execute_analysis()`, which gates everything through `src/analysis/validator.validate_pandas_code()`:
- `validator.py` walks the AST and blocks non-whitelisted imports, forbidden builtins (`eval`, `open`, `getattr`, etc.), and **any dunder attribute access** (`__class__`, …) to prevent sandbox escapes.
- `engine.py` then `exec()`s the code in a hand-built namespace with a restricted `__import__` and a curated builtins dict. Active datasets are pre-injected as variables named after their cleaned filename (`subscriber_profile.csv` → `subscriber_profile`); `pd`, `np`, `stats_engine`, `ml_engine` are pre-bound.
- **Contract:** generated code must produce its output by assigning `result_df` (or `result`), or by defining an `analyze(df)` function. Otherwise the final state of `df` is returned.
- **Gotcha:** the whitelists in `validator.ALLOWED_MODULES` and `engine.SAFE_MODULES` differ (engine allows `sklearn`/`ml_engine`, validator does not). Code that *imports* sklearn will pass `engine` but be blocked by `validator`. ML code is expected to call the pre-injected `ml_engine` module rather than import sklearn directly.

### Context assembly
`src/metadata/extractor.profile_dataframe()` deterministically profiles each uploaded CSV (types, cardinality, stats, ID detection). An optional business data dictionary (YAML/JSON) is validated and merged via `src/metadata/dictionary.py`. `src/context/builder.ContextBuilder.build_llm_context()` then compresses all active tables, inferred join relationships, KPIs, and learned rules into one token-efficient profile that is passed to every agent. `src/context/manager.py` handles file persistence to `data/`.

### Persistent learning stores (JSON files in `data/`)
- `golden_queries.json` (`src/llm/golden_queries.py`): caches high-confidence (≥0.75) query→pandas-code pairs as dynamic few-shot examples, retrieved by Jaccard token similarity.
- `semantic_memory.json` (`src/llm/semantic_memory.py`): user-taught business rules, injected into context on every call.

## Known incomplete areas

- In `_run_complex_path`, the deterministic stats execution loop is **mocked** — it iterates `stats_plan.requested_tests` but `pass`es instead of calling the `stats_engine` functions (see the comment in `pipeline.py`). Stats results are effectively empty in the complex path.
- `app.py` reads the architecture diagram and walkthrough from **hardcoded absolute paths** under `/Users/B0269091/.gemini/antigravity/brain/...` (the "System Architecture" and "Semantic KB" tabs). These will silently warn if absent on another machine.
- The complex/predictive paths build a `MockInterpretation` object to reuse `_build_final_package`, rather than a real `InsightInterpretation` schema instance.

## Conventions

- All modules use the shared singletons: `config`, `llm_client`, `context_manager`, `context_builder`, `golden_store`, `semantic_store`, and `get_logger(__name__)`.
- LLM I/O is always structured: define/extend a Pydantic model in `schemas.py` and pass it as `response_model`. The Gemini path in `client.py` does heavy schema rewriting (inlining `$refs`, flattening `anyOf`/`Optional`, stripping `default`/`title`) because Gemini's schema parser rejects standard Pydantic JSON Schema — preserve this when touching `_call_gemini`.
- Charts: agents emit `VisualSpec` objects (`schemas.py`); `src/visuals/generator.build_plotly_chart()` renders them. DataFrames are serialized to JSON for Streamlit session-state round-tripping.
