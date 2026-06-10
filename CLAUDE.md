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

`.env.example` is incomplete: the app's default model tier is **Gemini**, so `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is effectively required even though it's missing from the example file. `config.py` reads `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`/`GOOGLE_API_KEY`. Provider availability is auto-detected; the picker order is openai → gemini → anthropic. On Streamlit Community Cloud, `app.py` bridges `st.secrets` into environment variables at import time.

Per-provider models are hardcoded in `src/llm/client.py`: OpenAI uses `gpt-4o-mini`, Anthropic uses `claude-3-haiku-20240307`, Gemini uses the tier constants `MODEL_FAST = gemini-2.5-flash` and `MODEL_PRO = gemini-pro-latest` (an auto-tracking alias — pin an explicit version for deterministic runs). Note that `model_override` (the FAST/PRO tier routing) is **only wired through the Gemini path** — OpenAI/Anthropic ignore it.

Optional: setting `DATABASE_URL` (Postgres or `sqlite:///...`) makes the learning stores and query log durable across redeploys via `src/utils/persistence.JSONBlobStore`; without it they fall back to local JSON files in `data/`.

## Architecture

### Request flow
`app.py` (UI only) → `src/llm/pipeline.run_headless_pipeline()` (generator that yields intermediate UI states: `running`, `thought`, `code`, `clarify`, `error`, `complete`) → routed to one of three execution paths.

`pipeline.py` first calls `complexity_router.classify_query_complexity()`, which classifies the query as SIMPLE, COMPLEX, PREDICTIVE, or LEARN_RULE in a single call. LEARN_RULE (the user is teaching a business rule, not asking a question) is handled directly: the rule is saved to semantic memory and the pipeline returns early. The router also detects whether the user asked for recommendations (`wants_recommendations`) — the Recommender agent only runs when they did. Otherwise it dispatches:

- **SIMPLE** (`_run_simple_path`, Flash): golden-cache short-circuit (reuse validated code from a near-identical past query, still re-validated/re-executed) → single-shot code gen → execute → one self-correction retry → relevance check (`relevance.run_relevance_check`, a correctness gate that catches code that ran fine but computed the wrong thing; one refinement pass) → single-shot interpretation → charts. Target latency 3–5s.
- **COMPLEX** (`_run_complex_path`, Pro): a strategic ReAct investigation, not a fixed DAG:
  1. **Clarification check** (`clarifier.py`): if the question hinges on an undefined term, yield a `clarify` state with questions and stop; otherwise record the analyst's assumptions/approach.
  2. **Strategic blueprint** (`analysis_planner.py`, guided by keyword-matched `playbooks.py`): decomposes the goal into planned analytical steps.
  3. **Bounded ReAct loop** (≤ `MAX_INVESTIGATION_STEPS` = 5): per step, `schema_agent` generates code → `engine` executes it (earlier step results are injected as `result_step_N` variables so steps build on each other) → one self-correction retry. Planned blueprint steps run first; once exhausted, `agent.evaluate_agent_state()` decides CONTINUE (with a new focus) or COMPLETE.
  4. **Synthesis**: `stats_agent` plans tests which are executed deterministically by `stats_engine.run_requested_tests()` → `viz_agent` designs charts (de-duplicated by signature) → `insight_agent` synthesizes across all investigation steps → `critic.run_critique()` reviews the answer against the evidence (tightens unsupported claims, appends caveats, sets grounded confidence) → `recommender_agent` (only if requested).
- **PREDICTIVE** (`_run_predictive_path`, Pro): `ml_agent` generates code calling `ml_engine.train_and_predict` (RandomForest) → one self-correction retry → `insight_agent` → `recommender_agent` (only if requested, fed a compact evidence payload of the scored cohort).

Each agent is a thin wrapper in `src/llm/*_agent.py` (plus `clarifier`/`critic`/`relevance`/`planner`/`analysis_planner`/`interpreter`) that calls `llm_client.generate_structured_output()` with a Pydantic schema from `src/llm/schemas.py`.

### Deterministic execution sandbox (the security-critical core)
LLM-generated pandas code is run by `src/analysis/engine.execute_analysis()`, which gates everything through `src/analysis/validator.validate_pandas_code()`:
- `validator.py` walks the AST and blocks non-whitelisted imports, forbidden builtins (`eval`, `open`, `getattr`, etc.), and **any dunder attribute access** (`__class__`, …) to prevent sandbox escapes.
- `engine.py` then `exec()`s the code in a hand-built namespace with a restricted `__import__` and a curated builtins dict. Active datasets are pre-injected as variables named after their cleaned filename (`subscriber_profile.csv` → `subscriber_profile`); `pd`, `np`, `stats_engine`, `ml_engine` are pre-bound, and `extra_vars` lets the complex path inject prior-step results.
- **Contract:** generated code must produce its output by assigning `result_df` (or `result`), or by defining an `analyze(df)` function. Otherwise the final state of `df` is returned.
- `validator.ALLOWED_MODULES` is the **single source of truth** for importable modules — `engine._safe_import` enforces the same set. sklearn is deliberately not importable; ML code must call the pre-injected `ml_engine` module instead.

### Context assembly
`src/metadata/extractor.profile_dataframe()` deterministically profiles each uploaded CSV (types, cardinality, stats, ID detection). An optional business data dictionary (YAML/JSON) is validated and merged via `src/metadata/dictionary.py` (`src/llm/dictionary_generator.py` can draft one with the LLM). `src/context/builder.ContextBuilder.build_llm_context()` then compresses all active tables, inferred join relationships, KPIs, and learned rules into one token-efficient profile that is passed to every agent. `src/context/manager.py` handles file persistence to `data/`.

### Persistent learning stores
Backed by `src/utils/persistence.JSONBlobStore` (Postgres/SQLite when `DATABASE_URL` is set, local JSON files in `data/` otherwise):
- `golden_queries.json` (`src/llm/golden_queries.py`): caches high-confidence (≥0.75) query→pandas-code pairs as dynamic few-shot examples and for the SIMPLE-path cache short-circuit, retrieved by Jaccard token similarity.
- `semantic_memory.json` (`src/llm/semantic_memory.py`): user-taught business rules, injected into context on every call.
- `query_log.json` (`src/utils/query_log.py`): append-only global log of questions asked, viewable/downloadable in the app's Question Log tab.

## Conventions

- All modules use the shared singletons: `config`, `llm_client`, `context_manager`, `context_builder`, `golden_store`, `semantic_store`, `query_log_store`, and `get_logger(__name__)`.
- LLM I/O is always structured: define/extend a Pydantic model in `schemas.py` and pass it as `response_model`. The Gemini path in `client.py` does heavy schema rewriting (inlining `$refs`, flattening `anyOf`/`Optional`, stripping `default`/`title`) and deliberately embeds the schema in the prompt instead of using native constrained decoding (which drives Gemini 2.5 into repetition loops) — preserve both behaviors when touching `_call_gemini`.
- The complex/predictive paths build their interpretation object as a `SimpleNamespace` with the `InsightInterpretation`-shaped fields `_build_final_package` expects (`direct_answer`, `insights`, `recommendations`, `confidence_score`, `statistical_backing`).
- Charts: agents emit `VisualSpec` objects (`schemas.py`); `src/visuals/generator.build_plotly_chart()` renders them. DataFrames are serialized to JSON for Streamlit session-state round-tripping.
