import pandas as pd
from typing import Optional
from src.llm.client import llm_client
from src.llm.schemas import RelevanceCheckResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are a meticulous analytics reviewer. You are given a user's business question, the
pandas code that was run, and the resulting table. Your ONLY job is to decide whether the
RESULT actually answers the QUESTION — the code already executed without errors, so you are
NOT checking for crashes. You are checking for the silent, dangerous failure where code runs
fine but computes the WRONG thing.

Check for:
- Wrong column or metric (e.g. averaged the handset price when the question is about plan value).
- Wrong aggregation or grain (e.g. summed when it should have averaged, or per-row when it
  should be per-group).
- A missing filter or wrong segment (e.g. ignored a "for active users" qualifier).
- Answering a subtly different question than the one asked.
- An empty or degenerate result that doesn't actually address the question.

Be pragmatic, not pedantic: if the result reasonably and correctly answers the question, pass it
(answers_question = true). Only fail it when there is a concrete, defensible defect — and when you
fail it, name the issue and give a specific fix_instruction the code generator can act on.
"""


def run_relevance_check(
    query: str,
    pandas_code: str,
    result_df: pd.DataFrame,
    provider: Optional[str] = None,
    model_override: Optional[str] = None,
    max_rows: int = 25,
) -> RelevanceCheckResult:
    """Verify that an executed result genuinely answers the question before it is interpreted.

    Returns a RelevanceCheckResult; on any failure to evaluate, returns a permissive
    pass so the pipeline never blocks on the check itself.
    """
    logger.info("Relevance check: verifying the result answers the question...")

    if result_df is None or result_df.empty:
        return RelevanceCheckResult(
            answers_question=False,
            reasoning="The analysis produced no rows, so it does not answer the question.",
            issue="Empty result set.",
            fix_instruction="Re-check filters/joins; the aggregation returned no rows.",
        )

    try:
        table_md = result_df.head(max_rows).to_markdown(index=False)
    except Exception:
        table_md = str(result_df.head(max_rows))
    shape_note = f"(showing up to {max_rows} of {len(result_df)} rows, columns: {list(result_df.columns)})"

    prompt = (
        f"User Question:\n\"{query}\"\n\n"
        f"Pandas code that was executed:\n```python\n{pandas_code}\n```\n\n"
        f"Resulting table {shape_note}:\n{table_md}\n\n"
        "Does this result genuinely answer the user's question?"
    )

    return llm_client.generate_structured_output(
        prompt=prompt,
        response_model=RelevanceCheckResult,
        system_prompt=SYSTEM_PROMPT,
        provider=provider,
        model_override=model_override,
    )
