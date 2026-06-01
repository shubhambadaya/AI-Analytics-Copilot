import json
from typing import Dict, Any, List
from src.llm.client import llm_client
from src.llm.schemas import ClarificationCheck
from src.context.builder import format_business_context
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are a senior data analyst scoping a stakeholder's question BEFORE any analysis runs.
You are given the dataset's columns, its defined KPIs, learned business rules, and the question.

Decide whether you genuinely need to ASK the stakeholder a clarifying question first.
Set needs_clarification = True ONLY when the question depends on a metric, term, or segment that is:
  - NOT an existing column,
  - NOT a defined KPI, and
  - NOT covered by a learned business rule,
AND where guessing its meaning would materially change the answer
(e.g. the user asks about "ARPU" or "high-value customers" but no revenue column, KPI, or rule defines it).

Otherwise set needs_clarification = False and proceed.

ALWAYS provide:
- `assumptions`: the key assumptions you are making (e.g. "Using `Price` as the revenue measure").
- `approach`: 1-2 sentences, plain language, on how you'll tackle the question.

Be a decisive analyst, not a gatekeeper. Do NOT ask trivial, stylistic, or formatting questions.
Ask at most 2 questions, and only when truly necessary.
"""


def run_clarification_check(
    query: str,
    context_profile: Dict[str, Any],
    history: List[Dict[str, Any]] = None,
    provider: str = None,
    model_override: str = None
) -> ClarificationCheck:
    """
    Lightweight scoping pass for the COMPLEX path: decides whether to ask the
    stakeholder a clarifying question (only when a key term is genuinely undefined)
    and returns the analyst's stated assumptions and approach.
    """
    logger.info("Clarifier scoping the question...")

    columns = []
    for table in context_profile.get("tables", {}).values():
        columns.extend(table.get("columns", {}).keys())

    business_ctx = format_business_context(context_profile) or "No KPIs or learned rules are defined."

    prompt = (
        f"Available columns: {columns}\n\n"
        f"{business_ctx}\n\n"
        f'Stakeholder question: "{query}"\n'
    )

    if history:
        hist_str = "\n".join([f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in history[-3:]])
        prompt = f"Conversation so far:\n{hist_str}\n\n" + prompt

    return llm_client.generate_structured_output(
        prompt=prompt,
        response_model=ClarificationCheck,
        system_prompt=SYSTEM_PROMPT,
        provider=provider,
        model_override=model_override
    )
