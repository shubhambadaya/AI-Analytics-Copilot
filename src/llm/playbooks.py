"""
Strategy playbooks: proven analysis sequences for common open-ended business goals.

These GUIDE the strategic planner (they don't hard-code the analysis) — the planner
still adapts each step to the columns and KPIs that actually exist in the dataset,
and the plan is validated against the schema. Think of them as an expert analyst's
default approach to recurring questions.
"""
from typing import Optional, Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

PLAYBOOKS: List[Dict[str, Any]] = [
    {
        "name": "Improve ARPU / revenue",
        "keywords": ["arpu", "revenue", "spend", "spending", "monetiz", "average revenue", "wallet", "value per user"],
        "steps": [
            "Establish the revenue/ARPU metric and profile its overall level and distribution",
            "Compare average revenue across the key customer dimensions (demographics, plan, handset, geography, behavior)",
            "Rank which characteristics most drive higher revenue using driver analysis (effect size + significance)",
            "Identify the high-value vs low-value segments and quantify the size of the upside",
        ],
    },
    {
        "name": "Reduce churn / improve retention",
        "keywords": ["churn", "retention", "attrition", "leaving", "downgrade", "cancel", "lapse", "inactive"],
        "steps": [
            "Locate or define the churn/attrition signal and compute the overall rate",
            "Compute the churn rate by segment to see where it concentrates (within-group rates, not raw counts)",
            "Rank which characteristics are most associated with higher churn using driver analysis",
            "Size the at-risk segments and quantify the retention opportunity",
        ],
    },
    {
        "name": "Upgrade / cross-sell targeting",
        "keywords": ["upgrade", "cross-sell", "cross sell", "upsell", "who should we", "target", "propensity", "uptake", "higher plan", "premium"],
        "steps": [
            "Define the upgrade target (the higher-tier criterion) using the business definition/KPI",
            "Profile and compare customers who already meet it vs those who don't",
            "Rank the characteristics that most distinguish upgraders using driver analysis",
            "Identify the look-alike non-upgraders that match the upgrader profile as the target list",
        ],
    },
    {
        "name": "Boost engagement / usage",
        "keywords": ["usage", "engagement", "active users", "adoption", "consumption", "data usage", "utilization"],
        "steps": [
            "Profile the engagement/usage metric and its distribution",
            "Compare usage across the main customer segments",
            "Rank the drivers of high usage using driver analysis (effect size + significance)",
            "Identify the low-engagement segments with the most headroom to activate",
        ],
    },
]


def match_playbook(query: str) -> Optional[Dict[str, Any]]:
    """Return the best-matching playbook for a query (by keyword overlap), or None."""
    q = (query or "").lower()
    best, best_hits = None, 0
    for pb in PLAYBOOKS:
        hits = sum(1 for kw in pb["keywords"] if kw in q)
        if hits > best_hits:
            best, best_hits = pb, hits
    if best:
        logger.info(f"Matched strategy playbook: '{best['name']}' ({best_hits} keyword hits)")
    return best if best_hits > 0 else None


def format_playbook(pb: Dict[str, Any]) -> str:
    """Render a playbook as planner guidance."""
    steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(pb["steps"]))
    return (
        f"PROVEN ANALYSIS PLAYBOOK for goals like this ('{pb['name']}'). Use it as your "
        f"default structure, but ADAPT each step to the columns and KPIs that actually "
        f"exist in this dataset (skip a step if the data can't support it):\n{steps}"
    )
