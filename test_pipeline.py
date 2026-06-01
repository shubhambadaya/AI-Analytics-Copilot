import asyncio
from src.llm.pipeline import run_headless_pipeline
import pandas as pd

df = pd.DataFrame({"user_id": [1, 2, 3]})
metadata = {"dimensions": ["user_id"]}

pipeline = run_headless_pipeline(
    "How many users are there?",
    {"df": "profile"},
    "gemini",
    df,
    {"profile": df},
    metadata,
    []
)

for step in pipeline:
    print(step)
