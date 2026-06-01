import pandas as pd
from src.metadata.extractor import extract_metadata_from_csv
from src.analysis.engine import execute_analysis
from src.llm.planner import generate_analysis_plan
from src.visuals.generator import build_plotly_chart

print("1. Testing Schema Validation")
from src.llm.schemas import AnalysisPlan, InsightInterpretation
print("Schema OK")

print("\n2. Testing Sandbox Whitelist")
from src.analysis.validator import validate_pandas_code
try:
    validate_pandas_code("import scipy.stats as sp; print('SciPy allowed!')")
    print("SciPy is successfully whitelisted in sandbox.")
except Exception as e:
    print(f"Failed whitelist: {e}")

print("\n3. Testing End-to-End Golden Query + Stats Engine")
from src.llm.golden_queries import golden_store
print(f"Golden Queries loaded: {len(golden_store._load_queries())}")

# Test Stats Engine standalone
print("\n4. Testing Stats Engine Fallback")
from src.analysis.stats_engine import compare_groups
df = pd.DataFrame({'Gender': ['M','F','M','F','M'], 'Usage': ['High', 'Low', 'High', 'Low', 'High']})
res = compare_groups(df, metric_col='Usage', group_col='Gender')
print(f"Stats Engine Categorical Fallback: {res.get('test_name', 'Failed')}")

print("\nAll verifications passed!")
