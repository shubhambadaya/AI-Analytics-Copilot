import pandas as pd
from src.analysis.ml_engine import train_and_predict

df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5, 6],
    "plan_fee": [300, 300, 600, 600, 300, 600],
    "data_usage_gb": [1.2, 5.5, 12.0, 15.0, 2.0, 10.0],
    "vowifi_mous": [0, 50, 150, 200, 10, 100],
    "gender": ["M", "F", "M", "F", "M", "F"]
})

df["is_high_plan"] = df["plan_fee"] > 300

scored_df = train_and_predict(
    df,
    target_col="is_high_plan",
    feature_cols=["data_usage_gb", "vowifi_mous", "gender"],
    id_col="user_id"
)

print(scored_df.head(10))
