import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from typing import List, Dict, Any, Optional

def train_and_predict(df: pd.DataFrame, target_col: str, feature_cols: List[str], id_col: Optional[str] = None) -> pd.DataFrame:
    """
    Trains a Random Forest Classifier on the fly to predict the target_col using feature_cols.
    Handles missing values and categorical encoding automatically.
    Returns a scored dataframe with probability and top driving reasons.
    """
    if df.empty:
        raise ValueError("Cannot train on an empty DataFrame.")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    for f in feature_cols:
        if f not in df.columns:
            raise ValueError(f"Feature column '{f}' not found in DataFrame.")
            
    # Clean data: drop rows where target is NaN
    ml_df = df.dropna(subset=[target_col]).copy()
    
    if ml_df.empty:
        raise ValueError("No valid rows remaining after dropping missing target values.")
        
    X = ml_df[feature_cols].copy()
    y = ml_df[target_col].copy()
    
    # 1. Preprocessing - Imputation
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        
    if len(cat_cols) > 0:
        # Simple categorical handling for POC: Impute with mode, then label encode
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        
        for col in cat_cols:
            le = LabelEncoder()
            # Convert to string to prevent mixed type errors
            X[col] = le.fit_transform(X[col].astype(str))
            
    # Encode target if categorical
    is_target_cat = False
    if y.dtype == 'object' or str(y.dtype) == 'category':
        is_target_cat = True
        target_le = LabelEncoder()
        y = target_le.fit_transform(y.astype(str))
        
    # 2. Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X, y)
    
    # 3. Predict & Score
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    
    # Extract predicted classes and confidence
    if is_target_cat:
        pred_labels = target_le.inverse_transform(preds)
    else:
        pred_labels = preds
        
    confidence = np.max(probs, axis=1)
    
    # 4. Feature Importance & Reasons
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    # Generate generic "top reasons" based on global feature importance
    top_features = [feature_cols[i] for i in sorted_idx[:3]]
    reason_str = "Driven by: " + ", ".join(top_features)
    
    # 5. Build Result DataFrame
    result = pd.DataFrame()
    if id_col and id_col in ml_df.columns:
        result[id_col] = ml_df[id_col]
        
    result["predicted_" + target_col] = pred_labels
    result["confidence_score"] = np.round(confidence, 3)
    result["key_factors"] = reason_str
    
    # Attach features so user can see context
    for f in feature_cols:
        result[f] = ml_df[f]
        
    return result
