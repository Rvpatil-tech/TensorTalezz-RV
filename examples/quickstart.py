"""
TensorTalezz-RV: Quickstart Example
This script demonstrates the core functionality of the TensorTalezz-RV library.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from tensor_talezz_rv.preprocessing import auto_clean, split_dataset, feature_summary
from tensor_talezz_rv.lightweight import train, evaluate, save_model
from tensor_talezz_rv.explainability import plot_feature_influence, explain_prediction
from tensor_talezz_rv.education import explain
from tensor_talezz_rv.lowresource import compress_dataset

def main():
    print("Welcome to TensorTalezz-RV Quickstart!")
    
    # 1. Preprocessing
    print("\n--- 1. Preprocessing ---")
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(10)])
    df["Target"] = y
    
    # Inject some missing values
    df.loc[0, "Feature_0"] = np.nan
    
    print("Initial Data Summary:")
    feature_summary(df, top_n_categories=3)
    
    print("\nCleaning Data:")
    df_clean, report = auto_clean(df)
    
    X_clean = df_clean.drop("Target", axis=1).values
    y_clean = df_clean["Target"].values
    
    X_train, X_test, y_train, y_test = split_dataset(X_clean, y_clean)
    
    # 2. Lightweight ML
    print("\n--- 2. Lightweight ML ---")
    model = train(X_train, y_train, algorithm="random_forest")
    evaluate(model, X_test, y_test)
    save_model(model, "demo_model.joblib")
    
    # 3. Explainability
    print("\n--- 3. Explainability ---")
    plot_feature_influence(model, [f"Feature_{i}" for i in range(10)], X_test, top_n=5)
    result = explain_prediction(model, X_test[0], [f"Feature_{i}" for i in range(10)])
    print(result["summary"])
    
    # 4. Low Resource
    print("\n--- 4. Low Resource ---")
    X_tiny, y_tiny = compress_dataset(X_clean, y_clean, max_samples=50, n_components=3)
    print(f"Compressed shape: {X_tiny.shape}")
    
    # 5. Education
    print("\n--- 5. Education ---")
    explain("random_forest")

if __name__ == "__main__":
    main()
