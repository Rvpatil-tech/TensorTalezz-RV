"""
master_test.py
TensorTalezz-RV Master Test Suite (v0.2.0)

Run this script to test all the major functionalities of the TensorTalezz-RV library.
Includes core modeling, explainability, responsible AI, deployment, and reinforcement learning.
"""

import pandas as pd
import numpy as np
import time

import tensor_talezz_rv
from tensor_talezz_rv import (
    fit, predict, explain, 
    detect_bias, fairness_metrics, audit_report,
    serve_model, export_model, deploy_edge,
    rl_demo, policy_visualizer, reward_curve
)

def print_header(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def main():
    print_header(f"Starting Master Test for TensorTalezz-RV v{tensor_talezz_rv.__version__}")
    
    # Create synthetic dataset with a 'sensitive_feature' for Responsible AI testing
    np.random.seed(42)
    n_samples = 200
    df = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(60000, 15000, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Non-Binary'], n_samples, p=[0.48, 0.48, 0.04]),
        'region': np.random.choice(['North', 'South'], n_samples),
    })
    
    # Target variable 'approved' depends slightly on income and intentionally biased on gender
    noise = np.random.normal(0, 0.5, n_samples)
    bias = np.where(df['gender'] == 'Male', 0.8, -0.2)
    logits = (df['income'] - 60000) / 10000 + bias + noise
    df['approved'] = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    
    print("Synthetic dataset created. shape:", df.shape)
    
    print_header("1. Core API: Simplified Training & Prediction")
    # Train the pipeline
    model, report = fit(df, target='approved', task='classification', model='random_forest', explain=False, verbose=True)
    
    # Predict on unseen raw data
    test_rows = df.head(5)
    preds = predict(model, test_rows)
    print(f"\nSample Predictions: {preds}")

    print_header("2. Education & Explainability")
    explain('random_forest')
    
    print_header("3. Responsible AI: Auditing the Model")
    audit_report(model, df, target='approved', sensitive_feature='gender')
    
    print_header("4. Deployment & Edge ML")
    exported_path = export_model(model, filepath="test_model.joblib", format="joblib")
    
    # Try deploying edge simulation
    edge_model = deploy_edge(model)
    
    # Generate API script
    script_path = serve_model(model_path=exported_path, port=8080)
    
    print_header("5. Reinforcement Learning Basics")
    agent = rl_demo(env_name="GridWorld Simulator", episodes=50, display=True)
    
    policy_visualizer(agent)
    
    print("\nReward Curve plotted (Close the window to complete test)")
    reward_curve(agent)

    print_header("MASTER TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
