"""
TensorTalezz-RV Complete Demo (all_test.py)

Run this script to see exactly what an end user experiences when using the library!
It walks through preprocessing, training, explainability, education, and low-resource tools.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from tensor_talezz_rv.preprocessing import auto_clean, feature_summary, detect_outliers, split_dataset
from tensor_talezz_rv.lightweight import train, predict, evaluate, save_model
from tensor_talezz_rv.explainability import explain_prediction, plot_feature_influence, compare_models
from tensor_talezz_rv.lowresource import tiny_classifier, evaluate_small_data, compress_dataset
from tensor_talezz_rv.education import explain, visualize_algorithm, compare_algorithms

def main():
    print("\n" + "="*70)
    print(" 🚀 WELCOME TO TENSORTALEZZ-RV LIVE DEMO")
    print("="*70 + "\n")
    
    # ---------------------------------------------------------
    # 1. Preprocessing Module
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print(" 🧹 MODULE 1: PREPROCESSING")
    print("-"*40)
    
    # Generate messy toy data
    X, y = make_classification(n_samples=250, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=['Age', 'Income', 'Score', 'Engagement'])
    
    # Inject missing values and an outlier to make it messy
    df.loc[0, 'Income'] = np.nan
    df.loc[10, 'Age'] = np.nan
    df.loc[5, 'Score'] = 9999.0  # Outlier
    
    df['Category'] = np.random.choice(['A', 'B', None], size=250)
    df['Target'] = y

    print("\n1a. Before Cleaning (Feature Summary):")
    feature_summary(df, top_n_categories=2)
    
    # Detect outliers
    print("\n1b. Outlier Detection:")
    _, outlier_summary = detect_outliers(df, threshold=2.0)
    
    # Auto Clean
    print("\n1c. Auto Clean Action:")
    cleaned_df, report = auto_clean(df, verbose=True)
    
    X_clean = cleaned_df.drop('Target', axis=1).values
    y_clean = cleaned_df['Target'].values

    # Split
    X_train, X_test, y_train, y_test = split_dataset(X_clean, y_clean, test_size=0.2)
    feature_names = cleaned_df.drop('Target', axis=1).columns.tolist()

    # ---------------------------------------------------------
    # 2. Lightweight ML Module
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print(" ⚡ MODULE 2: LIGHTWEIGHT ML")
    print("-"*40)
    
    print("\n2a. One-Line Training:")
    model_rf = train(X_train, y_train, algorithm="random_forest", random_state=42)
    model_lr = train(X_train, y_train, algorithm="logistic_regression", random_state=42)
    
    print("\n2b. Easy Evaluation:")
    test_metrics = evaluate(model_rf, X_test, y_test)
    
    print("\n2c. Saving Model:")
    save_model(model_rf, "demo_random_forest.joblib")


    # ---------------------------------------------------------
    # 3. Explainability Module
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print(" 🔍 MODULE 3: EXPLAINABILITY")
    print("-"*40)
    
    print("\n3a. Explaining a Single Prediction:")
    explanation = explain_prediction(model_rf, X_test[0], feature_names)
    print(explanation['summary'])
    
    print("\n3b. Comparing Models On Test Set:")
    compare_models([model_rf, model_lr], ["Random Forest", "Logistic Regression"], X_test, y_test)
    
    print("\n3c. Plotting Feature Influence (Close window to proceed):")
    plot_feature_influence(model_rf, feature_names, X_test)


    # ---------------------------------------------------------
    # 4. Low-Resource Module
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print(" 📱 MODULE 4: LOW-RESOURCE ML")
    print("-"*40)
    
    # Tiny dataset
    X_tiny = X_train[:15]
    y_tiny = y_train[:15]
    
    print("\n4a. Compressing Dataset (e.g. for Edge Devices):")
    X_comp, y_comp = compress_dataset(X_train, y_train, max_samples=50, n_components=2)
    
    print("\n4b. Small Data Evaluation (Leave-One-Out CV):")
    tiny_mod = tiny_classifier(X_tiny, y_tiny, algorithm="fast_tree", max_depth=2)
    evaluate_small_data(tiny_mod, X_tiny, y_tiny)


    # ---------------------------------------------------------
    # 5. Education Module
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print(" 🎓 MODULE 5: EDUCATION")
    print("-"*40)
    
    print("\n5a. Explaining Concepts Analogy-Style:")
    explain("linear_regression")
    explain("knn")
    explain("naive_bayes")
    explain("confusion_matrix")
    explain("cnns")
    
    print("\n5b. Visualizing Algorithms Step-by-Step (Close window to exit):")
    visualize_algorithm("knn")

    print("\n" + "="*70)
    print(" 🎉 DEMO COMPLETE! ")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
