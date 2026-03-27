# TensorTalezz-RV  
*Accessible, educational, and practical AI/ML tools for everyone.*

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-Proprietary-red)

---

## 🌟 Overview
TensorTalezz-RV is a proprietary AI/ML library designed to make machine learning approachable, educational, and practical. Structured into five core modules, it combines powerful, easy-to-use ML tools with interactive educational features to bridge the gap between abstract concepts and real-world implementation.

---

## 📦 Modules & Features

### 1. 🔍 Explainability
Interpret and trust your models with intuitive visualizations.
* `plot_feature_influence`: Visualize how each feature contributes to predictions.
* `plot_decision_boundary`: Show classifier boundaries in two dimensions.
* `explain_prediction`: Provide a human-readable breakdown of a single prediction.
* `compare_models`: Contrast multiple models side by side.
* `plot_training_curve`: Track loss and accuracy trends during training.

### 2. ⚡ Lightweight ML
Simplicity and speed for rapid prototyping.
* `train`: One-line model training with minimal boilerplate.
* `predict`: Generate predictions effortlessly.
* `evaluate`: Print clear, friendly, and comprehensive performance metrics.
* `save_model` & `load_model`: Persist and reuse trained models easily.

### 3. 🧹 Preprocessing
Automate the messy work of data preparation.
* `auto_clean`: Handle missing values, scaling, and encoding automatically.
* `detect_outliers`: Flag anomalies in datasets.
* `normalize_text`: Clean text data for NLP tasks.
* `split_dataset`: Perform train/test splits in one call.
* `feature_summary`: Generate quick statistics for each column.

### 4. 🎓 Education
Interactive tools to teach and learn AI concepts tangibly.
* `explain`: Analogy-driven explanations of complex AI/ML concepts.
* `visualize_algorithm`: Step-by-step algorithm demonstrations.
* `concept_demo`: Toy-dataset demonstrations of ideas like over/underfitting.
* `quiz`: Generate short, interactive quizzes for learners.
* `compare_algorithms`: Run multiple algorithms and explain their differences.

### 5. 📱 Low-Resource ML
Built for constrained environments with limited data or compute.
* `tiny_classifier`: Train highly constrained, lightweight models.
* `quantize_model`: Reduce model serialization size for deployment.
* `evaluate_small_data`: Use robust metrics tailored for tiny datasets.
* `stream_train`: Support incremental, out-of-core training for streaming data.
* `compress_dataset`: Reduce dataset size via intelligent downsampling or dimensionality reduction.

---

## 🛠 Requirements
- Python 3.8+  
- Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/example/TensorTalezz-RV.git
cd TensorTalezz-RV

# Install the package
pip install .
```

---

## 🎯 Quick Start
Check out the `examples/quickstart.py` file to see a complete workflow combining all modules.

```python
from tensor_talezz_rv.lightweight import train, evaluate
from tensor_talezz_rv.education import explain

# Learn about an algorithm
explain("random_forest")

# Train it in one line!
model = train(X_train, y_train, algorithm="random_forest")
metrics = evaluate(model, X_test, y_test)
```

---

## 📚 Usage Examples

### Preprocessing
```python
from tensor_talezz_rv.preprocessing import auto_clean

cleaned_df, report = auto_clean(raw_dataset)
```

### Explainability
```python
from tensor_talezz_rv.explainability import explain_prediction

feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]
explanation = explain_prediction(model, X_test[0], feature_names)
```

### Low-Resource ML
```python
from tensor_talezz_rv.lowresource import tiny_classifier

tiny_model = tiny_classifier(X_train, y_train)
```

---

## tensor_talezz_rv in Action


======================================================================
 🚀 WELCOME TO TENSORTALEZZ-RV LIVE DEMO
======================================================================


----------------------------------------
 🧹 MODULE 1: PREPROCESSING
----------------------------------------

1a. Before Cleaning (Feature Summary):

📋 Feature Summary (250 rows, 6 columns)

    column   dtype  missing  missing_pct  unique    mean      std       min         max    median 
        top_values
       Age float64        1          0.4     249 -0.0396   1.3134 -3.422492    3.370100 -0.229298 
               NaN
    Income float64        1          0.4     249  0.0700   1.0655 -2.364563    2.367717  0.057826 
               NaN
     Score float64        0          0.0     250 39.9405 632.3972 -3.959492 9999.000000 -0.217356 
               NaN
Engagement float64        0          0.0     250 -0.0696   1.3463 -2.920976    3.498072 -0.041329 
               NaN
  Category  object       86         34.4       2     NaN      NaN       NaN         NaN       NaN 
{'B': 83, 'A': 81}
    Target   int64        0          0.0       2  0.5000   0.5010  0.000000    1.000000  0.500000 
               NaN

1b. Outlier Detection:
🔍 Detected 1 outlier(s) across 5 column(s).
   • Score: 1 outliers

1c. Auto Clean Action:
🩹 Filled missing: numeric=['Age', 'Income'], categorical=['Category']
📏 Scaled numeric columns: ['Age', 'Income', 'Score', 'Engagement', 'Target']
🏷️  Encoded categorical columns: ['Category']
✅ Cleaning complete. Shape: (250, 6)
✂️  Split: 200 train / 50 test  (test_size=0.2)

----------------------------------------
 ⚡ MODULE 2: LIGHTWEIGHT ML
----------------------------------------

2a. One-Line Training:
✅ Trained Random Forest (classification) on 200 samples.
✅ Trained Logistic Regression (classification) on 200 samples.

2b. Easy Evaluation:

📊 ── Evaluation Results ──────────────────
   accuracy    : 0.9600
   precision   : 0.9628
   recall      : 0.9600
   f1_score    : 0.9598

              precision    recall  f1-score   support

        -1.0       1.00      0.91      0.95        23
         1.0       0.93      1.00      0.96        27

    accuracy                           0.96        50
   macro avg       0.97      0.96      0.96        50
weighted avg       0.96      0.96      0.96        50

──────────────────────────────────────────


2c. Saving Model:
💾 Model saved to 'demo_random_forest.joblib' (348.9 KB)

----------------------------------------
 🔍 MODULE 3: EXPLAINABILITY
----------------------------------------

3a. Explaining a Single Prediction:
Prediction: 1.0
--- Feature Contributions ---
  + Age: 0.1445
  - Income: -0.0931
  + Engagement: 0.0195
  + Category: 0.0087
  - Score: -0.0076

3b. Comparing Models On Test Set:

Model               Accuracy       Precision      Recall         F1 Score
--------------------------------------------------------------------------------
Random Forest       0.9600         0.9628         0.9600         0.9598
Logistic Regression 0.9200         0.9232         0.9200         0.9201

3c. Plotting Feature Influence (Close window to proceed):

----------------------------------------
 📱 MODULE 4: LOW-RESOURCE ML
----------------------------------------

4a. Compressing Dataset (e.g. for Edge Devices):
🗜️ Original shape: (200, 5)
   📉 Downsampled rows to 50.
   📉 PCA reduced features to 2 components.
✅ Compressed shape: (50, 2)

4b. Small Data Evaluation (Leave-One-Out CV):
🚀 Trained 'fast_tree' tiny classifier on 15 samples.
🔬 Evaluating on small dataset (15 samples) using Leave-One-Out CV...

📊 ── Small Data Evaluation ──────
   LOO Accuracy      : 0.8000
   Balanced Accuracy : 0.7946 (Handles class imbalance)
   MCC               : 0.6001 (Robust score -1 to +1)
──────────────────────────────────


----------------------------------------
 🎓 MODULE 5: EDUCATION
----------------------------------------

5a. Explaining Concepts Analogy-Style:

============================================================
📖  Bias–Variance Tradeoff
============================================================

🎯 Analogy:
   Think of throwing darts. High *bias* means you consistently hit the wrong spot (systematic error). High *variance* means your throws are scattered all over the board (inconsistent). The sweet spot is low bias AND low variance — consistently hitting the bullseye.

📐 Formal Definition:
   Bias is the error from wrong assumptions; variance is sensitivity to small fluctuations in the 
training set. Total error = Bias² + Variance + Irreducible error. Reducing one often increases the other.

💡 Tips:
   • Complex models → low bias, high variance.
   • Simple models → high bias, low variance.
   • Use ensemble methods to balance both.


5b. Visualizing Algorithms Step-by-Step (Close window to exit):
⭐ The gold star is the query point. Notice how k affects the prediction and boundary smoothness.

======================================================================
 🎉 DEMO COMPLETE!
======================================================================


## 📄 License
This project is licensed under the **TensorTalezz-RV Proprietary License**.  
See the `LICENSE` file for details.

---

## 💼 Commercial Use
TensorTalezz-RV is proprietary software.  
For **commercial licensing, enterprise support, integration into paid services, or collaborations**, please refer to the `COMMERCIAL.md` file for full details and constraints. 

To request a commercial license, please contact:  
📧 rahulvpatil098@gmail.com

---

## 🤝 Contributing
Bug reports, feature requests, and suggestions are welcome.  
Code modifications require explicit approval due to the proprietary license.
