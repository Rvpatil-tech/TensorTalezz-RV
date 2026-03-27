"""
TensorTalezz-RV: Making AI Approachable, Interpretable, and Resource-Friendly.

A Python library that combines practical machine learning tools with educational clarity.

Modules:
    - explainability: Visualize and interpret model predictions.
    - lightweight: One-line model training, prediction, and evaluation.
    - preprocessing: Automated data cleaning and preparation.
    - education: Interactive AI/ML concept teaching.
    - lowresource: ML tools for limited data and compute environments.
"""

__version__ = "0.1.0"
__author__ = "Rahul Patil"

from tensor_talezz_rv.explainability import (
    plot_feature_influence,
    plot_decision_boundary,
    explain_prediction,
    compare_models,
    plot_training_curve,
)

from tensor_talezz_rv.lightweight import (
    train,
    predict,
    evaluate,
    save_model,
    load_model,
)

from tensor_talezz_rv.preprocessing import (
    auto_clean,
    detect_outliers,
    normalize_text,
    split_dataset,
    feature_summary,
)

from tensor_talezz_rv.education import (
    explain,
    visualize_algorithm,
    concept_demo,
    quiz,
    compare_algorithms,
)

from tensor_talezz_rv.lowresource import (
    tiny_classifier,
    quantize_model,
    evaluate_small_data,
    stream_train,
    compress_dataset,
)
