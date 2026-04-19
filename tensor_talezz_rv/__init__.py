"""
TensorTalezz-RV: Making AI Approachable, Interpretable, and Resource-Friendly.

A Python library that combines practical machine learning tools with educational clarity.

Modules:
    - explainability: Visualize and interpret model predictions.
    - lightweight: One-line model training, prediction, and evaluation.
    - preprocessing: Automated data cleaning and preparation.
    - education: Interactive AI/ML concept teaching.
    - lowresource: ML tools for limited data and compute environments.
    - responsible: Ethics, fairness metrics, and bias detection.
    - deployment: Edge deployment and API generation.
    - rl_basics: Simple reinforcement learning demonstrations.
"""

__version__ = "0.2.0"
__author__ = "RV Patil"

from tensor_talezz_rv.core import (
    fit,
    predict,
    TensorTalezzPipeline
)

from tensor_talezz_rv.explainability import (
    plot_feature_influence,
    plot_decision_boundary,
    explain_prediction,
    compare_models,
    plot_training_curve,
)

from tensor_talezz_rv.lightweight import (
    train,
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

from tensor_talezz_rv.responsible import (
    detect_bias,
    fairness_metrics,
    audit_report,
)

from tensor_talezz_rv.deployment import (
    serve_model,
    export_model,
    deploy_edge,
)

from tensor_talezz_rv.rl_basics import (
    rl_demo,
    policy_visualizer,
    reward_curve,
)
