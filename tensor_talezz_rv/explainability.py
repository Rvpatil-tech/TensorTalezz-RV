"""
Explainability Module
=====================

Tools to visualize, interpret, and trust machine learning models.

Functions:
    - plot_feature_influence: Visualize feature contributions to predictions.
    - plot_decision_boundary: Show classifier boundaries in 2D space.
    - explain_prediction: Human-readable breakdown of a single prediction.
    - compare_models: Side-by-side comparison of multiple models.
    - plot_training_curve: Track loss/accuracy over training epochs.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def plot_feature_influence(model, feature_names, X, top_n=None, figsize=(10, 6)):
    """
    Visualize how each feature contributes to model predictions.

    For tree-based models, uses built-in ``feature_importances_``.  For linear
    models, uses the absolute value of ``coef_``.  Falls back to a simple
    permutation-style importance estimate for other model types.

    Parameters
    ----------
    model : fitted sklearn estimator
        A trained model that exposes ``feature_importances_`` or ``coef_``.
    feature_names : list of str
        Names corresponding to each feature column.
    X : array-like of shape (n_samples, n_features)
        The dataset used to evaluate importance (used in fallback mode).
    top_n : int or None, optional
        If provided, only the top N most influential features are shown.
    figsize : tuple, optional
        Matplotlib figure size. Default is ``(10, 6)``.

    Returns
    -------
    importances : numpy.ndarray
        Array of importance values for each feature.

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>> X = np.random.rand(100, 4)
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(int)
    >>> model = RandomForestClassifier(random_state=42).fit(X, y)
    >>> importances = plot_feature_influence(model, ['a', 'b', 'c', 'd'], X)
    """
    X = np.array(X)

    # Determine importance source
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        # Permutation-style fallback
        baseline = model.score(X, np.zeros(X.shape[0]))  # crude baseline
        importances = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, i])
            importances[i] = abs(baseline - model.score(X_perm, np.zeros(X.shape[0])))

    # Sort by importance
    sorted_idx = np.argsort(importances)
    if top_n is not None:
        sorted_idx = sorted_idx[-top_n:]

    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
    ax.barh(sorted_names, sorted_importances, color=colors)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Influence on Predictions")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return importances


def plot_decision_boundary(model, X, y, resolution=200, figsize=(10, 8), feature_indices=(0, 1)):
    """
    Display the decision boundary of a classifier in two dimensions.

    Parameters
    ----------
    model : fitted sklearn classifier
        A trained classifier with a ``predict`` method.
    X : array-like of shape (n_samples, n_features)
        Training data (at least 2 features).
    y : array-like of shape (n_samples,)
        True labels.
    resolution : int, optional
        Number of grid points along each axis. Default is 200.
    figsize : tuple, optional
        Figure size. Default is ``(10, 8)``.
    feature_indices : tuple of int, optional
        Which two feature columns to use for the 2D plot. Default is ``(0, 1)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.

    Example
    -------
    >>> from sklearn.svm import SVC
    >>> import numpy as np
    >>> X = np.random.randn(200, 2)
    >>> y = (X[:, 0] * X[:, 1] > 0).astype(int)
    >>> model = SVC(kernel='rbf').fit(X, y)
    >>> fig = plot_decision_boundary(model, X, y)
    """
    X = np.array(X)
    y = np.array(y)
    fi, fj = feature_indices

    x_min, x_max = X[:, fi].min() - 1, X[:, fi].max() + 1
    y_min, y_max = X[:, fj].min() - 1, X[:, fj].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # Build grid input – fill non-plotted features with their mean
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    if X.shape[1] > 2:
        filler = np.tile(X.mean(axis=0), (grid.shape[0], 1))
        filler[:, fi] = grid[:, 0]
        filler[:, fj] = grid[:, 1]
        grid = filler

    Z = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
    scatter = ax.scatter(X[:, fi], X[:, fj], c=y, edgecolors="k", cmap="coolwarm", s=30)
    ax.set_xlabel(f"Feature {fi}")
    ax.set_ylabel(f"Feature {fj}")
    ax.set_title("Decision Boundary")
    plt.colorbar(scatter, ax=ax, label="Class")
    plt.tight_layout()
    plt.show()

    return fig


def explain_prediction(model, X_single, feature_names, model_type="auto"):
    """
    Provide a human-readable breakdown of a single prediction.

    For linear models the contribution of each feature is calculated as
    ``coef * feature_value``.  For tree-based models the feature importances
    are weighted by the feature values.

    Parameters
    ----------
    model : fitted sklearn estimator
        A trained model.
    X_single : array-like of shape (n_features,)
        A single sample to explain.
    feature_names : list of str
        Feature names corresponding to the input columns.
    model_type : str, optional
        ``"linear"``, ``"tree"``, or ``"auto"`` (default). In auto mode,
        the function tries to detect the model type automatically.

    Returns
    -------
    explanation : dict
        Dictionary containing ``prediction``, ``contributions`` (per feature),
        and a ``summary`` string.

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X = np.random.rand(100, 3)
    >>> y = (X.sum(axis=1) > 1.5).astype(int)
    >>> model = LogisticRegression().fit(X, y)
    >>> result = explain_prediction(model, X[0], ['a', 'b', 'c'])
    >>> print(result['summary'])
    """
    X_single = np.array(X_single).flatten()
    prediction = model.predict(X_single.reshape(1, -1))[0]

    # Detect model type
    if model_type == "auto":
        if hasattr(model, "coef_"):
            model_type = "linear"
        elif hasattr(model, "feature_importances_"):
            model_type = "tree"
        else:
            model_type = "generic"

    contributions = {}
    if model_type == "linear":
        coef = np.array(model.coef_).flatten()
        for name, c, val in zip(feature_names, coef, X_single):
            contributions[name] = round(float(c * val), 4)
    elif model_type == "tree":
        importances = np.array(model.feature_importances_)
        for name, imp, val in zip(feature_names, importances, X_single):
            contributions[name] = round(float(imp * val), 4)
    else:
        for name, val in zip(feature_names, X_single):
            contributions[name] = round(float(val), 4)

    # Build summary
    sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = [f"Prediction: {prediction}", "--- Feature Contributions ---"]
    for name, val in sorted_contrib:
        direction = "+" if val >= 0 else "-"
        lines.append(f"  {direction} {name}: {val}")
    summary = "\n".join(lines)

    return {"prediction": prediction, "contributions": contributions, "summary": summary}


def compare_models(models, model_names, X_test, y_test, figsize=(12, 6)):
    """
    Compare multiple models side by side on the same test set.

    Parameters
    ----------
    models : list of fitted sklearn estimators
        Trained models to compare.
    model_names : list of str
        Human-readable names for each model.
    X_test : array-like of shape (n_samples, n_features)
        Test features.
    y_test : array-like of shape (n_samples,)
        True labels.
    figsize : tuple, optional
        Figure size. Default is ``(12, 6)``.

    Returns
    -------
    results : list of dict
        Per-model metrics (accuracy, precision, recall, f1).

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X = np.random.rand(200, 4); y = (X[:, 0] > 0.5).astype(int)
    >>> rf = RandomForestClassifier().fit(X[:150], y[:150])
    >>> lr = LogisticRegression().fit(X[:150], y[:150])
    >>> results = compare_models([rf, lr], ['RF', 'LR'], X[150:], y[150:])
    """
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    results = []
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for model in models:
        preds = model.predict(X_test)
        results.append({
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, preds, average="weighted", zero_division=0),
        })

    # Grouped bar chart
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, res) in enumerate(zip(model_names, results)):
        values = [res[m] for m in metric_names]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print table
    header = f"{'Model':<20}" + "".join(f"{m:<15}" for m in metric_names)
    print("\n" + header)
    print("-" * len(header))
    for name, res in zip(model_names, results):
        row = f"{name:<20}" + "".join(f"{res[m]:<15.4f}" for m in metric_names)
        print(row)

    return results


def plot_training_curve(history, metrics=None, figsize=(12, 5)):
    """
    Track loss and/or accuracy trends during training.

    Parameters
    ----------
    history : dict
        Dictionary with metric names as keys and lists of per-epoch values.
        Common keys: ``"loss"``, ``"val_loss"``, ``"accuracy"``, ``"val_accuracy"``.
    metrics : list of str or None, optional
        Which metrics to plot. If ``None``, all keys in *history* are plotted.
    figsize : tuple, optional
        Figure size. Default is ``(12, 5)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    Example
    -------
    >>> history = {
    ...     "loss": [0.9, 0.7, 0.5, 0.3],
    ...     "val_loss": [0.95, 0.75, 0.6, 0.55],
    ...     "accuracy": [0.5, 0.65, 0.75, 0.85],
    ...     "val_accuracy": [0.48, 0.60, 0.70, 0.78],
    ... }
    >>> fig = plot_training_curve(history)
    """
    if metrics is None:
        metrics = list(history.keys())

    # Group loss and accuracy metrics
    loss_metrics = [m for m in metrics if "loss" in m.lower()]
    acc_metrics = [m for m in metrics if "loss" not in m.lower()]

    n_plots = (1 if loss_metrics else 0) + (1 if acc_metrics else 0)
    if n_plots == 0:
        print("No metrics to plot.")
        return None

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    if loss_metrics:
        ax = axes[plot_idx]
        for m in loss_metrics:
            epochs = range(1, len(history[m]) + 1)
            style = "--" if "val" in m else "-"
            ax.plot(epochs, history[m], style, label=m, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        plot_idx += 1

    if acc_metrics:
        ax = axes[plot_idx]
        for m in acc_metrics:
            epochs = range(1, len(history[m]) + 1)
            style = "--" if "val" in m else "-"
            ax.plot(epochs, history[m], style, label=m, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Training & Validation Metrics")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig
