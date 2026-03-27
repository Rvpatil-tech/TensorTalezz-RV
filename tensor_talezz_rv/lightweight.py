"""
Lightweight ML Module
=====================

Simplify model training, prediction, evaluation, and persistence.

Functions:
    - train: One-line model training.
    - predict: Generate predictions with minimal effort.
    - evaluate: Print performance metrics in a friendly format.
    - save_model: Persist trained models to disk.
    - load_model: Reload a saved model for reuse.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
)


# ── Model registry ────────────────────────────────────────────────────────────

_CLASSIFIERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
}

_REGRESSORS = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "svr": SVR,
    "knn": KNeighborsRegressor,
    "decision_tree": DecisionTreeRegressor,
}


def _resolve_model(algorithm, task, **kwargs):
    """Return an instantiated sklearn estimator."""
    registry = _CLASSIFIERS if task == "classification" else _REGRESSORS
    algo = algorithm.lower().replace(" ", "_")
    if algo not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(
            f"Unknown algorithm '{algorithm}' for {task}. Available: {available}"
        )
    return registry[algo](**kwargs)


# ── Public API ─────────────────────────────────────────────────────────────────


def train(X, y, algorithm="random_forest", task="classification", random_state=42, **kwargs):
    """
    Train a model in one line.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target values.
    algorithm : str, optional
        Algorithm name (e.g. ``"random_forest"``, ``"logistic_regression"``).
        Default is ``"random_forest"``.
    task : str, optional
        ``"classification"`` or ``"regression"``. Default is ``"classification"``.
    random_state : int or None, optional
        Seed for reproducibility. Default is ``42``.
    **kwargs
        Extra keyword arguments forwarded to the sklearn estimator.

    Returns
    -------
    model : fitted sklearn estimator

    Example
    -------
    >>> import numpy as np
    >>> X = np.random.rand(100, 4)
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> model = train(X, y, algorithm="random_forest")
    """
    # Inject random_state where supported
    model = _resolve_model(algorithm, task, **kwargs)
    if random_state is not None and hasattr(model, "random_state"):
        model.set_params(random_state=random_state)

    model.fit(X, y)
    algo_display = algorithm.replace("_", " ").title()
    print(f"✅ Trained {algo_display} ({task}) on {np.array(X).shape[0]} samples.")
    return model


def predict(model, X):
    """
    Generate predictions with minimal effort.

    Parameters
    ----------
    model : fitted sklearn estimator
        A previously trained model.
    X : array-like of shape (n_samples, n_features)
        Samples to predict.

    Returns
    -------
    predictions : numpy.ndarray

    Example
    -------
    >>> preds = predict(model, X_test)
    """
    X = np.array(X)
    preds = model.predict(X)
    print(f"🔮 Generated {len(preds)} predictions.")
    return preds


def evaluate(model, X, y, task="classification"):
    """
    Print performance metrics in a friendly, readable format.

    Parameters
    ----------
    model : fitted sklearn estimator
        A trained model.
    X : array-like of shape (n_samples, n_features)
        Test features.
    y : array-like of shape (n_samples,)
        True labels / values.
    task : str, optional
        ``"classification"`` or ``"regression"``. Default is ``"classification"``.

    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.

    Example
    -------
    >>> metrics = evaluate(model, X_test, y_test)
    """
    X, y = np.array(X), np.array(y)
    preds = model.predict(X)

    print("\n📊 ── Evaluation Results ──────────────────")

    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average="weighted", zero_division=0),
            "recall": recall_score(y, preds, average="weighted", zero_division=0),
            "f1_score": f1_score(y, preds, average="weighted", zero_division=0),
        }
        for k, v in metrics.items():
            print(f"   {k:<12}: {v:.4f}")
        print("\n" + classification_report(y, preds, zero_division=0))
    else:
        metrics = {
            "mse": mean_squared_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "mae": mean_absolute_error(y, preds),
            "r2": r2_score(y, preds),
        }
        for k, v in metrics.items():
            print(f"   {k:<12}: {v:.4f}")

    print("──────────────────────────────────────────\n")
    return metrics


def save_model(model, filepath="model.joblib"):
    """
    Persist a trained model to disk.

    Parameters
    ----------
    model : fitted sklearn estimator
        The model to save.
    filepath : str, optional
        Destination path. Default is ``"model.joblib"``.

    Returns
    -------
    filepath : str
        The path where the model was saved.

    Example
    -------
    >>> save_model(model, "my_model.joblib")
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, filepath)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"💾 Model saved to '{filepath}' ({size_kb:.1f} KB)")
    return filepath


def load_model(filepath="model.joblib"):
    """
    Reload a previously saved model.

    Parameters
    ----------
    filepath : str, optional
        Path to the saved model file. Default is ``"model.joblib"``.

    Returns
    -------
    model : fitted sklearn estimator
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Example
    -------
    >>> model = load_model("my_model.joblib")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at '{filepath}'.")
    model = joblib.load(filepath)
    print(f"📦 Model loaded from '{filepath}'")
    return model
