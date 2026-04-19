"""
Low-Resource ML Module
======================

Tools designed for environments with limited data or computational power.

Functions:
    - tiny_classifier: Train lightweight models for small datasets.
    - quantize_model: Reduce model size via precision reduction.
    - evaluate_small_data: Robust metrics tailored for tiny datasets.
    - stream_train: Incremental training on streaming data.
    - compress_dataset: Reduce dataset size via downsampling/PCA.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneOut, cross_val_score


# ── Modest Classifiers ─────────────────────────────────────────────────────────

def tiny_classifier(X, y, algorithm="fast_tree", max_depth=3, random_state=42):
    """
    Train a highly constrained lightweight model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target values.
    algorithm : str, optional
        ``"fast_tree"`` or ``"simple_linear"``. Default is ``"fast_tree"``.
    max_depth : int, optional
        Constraint for tree depth (used if algorithm is ``"fast_tree"``). Default is 3.
    random_state : int, optional
        Seed for reproducibility. Default is 42.

    Returns
    -------
    model : fitted sklearn estimator
        The trained lightweight model.

    Example
    -------
    >>> import numpy as np
    >>> X = np.random.rand(50, 4)
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> model = tiny_classifier(X, y)
    """
    if algorithm == "fast_tree":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    elif algorithm == "simple_linear":
        model = LogisticRegression(max_iter=100, solver="liblinear", random_state=random_state)
    else:
        raise ValueError(f"Unknown tiny algorithm: {algorithm}. Try 'fast_tree' or 'simple_linear'.")

    model.fit(X, y)
    print(f"🚀 Trained '{algorithm}' tiny classifier on {len(X)} samples.")
    return model


def quantize_model(model):
    """
    Reduce model size for deployment (Simulated Quantization).

    In traditional scikit-learn models, true INT8 quantization isn't natively
    supported for execution. This function *simulates* quantization by downcasting
    the model's internal float64 arrays to float32 or float16, significantly
    reducing the serialized memory footprint.

    Parameters
    ----------
    model : fitted sklearn estimator
        The model to compress.

    Returns
    -------
    quantized_model : fitted sklearn estimator
        The memory-reduced model in-place.

    Example
    -------
    >>> model = quantize_model(model)
    """
    print("🗜️ Shrinking model precision...")

    original_size = 0
    new_size = 0
    cast_count = 0

    import copy
    q_model = copy.deepcopy(model)

    for attr in dir(q_model):
        # Only touch public attributes ending with '_' (fitted params)
        if attr.endswith("_") and not attr.startswith("__"):
            val = getattr(q_model, attr)
            if isinstance(val, np.ndarray) and val.dtype in (np.float64, np.float32):
                # Downcast to float16 (or float32 if needed)
                new_val = val.astype(np.float16)
                try:
                    setattr(q_model, attr, new_val)
                    original_size += val.nbytes
                    new_size += new_val.nbytes
                    cast_count += 1
                except AttributeError:
                    # Ignore read-only properties (like feature_importances_)
                    pass

    if cast_count > 0:
        saved = (original_size - new_size) / 1024
        print(f"📉 Downcasted {cast_count} arrays. Saved approx {saved:.2f} KB in memory.")
    else:
        print("ℹ️ No eligible arrays found for downcasting in this model type.")

    import pickle
    sz1 = len(pickle.dumps(model)) / 1024
    sz2 = len(pickle.dumps(q_model)) / 1024
    print(f"📦 Serialized size changed from {sz1:.1f} KB to {sz2:.1f} KB.")

    return q_model


def evaluate_small_data(model, X, y):
    """
    Provide evaluation metrics tailored for tiny datasets.

    On very small datasets, a standard train/test split might leave too few
    samples for an accurate test score. This function uses Leave-One-Out (LOO)
    cross-validation and robust metrics like Balanced Accuracy and MCC.

    Parameters
    ----------
    model : sklearn estimator
        An *unfitted* model instance to validate.
    X : array-like of shape (n_samples, n_features)
        Features.
    y : array-like of shape (n_samples,)
        Targets.

    Returns
    -------
    metrics : dict
        LOO accuracy, Balanced Accuracy, and MCC.

    Example
    -------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> evaluate_small_data(DecisionTreeClassifier(max_depth=3), X_small, y_small)
    """
    print(f"🔬 Evaluating on small dataset ({len(X)} samples) using Leave-One-Out CV...")

    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    X = np.array(X)
    y = np.array(y)

    import copy
    import warnings
    warnings.filterwarnings("ignore") # Ignore warnings for single-sample splits

    for train_idx, test_idx in loo.split(X):
        # We need a fresh model clone for each split
        clone = copy.deepcopy(model)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clone.fit(X_train, y_train)
        pred = clone.predict(X_test)[0]

        y_true.append(y_test[0])
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n📊 ── Small Data Evaluation ──────")
    print(f"   LOO Accuracy      : {acc:.4f}")
    print(f"   Balanced Accuracy : {b_acc:.4f} (Handles class imbalance)")
    print(f"   MCC               : {mcc:.4f} (Robust score -1 to +1)")
    print("──────────────────────────────────\n")

    return {"loo_accuracy": acc, "balanced_accuracy": b_acc, "mcc": mcc}


def stream_train(stream_iterator, classes, batch_size=50, max_batches=None):
    """
    Perform incremental (out-of-core) training on streaming data.

    This uses algorithms that support `partial_fit` (like SGDClassifier),
    allowing models to learn without loading all data into memory at once.

    Parameters
    ----------
    stream_iterator : iterator
        An iterator/generator yielding (X_batch, y_batch) tuples.
    classes : array-like
        List of all possible unique class labels in the entire dataset.
    batch_size : int, optional
        Used mainly for logging purposes here.
    max_batches : int or None, optional
        Maximum number of batches to consume. If None, consumes until exhausted.

    Returns
    -------
    model : fitted SGDClassifier
        The incrementally trained model.

    Example
    -------
    >>> def mock_stream():
    ...     for _ in range(5):
    ...         yield (np.random.rand(10, 3), np.random.randint(0, 2, 10))
    >>> model = stream_train(mock_stream(), classes=[0, 1])
    """
    model = SGDClassifier(loss="log_loss", random_state=42)
    print("🌊 Starting stream training (out-of-core learning)...")

    batch_count = 0
    total_samples = 0

    for X_batch, y_batch in stream_iterator:
        X_batch, y_batch = np.array(X_batch), np.array(y_batch)

        model.partial_fit(X_batch, y_batch, classes=classes)

        total_samples += len(X_batch)
        batch_count += 1
        print(f"   🔄 Processed batch {batch_count}: +{len(X_batch)} samples (Total: {total_samples})")

        if max_batches and batch_count >= max_batches:
            break

    print(f"✅ Stream training complete. Model trained on {total_samples} samples across {batch_count} batches.")
    return model


def compress_dataset(X, y=None, max_samples=None, n_components=None, random_state=42):
    """
    Reduce dataset size through downsampling and/or dimensionality reduction.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Features.
    y : array-like or pandas.Series, optional
        Targets. Required if stratifying the downsample.
    max_samples : int, optional
        Maximum number of rows to keep.
    n_components : int or float, optional
        Target dimensions via PCA. If float (0, 1), it represents variance retention.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    compressed_data : tuple or array
        (X_compressed, y_compressed) if y was provided, else X_compressed.

    Example
    -------
    >>> import numpy as np
    >>> X = np.random.rand(1000, 50); y = np.random.randint(0, 2, 1000)
    >>> X_tiny, y_tiny = compress_dataset(X, y, max_samples=100, n_components=10)
    """
    X_comp = np.array(X)
    y_comp = np.array(y) if y is not None else None

    print(f"🗜️ Original shape: {X_comp.shape}")

    # 1. Downsampling (Rows)
    if max_samples is not None and len(X_comp) > max_samples:
        from sklearn.model_selection import train_test_split
        # Using train_test_split to handle stratified shrinking elegantly
        stratify = y_comp if y_comp is not None else None
        try:
            # train size is the fraction we want to keep
            train_size = max_samples / len(X_comp)
            if y_comp is not None:
                X_comp, _, y_comp, _ = train_test_split(
                    X_comp, y_comp, train_size=train_size, random_state=random_state, stratify=stratify
                )
            else:
                X_comp, _ = train_test_split(X_comp, train_size=train_size, random_state=random_state)
            print(f"   📉 Downsampled rows to {len(X_comp)}.")
        except ValueError:
            # Fallback if stratify fails (e.g., classes too small)
            idx = np.random.RandomState(random_state).choice(len(X_comp), max_samples, replace=False)
            X_comp = X_comp[idx]
            if y_comp is not None:
                y_comp = y_comp[idx]
            print(f"   📉 Randomly downsampled rows to {len(X_comp)} (stratification failed).")

    # 2. Dimensionality Reduction (Columns)
    if n_components is not None and n_components < X_comp.shape[1]:
        pca = PCA(n_components=n_components, random_state=random_state)
        X_comp = pca.fit_transform(X_comp)
        if isinstance(n_components, float):
            print(f"   📉 PCA reduced features to {X_comp.shape[1]} (Retained {sum(pca.explained_variance_ratio_)*100:.1f}% variance).")
        else:
            print(f"   📉 PCA reduced features to {X_comp.shape[1]} components.")

    print(f"✅ Compressed shape: {X_comp.shape}")

    if y is not None:
        return X_comp, y_comp
    return X_comp
