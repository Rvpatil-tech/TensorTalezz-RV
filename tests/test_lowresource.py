import pytest
import numpy as np

from tensor_talezz_rv.lowresource import (
    tiny_classifier,
    quantize_model,
    evaluate_small_data,
    compress_dataset
)

from sklearn.tree import DecisionTreeClassifier

@pytest.fixture
def small_data():
    np.random.seed(42)
    X = np.random.rand(20, 4)
    y = (X[:, 0] > 0.5).astype(int)
    return X, y

def test_tiny_classifier(small_data):
    X, y = small_data
    model = tiny_classifier(X, y, algorithm="fast_tree")
    assert isinstance(model, DecisionTreeClassifier)
    preds = model.predict(X[:5])
    assert len(preds) == 5

def test_quantize_model(small_data):
    X, y = small_data
    model = tiny_classifier(X, y, algorithm="simple_linear")
    
    # Store original dtype
    orig_dtype = model.coef_.dtype
    
    q_model = quantize_model(model)
    
    assert q_model.coef_.dtype == np.float16

def test_evaluate_small_data(small_data):
    X, y = small_data
    model = tiny_classifier(X, y, algorithm="fast_tree")
    metrics = evaluate_small_data(model, X, y)
    
    assert "loo_accuracy" in metrics
    assert "balanced_accuracy" in metrics

def test_compress_dataset(small_data):
    X, y = small_data
    
    X_comp, y_comp = compress_dataset(X, y, max_samples=10)
    assert len(X_comp) == 10
    
    X_pca = compress_dataset(X, n_components=2)
    assert X_pca.shape[1] == 2
