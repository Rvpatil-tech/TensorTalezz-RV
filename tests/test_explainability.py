import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from tensor_talezz_rv.explainability import (
    plot_feature_influence,
    explain_prediction,
    compare_models,
)

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 4)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y

def test_explain_prediction_linear(sample_data):
    X, y = sample_data
    model = LogisticRegression().fit(X, y)
    result = explain_prediction(model, X[0], ["a", "b", "c", "d"], model_type="linear")
    
    assert "prediction" in result
    assert "contributions" in result
    assert "summary" in result
    assert len(result["contributions"]) == 4

def test_explain_prediction_tree(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(random_state=42).fit(X, y)
    result = explain_prediction(model, X[0], ["a", "b", "c", "d"], model_type="tree")
    
    assert "prediction" in result
    assert len(result["contributions"]) == 4

def test_compare_models(sample_data):
    X, y = sample_data
    model1 = LogisticRegression().fit(X[:80], y[:80])
    model2 = RandomForestClassifier(random_state=42).fit(X[:80], y[:80])
    
    # We will mock plt.show or something, but actually matplotlib often just works in headless pytest without blocking if non-interactive
    import matplotlib
    matplotlib.use("Agg")
    
    results = compare_models([model1, model2], ["LR", "RF"], X[80:], y[80:])
    
    assert len(results) == 2
    assert "Accuracy" in results[0]
