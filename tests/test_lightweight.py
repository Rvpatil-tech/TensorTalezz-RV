import os
import pytest
import numpy as np

from tensor_talezz_rv.lightweight import train, predict, evaluate, save_model, load_model

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 4)
    y = (X[:, 0] > 0.5).astype(int)
    return X, y

def test_train_predict(sample_data):
    X, y = sample_data
    model = train(X, y, algorithm="logistic_regression")
    preds = predict(model, X[:10])
    assert len(preds) == 10

def test_evaluate(sample_data):
    X, y = sample_data
    model = train(X, y, algorithm="random_forest")
    metrics = evaluate(model, X, y)
    assert "accuracy" in metrics
    assert "f1_score" in metrics

def test_save_load_model(sample_data, tmp_path):
    X, y = sample_data
    model = train(X, y, algorithm="logistic_regression")
    
    model_path = tmp_path / "test_model.joblib"
    save_model(model, filepath=str(model_path))
    
    assert os.path.exists(model_path)
    
    loaded_model = load_model(filepath=str(model_path))
    preds = predict(loaded_model, X[:5])
    assert len(preds) == 5
