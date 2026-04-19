import pytest
import pandas as pd
import numpy as np
import os
from tensor_talezz_rv.core import fit, predict
from tensor_talezz_rv.responsible import detect_bias, fairness_metrics
from tensor_talezz_rv.deployment import export_model, serve_model
from tensor_talezz_rv.lowresource import quantize_model

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 65, 100),
        'income': np.random.randint(30000, 100000, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    return df

def test_fit_predict(dummy_data):
    model, report = fit(dummy_data, target='target', model='random_forest', verbose=False)
    assert 'task' in report
    assert report['task'] == 'classification'
    
    preds = predict(model, dummy_data.head())
    assert len(preds) == 5

def test_detect_bias(dummy_data):
    report = detect_bias(dummy_data, target='target', sensitive_feature='gender')
    assert 'group_stats' in report
    assert 'max_difference' in report

def test_fairness_metrics(dummy_data):
    model, _ = fit(dummy_data, target='target', verbose=False)
    X = dummy_data.drop(columns=['target'])
    y = dummy_data['target']
    metrics = fairness_metrics(model, X, y, sensitive_feature='gender')
    assert 'M' in metrics
    assert 'F' in metrics

def test_export_and_quantize(dummy_data):
    model, _ = fit(dummy_data, target='target', verbose=False)
    filepath = export_model(model, filepath="test_export.joblib", format="joblib")
    assert os.path.exists(filepath)
    os.remove(filepath)
    
    q_model = quantize_model(model.pipeline.named_steps["model"])
    assert q_model is not None
