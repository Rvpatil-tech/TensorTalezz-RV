import pytest
import numpy as np
import pandas as pd

from tensor_talezz_rv.preprocessing import (
    auto_clean,
    detect_outliers,
    normalize_text,
    split_dataset,
    feature_summary
)

def test_auto_clean():
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": ["A", "B", "A", None],
        "drop": [np.nan, np.nan, np.nan, "keep"]
    })
    
    cleaned, report = auto_clean(df, verbose=False)
    
    assert "drop" not in cleaned.columns
    assert not cleaned["num"].isnull().any()
    assert not cleaned["cat"].isnull().any()
    
    # Check scaling and encoding
    assert np.issubdtype(cleaned["num"].dtype, np.floating)
    assert np.issubdtype(cleaned["cat"].dtype, np.integer)

def test_normalize_text():
    text = "  <p>Hello,   World! 123</p>  "
    clean = normalize_text(text)
    assert clean == "hello world 123"
    
def test_split_dataset():
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)
    
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, verbose=False)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
