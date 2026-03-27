"""
Preprocessing Module
====================

Automated data cleaning and preparation utilities.

Functions:
    - auto_clean: Handle missing values, scaling, and encoding.
    - detect_outliers: Flag anomalies in datasets.
    - normalize_text: Clean text data for NLP tasks.
    - split_dataset: Train/test split in one call.
    - feature_summary: Quick statistics for each column.
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def auto_clean(
    df,
    numeric_strategy="mean",
    categorical_strategy="mode",
    scale=True,
    encode=True,
    drop_threshold=0.5,
    verbose=True,
):
    """
    Automatically handle missing values, scale numerics, and encode categoricals.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    numeric_strategy : str, optional
        Strategy for filling numeric NaNs: ``"mean"``, ``"median"``, or ``"zero"``.
        Default is ``"mean"``.
    categorical_strategy : str, optional
        Strategy for filling categorical NaNs: ``"mode"`` or ``"unknown"``.
        Default is ``"mode"``.
    scale : bool, optional
        If ``True``, standard-scale numeric columns. Default is ``True``.
    encode : bool, optional
        If ``True``, label-encode categorical columns. Default is ``True``.
    drop_threshold : float, optional
        Drop columns with more than this fraction of missing values. Default ``0.5``.
    verbose : bool, optional
        Print progress information. Default is ``True``.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        The cleaned dataframe.
    report : dict
        Cleaning report with details of actions taken.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2, None], "b": ["x", None, "y"], "c": [4, 5, 6]})
    >>> cleaned, report = auto_clean(df)
    """
    df = df.copy()
    report = {"dropped_columns": [], "filled_numeric": [], "filled_categorical": [], "scaled": [], "encoded": []}

    # 1. Drop columns above missing threshold
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > drop_threshold].index.tolist()
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        report["dropped_columns"] = drop_cols
        if verbose:
            print(f"🗑️  Dropped columns (>{drop_threshold*100:.0f}% missing): {drop_cols}")

    # 2. Fill numeric missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            if numeric_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif numeric_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif numeric_strategy == "zero":
                df[col] = df[col].fillna(0)
            report["filled_numeric"].append(col)

    # 3. Fill categorical missing values
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            if categorical_strategy == "mode":
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "unknown")
            elif categorical_strategy == "unknown":
                df[col] = df[col].fillna("unknown")
            report["filled_categorical"].append(col)

    if verbose and (report["filled_numeric"] or report["filled_categorical"]):
        print(f"🩹 Filled missing: numeric={report['filled_numeric']}, categorical={report['filled_categorical']}")

    # 4. Scale numeric columns
    if scale and num_cols:
        scaler = StandardScaler()
        # Only scale columns that still exist
        existing_num = [c for c in num_cols if c in df.columns]
        df[existing_num] = scaler.fit_transform(df[existing_num])
        report["scaled"] = existing_num
        if verbose:
            print(f"📏 Scaled numeric columns: {existing_num}")

    # 5. Encode categorical columns
    if encode and cat_cols:
        existing_cat = [c for c in cat_cols if c in df.columns]
        for col in existing_cat:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            report["encoded"].append(col)
        if verbose:
            print(f"🏷️  Encoded categorical columns: {existing_cat}")

    if verbose:
        print(f"✅ Cleaning complete. Shape: {df.shape}")

    return df, report


def detect_outliers(df, method="iqr", threshold=1.5, columns=None):
    """
    Flag anomalies in a dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    method : str, optional
        Detection method: ``"iqr"`` (interquartile range) or ``"zscore"``.
        Default is ``"iqr"``.
    threshold : float, optional
        Sensitivity: IQR multiplier or z-score cutoff. Default is ``1.5``.
    columns : list of str or None, optional
        Numeric columns to check. ``None`` means all numeric columns.

    Returns
    -------
    outlier_mask : pandas.DataFrame
        Boolean dataframe (same shape as input numeric columns) where ``True``
        indicates an outlier.
    summary : dict
        Per-column outlier counts.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({"a": np.append(np.random.randn(100), [10, -10])})
    >>> mask, summary = detect_outliers(df)
    """
    df_num = df.select_dtypes(include=[np.number])
    if columns is not None:
        df_num = df_num[columns]

    outlier_mask = pd.DataFrame(False, index=df_num.index, columns=df_num.columns)

    for col in df_num.columns:
        if method == "iqr":
            q1 = df_num[col].quantile(0.25)
            q3 = df_num[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
            outlier_mask[col] = (df_num[col] < lower) | (df_num[col] > upper)
        elif method == "zscore":
            mean, std = df_num[col].mean(), df_num[col].std()
            if std > 0:
                z = np.abs((df_num[col] - mean) / std)
                outlier_mask[col] = z > threshold
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'zscore'.")

    summary = {col: int(outlier_mask[col].sum()) for col in outlier_mask.columns}
    total = sum(summary.values())
    print(f"🔍 Detected {total} outlier(s) across {len(summary)} column(s).")
    for col, count in summary.items():
        if count > 0:
            print(f"   • {col}: {count} outliers")

    return outlier_mask, summary


def normalize_text(text, lowercase=True, remove_punctuation=True, remove_numbers=False,
                   remove_extra_whitespace=True, strip_html=True):
    """
    Clean text data for NLP tasks.

    Parameters
    ----------
    text : str or list of str
        Raw text or list of raw texts.
    lowercase : bool, optional
        Convert to lowercase. Default is ``True``.
    remove_punctuation : bool, optional
        Remove punctuation characters. Default is ``True``.
    remove_numbers : bool, optional
        Remove digits. Default is ``False``.
    remove_extra_whitespace : bool, optional
        Collapse extra whitespace. Default is ``True``.
    strip_html : bool, optional
        Remove HTML tags. Default is ``True``.

    Returns
    -------
    cleaned : str or list of str
        Cleaned text(s).

    Example
    -------
    >>> normalize_text("  <p>Hello,   World! 123</p>  ")
    'hello world 123'
    """
    def _clean(t):
        if strip_html:
            t = re.sub(r"<[^>]+>", " ", t)
        if lowercase:
            t = t.lower()
        if remove_punctuation:
            t = re.sub(r"[^\w\s]", " ", t)
        if remove_numbers:
            t = re.sub(r"\d+", " ", t)
        if remove_extra_whitespace:
            t = re.sub(r"\s+", " ", t).strip()
        return t

    if isinstance(text, list):
        return [_clean(t) for t in text]
    return _clean(text)


def split_dataset(X, y, test_size=0.2, random_state=42, stratify=True, verbose=True):
    """
    Perform a train/test split in one call.

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix.
    y : array-like
        Target vector.
    test_size : float, optional
        Fraction of data to reserve for testing. Default is ``0.2``.
    random_state : int, optional
        Seed for reproducibility. Default is ``42``.
    stratify : bool, optional
        If ``True`` and the target is categorical, stratified splitting is used.
        Default is ``True``.
    verbose : bool, optional
        Print split sizes. Default is ``True``.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split datasets.

    Example
    -------
    >>> import numpy as np
    >>> X = np.random.rand(100, 3); y = np.random.randint(0, 2, 100)
    >>> X_train, X_test, y_train, y_test = split_dataset(X, y)
    """
    y_arr = np.array(y)
    strat = y_arr if (stratify and np.issubdtype(y_arr.dtype, np.integer)) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    if verbose:
        print(f"✂️  Split: {len(X_train)} train / {len(X_test)} test  (test_size={test_size})")

    return X_train, X_test, y_train, y_test


def feature_summary(df, top_n_categories=10):
    """
    Generate quick statistics for each column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    top_n_categories : int, optional
        For categorical columns, show the top N most frequent values.
        Default is ``10``.

    Returns
    -------
    summary : pandas.DataFrame
        Summary dataframe with dtype, missing count / percentage, unique values,
        and key statistics.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    >>> summary = feature_summary(df)
    """
    records = []
    for col in df.columns:
        info = {
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isnull().sum()),
            "missing_pct": round(df[col].isnull().mean() * 100, 2),
            "unique": int(df[col].nunique()),
        }

        if np.issubdtype(df[col].dtype, np.number):
            info["mean"] = round(df[col].mean(), 4)
            info["std"] = round(df[col].std(), 4)
            info["min"] = df[col].min()
            info["max"] = df[col].max()
            info["median"] = df[col].median()
        else:
            top = df[col].value_counts().head(top_n_categories).to_dict()
            info["top_values"] = str(top)

        records.append(info)

    summary_df = pd.DataFrame(records)
    print(f"\n📋 Feature Summary ({len(df)} rows, {len(df.columns)} columns)\n")
    print(summary_df.to_string(index=False))
    return summary_df
