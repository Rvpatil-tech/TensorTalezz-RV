import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV

from .lightweight import _CLASSIFIERS, _REGRESSORS, evaluate as lw_evaluate
from .education import explain as ed_explain

def _detect_task(y):
    """Auto-detect classification vs regression."""
    if pd.api.types.is_numeric_dtype(y):
        if len(y.unique()) <= 20: 
            # Could be categorical numbers
            return "classification"
        return "regression"
    return "classification"

def _build_preprocessor(X):
    """Builds a scikit-learn ColumnTransformer for preprocessing."""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = []
    if len(numeric_features) > 0:
        transformers.append(('num', numeric_transformer, numeric_features))
    if len(categorical_features) > 0:
        transformers.append(('cat', categorical_transformer, categorical_features))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    return preprocessor, numeric_features.tolist(), categorical_features.tolist()

class TensorTalezzPipeline:
    """Wrapper to hold the pipeline and metadata."""
    def __init__(self, pipeline, task, target, features, report):
        self.pipeline = pipeline
        self.task = task
        self.target = target
        self.features = features
        self.report = report

def fit(df, target, task="auto", model="auto", handle_outliers=False, tune=False, explain=False, mode="fast", verbose=True):
    """
    Train a complete ML pipeline in a single step.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The full dataset.
    target : str
        The name of the target column.
    task : str
        'classification', 'regression', or 'auto'.
    model : str
        Model name (e.g., 'random_forest') or 'auto' for best selection.
    handle_outliers : bool
        If True, drops obvious numeric outliers before training.
    tune : bool
        If True, performs basic hyperparameter tuning.
    explain : bool
        If True, prints educational explanations.
    mode : str
        'fast' or 'learn'. 'learn' provides detailed educational guidance.
    verbose : bool
        If True, prints logging output.
    """
    if mode == "learn" or explain:
        verbose = True
        print("\n🎓 [LEARN MODE] Let's build your machine learning model!")
        if explain:
            ed_explain("feature_engineering")
            time.sleep(1)

    df_train = df.copy()
    
    if target not in df_train.columns:
        raise ValueError(
            f"❌ [Error] Target column '{target}' not found in your dataframe.\n"
            f"🎓 [Tip] The 'target' is what you want the model to predict. "
            f"Double-check your spelling, or verify it exists using `print(df.columns)`."
        )

    # Drop missing targets immediately
    initial_len = len(df_train)
    df_train = df_train.dropna(subset=[target])
    if len(df_train) < initial_len and verbose:
        print(f"⚠️ Dropped {initial_len - len(df_train)} rows where the target was missing (a model can't learn without the answer!).")

    y = df_train[target]
    X = df_train.drop(columns=[target])

    if task == "auto":
        task = _detect_task(y)
        if verbose:
            print(f"🤖 [Auto-Detect] target '{target}' -> Task: {task.upper()}")

    # Outlier handling (Optional)
    if handle_outliers and len(X.select_dtypes(include=np.number).columns) > 0:
        num_cols = X.select_dtypes(include=np.number).columns
        # Simple IQR filter targeting extreme outliers for stability
        Q1 = X[num_cols].quantile(0.25)
        Q3 = X[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        condition = ~((X[num_cols] < (Q1 - 3 * IQR)) | (X[num_cols] > (Q3 + 3 * IQR))).any(axis=1)
        X = X[condition]
        y = y[condition]
        if verbose:
            print(f"🧹 Removed {len(df_train) - len(X)} extreme outliers.")

    # Feature Engineering (Automated basic steps)
    # 1. Date variables
    date_cols = X.select_dtypes(include=['datetime64']).columns
    for d_col in date_cols:
        X[f'{d_col}_year'] = X[d_col].dt.year
        X[f'{d_col}_month'] = X[d_col].dt.month
        X = X.drop(columns=[d_col])
        if verbose:
             print(f"⚙️ Auto-engineered date features for {d_col}.")
             
    # 2. High-cardinality grouping
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for c_col in cat_cols:
        unique_count = X[c_col].nunique()
        if unique_count > 15:
            # Keep top 10 categories, group rest into 'Other'
            top_cats = X[c_col].value_counts().nlargest(10).index
            X[c_col] = X[c_col].apply(lambda x: x if x in top_cats else 'Other')
            if verbose:
                print(f"⚙️ Grouped high-cardinality feature '{c_col}' into top 10 + 'Other'.")

    # Build Pipeline
    preprocessor, num_feats, cat_feats = _build_preprocessor(X)

    if model == "auto":
        # Auto-select best robust model based on data size
        if len(X) < 1000:
            model_name = "random_forest"
        else:
            model_name = "gradient_boosting"
        if verbose:
            print(f"🧠 [Auto-Select] Picked '{model_name}' for dataset size {len(X)}.")
    else:
        model_name = model

    # Resolve estimator
    registry = _CLASSIFIERS if task == "classification" else _REGRESSORS
    if model_name not in registry:
        raise ValueError(f"Model '{model_name}' not supported for {task}.")
    
    estimator = registry[model_name]()

    # Construct the final sklearn Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', estimator)
    ])

    if explain or mode == "learn":
        print("\n🎓 [EXPLAIN] Preprocessing:")
        print("Numeric data will have missing values filled with the median, then standardized (mean=0, variance=1).")
        print("Categorical data will be converted into one-hot encoded numbers so the model can understand them.\n")

    if tune:
        if verbose:
            print("🔧 Tuning hyperparameters...")
        # Very simple random grid for robust models to save time
        param_grid = {}
        if model_name == "random_forest":
            param_grid = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20]}
        elif model_name == "gradient_boosting":
            param_grid = {'model__learning_rate': [0.01, 0.1, 0.2], 'model__n_estimators': [50, 100, 200]}
        
        if param_grid:
            search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=5, cv=3, random_state=42)
            search.fit(X, y)
            pipeline = search.best_estimator_
            if verbose:
                print(f"🎯 Best params found: {search.best_params_}")
        else:
            pipeline.fit(X, y)
    else:
        if verbose:
            print(f"🚀 Training {model_name}...")
        pipeline.fit(X, y)

    # Feature Importance (if supported by model)
    top_features = None
    try:
        if hasattr(pipeline.named_steps["model"], "feature_importances_"):
            importances = pipeline.named_steps["model"].feature_importances_
            feature_names = []
            if 'num' in pipeline.named_steps["preprocessor"].named_transformers_:
                feature_names.extend(num_feats)
            if 'cat' in pipeline.named_steps["preprocessor"].named_transformers_:
                cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_['cat'].named_steps['onehot']
                feature_names.extend(cat_encoder.get_feature_names_out(cat_feats))
            
            # Match
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            top_features = feat_imp.head(5).to_dict()
            if verbose:
                print(f"\n🌟 Top 3 Important Features: {feat_imp.head(3).index.tolist()}")
    except Exception as e:
        pass # Not all transformers perfectly map names back yet depending on versions

    if explain or mode == "learn":
        ed_explain(model_name)
        if top_features:
            print("💡 The model relied heavily on these features to make its decisions!")

    if verbose:
        print("\n✅ Training Complete!")
        # Mini evaluation
        preds = pipeline.predict(X)
        print("   Training Performance (Quick check):")
        if task == "classification":
            acc = (preds == y).mean()
            print(f"   Accuracy: {acc:.2%}")
        else:
            mse = mean_squared_error(y, preds)
            print(f"   RMSE: {np.sqrt(mse):.4f}")

    report = {
        "task": task,
        "model": model_name,
        "features": list(X.columns),
        "target": target,
        "top_features": top_features,
        "preprocessing": ["Imputation", "Scaling", "One-Hot Encoding"]
    }

    if verbose and mode != "learn":
        print(f"\nTip: Want to understand this model better? Try `from tensor_talezz_rv import explain; explain('{model_name}')`")

    return TensorTalezzPipeline(pipeline, task, target, list(X.columns), report), report


def predict(model_obj, df):
    """
    Generate predictions using the trained pipeline on raw, unprocessed data.
    
    Parameters:
    -----------
    model_obj : TensorTalezzPipeline
        The completed model object returned by `fit()`.
    df : pandas.DataFrame
        Raw data to predict on.
        
    Returns:
    --------
    predictions : numpy.ndarray
    """
    if isinstance(model_obj, TensorTalezzPipeline):
        pipeline = model_obj.pipeline
        expected_features = model_obj.features
    else:
        pipeline = model_obj
        expected_features = df.columns

    # Handle Date features if they were auto-engineered
    df_eval = df.copy()
    date_cols = df_eval.select_dtypes(include=['datetime64']).columns
    for d_col in date_cols:
        if f'{d_col}_year' in expected_features:
            df_eval[f'{d_col}_year'] = df_eval[d_col].dt.year
            df_eval[f'{d_col}_month'] = df_eval[d_col].dt.month
            df_eval = df_eval.drop(columns=[d_col])

    # Add missing columns with None to avoid pipeline errors if predicting on single rows
    for col in expected_features:
        if col not in df_eval.columns:
            df_eval[col] = np.nan

    # Subset to correct columns
    X = df_eval[expected_features]

    preds = pipeline.predict(X)
    return preds
