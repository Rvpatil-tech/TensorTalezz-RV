import pandas as pd
import numpy as np

def detect_bias(dataset, target, sensitive_feature):
    """
    Automatically checks whether outcomes differ significantly across sensitive groups.
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset containing both features and target.
    target : str
        The target column name.
    sensitive_feature : str
        The categorical feature representing groups (e.g., 'gender', 'region').
        
    Returns:
    --------
    bias_report : dict
        A dictionary containing baseline averages across the groups and the maximum difference detected.
        
    Example:
    --------
    >>> df = pd.DataFrame({'gender': ['M','F','M','F'], 'hired': [1,0,1,1]})
    >>> report = detect_bias(df, target='hired', sensitive_feature='gender')
    """
    print(f"\n🛡️  [Responsible AI] Detecting bias for '{target}' across '{sensitive_feature}' groups...")
    
    if sensitive_feature not in dataset.columns or target not in dataset.columns:
        raise ValueError("Target or sensitive feature not found in dataset.")
        
    group_stats = dataset.groupby(sensitive_feature)[target].agg(['mean', 'count']).to_dict(orient="index")
    
    print(f"   Baseline Outcome Averages:")
    for group, stats in group_stats.items():
        print(f"     - Group [{group}]: Average = {stats['mean']:.3f} (N={stats['count']})")
        
    means = [stats['mean'] for stats in group_stats.values()]
    diff = max(means) - min(means)
    
    print("\n   Observation:")
    if diff > 0.1:
        print(f"   ⚠️ Significant disparity detected! Max difference between groups is {diff:.3f}.")
        print("      This strongly indicates potential historical or sampling bias in your dataset.")
    else:
        print(f"   ✅ Outcomes appear relatively balanced. Max difference is {diff:.3f}.")
        
    return {"group_stats": group_stats, "max_difference": diff}


def fairness_metrics(model_obj, X, y, sensitive_feature):
    """
    Compute fairness metrics such as Demographic Parity and Equal Opportunity.
    
    Parameters:
    -----------
    model_obj : TensorTalezzPipeline or standard model
    X : pandas.DataFrame
        Test features, must include the sensitive feature.
    y : array-like
        True labels.
    sensitive_feature : str
        Column name in X to evaluate.
        
    Returns:
    --------
    metrics : dict
        A dictionary mapping each group to their 'demographic_parity' and 'true_positive_rate'.
        
    Example:
    --------
    >>> metrics = fairness_metrics(model, X_test, y_test, sensitive_feature='gender')
    >>> print(metrics['M']['demographic_parity'])
    """
    from .core import predict
    print(f"\n⚖️  [Fairness Analytics] Evaluating model fairness across '{sensitive_feature}'...")
    
    if sensitive_feature not in X.columns:
        raise ValueError(f"'{sensitive_feature}' must be present in X.")
        
    preds = predict(model_obj, X)
    
    # Needs to be a classification for parity metrics usually
    is_classification = len(np.unique(preds)) <= 10
    if not is_classification:
        print("   Metrics currently optimized for classification. Returning raw mean differences.")
        
    df = X.copy()
    df['y_true'] = np.array(y)
    df['y_pred'] = preds
    
    groups = df[sensitive_feature].unique()
    metrics = {}
    
    print("   Metrics Breakdown (Ideal difference = 0.0):")
    for g in groups:
        group_df = df[df[sensitive_feature] == g]
        if len(group_df) == 0:
            continue
        
        # Demographic Parity: P(Y_hat = 1 | s = g)
        demo_parity = group_df['y_pred'].mean()
        
        # Equal Opportunity: TPR for group
        positives = group_df[group_df['y_true'] == 1]
        tpr = positives['y_pred'].mean() if len(positives) > 0 else np.nan
        
        metrics[g] = {'demographic_parity': demo_parity, 'true_positive_rate': tpr}
        print(f"     - [{g}] Predictive Positivity Rate: {demo_parity:.3f} | TPR: {tpr:.3f}")
        
    return metrics


def audit_report(model_obj, dataset, target, sensitive_feature):
    """
    Generates a structured report combining bias detection and fairness scores.
    """
    print("\n" + "="*60)
    print(" 📑 RESPONSIBLE AI AUDIT REPORT")
    print("="*60)
    
    X = dataset.drop(columns=[target])
    y = dataset[target]
    
    print("\n[PART 1: DATA BIAS]")
    detect_bias(dataset, target, sensitive_feature)
    
    print("\n[PART 2: MODEL FAIRNESS]")
    fairness_metrics(model_obj, X, y, sensitive_feature)
    
    print("\n[RECOMMENDATION]")
    print(f"   If disparities exist, reconsider using '{sensitive_feature}' explicitly or")
    print("   apply reweighing techniques to balance the training data before `fit()`.")
    print("="*60 + "\n")
