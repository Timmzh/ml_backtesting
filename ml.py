from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

import xgboost as xgb


# ==== CONFIG ====
INPUT_CSV = "cleaned_with_labels.csv"   # <-- change this to your CSV
TARGET_COL = "label_SM_minmax"          # <-- continuous pct-return label (e.g. trade_ret_pct)
TEST_SIZE = 0.2                         # last 20% of samples as test
RANDOM_STATE = 42
N_FEATURES = 10                         # desired number of factors
CORR_THRESHOLD = 0.7                    # max allowed pairwise |corr| between chosen factors
# ================


def load_data(path: str) -> pd.DataFrame:
    """Load the labeled factor file."""
    df = pd.read_csv(path)
    return df


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare X (features) and y (continuous target):

    - Sort by time (no shuffling).
    - Drop rows with NaN in target.
    - Convert features to numeric and drop rows with NaNs in features.
    """
    df = df.copy()

    # Sort by time to respect chronology
    time_cols = [c for c in ["open_time", "timestamp", "openDate"] if c in df.columns]
    if time_cols:
        df = df.sort_values(time_cols[0])

    # Remove infinities and obvious NaNs in label
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COL])

    # Continuous target: pct return per trade/bar
    y = df[TARGET_COL].astype(float)
    print(f"[INFO] Using continuous target '{TARGET_COL}' (min={y.min():.6f}, max={y.max():.6f})")

    # Columns we definitely do NOT want as features
    non_feature_cols = {
        TARGET_COL,
        "open_time",
        "close_time",
        "timestamp",
        "openDate",
    }

    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Convert features to numeric; coerce non-numeric to NaN
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows where any feature is NaN after conversion
    mask = ~X.isna().any(axis=1)
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"[INFO] Dropping {dropped} rows due to NaNs in features after conversion.")

    X = X[mask]
    y = y[mask]

    return X, y, feature_cols


def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Simple chronological split: first part train, last part test.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ========= FEATURE SELECTION =========

def select_top_uncorrelated_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    corr_threshold: float = 0.7,
) -> List[str]:
    """
    1) Fit a RandomForestRegressor on full X,y to get feature importances.
    2) Sort features by importance.
    3) Greedily pick features, skipping any that have |corr| > corr_threshold
       with already selected features.
    4) Return up to top_k features.
    """
    print("\n[INFO] Selecting top uncorrelated features...")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]  # descending importance
    ordered_features = [X.columns[i] for i in order]

    # Precompute correlation matrix
    corr = X.corr().abs().fillna(0.0)

    selected: List[str] = []
    for feat in ordered_features:
        if len(selected) >= top_k:
            break
        if not selected:
            selected.append(feat)
            continue
        ok = True
        for s in selected:
            if corr.loc[feat, s] > corr_threshold:
                ok = False
                break
        if ok:
            selected.append(feat)

    if len(selected) < top_k:
        print(f"[INFO] Only found {len(selected)} features under correlation threshold {corr_threshold}.")

    print(f"[INFO] Selected {len(selected)} features:")
    for i, f in enumerate(selected, 1):
        print(f"  {i:2d}. {f}")

    return selected


# ========= SIMPLE REGRESSORS =========

def train_dummy_mean_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> DummyRegressor:
    """
    Baseline regression: always predicts the mean of y_train.
    """
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    return dummy


def train_linear_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Ordinary Least Squares linear regression with scaling.
    """
    lin = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])
    lin.fit(X_train, y_train)
    return lin


def train_ridge_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 10.0
) -> Pipeline:
    """
    Ridge regression (L2-regularized) with scaling.
    """
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=alpha, random_state=RANDOM_STATE))
    ])
    ridge.fit(X_train, y_train)
    return ridge


def train_lasso_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 0.01
) -> Pipeline:
    """
    Lasso regression (L1-regularized) with scaling.
    """
    lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=50000))
    ])
    lasso.fit(X_train, y_train)
    return lasso


# ========= TREE / BOOSTING REGRESSORS =========

def train_random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestRegressor:
    """
    Random Forest regressor for continuous return prediction.
    """
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> xgb.XGBRegressor:
    """
    XGBoost regressor for continuous return prediction.
    """
    xgb_reg = xgb.XGBRegressor(
        n_estimators=200,          # was 400
        learning_rate=0.05,
        max_depth=3,              # was 4
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,           # add more L2
        reg_alpha=0.0,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
        random_state=RANDOM_STATE,
    )
    xgb_reg.fit(X_train, y_train)
    return xgb_reg


# ========= EVALUATION & CV =========

def evaluate_regression_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Evaluate regression model with standard metrics + trading-relevant ones.
    Returns y_pred for visualization.
    """
    print(f"\n==== {name} Evaluation ====")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Pearson correlation between predicted and realized returns
    if len(y_test) > 1 and np.std(y_pred) > 0 and np.std(y_test) > 0:
        corr = np.corrcoef(y_test, y_pred)[0, 1]
    else:
        corr = np.nan

    # Sign hit rate: how often sign(pred) == sign(real)
    sign_hit = np.mean(np.sign(y_pred) == np.sign(y_test))

    print(f"MAE:         {mae:.6f}")
    print(f"RMSE:        {rmse:.6f}")
    print(f"R^2:         {r2:.6f}")
    print(f"Corr(pred,y):{corr:.6f}")
    print(f"Sign hit rate: {sign_hit:.4f}")

    # Optional: show basic distribution info
    print(f"y_test mean: {y_test.mean():.6f}, y_pred mean: {y_pred.mean():.6f}")

    return y_pred


def print_feature_importance(
    name: str,
    model,
    feature_names: List[str],
    top_n: int = 25,
):
    """
    Print top N features by importance (works for tree-based regressors).
    """
    if not hasattr(model, "feature_importances_"):
        print(f"\n[INFO] {name} has no feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    print(f"\n==== Top {top_n} Features ({name}) ====")
    for rank, i in enumerate(idx, start=1):
        print(f"{rank:2d}. {feature_names[i]:30s} {importances[i]:.6f}")


def cross_validate_time_series_reg(
    base_model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
):
    """
    Time-series cross-validation for regression models.

    Reports correlation and RMSE per fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    corrs = []
    rmses = []

    print(f"\n==== Time-Series CV ({base_model.__class__.__name__}) ====")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_fold = clone(base_model)
        model_fold.fit(X_tr, y_tr)
        y_pred = model_fold.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        if len(y_val) > 1 and np.std(y_pred) > 0 and np.std(y_val) > 0:
            corr = np.corrcoef(y_val, y_pred)[0, 1]
        else:
            corr = np.nan

        corrs.append(corr)
        rmses.append(rmse)

        print(f"[TS-CV] Fold {fold}: Corr = {corr:.4f}, RMSE = {rmse:.6f}")

    if corrs:
        print(f"[TS-CV] Corr mean ± std: {np.nanmean(corrs):.4f} ± {np.nanstd(corrs):.4f}")
        print(f"[TS-CV] RMSE mean ± std: {np.mean(rmses):.6f} ± {np.std(rmses):.6f}")


# ========= VISUALIZATION =========

def plot_target_distribution(
    y_train: pd.Series,
    y_test: pd.Series,
    bins: int = 50
):
    """
    Plot distribution (histogram) of target for train vs test.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(y_train, bins=bins, alpha=0.5, label="y_train")
    plt.hist(y_test, bins=bins, alpha=0.5, label="y_test")
    plt.title("Target distribution: train vs test")
    plt.xlabel("Target value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_prediction_distributions(
    y_test: pd.Series,
    preds_dict: Dict[str, np.ndarray],
    bins: int = 50
):
    """
    Plot distribution of y_test vs predictions for each model.
    """
    for name, y_pred in preds_dict.items():
        plt.figure(figsize=(8, 4))
        plt.hist(y_test, bins=bins, alpha=0.5, label="y_test")
        plt.hist(y_pred, bins=bins, alpha=0.5, label=f"y_pred ({name})")
        plt.title(f"Distribution: y_test vs y_pred ({name})")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_pred_vs_actual_scatter(
    y_test: pd.Series,
    preds_dict: Dict[str, np.ndarray],
    max_points: int = 5000
):
    """
    Scatter plot: actual vs predicted for each model (downsampled if needed).
    """
    y_test_np = np.asarray(y_test)

    for name, y_pred in preds_dict.items():
        y_pred_np = np.asarray(y_pred)

        n = len(y_test_np)
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(int)
            x = y_test_np[idx]
            y = y_pred_np[idx]
        else:
            x = y_test_np
            y = y_pred_np

        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, s=5, alpha=0.4)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted ({name})")
        plt.tight_layout()
        plt.show()


def main():
    df = load_data(INPUT_CSV)
    print(f"[INFO] Loaded data with shape: {df.shape}")

    X, y, feature_cols = prepare_dataset(df)

    # ----- Feature selection to 10 uncorrelated factors -----
    selected_features = select_top_uncorrelated_features(
        X, y, top_k=N_FEATURES, corr_threshold=CORR_THRESHOLD
    )
    X = X[selected_features]

    X_train, X_test, y_train, y_test = time_series_train_test_split(
        X, y, test_size=TEST_SIZE
    )

    # Store predictions for visualization
    preds_dict: Dict[str, np.ndarray] = {}

    # ----- Dummy Mean Regressor -----
    dummy = train_dummy_mean_regressor(X_train, y_train)
    y_pred_dummy = evaluate_regression_model("Dummy Mean Regressor", dummy, X_test, y_test)
    preds_dict["Dummy"] = y_pred_dummy
    cross_validate_time_series_reg(dummy, X, y, n_splits=5)

    # ----- Linear Regression -----
    lin = train_linear_regressor(X_train, y_train)
    y_pred_lin = evaluate_regression_model("Linear Regression", lin, X_test, y_test)
    preds_dict["Linear Regression"] = y_pred_lin
    cross_validate_time_series_reg(lin, X, y, n_splits=5)

    # ----- Ridge Regression -----
    ridge = train_ridge_regressor(X_train, y_train, alpha=10.0)
    y_pred_ridge = evaluate_regression_model("Ridge Regression (alpha=10.0)", ridge, X_test, y_test)
    preds_dict["Ridge (10.0)"] = y_pred_ridge
    cross_validate_time_series_reg(ridge, X, y, n_splits=5)

    # ----- Lasso Regression -----
    lasso = train_lasso_regressor(X_train, y_train, alpha=0.01)
    y_pred_lasso = evaluate_regression_model("Lasso Regression (alpha=0.01)", lasso, X_test, y_test)
    preds_dict["Lasso (0.01)"] = y_pred_lasso
    cross_validate_time_series_reg(lasso, X, y, n_splits=5)

    # ----- Random Forest Regressor -----
    rf = train_random_forest_regressor(X_train, y_train)
    y_pred_rf = evaluate_regression_model("Random Forest Regressor", rf, X_test, y_test)
    # ----- RF Tail Bucket PnL diagnostic -----
    rf_eval = pd.DataFrame({
        "y_test": y_test.values,
        "y_pred": y_pred_rf
    }).sort_values("y_pred")

    n = len(rf_eval)
    k = max(5, n // 10)  # top/bottom 10%

    bottom_mean = rf_eval.iloc[:k]["y_test"].mean()
    top_mean = rf_eval.iloc[-k:]["y_test"].mean()

    print(f"\n[RF Tail Buckets] bottom 10% mean y: {bottom_mean:.6f}, top 10% mean y: {top_mean:.6f}")

    preds_dict["Random Forest"] = y_pred_rf
    print_feature_importance("Random Forest Regressor", rf, selected_features, top_n=min(30, len(selected_features)))
    cross_validate_time_series_reg(rf, X, y, n_splits=5)

    # ----- XGBoost Regressor -----
    xgb_reg = train_xgboost_regressor(X_train, y_train)
    y_pred_xgb = evaluate_regression_model("XGBoost Regressor", xgb_reg, X_test, y_test)
    # ----- XGB Tail Bucket PnL diagnostic -----
    xgb_eval = pd.DataFrame({
        "y_test": y_test.values,
        "y_pred": y_pred_xgb
    }).sort_values("y_pred")

    n = len(xgb_eval)
    k = max(5, n // 10)

    bottom_mean_x = xgb_eval.iloc[:k]["y_test"].mean()
    top_mean_x = xgb_eval.iloc[-k:]["y_test"].mean()

    print(f"[XGB Tail Buckets] bottom 10% mean y: {bottom_mean_x:.6f}, top 10% mean y: {top_mean_x:.6f}")

    preds_dict["XGBoost"] = y_pred_xgb
    print_feature_importance("XGBoost Regressor", xgb_reg, selected_features, top_n=min(30, len(selected_features)))
    cross_validate_time_series_reg(xgb_reg, X, y, n_splits=5)

    # ----- Visualizations -----
    plot_target_distribution(y_train, y_test, bins=50)
    plot_prediction_distributions(y_test, preds_dict, bins=50)
    # Optional scatter (can be slow if huge dataset)
    plot_pred_vs_actual_scatter(y_test, preds_dict, max_points=5000)


if __name__ == "__main__":
    main()


