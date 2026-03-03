#!/usr/bin/env python3
"""
Data Science Portfolio Demo (Python)

Purpose
-------
This single-file script showcases practical, job-ready Data Science skills:
- Reproducible experiment setup (argparse + random seeds + logging)
- Data loading (scikit-learn built-in datasets; works offline)
- Feature engineering (including a synthetic categorical feature)
- Preprocessing with ColumnTransformer (scaling + one-hot encoding)
- Model training with Pipelines
- Cross-validation + robust metrics
- Hyperparameter tuning (RandomizedSearchCV)
- Model interpretation (permutation importance)
- Artifact generation (JSON report + model pickle + optional plots)

How to run
----------
Classification (default): Breast cancer dataset
    python data_science_portfolio_demo.py --task classification

Regression: California housing dataset
    python data_science_portfolio_demo.py --task regression

Outputs (created in ./artifacts by default)
------------------------------------------
- report.json      : key metrics + config
- model.joblib     : trained best model pipeline
- importance.csv   : permutation importances
- cv_results.csv   : CV results (tuning)
- (optional) plots : confusion matrix / residual plot

Notes
-----
This is designed to be attached to applications to demonstrate competence in:
Pandas, scikit-learn, pipelines, evaluation, and clean engineering practices.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance


# -----------------------------
# Configuration / Data classes
# -----------------------------

@dataclass(frozen=True)
class RunConfig:
    task: str
    test_size: float
    random_state: int
    artifacts_dir: str
    n_iter_search: int
    cv_folds: int
    n_jobs: int
    plot: bool


# -----------------------------
# Logging
# -----------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# -----------------------------
# Data loading & feature eng.
# -----------------------------

def load_dataset(task: str) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Returns
    -------
    X : DataFrame
    y : Series
    target_name : str
    """
    if task == "classification":
        ds = load_breast_cancer(as_frame=True)
        X = ds.data.copy()
        y = ds.target.copy()
        target_name = "malignant(0)/benign(1)"
        return X, y, target_name

    if task == "regression":
        ds = fetch_california_housing(as_frame=True)
        X = ds.data.copy()
        y = ds.target.copy()
        target_name = "median_house_value"
        return X, y, target_name

    raise ValueError(f"Unknown task: {task}. Use 'classification' or 'regression'.")


def add_synthetic_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lightweight feature engineering to demonstrate practical work.

    - Adds a categorical feature by binning a numeric column.
    - Adds an interaction feature between two numeric columns (if available).
    """
    X = X.copy()

    # Choose a stable numeric column for binning
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return X

    base_col = numeric_cols[0]
    # Quantile-based bins (robust to scale)
    X[f"{base_col}_bin"] = pd.qcut(X[base_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    # Simple interaction feature (if at least two numeric cols exist)
    if len(numeric_cols) >= 2:
        c1, c2 = numeric_cols[0], numeric_cols[1]
        X[f"{c1}_x_{c2}"] = X[c1] * X[c2]

    return X


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_features, categorical_features


# -----------------------------
# Modeling
# -----------------------------

def get_models(task: str, random_state: int) -> Dict[str, Any]:
    """
    Returns candidate estimators (unfitted).
    """
    if task == "classification":
        return {
            "logreg": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
            "rf": RandomForestClassifier(
                n_estimators=400, random_state=random_state, n_jobs=-1, class_weight="balanced_subsample"
            ),
        }

    if task == "regression":
        return {
            "ridge": Ridge(random_state=random_state),
            "rf": RandomForestRegressor(
                n_estimators=500, random_state=random_state, n_jobs=-1
            ),
        }

    raise ValueError(f"Unknown task: {task}")


def get_search_space(task: str) -> Dict[str, Dict[str, Any]]:
    """
    Hyperparameter spaces per model key for RandomizedSearchCV.
    (Keeps spaces compact so it runs quickly on most machines.)
    """
    if task == "classification":
        return {
            "logreg": {
                "model__C": np.logspace(-3, 2, 30),
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
            "rf": {
                "model__max_depth": [None, 3, 5, 8, 12],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        }

    if task == "regression":
        return {
            "ridge": {
                "model__alpha": np.logspace(-4, 3, 40),
            },
            "rf": {
                "model__max_depth": [None, 5, 8, 12, 18],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": [1.0, "sqrt", "log2"],
            },
        }

    raise ValueError(f"Unknown task: {task}")


def build_pipeline(preprocessor: ColumnTransformer, model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def choose_scoring(task: str) -> Dict[str, str]:
    if task == "classification":
        return {"f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"}
    return {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def get_cv(task: str, folds: int, random_state: int):
    if task == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


# -----------------------------
# Evaluation helpers
# -----------------------------

def evaluate_holdout(task: str, y_true: pd.Series, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    if task == "classification":
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }
        if y_proba is not None:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        return out

    # regression
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def safe_get_feature_names(pipe: Pipeline) -> List[str]:
    """
    Extract feature names after preprocessing for interpretability.
    Works for ColumnTransformer with OneHotEncoder.
    """
    pre = pipe.named_steps["preprocess"]
    try:
        names = pre.get_feature_names_out().tolist()
        return [str(n) for n in names]
    except Exception:
        return []


# -----------------------------
# Optional plotting
# -----------------------------

def maybe_plot_classification(art_dir: Path, y_true: pd.Series, y_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(art_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def maybe_plot_regression(art_dir: Path, y_true: pd.Series, y_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    residuals = y_true.to_numpy() - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, s=12)
    ax.axhline(0.0)
    ax.set_title("Residual plot")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (true - pred)")
    fig.tight_layout()
    fig.savefig(art_dir / "residual_plot.png", dpi=160)
    plt.close(fig)


# -----------------------------
# Main run
# -----------------------------

def run(config: RunConfig) -> Dict[str, Any]:
    setup_logging()
    seed_everything(config.random_state)

    art_dir = Path(config.artifacts_dir).resolve()
    ensure_dir(art_dir)
    logging.info("Artifacts will be saved to: %s", art_dir)

    # Load & engineer features
    X, y, target_name = load_dataset(config.task)
    X = add_synthetic_features(X)

    preprocessor, numeric_cols, cat_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y if config.task == "classification" else None,
    )

    models = get_models(config.task, config.random_state)
    spaces = get_search_space(config.task)
    scoring = choose_scoring(config.task)
    cv = get_cv(config.task, config.cv_folds, config.random_state)

    # Tune each model and keep the best by primary metric
    # Primary metric: F1 for classification, RMSE for regression
    primary_metric = "f1" if config.task == "classification" else "rmse"
    best_score = -np.inf
    best_key = None
    best_search = None

    cv_rows = []

    for key, estimator in models.items():
        pipe = build_pipeline(preprocessor, estimator)
        logging.info("Tuning model: %s", key)

        # Choose refit metric aligned to primary
        refit_metric = primary_metric if config.task == "classification" else "rmse"

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=spaces[key],
            n_iter=config.n_iter_search,
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            verbose=0,
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        # Collect CV results (small subset)
        res = pd.DataFrame(search.cv_results_)
        keep_cols = [c for c in res.columns if c.startswith("mean_test_")] + ["rank_test_" + refit_metric, "params"]
        res_small = res[keep_cols].copy()
        res_small.insert(0, "model", key)
        cv_rows.append(res_small)

        # Convert to comparable "higher is better"
        if config.task == "classification":
            score = float(search.best_score_)  # F1
        else:
            # rmse is negative in scoring; best_score_ is negative -> convert to positive goodness by negating
            score = float(-search.best_score_)  # smaller RMSE is better -> invert for comparison
            score = -score  # now higher is better (less RMSE -> closer to 0 -> higher after negation)

        if score > best_score:
            best_score = score
            best_key = key
            best_search = search

    assert best_search is not None and best_key is not None

    best_pipe: Pipeline = best_search.best_estimator_
    logging.info("Best model: %s | best params: %s", best_key, best_search.best_params_)

    # Holdout evaluation
    if config.task == "classification":
        y_pred = best_pipe.predict(X_test)
        # Some models offer predict_proba; use it for ROC-AUC
        y_proba = None
        if hasattr(best_pipe.named_steps["model"], "predict_proba"):
            y_proba = best_pipe.predict_proba(X_test)[:, 1]
        holdout = evaluate_holdout(config.task, y_test, y_pred, y_proba=y_proba)
        cls_report = classification_report(y_test, y_pred, output_dict=True)
    else:
        y_pred = best_pipe.predict(X_test)
        holdout = evaluate_holdout(config.task, y_test, y_pred)
        cls_report = None

    # Permutation importance on holdout set (model-agnostic interpretability)
    feat_names = safe_get_feature_names(best_pipe)
    logging.info("Computing permutation importance on holdout set...")
    pi = permutation_importance(
        best_pipe,
        X_test,
        y_test,
        n_repeats=10,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        scoring=("f1" if config.task == "classification" else "neg_root_mean_squared_error"),
    )

    imp = pd.DataFrame(
        {
            "feature": feat_names if feat_names else [f"f_{i}" for i in range(len(pi.importances_mean))],
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # Save artifacts
    report: Dict[str, Any] = {
        "created_utc": utc_now_iso(),
        "task": config.task,
        "target": target_name,
        "data_shape": {"rows": int(X.shape[0]), "cols": int(X.shape[1])},
        "feature_groups": {"numeric": numeric_cols, "categorical": cat_cols},
        "best_model": best_key,
        "best_params": best_search.best_params_,
        "holdout_metrics": holdout,
    }
    if cls_report is not None:
        report["classification_report"] = cls_report

    (art_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    imp.to_csv(art_dir / "importance.csv", index=False)

    cv_results = pd.concat(cv_rows, ignore_index=True)
    cv_results.to_csv(art_dir / "cv_results.csv", index=False)

    dump(best_pipe, art_dir / "model.joblib")

    if config.plot:
        if config.task == "classification":
            maybe_plot_classification(art_dir, y_test, y_pred)
        else:
            maybe_plot_regression(art_dir, y_test, y_pred)

    logging.info("Done. Holdout metrics: %s", holdout)
    return report


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="End-to-end Data Science demo script (offline-ready).")
    p.add_argument("--task", choices=["classification", "regression"], default="classification",
                   help="Choose problem type.")
    p.add_argument("--test-size", type=float, default=0.2, help="Holdout test size fraction.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--artifacts-dir", type=str, default="artifacts", help="Where to save outputs.")
    p.add_argument("--n-iter-search", type=int, default=25, help="RandomizedSearchCV iterations per model.")
    p.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds.")
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for sklearn.")
    p.add_argument("--plot", action="store_true", help="Save simple plots as PNG.")
    args = p.parse_args()

    return RunConfig(
        task=args.task,
        test_size=args.test_size,
        random_state=args.random_state,
        artifacts_dir=args.artifacts_dir,
        n_iter_search=args.n_iter_search,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        plot=bool(args.plot),
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
