"""
Binary classifier to predict whether an adversarial attack is effective
on a given trading day.

Labels
------
  0 → "non attack"  (attack NOT effective)
  1 → "attack"      (attack IS effective)

Input (X)
---------
  114 features per day, semicolon-separated CSV:
  - Calendar:        day_of_week, month, quarter
  - Returns:         return_1d, return_5d, return_10d
  - Volatility:      volatility_5d, volatility_20d
  - Technical:       rsi_14, atr_14, bollinger_width_20, macd,
                     trend_strength, volume_ratio
  - OHLCV lags 0-19: ohlcv_lag_N_{open,high,low,close,volume}_rel

Output (y)
----------
  Single column "target": 0 or 1
"""

from __future__ import annotations

import logging
import os
import pickle
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_MAP = {0: "non attack", 1: "attack"}
X_SEP = ";"
TARGET_COL = "target"

CALENDAR_FEATURES = ["day_of_week", "month", "quarter"]
RETURN_FEATURES = ["return_1d", "return_5d", "return_10d"]
VOLATILITY_FEATURES = ["volatility_5d", "volatility_20d"]
TECHNICAL_FEATURES = [
    "rsi_14", "atr_14", "bollinger_width_20", "macd",
    "trend_strength", "volume_ratio",
]
N_OHLCV_LAGS = 20
OHLCV_COMPONENTS = ["open_rel", "high_rel", "low_rel", "close_rel", "volume_rel"]
OHLCV_FEATURES = [
    f"ohlcv_lag_{lag}_{comp}"
    for lag in range(N_OHLCV_LAGS)
    for comp in OHLCV_COMPONENTS
]
ALL_FEATURES = (
    CALENDAR_FEATURES + RETURN_FEATURES + VOLATILITY_FEATURES
    + TECHNICAL_FEATURES + OHLCV_FEATURES
)


# ---------------------------------------------------------------------------
# Imbalance strategy
# ---------------------------------------------------------------------------

class ImbalanceStrategy(str, Enum):
    """Strategy to handle class imbalance during training."""
    NONE = "none"
    """No correction. Use only if classes are roughly balanced."""

    CLASS_WEIGHT = "class_weight"
    """
    Penalise errors on the minority class more heavily during training.
    - RandomForest / LogisticRegression: native class_weight="balanced"
    - GradientBoosting: achieved via sample_weight in fit()
    """

    SMOTE = "smote"
    """
    Synthetic Minority Over-sampling TEchnique.
    Generates synthetic 'attack' samples by interpolating existing ones.
    Applied inside the cross-validation fold to avoid data leakage.
    """

    UNDERSAMPLE = "undersample"
    """
    Randomly removes 'non attack' samples to rebalance the dataset.
    Fast; loses information from the majority class.
    """

    SMOTE_UNDERSAMPLE = "smote_undersample"
    """
    Combines SMOTE (oversample minority) + RandomUnderSampler (reduce majority).
    """


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_X(path: str | Path) -> pd.DataFrame:
    """Load the feature matrix from a semicolon-separated CSV."""
    df = pd.read_csv(path, sep=X_SEP)
    logger.info("Loaded X from '%s' – shape %s", path, df.shape)
    _validate_X(df)
    return df


def load_y(path: str | Path) -> pd.Series:
    """Load the target vector from a CSV (single column 'target')."""
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}.")
    y = df[TARGET_COL].astype(int)
    counts = dict(y.value_counts().sort_index())
    total = len(y)
    logger.info("Loaded y from '%s' – %d samples", path, total)
    for cls, cnt in counts.items():
        logger.info("  Class %d (%s): %d (%.1f%%)", cls, LABEL_MAP[cls], cnt, cnt / total * 100)
    if 0 in counts and 1 in counts:
        ratio = counts[0] / counts[1]
        logger.info("  Imbalance ratio: 1:%.0f (non attack : attack)", ratio)
        if ratio > 20:
            logger.warning(
                "Severe imbalance detected (1:%.0f). "
                "Recommend ImbalanceStrategy.SMOTE_UNDERSAMPLE.", ratio
            )
        elif ratio > 5:
            logger.warning(
                "Moderate imbalance detected (1:%.0f). "
                "Recommend ImbalanceStrategy.SMOTE or CLASS_WEIGHT.", ratio
            )
    return y


def _validate_X(df: pd.DataFrame) -> None:
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        logger.warning(
            "%d expected feature(s) missing: %s%s",
            len(missing), missing[:5], " ..." if len(missing) > 5 else ""
        )


# ---------------------------------------------------------------------------
# Model / pipeline construction
# ---------------------------------------------------------------------------

def build_model(
    model_type: str = "gradient_boosting",
    imbalance_strategy: ImbalanceStrategy = ImbalanceStrategy.CLASS_WEIGHT,
) -> Pipeline | ImbPipeline:
    
    use_balanced_weight = imbalance_strategy == ImbalanceStrategy.CLASS_WEIGHT

    classifiers = {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_split=10,
            random_state=42,
            # GradientBoosting does not support class_weight natively;
            # sample_weight is injected at fit() time when CLASS_WEIGHT is used.
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced" if use_balanced_weight else None,
            random_state=42,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced" if use_balanced_weight else None,
            random_state=42,
        ),
    }

    if model_type not in classifiers:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from {list(classifiers.keys())}."
        )

    clf = classifiers[model_type]

    # --- Resampling steps (only for SMOTE / UNDERSAMPLE strategies) ----------
    resamplers = []
    if imbalance_strategy in (ImbalanceStrategy.SMOTE, ImbalanceStrategy.SMOTE_UNDERSAMPLE):
        resamplers.append(("smote", SMOTE(random_state=42)))
    if imbalance_strategy in (ImbalanceStrategy.UNDERSAMPLE, ImbalanceStrategy.SMOTE_UNDERSAMPLE):
        resamplers.append(("undersampler", RandomUnderSampler(random_state=42)))

    steps = [("scaler", StandardScaler())] + resamplers + [("clf", clf)]

    # imbalanced-learn Pipeline supports resamplers; sklearn Pipeline does not
    PipelineClass = ImbPipeline if resamplers else Pipeline
    pipeline = PipelineClass(steps=steps)

    logger.info(
        "Built pipeline: model_type='%s', imbalance_strategy='%s'.",
        model_type, imbalance_strategy.value,
    )
    return pipeline


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "gradient_boosting",
    imbalance_strategy: ImbalanceStrategy = ImbalanceStrategy.CLASS_WEIGHT,
    test_size: float = 0.2,
    cv_folds: int = 5,
    cv_scoring: str = "f1",
    cv_strategy: str = "temporal",
    random_state: int = 42,
    X_train: Optional[pd.DataFrame] = None,
    X_test:  Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series]    = None,
    y_test:  Optional[pd.Series]    = None,
) -> tuple[Pipeline | ImbPipeline, dict]:
    """
    Train the attack-detection model.
 
    Parameters
    ----------
    X : pd.DataFrame
        Ignored when pre-split data is provided.
    y : pd.Series
        Ignored when pre-split data is provided.
    model_type : str
        Classifier variant.
    imbalance_strategy : ImbalanceStrategy
        Strategy for handling class imbalance.
    test_size : float
        Ignored when pre-split data is provided.
    cv_folds : int
    cv_scoring : str
        Metric optimised during CV. Recommended: "f1", "roc_auc",
        "average_precision".
    cv_strategy : str
        Cross-validation strategy on the training set.
        - "temporal"
        - "stratified"
    random_state : int
    X_train, X_test, y_train, y_test : optional
        Pre-computed splits. When all four are
        provided the internal train_test_split is skipped entirely.
 
    Returns
    -------
    pipeline : fitted Pipeline/ImbPipeline
    metrics  : evaluation dict from evaluate()
    """
    if cv_strategy not in ("temporal", "stratified"):
        raise ValueError(
            f"cv_strategy must be 'temporal' or 'stratified', got '{cv_strategy}'."
        )
 
    pre_split = all(v is not None for v in (X_train, X_test, y_train, y_test))
 
    if pre_split:
        logger.info(
            "Using pre-computed split → train: %d | test: %d  "
            "(attack in train: %d, in test: %d)",
            len(X_train), len(X_test), y_train.sum(), y_test.sum(),
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state,
        )
        logger.info(
            "Random split → train: %d | test: %d  (attack in test: %d)",
            len(X_train), len(X_test), y_test.sum(),
        )
 
    pipeline = build_model(model_type, imbalance_strategy)
 
    # Cross-validation on train set
    if cv_strategy == "temporal":
        from sklearn.model_selection import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=cv_folds)
        logger.info(
            "CV strategy: TimeSeriesSplit (n_splits=%d) — temporal order preserved.",
            cv_folds,
        )
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        logger.info(
            "CV strategy: StratifiedKFold (n_splits=%d, shuffle=True).",
            cv_folds,
        )
 
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring=cv_scoring, n_jobs=-1
    )
    logger.info(
        "CV %s: %.4f ± %.4f", cv_scoring, cv_scores.mean(), cv_scores.std()
    )
 
    # Final fit — inject sample_weight for GradientBoosting + CLASS_WEIGHT
    fit_params = {}
    if (
        model_type == "gradient_boosting"
        and imbalance_strategy == ImbalanceStrategy.CLASS_WEIGHT
    ):
        sw = compute_sample_weight("balanced", y_train)
        fit_params["clf__sample_weight"] = sw
 
    pipeline.fit(X_train, y_train, **fit_params)
    logger.info("Model fitted.")
 
    metrics = evaluate(pipeline, X_test, y_test)
    return pipeline, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    pipeline: Pipeline | ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate the model on a labelled dataset.

    Parameters
    ----------
    threshold : float
        Decision threshold for the "attack" class.
        Lower → higher recall, more false positives.
        Tip: sweep from 0.2 to 0.7 to find the best operating point.
    """
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    has_both_classes = len(np.unique(y)) > 1

    metrics = {
        "accuracy":  float((y_pred == y).mean()),
        "f1_attack": float(f1_score(y, y_pred, pos_label=1, zero_division=0)),
        "f1_macro":  float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "roc_auc":   float(roc_auc_score(y, y_proba)) if has_both_classes else float("nan"),
        "pr_auc":    float(average_precision_score(y, y_proba)) if has_both_classes else float("nan"),
        "confusion_matrix":       confusion_matrix(y, y_pred).tolist(),
        "classification_report":  classification_report(
            y, y_pred, target_names=["non attack", "attack"], zero_division=0
        ),
        "threshold": threshold,
    }

    logger.info("── Evaluation (threshold=%.2f) ──────────────────", threshold)
    logger.info("Accuracy   : %.4f  ← not reliable for imbalanced data", metrics["accuracy"])
    logger.info("F1 (attack): %.4f  ← primary metric", metrics["f1_attack"])
    logger.info("F1 (macro) : %.4f", metrics["f1_macro"])
    logger.info("ROC-AUC    : %.4f", metrics["roc_auc"])
    logger.info("PR-AUC     : %.4f  ← best metric when attacks are rare", metrics["pr_auc"])
    logger.info("\n%s", metrics["classification_report"])

    return metrics


def find_best_threshold(
    pipeline: Pipeline | ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """
    Sweep decision thresholds and return the one maximising `metric`.

    Parameters
    ----------
    metric : str
        "f1", "recall", or "precision" on the attack class.
    thresholds : array-like, optional
        Values to try. Default: 0.05 to 0.95 step 0.05.

    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    y_proba = pipeline.predict_proba(X)[:, 1]
    best_t, best_score = 0.5, -1.0

    results = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y, y_pred, pos_label=1, zero_division=0)
        elif metric == "recall":
            from sklearn.metrics import recall_score
            score = recall_score(y, y_pred, pos_label=1, zero_division=0)
        elif metric == "precision":
            from sklearn.metrics import precision_score
            score = precision_score(y, y_pred, pos_label=1, zero_division=0)
        else:
            raise ValueError(f"Unknown metric '{metric}'.")
        results.append((t, score))
        if score > best_score:
            best_score, best_t = score, t

    logger.info("Best threshold (by %s): %.2f → %.4f", metric, best_t, best_score)
    return float(best_t), float(best_score)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    pipeline: Pipeline | ImbPipeline,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Predict attack / non attack for each day in X.

    """
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pd.DataFrame(
        {
            "label":       [LABEL_MAP[p] for p in pred],
            "probability": proba,
            "prediction":  pred,
        },
        index=X.index,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(pipeline: Pipeline | ImbPipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Model saved to '%s'.", path)


def load_model(path: str | Path) -> Pipeline | ImbPipeline:
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    logger.info("Model loaded from '%s'.", path)
    return pipeline


# ---------------------------------------------------------------------------
# Feature importance (tree-based models only)
# ---------------------------------------------------------------------------

def feature_importance(
    pipeline: Pipeline | ImbPipeline,
    feature_names: Optional[list[str]] = None,
    top_n: int = 20,
) -> pd.DataFrame:
    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    names = feature_names or ALL_FEATURES
    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )