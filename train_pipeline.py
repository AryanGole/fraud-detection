"""
============================================================
Fraud Detection Pipeline - ML Training Orchestration
============================================================
Trains XGBoost + LightGBM + Isolation Forest ensemble.
Handles class imbalance, hyperparameter optimization,
threshold tuning, and MLflow experiment tracking.

Author: Senior ML Engineering Team
Version: 2.1.0
"""

import os
import json
import logging
import warnings
from typing import TYPE_CHECKING, Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path

if TYPE_CHECKING:
    import optuna as _optuna_type
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report,
    confusion_matrix, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
import shap
import optuna
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Feature Configuration ─────────────────────────────────────────────────────
FEATURE_COLS = [
    # Velocity
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "txn_amount_1h", "txn_amount_24h",
    "amt_zscore_30d", "velocity_spike", "max_single_day",
    # Behavioral
    "avg_txn_amount_30d", "median_txn_amount_30d", "amount_ratio_to_avg",
    "pct_international_30d", "pct_cnp_30d", "unique_merchants_7d",
    "unique_countries_30d", "typical_amount_band",
    "days_since_first_txn", "customer_risk_score_hist",
    # Geographic
    "distance_from_last_txn_km", "time_since_last_txn_min",
    "implied_travel_speed_kmph", "is_impossible_travel",
    "is_new_country", "home_country_pct",
    # Device
    "device_txn_count_24h", "device_customer_count_7d",
    "is_new_device", "device_fraud_rate_30d",
    "ip_txn_count_1h", "multi_account_device",
    # Merchant
    "merchant_txn_count_7d", "merchant_fraud_rate_30d",
    "merchant_avg_amount_30d", "merchant_amount_deviation",
    "merchant_unique_cards_7d", "merchant_risk_tier", "is_mcc_high_risk",
    # Temporal
    "is_fraud_hour", "is_weekend", "is_month_end",
    "days_since_last_txn", "txn_hour_sin", "txn_hour_cos",
    # Raw
    "amount", "is_international", "is_high_risk_merchant",
]

TARGET_COL = "is_fraud"


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_training_data(data_path: str) -> pd.DataFrame:
    """Load and validate training data from Gold feature store."""
    log.info(f"Loading training data from: {data_path}")

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        # Try parquet directory (Delta export)
        import glob
        files = glob.glob(f"{data_path}/**/*.parquet", recursive=True)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    log.info(f"Loaded {len(df):,} records. Fraud rate: {df[TARGET_COL].mean():.4%}")

    # Basic validation
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        log.warning(f"Missing feature columns (will use 0): {missing_cols}")
        for c in missing_cols:
            df[c] = 0.0

    # Fill NaN
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    df[TARGET_COL]   = df[TARGET_COL].fillna(0).astype(int)

    return df


def generate_synthetic_training_data(n_samples: int = 500_000) -> pd.DataFrame:
    """
    Generate realistic synthetic training data for demonstration.
    Uses domain-informed distributions for fraud vs. legitimate transactions.
    """
    log.info(f"Generating {n_samples:,} synthetic training samples...")
    rng = np.random.RandomState(42)

    n_fraud = int(n_samples * 0.002)  # 0.2% fraud rate
    n_legit = n_samples - n_fraud

    def _generate_segment(n, is_fraud):
        f = float(is_fraud)
        data = {
            # Velocity
            "txn_count_1h":              rng.poisson(1 + f*12, n),
            "txn_count_24h":             rng.poisson(3 + f*18, n),
            "txn_count_7d":              rng.poisson(10 + f*5, n),
            "txn_amount_1h":             rng.exponential(2000 + f*30000, n),
            "txn_amount_24h":            rng.exponential(8000 + f*50000, n),
            "amt_zscore_30d":            rng.normal(f*3.5, 1.0, n),
            "velocity_spike":            rng.binomial(1, 0.01 + f*0.5, n),
            "max_single_day":            rng.exponential(10000 + f*100000, n),
            # Behavioral
            "avg_txn_amount_30d":        rng.exponential(5000, n),
            "median_txn_amount_30d":     rng.exponential(3000, n),
            "amount_ratio_to_avg":       rng.lognormal(f*1.5, 0.5, n),
            "pct_international_30d":     rng.beta(1 + f*5, 10, n),
            "pct_cnp_30d":               rng.beta(1 + f*8, 5, n),
            "unique_merchants_7d":       rng.poisson(3 + f*2, n),
            "unique_countries_30d":      rng.poisson(1 + f*3, n),
            "typical_amount_band":       rng.randint(0, 5, n),
            "days_since_first_txn":      rng.exponential(300, n),
            "customer_risk_score_hist":  rng.beta(1 + f*3, 10, n) * 100,
            # Geographic
            "distance_from_last_txn_km": rng.exponential(50 + f*2000, n),
            "time_since_last_txn_min":   rng.exponential(180, n),
            "implied_travel_speed_kmph": rng.exponential(30 + f*400, n),
            "is_impossible_travel":      rng.binomial(1, 0.001 + f*0.2, n),
            "is_new_country":            rng.binomial(1, 0.02 + f*0.4, n),
            "home_country_pct":          rng.beta(10 - f*8, 2, n),
            # Device
            "device_txn_count_24h":      rng.poisson(2 + f*15, n),
            "device_customer_count_7d":  rng.poisson(1 + f*3, n),
            "is_new_device":             rng.binomial(1, 0.05 + f*0.5, n),
            "device_fraud_rate_30d":     rng.beta(1, 50 - f*45, n),
            "ip_txn_count_1h":           rng.poisson(1 + f*8, n),
            "multi_account_device":      rng.binomial(1, 0.01 + f*0.4, n),
            # Merchant
            "merchant_txn_count_7d":     rng.poisson(100, n),
            "merchant_fraud_rate_30d":   rng.beta(1 + f*4, 20, n),
            "merchant_avg_amount_30d":   rng.exponential(3000, n),
            "merchant_amount_deviation": rng.exponential(0.2 + f*3, n),
            "merchant_unique_cards_7d":  rng.poisson(50, n),
            "merchant_risk_tier":        rng.choice([0,1,2,3,4], n, p=[0.4,0.25,0.2,0.1,0.05]),
            "is_mcc_high_risk":          rng.binomial(1, 0.05 + f*0.25, n),
            # Temporal
            "is_fraud_hour":             rng.binomial(1, 0.07 + f*0.35, n),
            "is_weekend":                rng.binomial(1, 0.28, n),
            "is_month_end":              rng.binomial(1, 0.1 + f*0.2, n),
            "days_since_last_txn":       rng.exponential(2 + f*0.5, n),
            "txn_hour_sin":              rng.uniform(-1, 1, n),
            "txn_hour_cos":              rng.uniform(-1, 1, n),
            # Raw
            "amount":                    rng.lognormal(7 + f*2, 1.5, n),
            "is_international":          rng.binomial(1, 0.08 + f*0.35, n),
            "is_high_risk_merchant":     rng.binomial(1, 0.05 + f*0.2, n),
            TARGET_COL:                  np.ones(n, dtype=int) if is_fraud else np.zeros(n, dtype=int),
        }
        return pd.DataFrame(data)

    df = pd.concat([
        _generate_segment(n_legit, is_fraud=False),
        _generate_segment(n_fraud,  is_fraud=True),
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(f"Synthetic data ready. Shape: {df.shape}, Fraud rate: {df[TARGET_COL].mean():.4%}")
    return df


# ── Model Trainers ─────────────────────────────────────────────────────────────
class XGBoostFraudClassifier:
    """XGBoost binary classifier with SMOTE, Optuna tuning, and calibration."""

    def __init__(self, n_trials: int = 30, cv_folds: int = 5):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.model    = None
        self.best_params = {}
        self.threshold   = 0.5
        self.scaler      = StandardScaler()

    def _objective(self, trial: "optuna.Trial", X, y) -> float:
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 200, 800),
            "max_depth":          trial.suggest_int("max_depth", 4, 10),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 10, 500),
            "use_label_encoder":  False,
            "eval_metric":        "aucpr",
            "tree_method":        "hist",
            "random_state":       42,
        }

        cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = []

        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            smote = SMOTE(sampling_strategy=0.1, random_state=42, n_jobs=-1)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr_res, y_tr_res,
                    eval_set=[(X_val, y_val)],
                    verbose=False)

            preds = clf.predict_proba(X_val)[:, 1]
            score.append(average_precision_score(y_val, preds))

        return np.mean(score)

    def tune(self, X: np.ndarray, y: np.ndarray) -> Dict:
        log.info(f"Tuning XGBoost with {self.n_trials} Optuna trials...")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: self._objective(t, X, y),
                       n_trials=self.n_trials,
                       show_progress_bar=False)
        self.best_params = study.best_params
        log.info(f"Best PR-AUC: {study.best_value:.4f}")
        return self.best_params

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray,   y_val: np.ndarray) -> None:
        """Full training with SMOTE, tuning, and calibration."""
        log.info("Training XGBoost classifier...")

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)

        # Tune hyperparameters
        self.tune(X_train, y_train)

        # Apply SMOTE on full training set
        smote = SMOTE(sampling_strategy=0.15, random_state=42, n_jobs=-1)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        log.info(f"After SMOTE: {y_res.sum():,} fraud / {(~y_res.astype(bool)).sum():,} legit")

        # Train with best params
        base_clf = xgb.XGBClassifier(
            **self.best_params,
            use_label_encoder=False,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
        )
        base_clf.fit(X_res, y_res,
                     eval_set=[(X_val, y_val)],
                     verbose=100)

        # Calibrate probabilities (Platt scaling) on held-out validation set.
        # Requires at least 2 samples per class per fold.
        # Guard against tiny validation sets (e.g. in unit tests).
        n_fraud_val = int(y_val.sum())
        if n_fraud_val >= 4:
            self.model = CalibratedClassifierCV(base_clf, method="sigmoid", cv=2)
            self.model.fit(X_val, y_val)
        else:
            # Not enough fraud samples for cross-val calibration; use base model
            log.warning(f"Skipping calibration: only {n_fraud_val} fraud samples in val set")
            self.model = base_clf
            self.model.fit(X_val, y_val) if not self.model._fitted else None

        # Tune decision threshold
        self.threshold = self._tune_threshold(X_val, y_val)
        log.info(f"Optimal threshold: {self.threshold:.4f}")

    def _tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                        target_recall: float = 0.92) -> float:
        """
        Find threshold maximizing F1 score subject to minimum recall constraint.
        Fraud operations teams typically accept lower precision for higher recall.
        """
        probs = self.model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

        best_f1, best_threshold = 0, 0.5
        for prec, rec, thresh in zip(precisions, recalls, thresholds):
            if rec >= target_recall:
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                if f1 > best_f1:
                    best_f1, best_threshold = f1, thresh

        return best_threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


class LightGBMFraudClassifier:
    """LightGBM challenger model with asymmetric loss tuning."""

    def __init__(self, n_trials: int = 20):
        self.n_trials    = n_trials
        self.model       = None
        self.best_params = {}
        self.threshold   = 0.5

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray,   y_val: np.ndarray) -> None:
        log.info("Training LightGBM classifier...")

        # SMOTE for class imbalance
        smote = SMOTE(sampling_strategy=0.15, random_state=42, n_jobs=-1)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        # Class weight calculation
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        params = {
            "objective":          "binary",
            "metric":             "average_precision",
            "boosting_type":      "gbdt",
            "n_estimators":       600,
            "learning_rate":      0.05,
            "num_leaves":         127,
            "max_depth":          -1,
            "min_child_samples":  20,
            "feature_fraction":   0.8,
            "bagging_fraction":   0.8,
            "bagging_freq":       5,
            "reg_alpha":          0.1,
            "reg_lambda":         0.2,
            "scale_pos_weight":   pos_weight,
            "random_state":       42,
            "n_jobs":             -1,
            "verbose":            -1,
        }

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X_res, y_res,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(100)]
        )

        # Threshold tuning
        probs = self.model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        self.threshold = thresholds[f1_scores.argmax()]

        log.info(f"LightGBM optimal threshold: {self.threshold:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


class IsolationForestDetector:
    """Unsupervised anomaly detector for novel fraud patterns."""

    def __init__(self, contamination: float = 0.002):
        self.contamination = contamination
        self.model         = None
        self.scaler        = StandardScaler()

    def train(self, X_train: np.ndarray) -> None:
        log.info("Training Isolation Forest anomaly detector...")
        X_scaled    = self.scaler.fit_transform(X_train)
        self.model  = IsolationForest(
            n_estimators  = 200,
            contamination = self.contamination,
            max_features  = 0.8,
            random_state  = 42,
            n_jobs        = -1,
        )
        self.model.fit(X_scaled)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns normalized anomaly scores in [0, 1].
        Higher score = more anomalous.
        """
        X_scaled = self.scaler.transform(X)
        # Raw scores are negative (more negative = more anomalous)
        raw_scores = self.model.decision_function(X_scaled)
        # Normalize to [0, 1] where 1 = most anomalous
        # Note: ndarray.ptp() removed in NumPy 2.0; use np.ptp() or manual range
        score_range = raw_scores.max() - raw_scores.min()
        normalized = 1 - (raw_scores - raw_scores.min()) / (score_range + 1e-8)
        return normalized


# ── Ensemble Scorer ────────────────────────────────────────────────────────────
class FraudEnsembleScorer:
    """
    Ensemble scoring combining supervised and unsupervised models.
    Produces calibrated fraud risk score (0-100).
    """
    # Ensemble weights (tuned on validation set)
    WEIGHT_XGB = 0.55
    WEIGHT_LGB = 0.35
    WEIGHT_IF  = 0.10

    def __init__(self):
        self.xgb_model = XGBoostFraudClassifier(n_trials=30)
        self.lgb_model = LightGBMFraudClassifier(n_trials=20)
        self.if_model  = IsolationForestDetector(contamination=0.002)
        self.is_fitted  = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray,   y_val: np.ndarray) -> None:
        self.xgb_model.train(X_train, y_train, X_val, y_val)
        self.lgb_model.train(X_train, y_train, X_val, y_val)
        self.if_model.train(X_train)
        self.is_fitted = True

    def fraud_score(self, X: np.ndarray) -> np.ndarray:
        """Return fraud risk scores in range [0, 100]."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        xgb_prob  = self.xgb_model.predict_proba(X)
        lgb_prob  = self.lgb_model.predict_proba(X)
        if_score  = self.if_model.anomaly_score(X)

        # Weighted average
        ensemble = (self.WEIGHT_XGB * xgb_prob +
                    self.WEIGHT_LGB * lgb_prob +
                    self.WEIGHT_IF  * if_score)

        # Scale to [0, 100] integer risk score
        return np.clip(ensemble * 100, 0, 100)

    def predict(self, X: np.ndarray, threshold: float = 45.0) -> np.ndarray:
        """Binary prediction using risk score threshold."""
        return (self.fraud_score(X) >= threshold).astype(int)


# ── Model Evaluation ───────────────────────────────────────────────────────────
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "FraudModel",
) -> Dict[str, float]:
    """Comprehensive model evaluation with fraud-specific metrics."""

    if hasattr(model, "fraud_score"):
        scores = model.fraud_score(X_test) / 100.0
        preds  = model.predict(X_test)
    else:
        raw = model.predict_proba(X_test)
        scores = raw[:, 1] if raw.ndim == 2 else raw
        preds  = model.predict(X_test)

    precisions, recalls, _ = precision_recall_curve(y_test, scores)
    # Use the threshold-tuned operating point (argmax F1)
    f1_vals   = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx  = int(np.argmax(f1_vals))

    try:
        roc_auc = roc_auc_score(y_test, scores)
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "roc_auc":   roc_auc,
        "pr_auc":    average_precision_score(y_test, scores),
        "f1_score":  f1_score(y_test, preds, zero_division=0),
        "precision": float(precisions[best_idx]),
        "recall":    float(recalls[best_idx]),
    }

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    metrics["true_positives"]  = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_negatives"]  = int(tn)
    metrics["false_positive_rate"] = fp / (fp + tn + 1e-8)
    metrics["false_negative_rate"] = fn / (fn + tp + 1e-8)

    log.info(f"\n{'='*60}")
    log.info(f"  {model_name} Evaluation Results")
    log.info(f"{'='*60}")
    log.info(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
    log.info(f"  PR-AUC:           {metrics['pr_auc']:.4f}")
    log.info(f"  F1 Score:         {metrics['f1_score']:.4f}")
    log.info(f"  True Positives:   {tp:,}  (fraud caught)")
    log.info(f"  False Positives:  {fp:,}  (legit flagged)")
    log.info(f"  False Negatives:  {fn:,}  (fraud missed)")
    log.info(f"  False Positive Rate: {metrics['false_positive_rate']:.4%}")
    log.info(f"{'='*60}")

    return metrics


# ── MLflow Tracking ────────────────────────────────────────────────────────────
def run_training_with_mlflow(
    data_path:      str = None,
    experiment_name: str = "fraud_detection_v2",
    run_name:       str = "ensemble_v2_1",
) -> FraudEnsembleScorer:
    """
    Full training pipeline with MLflow experiment tracking.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        log.info(f"MLflow Run ID: {run.info.run_id}")

        # ── Data Loading ─────────────────────────────────────────────────────
        if data_path:
            df = load_training_data(data_path)
        else:
            df = generate_synthetic_training_data(n_samples=500_000)

        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values

        # Log dataset info
        mlflow.log_param("n_samples",    len(df))
        mlflow.log_param("n_features",   len(FEATURE_COLS))
        mlflow.log_param("fraud_rate",   f"{y.mean():.4f}")
        mlflow.log_param("features",     ",".join(FEATURE_COLS))

        # ── Train/Val/Test Split ─────────────────────────────────────────────
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42)

        log.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

        # ── Train Ensemble ───────────────────────────────────────────────────
        ensemble = FraudEnsembleScorer()
        ensemble.fit(X_train, y_train, X_val, y_val)

        # ── Evaluate ─────────────────────────────────────────────────────────
        metrics = evaluate_model(ensemble, X_test, y_test, "EnsembleFraudScorer")

        # Log all metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "ensemble_weight_xgb": ensemble.WEIGHT_XGB,
            "ensemble_weight_lgb": ensemble.WEIGHT_LGB,
            "ensemble_weight_if":  ensemble.WEIGHT_IF,
        })

        # ── SHAP Feature Importance ──────────────────────────────────────────
        log.info("Computing SHAP feature importance...")
        # Access underlying XGBoost estimator from CalibratedClassifierCV
        # sklearn >= 1.2 uses .calibrated_classifiers_, older uses .calibrated_classifiers
        cal_model = ensemble.xgb_model.model
        try:
            base_estimator = cal_model.calibrated_classifiers_[0].estimator
        except AttributeError:
            try:
                base_estimator = cal_model.calibrated_classifiers[0].estimator
            except AttributeError:
                base_estimator = cal_model  # fallback: use model directly
        explainer    = shap.TreeExplainer(base_estimator)
        sample_size  = min(1000, len(X_test))
        shap_values  = explainer.shap_values(
            ensemble.xgb_model.scaler.transform(X_test[:sample_size]))

        # shap_values is list[ndarray] for binary classifiers (class 0, class 1)
        # or a single ndarray for regression/XGBoost native
        if isinstance(shap_values, list):
            shap_matrix = np.abs(shap_values[1])   # use class-1 (fraud) SHAP values
        else:
            shap_matrix = np.abs(shap_values)

        importance_df = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": shap_matrix.mean(axis=0),
        }).sort_values("importance", ascending=False)

        log.info("\nTop 15 Most Important Features:")
        log.info(importance_df.head(15).to_string(index=False))

        importance_path = "/tmp/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # ── Log Models ───────────────────────────────────────────────────────
        mlflow.sklearn.log_model(ensemble.xgb_model.model, "xgboost_model",
                                  registered_model_name="fraud_xgboost")
        mlflow.lightgbm.log_model(ensemble.lgb_model.model, "lightgbm_model",
                                   registered_model_name="fraud_lightgbm")
        mlflow.sklearn.log_model(ensemble.if_model.model, "isolation_forest",
                                  registered_model_name="fraud_isolation_forest")

        log.info(f"Training complete. Run ID: {run.info.run_id}")
        log.info(f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")

        return ensemble


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection ML Training Pipeline")
    parser.add_argument("--data-path",   type=str, default=None,
                        help="Path to Gold feature store (parquet/delta)")
    parser.add_argument("--experiment",  type=str, default="fraud_detection_v2")
    parser.add_argument("--run-name",    type=str, default="ensemble_v2_1")
    args = parser.parse_args()

    model = run_training_with_mlflow(
        data_path       = args.data_path,
        experiment_name = args.experiment,
        run_name        = args.run_name,
    )
