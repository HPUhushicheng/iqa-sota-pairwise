from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainConfig:
    n_splits: int = 5
    random_state: int = 42
    use_lr: bool = True
    use_hgb: bool = True
    use_xgboost: bool = True
    lr_c: float = 0.2
    hgb_max_depth: int = 3
    hgb_lr: float = 0.05
    hgb_max_iter: int = 400


@dataclass
class ModelBundle:
    config: dict[str, Any]
    base_models: dict[str, Any]
    base_order: list[str]
    blender: Optional[Any]
    threshold: float
    cv_report: dict[str, Any]


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def _pred_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        if p.ndim == 2:
            return p[:, 1]
        return p.astype(np.float64)
    if hasattr(model, "decision_function"):
        z = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-z))
    y = model.predict(x)
    return y.astype(np.float64)


def _build_estimators(cfg: TrainConfig) -> dict[str, Any]:
    estimators: dict[str, Any] = {}

    if cfg.use_lr:
        estimators["lr"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=cfg.lr_c,
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=cfg.random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        )

    if cfg.use_hgb:
        estimators["hgb"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=cfg.hgb_max_depth,
                        learning_rate=cfg.hgb_lr,
                        max_iter=cfg.hgb_max_iter,
                        min_samples_leaf=4,
                        l2_regularization=1.0,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )

    if cfg.use_xgboost:
        try:
            from xgboost import XGBClassifier

            estimators["xgb"] = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=500,
                            max_depth=3,
                            learning_rate=0.03,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_lambda=2.0,
                            min_child_weight=3,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            random_state=cfg.random_state,
                        ),
                    ),
                ]
            )
        except Exception:
            # XGBoost is optional; keep pipeline runnable without it.
            pass

    if not estimators:
        raise ValueError("No base estimator enabled.")
    return estimators


def _optimize_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float, float]:
    best_t = 0.5
    best_bal = -1.0
    best_acc = -1.0
    for t in np.linspace(0.2, 0.8, 241):
        pred = (prob >= t).astype(np.int64)
        bal = balanced_accuracy_score(y, pred)
        acc = accuracy_score(y, pred)
        if bal > best_bal or (abs(bal - best_bal) < 1e-12 and acc > best_acc):
            best_bal = float(bal)
            best_acc = float(acc)
            best_t = float(t)
    return best_t, best_bal, best_acc


def train_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: TrainConfig,
) -> tuple[ModelBundle, dict[str, np.ndarray], np.ndarray]:
    if X.shape[0] != y.shape[0] or X.shape[0] != groups.shape[0]:
        raise ValueError("X/y/groups size mismatch.")

    estimators = _build_estimators(cfg)
    base_order = list(estimators.keys())

    cv = StratifiedGroupKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    n = X.shape[0]
    oof_base: dict[str, np.ndarray] = {name: np.zeros(n, dtype=np.float64) for name in base_order}
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), start=1):
        x_tr, x_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        one_fold: dict[str, Any] = {
            "fold": fold_idx,
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
            "models": {},
        }

        for name, est in estimators.items():
            model = clone(est)
            model.fit(x_tr, y_tr)
            p = _pred_proba(model, x_va)
            oof_base[name][va_idx] = p

            pred = (p >= 0.5).astype(np.int64)
            one_fold["models"][name] = {
                "acc@0.5": float(accuracy_score(y_va, pred)),
                "bal_acc@0.5": float(balanced_accuracy_score(y_va, pred)),
                "f1@0.5": float(f1_score(y_va, pred, zero_division=0)),
                "auc": _safe_auc(y_va, p),
            }

        fold_reports.append(one_fold)

    # Build blender with OOF predictions.
    oof_stack = np.column_stack([oof_base[name] for name in base_order])
    blender = None
    if oof_stack.shape[1] >= 2:
        blender = LogisticRegression(
            C=1.0,
            max_iter=3000,
            random_state=cfg.random_state,
            solver="lbfgs",
        )
        blender.fit(oof_stack, y)
        oof_final = blender.predict_proba(oof_stack)[:, 1]
    else:
        oof_final = oof_stack[:, 0]

    threshold, best_bal, best_acc = _optimize_threshold(y, oof_final)
    oof_pred = (oof_final >= threshold).astype(np.int64)

    full_models: dict[str, Any] = {}
    for name, est in estimators.items():
        model = clone(est)
        model.fit(X, y)
        full_models[name] = model

    per_model_report: dict[str, Any] = {}
    for name in base_order:
        p = oof_base[name]
        pred = (p >= 0.5).astype(np.int64)
        per_model_report[name] = {
            "acc@0.5": float(accuracy_score(y, pred)),
            "bal_acc@0.5": float(balanced_accuracy_score(y, pred)),
            "f1@0.5": float(f1_score(y, pred, zero_division=0)),
            "auc": _safe_auc(y, p),
        }

    cv_report = {
        "config": asdict(cfg),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "base_models": base_order,
        "per_model_oof": per_model_report,
        "blended_oof": {
            "threshold": threshold,
            "acc": float(accuracy_score(y, oof_pred)),
            "bal_acc": float(balanced_accuracy_score(y, oof_pred)),
            "f1": float(f1_score(y, oof_pred, zero_division=0)),
            "auc": _safe_auc(y, oof_final),
            "best_bal_acc": best_bal,
            "best_acc": best_acc,
        },
        "folds": fold_reports,
    }

    bundle = ModelBundle(
        config=asdict(cfg),
        base_models=full_models,
        base_order=base_order,
        blender=blender,
        threshold=threshold,
        cv_report=cv_report,
    )
    return bundle, oof_base, oof_final


def predict_proba(bundle: ModelBundle, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per_model: dict[str, np.ndarray] = {}
    for name in bundle.base_order:
        per_model[name] = _pred_proba(bundle.base_models[name], X)

    stack = np.column_stack([per_model[name] for name in bundle.base_order])
    if bundle.blender is not None:
        p = bundle.blender.predict_proba(stack)[:, 1]
    else:
        p = stack[:, 0]
    return p.astype(np.float64), per_model


def proba_to_label(prob_a: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(prob_a >= threshold, "A", "B")
