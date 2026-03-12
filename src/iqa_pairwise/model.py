from __future__ import annotations

import time
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
    verbose: int = 1
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
    blend_method: str = "logreg"
    blend_weights: Optional[list[float]] = None


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


def _fit_with_sample_weight(model: Any, x: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> None:
    if sample_weight is None:
        model.fit(x, y)
        return
    sw = np.asarray(sample_weight, dtype=np.float64)
    # Most estimators are wrapped in a Pipeline with final step name "clf".
    try:
        model.fit(x, y, clf__sample_weight=sw)
        return
    except Exception:
        pass
    # Fallback for plain estimators.
    try:
        model.fit(x, y, sample_weight=sw)
        return
    except Exception:
        model.fit(x, y)


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
                            verbosity=0,
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


def _log(verbose: int, msg: str) -> None:
    if verbose > 0:
        print(msg, flush=True)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.clip(weights.astype(np.float64), 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return np.ones_like(w) / float(len(w))
    return w / s


def _grid_weight_candidates(n_models: int, step: float = 0.05) -> list[np.ndarray]:
    if n_models == 1:
        return [np.array([1.0], dtype=np.float64)]
    if n_models == 2:
        n = int(round(1.0 / step))
        return [
            np.array([i * step, 1.0 - i * step], dtype=np.float64)
            for i in range(n + 1)
        ]
    if n_models == 3:
        n = int(round(1.0 / step))
        cands: list[np.ndarray] = []
        for i in range(n + 1):
            w0 = i * step
            for j in range(n + 1 - i):
                w1 = j * step
                w2 = 1.0 - w0 - w1
                cands.append(np.array([w0, w1, w2], dtype=np.float64))
        return cands
    return []


def _optimize_blend_weights(
    oof_stack: np.ndarray,
    y: np.ndarray,
    random_state: int,
    verbose: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    n_models = oof_stack.shape[1]
    grid = _grid_weight_candidates(n_models=n_models, step=0.05)
    if not grid:
        rng = np.random.default_rng(random_state)
        grid = [rng.dirichlet(np.ones(n_models)).astype(np.float64) for _ in range(6000)]
        # Keep single-model candidates.
        for i in range(n_models):
            one = np.zeros(n_models, dtype=np.float64)
            one[i] = 1.0
            grid.append(one)

    best_weights = _normalize_weights(grid[0])
    best_prob = np.clip(oof_stack @ best_weights, 0.0, 1.0)
    best_t, best_bal, best_acc = _optimize_threshold(y, best_prob)

    for w in grid[1:]:
        ww = _normalize_weights(w)
        p = np.clip(oof_stack @ ww, 0.0, 1.0)
        t, bal, acc = _optimize_threshold(y, p)
        if bal > best_bal or (abs(bal - best_bal) < 1e-12 and acc > best_acc):
            best_weights = ww
            best_prob = p
            best_t = t
            best_bal = bal
            best_acc = acc

    _log(
        verbose,
        (
            "[CV] Best weighted blend "
            f"| weights={best_weights.round(4).tolist()} "
            f"| threshold={best_t:.3f} bal_acc={best_bal:.4f} acc={best_acc:.4f}"
        ),
    )
    return best_weights, best_prob, best_t, best_bal, best_acc


def train_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: TrainConfig,
    stratify_labels: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[ModelBundle, dict[str, np.ndarray], np.ndarray]:
    if X.shape[0] != y.shape[0] or X.shape[0] != groups.shape[0]:
        raise ValueError("X/y/groups size mismatch.")
    if stratify_labels is not None and stratify_labels.shape[0] != X.shape[0]:
        raise ValueError("stratify_labels size mismatch.")
    sw_all: Optional[np.ndarray] = None
    if sample_weight is not None:
        sw_all = np.asarray(sample_weight, dtype=np.float64)
        if sw_all.shape[0] != X.shape[0]:
            raise ValueError("sample_weight size mismatch.")

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
    split_y = stratify_labels if stratify_labels is not None else y

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, split_y, groups), start=1):
        _log(
            cfg.verbose,
            f"[CV] Fold {fold_idx}/{cfg.n_splits} | train={len(tr_idx)} valid={len(va_idx)}",
        )
        x_tr, x_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        one_fold: dict[str, Any] = {
            "fold": fold_idx,
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
            "models": {},
        }

        for name, est in estimators.items():
            _log(cfg.verbose, f"[CV] Fold {fold_idx}/{cfg.n_splits} -> fitting {name} ...")
            t0 = time.perf_counter()
            model = clone(est)
            sw_tr = sw_all[tr_idx] if sw_all is not None else None
            _fit_with_sample_weight(model, x_tr, y_tr, sw_tr)
            elapsed = time.perf_counter() - t0
            p = _pred_proba(model, x_va)
            oof_base[name][va_idx] = p

            pred = (p >= 0.5).astype(np.int64)
            fold_acc = float(accuracy_score(y_va, pred))
            fold_bal_acc = float(balanced_accuracy_score(y_va, pred))
            fold_f1 = float(f1_score(y_va, pred, zero_division=0))
            fold_auc = _safe_auc(y_va, p)
            one_fold["models"][name] = {
                "acc@0.5": fold_acc,
                "bal_acc@0.5": fold_bal_acc,
                "f1@0.5": fold_f1,
                "auc": fold_auc,
            }
            _log(
                cfg.verbose,
                (
                    f"[CV] Fold {fold_idx}/{cfg.n_splits} -> {name} done in {elapsed:.1f}s "
                    f"| acc={fold_acc:.4f} bal_acc={fold_bal_acc:.4f} f1={fold_f1:.4f} "
                    f"auc={fold_auc if fold_auc is not None else 'NA'}"
                ),
            )

        fold_reports.append(one_fold)

    # Build final blend from OOF predictions:
    #   1) logreg blender
    #   2) non-negative weighted average via grid/random search
    # Pick whichever gives better OOF balanced accuracy.
    _log(cfg.verbose, "[CV] Building OOF blend ...")
    oof_stack = np.column_stack([oof_base[name] for name in base_order])

    blender = None
    blend_method = "weighted_mean"
    blend_weights: Optional[list[float]] = None

    weighted_w, weighted_prob, weighted_t, weighted_bal, weighted_acc = _optimize_blend_weights(
        oof_stack=oof_stack,
        y=y,
        random_state=cfg.random_state,
        verbose=cfg.verbose,
    )

    candidate_report: dict[str, Any] = {
        "weighted_mean": {
            "weights": weighted_w.tolist(),
            "threshold": float(weighted_t),
            "best_bal_acc": float(weighted_bal),
            "best_acc": float(weighted_acc),
            "auc": _safe_auc(y, weighted_prob),
        }
    }

    selected_prob = weighted_prob
    threshold = weighted_t
    best_bal = weighted_bal
    best_acc = weighted_acc
    blend_weights = weighted_w.tolist()

    if oof_stack.shape[1] >= 2:
        _log(cfg.verbose, "[CV] Fitting logreg blender on OOF predictions ...")
        logreg_blender = LogisticRegression(
            C=1.0,
            max_iter=3000,
            random_state=cfg.random_state,
            solver="lbfgs",
        )
        logreg_blender.fit(oof_stack, y)
        logreg_prob = logreg_blender.predict_proba(oof_stack)[:, 1]
        logreg_t, logreg_bal, logreg_acc = _optimize_threshold(y, logreg_prob)
        candidate_report["logreg"] = {
            "threshold": float(logreg_t),
            "best_bal_acc": float(logreg_bal),
            "best_acc": float(logreg_acc),
            "auc": _safe_auc(y, logreg_prob),
        }

        if logreg_bal > best_bal or (abs(logreg_bal - best_bal) < 1e-12 and logreg_acc > best_acc):
            blender = logreg_blender
            blend_method = "logreg"
            blend_weights = None
            selected_prob = logreg_prob
            threshold = logreg_t
            best_bal = logreg_bal
            best_acc = logreg_acc

    oof_final = selected_prob
    oof_pred = (oof_final >= threshold).astype(np.int64)
    _log(
        cfg.verbose,
        (
            f"[CV] OOF done | method={blend_method} threshold={threshold:.3f} "
            f"best_bal_acc={best_bal:.4f} best_acc={best_acc:.4f}"
        ),
    )

    full_models: dict[str, Any] = {}
    for name, est in estimators.items():
        _log(cfg.verbose, f"[FIT] Training full model: {name} ...")
        t0 = time.perf_counter()
        model = clone(est)
        _fit_with_sample_weight(model, X, y, sw_all)
        elapsed = time.perf_counter() - t0
        full_models[name] = model
        _log(cfg.verbose, f"[FIT] {name} done in {elapsed:.1f}s")

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
        "split_target": "stratify_labels" if stratify_labels is not None else "label",
        "base_models": base_order,
        "per_model_oof": per_model_report,
        "blend_candidates": candidate_report,
        "blended_oof": {
            "method": blend_method,
            "weights": blend_weights,
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
        blend_method=blend_method,
        blend_weights=blend_weights,
        threshold=threshold,
        cv_report=cv_report,
    )
    return bundle, oof_base, oof_final


def predict_proba(bundle: ModelBundle, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per_model: dict[str, np.ndarray] = {}
    for name in bundle.base_order:
        per_model[name] = _pred_proba(bundle.base_models[name], X)

    stack = np.column_stack([per_model[name] for name in bundle.base_order])
    blend_method = getattr(bundle, "blend_method", "logreg")
    blend_weights = getattr(bundle, "blend_weights", None)

    if blend_method == "weighted_mean":
        if blend_weights is None:
            weights = np.ones(stack.shape[1], dtype=np.float64) / float(stack.shape[1])
        else:
            weights = _normalize_weights(np.array(blend_weights, dtype=np.float64))
        p = np.clip(stack @ weights, 0.0, 1.0)
    elif bundle.blender is not None:
        p = bundle.blender.predict_proba(stack)[:, 1]
    else:
        p = stack[:, 0]
    return p.astype(np.float64), per_model


def proba_to_label(prob_a: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(prob_a >= threshold, "A", "B")
