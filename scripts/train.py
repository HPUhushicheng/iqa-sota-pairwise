#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iqa_pairwise.data import load_many, parse_dataset_spec
from iqa_pairwise.features import ImageFeatureExtractor, build_pair_dataset
from iqa_pairwise.model import TrainConfig, predict_proba, proba_to_label, train_with_cv


def _sanitize_thread_env(verbose: int) -> None:
    keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for key in keys:
        val = os.environ.get(key)
        if val is None:
            continue
        try:
            n = int(val)
            if n <= 0:
                raise ValueError("not positive")
        except Exception:
            os.environ.pop(key, None)
            if verbose > 0:
                print(
                    f"[WARN] Invalid {key}={val!r}. Unset it to avoid libgomp issues.",
                    flush=True,
                )


def _parse_sources(values: list[str]) -> set[str]:
    return {v.strip() for v in values if v and v.strip()}


def _build_stratify_labels(y: np.ndarray, sources: np.ndarray, by_source: bool) -> np.ndarray:
    if not by_source:
        return y.astype(np.int64)
    uniq_src = sorted(set(sources.tolist()))
    src_to_idx = {s: i for i, s in enumerate(uniq_src)}
    src_idx = np.array([src_to_idx[s] for s in sources.tolist()], dtype=np.int64)
    return y.astype(np.int64) * 100 + src_idx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train pairwise IQA A/B classifier with small-data robust CV.")
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec: source:jsonl_path:image_root (repeatable)",
    )
    ap.add_argument("--artifact-dir", default="artifacts/run1", help="Output directory")

    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", type=int, default=1, help="0: quiet, 1: show fold/model progress")
    ap.add_argument(
        "--stratify-by-source",
        type=int,
        default=1,
        help="1: use (label,source) stratification for folds; 0: stratify by label only",
    )

    ap.add_argument("--use-handcrafted", type=int, default=1)
    ap.add_argument("--use-clip", type=int, default=1)
    ap.add_argument("--clip-model", default="vit_base_patch32_clip_224.openai")
    ap.add_argument("--device", default="cpu", help="cpu/cuda/mps")

    ap.add_argument("--use-pyiqa", type=int, default=0)
    ap.add_argument("--pyiqa-metrics", default="musiq,clipiqa")

    ap.add_argument("--include-raw", type=int, default=0, help="Include raw left/right vectors in pair feature")

    ap.add_argument("--use-lr", type=int, default=1)
    ap.add_argument("--use-hgb", type=int, default=1)
    ap.add_argument("--use-xgboost", type=int, default=1)

    ap.add_argument(
        "--trusted-source",
        action="append",
        default=[],
        help="Clean-label source name (repeatable), e.g. --trusted-source train1536",
    )
    ap.add_argument(
        "--noisy-source",
        action="append",
        default=[],
        help="Potentially noisy-label source name (repeatable), e.g. --noisy-source validNew",
    )
    ap.add_argument(
        "--denoise-action",
        default="none",
        choices=["none", "flip", "drop"],
        help="How to handle high-confidence label conflicts in noisy sources",
    )
    ap.add_argument(
        "--denoise-threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for denoise action (|p-0.5|*2)",
    )
    ap.add_argument(
        "--trusted-weight",
        type=float,
        default=1.0,
        help="Sample weight for trusted-source samples",
    )
    ap.add_argument(
        "--noisy-weight",
        type=float,
        default=0.6,
        help="Base sample weight for noisy-source samples",
    )
    ap.add_argument(
        "--flipped-weight",
        type=float,
        default=0.8,
        help="Sample weight after denoise flip",
    )
    ap.add_argument(
        "--teacher-verbose",
        type=int,
        default=0,
        help="Verbose level for teacher model used in denoise stage",
    )

    ap.add_argument("--no-progress", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    _sanitize_thread_env(verbose=args.verbose)

    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.verbose > 0:
        print(f"[Stage] Artifacts -> {artifact_dir}", flush=True)

    if args.verbose > 0:
        print("[Stage] Loading datasets ...", flush=True)
    specs = [parse_dataset_spec(s) for s in args.dataset]
    samples = load_many(specs=specs, require_label=True)
    if not samples:
        raise RuntimeError("No samples loaded.")
    if args.verbose > 0:
        print(f"[Stage] Loaded samples: {len(samples)}", flush=True)

    pyiqa_metrics = [m.strip() for m in args.pyiqa_metrics.split(",") if m.strip()]

    extractor = ImageFeatureExtractor(
        use_handcrafted=bool(args.use_handcrafted),
        use_clip=bool(args.use_clip),
        clip_model=args.clip_model,
        device=args.device,
        use_pyiqa=bool(args.use_pyiqa),
        pyiqa_metrics=pyiqa_metrics,
    )

    if args.verbose > 0:
        print("[Stage] Extracting pair features ...", flush=True)
    pair_ds = build_pair_dataset(
        samples=samples,
        extractor=extractor,
        include_raw=bool(args.include_raw),
        show_progress=not args.no_progress,
    )
    if pair_ds.y is None:
        raise RuntimeError("Training labels are missing.")

    cfg = TrainConfig(
        n_splits=args.n_splits,
        random_state=args.seed,
        verbose=args.verbose,
        use_lr=bool(args.use_lr),
        use_hgb=bool(args.use_hgb),
        use_xgboost=bool(args.use_xgboost),
    )

    y_original = pair_ds.y.astype(np.int64).copy()
    y_work = y_original.copy()
    n_total = int(len(y_work))
    keep_mask = np.ones(n_total, dtype=bool)
    corrected_mask = np.zeros(n_total, dtype=bool)
    sample_weight = np.ones(n_total, dtype=np.float64)

    trusted_sources = _parse_sources(args.trusted_source)
    noisy_sources = _parse_sources(args.noisy_source)
    if trusted_sources:
        trusted_mask = np.isin(pair_ds.sources, list(trusted_sources))
        sample_weight[trusted_mask] = float(args.trusted_weight)
    if noisy_sources:
        noisy_mask = np.isin(pair_ds.sources, list(noisy_sources))
        sample_weight[noisy_mask] = float(args.noisy_weight)

    denoise_report: dict[str, object] = {
        "trusted_sources": sorted(trusted_sources),
        "noisy_sources": sorted(noisy_sources),
        "denoise_action": args.denoise_action,
        "denoise_threshold": float(args.denoise_threshold),
        "n_total": n_total,
        "n_corrected": 0,
        "n_dropped": 0,
    }

    if noisy_sources and args.denoise_action != "none":
        if not trusted_sources:
            print(
                "[WARN] --denoise-action enabled but no --trusted-source provided; skip denoise.",
                flush=True,
            )
        else:
            trusted_mask = np.isin(pair_ds.sources, list(trusted_sources))
            noisy_mask = np.isin(pair_ds.sources, list(noisy_sources))
            if int(trusted_mask.sum()) < max(20, args.n_splits * 4):
                print(
                    "[WARN] Trusted samples too few for teacher denoise; skip denoise.",
                    flush=True,
                )
            else:
                if args.verbose > 0:
                    print(
                        f"[Stage] Denoise teacher training on trusted sources: {sorted(trusted_sources)}",
                        flush=True,
                    )
                teacher_cfg = TrainConfig(
                    n_splits=args.n_splits,
                    random_state=args.seed,
                    verbose=args.teacher_verbose,
                    use_lr=bool(args.use_lr),
                    use_hgb=bool(args.use_hgb),
                    use_xgboost=bool(args.use_xgboost),
                )
                teacher_strat = _build_stratify_labels(
                    y=y_work[trusted_mask],
                    sources=pair_ds.sources[trusted_mask],
                    by_source=bool(args.stratify_by_source),
                )
                teacher_bundle, _, _ = train_with_cv(
                    X=pair_ds.X[trusted_mask],
                    y=y_work[trusted_mask],
                    groups=pair_ds.groups[trusted_mask],
                    cfg=teacher_cfg,
                    stratify_labels=teacher_strat,
                    sample_weight=sample_weight[trusted_mask],
                )
                teacher_prob, _ = predict_proba(teacher_bundle, pair_ds.X)
                teacher_pred = (teacher_prob >= 0.5).astype(np.int64)
                teacher_conf = np.abs(teacher_prob - 0.5) * 2.0

                disagree = teacher_pred != y_work
                high_conf = teacher_conf >= float(args.denoise_threshold)
                fix_mask = noisy_mask & disagree & high_conf

                denoise_report["n_noisy"] = int(noisy_mask.sum())
                denoise_report["n_disagree"] = int((noisy_mask & disagree).sum())
                denoise_report["n_high_conf_disagree"] = int(fix_mask.sum())

                if args.denoise_action == "flip":
                    y_work[fix_mask] = teacher_pred[fix_mask]
                    corrected_mask[fix_mask] = True
                    sample_weight[fix_mask] = np.maximum(
                        sample_weight[fix_mask], float(args.flipped_weight)
                    )
                    denoise_report["n_corrected"] = int(fix_mask.sum())
                    if args.verbose > 0:
                        print(
                            f"[Stage] Denoise flip corrected: {int(fix_mask.sum())}",
                            flush=True,
                        )
                elif args.denoise_action == "drop":
                    keep_mask[fix_mask] = False
                    denoise_report["n_dropped"] = int(fix_mask.sum())
                    if args.verbose > 0:
                        print(
                            f"[Stage] Denoise drop removed: {int(fix_mask.sum())}",
                            flush=True,
                        )

    train_idx = np.where(keep_mask)[0]
    if len(train_idx) == 0:
        raise RuntimeError("No samples remain after denoise/drop.")

    X_train = pair_ds.X[train_idx]
    y_train = y_work[train_idx]
    groups_train = pair_ds.groups[train_idx]
    sources_train = pair_ds.sources[train_idx]
    sample_weight_train = sample_weight[train_idx]

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training labels collapse to one class after denoise; cannot train.")

    stratify_labels = _build_stratify_labels(
        y=y_train,
        sources=sources_train,
        by_source=bool(args.stratify_by_source),
    )
    if args.verbose > 0:
        if bool(args.stratify_by_source):
            uniq_src = sorted(set(sources_train.tolist()))
            print(f"[Stage] Stratify mode: label+source ({uniq_src})", flush=True)
        else:
            print("[Stage] Stratify mode: label only", flush=True)

    if args.verbose > 0:
        print("[Stage] Training with group 5-fold CV ...", flush=True)
    model_bundle, oof_base, oof_final = train_with_cv(
        X=X_train,
        y=y_train,
        groups=groups_train,
        cfg=cfg,
        stratify_labels=stratify_labels,
        sample_weight=sample_weight_train,
    )

    pred_label = proba_to_label(oof_final, threshold=model_bundle.threshold)

    rows = []
    for local_idx, global_idx in enumerate(train_idx):
        s = pair_ds.samples[global_idx]
        row = {
            "source": s.source,
            "line_id": s.line_id,
            "pair_id": s.pair_id,
            "left": s.left_name,
            "right": s.right_name,
            "label": s.label,
            "y_true_original": int(y_original[global_idx]),
            "y_train": int(y_train[local_idx]),
            "cleaned_label": int(corrected_mask[global_idx]),
            "sample_weight": float(sample_weight_train[local_idx]),
            "prob_A": float(oof_final[local_idx]),
            "pred": str(pred_label[local_idx]),
        }
        for name, arr in oof_base.items():
            row[f"prob_{name}"] = float(arr[local_idx])
        rows.append(row)

    oof_csv = artifact_dir / "oof_predictions.csv"
    pd.DataFrame(rows).to_csv(oof_csv, index=False)

    if int((~keep_mask).sum()) > 0:
        dropped_rows = []
        for idx in np.where(~keep_mask)[0]:
            s = pair_ds.samples[idx]
            dropped_rows.append(
                {
                    "source": s.source,
                    "line_id": s.line_id,
                    "pair_id": s.pair_id,
                    "left": s.left_name,
                    "right": s.right_name,
                    "label": s.label,
                    "y_true_original": int(y_original[idx]),
                }
            )
        dropped_csv = artifact_dir / "dropped_samples.csv"
        pd.DataFrame(dropped_rows).to_csv(dropped_csv, index=False)

    report_path = artifact_dir / "cv_report.json"
    report_path.write_text(json.dumps(model_bundle.cv_report, ensure_ascii=False, indent=2), encoding="utf-8")

    bundle_path = artifact_dir / "model_bundle.joblib"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_config": {
            "use_handcrafted": bool(args.use_handcrafted),
            "use_clip": bool(args.use_clip),
            "clip_model": args.clip_model,
            "device": args.device,
            "use_pyiqa": bool(args.use_pyiqa),
            "pyiqa_metrics": pyiqa_metrics,
            "feature_names": extractor.feature_names,
        },
        "pair_feature_config": {
            "include_raw": bool(args.include_raw),
        },
        "training_policy": {
            "trusted_sources": sorted(trusted_sources),
            "noisy_sources": sorted(noisy_sources),
            "denoise_action": args.denoise_action,
            "denoise_threshold": float(args.denoise_threshold),
            "trusted_weight": float(args.trusted_weight),
            "noisy_weight": float(args.noisy_weight),
            "flipped_weight": float(args.flipped_weight),
            "denoise_report": denoise_report,
            "n_total_samples": int(n_total),
            "n_train_samples": int(len(train_idx)),
        },
        "train_datasets": [
            {
                "source": s.source,
                "jsonl_path": str(s.jsonl_path),
                "image_root": str(s.image_root),
            }
            for s in specs
        ],
        "model_bundle": model_bundle,
    }
    joblib.dump(payload, bundle_path)
    if args.verbose > 0:
        print(f"[Stage] Saved model bundle -> {bundle_path}", flush=True)

    print(f"[OK] samples: train={len(train_idx)} total={n_total}")
    print(f"[OK] features: {X_train.shape[1]}")
    if noisy_sources and args.denoise_action != "none":
        print(
            "[OK] denoise: corrected={} dropped={}".format(
                int(denoise_report.get("n_corrected", 0)),
                int(denoise_report.get("n_dropped", 0)),
            )
        )
    print(f"[OK] base models: {', '.join(model_bundle.base_order)}")
    print(
        "[OK] blended OOF method={} bal_acc={:.4f} acc={:.4f} threshold={:.3f}".format(
            model_bundle.cv_report["blended_oof"].get("method", "unknown"),
            model_bundle.cv_report["blended_oof"]["bal_acc"],
            model_bundle.cv_report["blended_oof"]["acc"],
            model_bundle.threshold,
        )
    )
    print(f"[OK] artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
