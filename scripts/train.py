#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iqa_pairwise.data import load_many, parse_dataset_spec
from iqa_pairwise.features import ImageFeatureExtractor, build_pair_dataset
from iqa_pairwise.model import TrainConfig, proba_to_label, train_with_cv


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

    if args.verbose > 0:
        print("[Stage] Training with group 5-fold CV ...", flush=True)
    model_bundle, oof_base, oof_final = train_with_cv(
        X=pair_ds.X,
        y=pair_ds.y,
        groups=pair_ds.groups,
        cfg=cfg,
    )

    pred_label = proba_to_label(oof_final, threshold=model_bundle.threshold)

    rows = []
    for idx, s in enumerate(pair_ds.samples):
        row = {
            "source": s.source,
            "line_id": s.line_id,
            "pair_id": s.pair_id,
            "left": s.left_name,
            "right": s.right_name,
            "label": s.label,
            "y_true": int(pair_ds.y[idx]),
            "prob_A": float(oof_final[idx]),
            "pred": str(pred_label[idx]),
        }
        for name, arr in oof_base.items():
            row[f"prob_{name}"] = float(arr[idx])
        rows.append(row)

    oof_csv = artifact_dir / "oof_predictions.csv"
    pd.DataFrame(rows).to_csv(oof_csv, index=False)

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

    print(f"[OK] samples: {len(pair_ds.samples)}")
    print(f"[OK] features: {pair_ds.X.shape[1]}")
    print(f"[OK] base models: {', '.join(model_bundle.base_order)}")
    print(
        "[OK] blended OOF bal_acc={:.4f} acc={:.4f} threshold={:.3f}".format(
            model_bundle.cv_report["blended_oof"]["bal_acc"],
            model_bundle.cv_report["blended_oof"]["acc"],
            model_bundle.threshold,
        )
    )
    print(f"[OK] artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
