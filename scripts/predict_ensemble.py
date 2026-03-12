#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iqa_pairwise.data import PairSample, load_many, parse_dataset_spec
from iqa_pairwise.features import ImageFeatureExtractor, build_pair_dataset
from iqa_pairwise.model import predict_proba, proba_to_label
from iqa_pairwise.thinking import build_thinking


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ensemble prediction for pairwise IQA model bundles.")
    ap.add_argument(
        "--model-bundle",
        action="append",
        required=True,
        help="Path to model_bundle.joblib (repeatable)",
    )
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec: source:jsonl_path:image_root (repeatable)",
    )
    ap.add_argument("--output", default="predictions_ensemble.jsonl")
    ap.add_argument(
        "--output-style",
        default="competition",
        choices=["competition", "answer"],
        help="competition: images+solution; answer: images+answer",
    )
    ap.add_argument("--with-thinking", type=int, default=1)
    ap.add_argument("--include-prob", type=int, default=0)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If not set, use average of model thresholds.",
    )
    ap.add_argument("--device", default=None, help="Override device in model bundles")
    ap.add_argument("--swap-tta", type=int, default=1, help="1: average P(A,B) with 1-P(B,A)")
    ap.add_argument("--no-progress", action="store_true")
    return ap.parse_args()


def _assert_compatible_feature_cfg(base: dict, other: dict, idx: int) -> None:
    keys = [
        "use_handcrafted",
        "use_clip",
        "clip_model",
        "use_pyiqa",
        "pyiqa_metrics",
    ]
    for k in keys:
        if base.get(k) != other.get(k):
            raise ValueError(
                f"Incompatible feature config among bundles. key={k} mismatch at bundle #{idx}."
            )


def main() -> None:
    args = parse_args()

    payloads = [joblib.load(Path(p).resolve()) for p in args.model_bundle]
    if not payloads:
        raise RuntimeError("No model bundle provided.")

    feature_cfg = dict(payloads[0]["feature_config"])
    if args.device:
        feature_cfg["device"] = args.device

    for i, payload in enumerate(payloads[1:], start=2):
        _assert_compatible_feature_cfg(payloads[0]["feature_config"], payload["feature_config"], i)

    pair_cfg = payloads[0].get("pair_feature_config", {})
    for i, payload in enumerate(payloads[1:], start=2):
        if payload.get("pair_feature_config", {}) != pair_cfg:
            raise ValueError(f"Incompatible pair feature config at bundle #{i}.")

    extractor = ImageFeatureExtractor(
        use_handcrafted=feature_cfg["use_handcrafted"],
        use_clip=feature_cfg["use_clip"],
        clip_model=feature_cfg["clip_model"],
        device=feature_cfg["device"],
        use_pyiqa=feature_cfg["use_pyiqa"],
        pyiqa_metrics=feature_cfg.get("pyiqa_metrics", []),
    )

    specs = [parse_dataset_spec(s) for s in args.dataset]
    samples = load_many(specs=specs, require_label=False)
    if not samples:
        raise RuntimeError("No samples loaded for prediction.")

    pair_ds = build_pair_dataset(
        samples=samples,
        extractor=extractor,
        include_raw=bool(pair_cfg.get("include_raw", False)),
        show_progress=not args.no_progress,
    )
    swapped_ds = None
    if bool(args.swap_tta):
        swapped_samples = [
            PairSample(
                source=s.source,
                line_id=s.line_id,
                pair_id=s.pair_id,
                left_name=s.right_name,
                right_name=s.left_name,
                left_path=s.right_path,
                right_path=s.left_path,
                label=None,
            )
            for s in pair_ds.samples
        ]
        swapped_ds = build_pair_dataset(
            samples=swapped_samples,
            extractor=extractor,
            include_raw=bool(pair_cfg.get("include_raw", False)),
            show_progress=False,
        )

    model_probs = []
    for payload in payloads:
        bundle = payload["model_bundle"]
        p, _ = predict_proba(bundle, pair_ds.X)

        if swapped_ds is not None:
            p_swap, _ = predict_proba(bundle, swapped_ds.X)
            p = 0.5 * (p + (1.0 - p_swap))

        model_probs.append(p)

    probs_mat = np.column_stack(model_probs)
    probs = probs_mat.mean(axis=1)

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(np.mean([float(payload["model_bundle"].threshold) for payload in payloads]))

    labels = proba_to_label(probs, threshold=threshold)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(pair_ds.samples):
            ans = str(labels[i])
            if args.output_style == "answer":
                row = {
                    "images": [s.left_name, s.right_name],
                    "answer": ans,
                }
            else:
                if args.with_thinking:
                    solution = build_thinking(pair_ds.left_stats[i], pair_ds.right_stats[i], ans)
                else:
                    solution = f"<answer>{ans}</answer>"
                row = {
                    "images": [s.left_name, s.right_name],
                    "solution": solution,
                }

            if bool(args.include_prob):
                row["prob_A"] = float(probs[i])
                for j, p in enumerate(model_probs, start=1):
                    row[f"prob_model{j}"] = float(p[i])

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] ensembled models: {len(payloads)}")
    if bool(args.swap_tta):
        print("[OK] swap-tta applied")
    print(f"[OK] threshold={threshold:.4f}")
    print(f"[OK] wrote {len(pair_ds.samples)} rows -> {out_path}")

    if all(s.label in {"A", "B"} for s in pair_ds.samples):
        y_true = [1 if s.label == "A" else 0 for s in pair_ds.samples]
        y_pred = [1 if x == "A" else 0 for x in labels]
        print("[Eval] acc={:.4f} bal_acc={:.4f}".format(
            accuracy_score(y_true, y_pred),
            balanced_accuracy_score(y_true, y_pred),
        ))


if __name__ == "__main__":
    main()
