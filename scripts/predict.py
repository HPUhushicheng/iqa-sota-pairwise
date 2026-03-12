#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
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
    ap = argparse.ArgumentParser(description="Predict A/B for pairwise IQA dataset.")
    ap.add_argument("--model-bundle", required=True, help="Path to model_bundle.joblib")
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec: source:jsonl_path:image_root (repeatable)",
    )
    ap.add_argument("--output", default="predictions.jsonl")
    ap.add_argument(
        "--output-style",
        default="competition",
        choices=["competition", "answer"],
        help="competition: images+solution; answer: images+answer",
    )
    ap.add_argument("--with-thinking", type=int, default=1)
    ap.add_argument("--include-prob", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--swap-tta", type=int, default=1, help="1: average P(A,B) with 1-P(B,A)")
    ap.add_argument("--device", default=None, help="Override device in model bundle")
    ap.add_argument("--no-progress", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = joblib.load(Path(args.model_bundle).resolve())

    feature_cfg = dict(payload["feature_config"])
    if args.device:
        feature_cfg["device"] = args.device

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

    pair_cfg = payload.get("pair_feature_config", {})
    pair_ds = build_pair_dataset(
        samples=samples,
        extractor=extractor,
        include_raw=bool(pair_cfg.get("include_raw", False)),
        show_progress=not args.no_progress,
    )

    bundle = payload["model_bundle"]
    probs, per_model = predict_proba(bundle, pair_ds.X)

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
        probs_swap, _ = predict_proba(bundle, swapped_ds.X)
        probs = 0.5 * (probs + (1.0 - probs_swap))
        print("[OK] swap-tta applied", flush=True)

    threshold = float(args.threshold) if args.threshold is not None else float(bundle.threshold)
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
                for name, arr in per_model.items():
                    row[f"prob_{name}"] = float(arr[i])

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] threshold={threshold:.4f}")
    print(f"[OK] wrote {len(pair_ds.samples)} rows -> {out_path}")

    # If ground truth exists in this dataset, print quick offline metrics.
    if all(s.label in {"A", "B"} for s in pair_ds.samples):
        y_true = [1 if s.label == "A" else 0 for s in pair_ds.samples]
        y_pred = [1 if x == "A" else 0 for x in labels]
        print("[Eval] acc={:.4f} bal_acc={:.4f}".format(
            accuracy_score(y_true, y_pred),
            balanced_accuracy_score(y_true, y_pred),
        ))


if __name__ == "__main__":
    main()
