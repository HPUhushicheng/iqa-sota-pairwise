#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iqa_pairwise.data import PairSample
from iqa_pairwise.features import ImageFeatureExtractor, build_pair_dataset
from iqa_pairwise.model import predict_proba, proba_to_label
from iqa_pairwise.thinking import build_thinking

PAIR_RE = re.compile(r"^(?P<base>.+)_c(?P<cid>[01])(?P<ext>\.[^.]+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Ensemble predict A/B from folder paired *_c0/*_c1 images (single run)."
    )
    ap.add_argument(
        "--model-bundle",
        action="append",
        required=True,
        help="Path to model_bundle.joblib (repeatable)",
    )
    ap.add_argument("--test-dir", required=True, help="Folder containing paired c0/c1 images")
    ap.add_argument("--output", default="predictions_ensemble_from_folder.jsonl")
    ap.add_argument(
        "--output-style",
        default="answer",
        choices=["answer", "competition"],
        help="answer: images+answer; competition: images+solution",
    )
    ap.add_argument("--with-thinking", type=int, default=0)
    ap.add_argument("--include-prob", type=int, default=0)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If not set, use average of model thresholds",
    )
    ap.add_argument("--swap-tta", type=int, default=1, help="1: average P(c0,c1) with 1-P(c1,c0)")
    ap.add_argument("--recursive", type=int, default=0, help="1: recursively scan subfolders")
    ap.add_argument("--strict", type=int, default=1, help="1: error if any pair is incomplete")
    ap.add_argument("--device", default=None, help="Override device in all model bundles")
    ap.add_argument("--no-progress", action="store_true")
    return ap.parse_args()


def collect_pairs(test_dir: Path, recursive: bool, strict: bool) -> tuple[list[PairSample], list[str]]:
    globber = test_dir.rglob("*") if recursive else test_dir.glob("*")
    pair_map: dict[str, dict[str, Path]] = {}

    for p in globber:
        if not p.is_file():
            continue
        m = PAIR_RE.match(p.name)
        if not m:
            continue
        base = m.group("base")
        cid = m.group("cid")
        key = str((p.parent.resolve(), base))
        if key not in pair_map:
            pair_map[key] = {}
        pair_map[key][cid] = p.resolve()

    samples: list[PairSample] = []
    issues: list[str] = []

    ordered_keys = sorted(pair_map.keys())
    line_id = 1
    for key in ordered_keys:
        c0 = pair_map[key].get("0")
        c1 = pair_map[key].get("1")
        if c0 is None or c1 is None:
            issues.append(f"incomplete pair: {key}")
            continue

        samples.append(
            PairSample(
                source="test",
                line_id=line_id,
                pair_id=f"test:{key}",
                left_name=str(c0.name),
                right_name=str(c1.name),
                left_path=c0,
                right_path=c1,
                label=None,
            )
        )
        line_id += 1

    if strict and issues:
        msg = "\n".join(issues[:20])
        extra = "" if len(issues) <= 20 else f"\n... and {len(issues) - 20} more"
        raise RuntimeError(f"Found incomplete pairs ({len(issues)}):\n{msg}{extra}")

    return samples, issues


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
            raise ValueError(f"Incompatible feature config at bundle #{idx}, key={k}")


def main() -> None:
    args = parse_args()

    payloads = [joblib.load(Path(p).resolve()) for p in args.model_bundle]
    if not payloads:
        raise RuntimeError("No model bundle provided")

    feature_cfg = dict(payloads[0]["feature_config"])
    if args.device:
        feature_cfg["device"] = args.device

    pair_cfg = payloads[0].get("pair_feature_config", {})

    for i, payload in enumerate(payloads[1:], start=2):
        _assert_compatible_feature_cfg(payloads[0]["feature_config"], payload["feature_config"], i)
        if payload.get("pair_feature_config", {}) != pair_cfg:
            raise ValueError(f"Incompatible pair feature config at bundle #{i}")

    extractor = ImageFeatureExtractor(
        use_handcrafted=feature_cfg["use_handcrafted"],
        use_clip=feature_cfg["use_clip"],
        clip_model=feature_cfg["clip_model"],
        device=feature_cfg["device"],
        use_pyiqa=feature_cfg["use_pyiqa"],
        pyiqa_metrics=feature_cfg.get("pyiqa_metrics", []),
    )

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.exists() or not test_dir.is_dir():
        raise FileNotFoundError(f"test-dir not found or not a directory: {test_dir}")

    samples, issues = collect_pairs(test_dir=test_dir, recursive=bool(args.recursive), strict=bool(args.strict))
    if not samples:
        raise RuntimeError("No valid c0/c1 pairs found")

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

    model_probs: list[np.ndarray] = []
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
            ans = str(labels[i])  # A means c0 better, B means c1 better
            if args.output_style == "competition":
                if bool(args.with_thinking):
                    solution = build_thinking(pair_ds.left_stats[i], pair_ds.right_stats[i], ans)
                else:
                    solution = f"<answer>{ans}</answer>"
                row = {
                    "images": [s.left_name, s.right_name],
                    "solution": solution,
                }
            else:
                row = {
                    "images": [s.left_name, s.right_name],
                    "answer": ans,
                }

            if bool(args.include_prob):
                row["prob_A"] = float(probs[i])
                for j, p in enumerate(model_probs, start=1):
                    row[f"prob_model{j}"] = float(p[i])

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] models={len(payloads)}")
    print(f"[OK] pairs={len(pair_ds.samples)}")
    print(f"[OK] threshold={threshold:.4f}")
    if bool(args.swap_tta):
        print("[OK] swap-tta applied")
    if issues and not bool(args.strict):
        print(f"[WARN] skipped incomplete pairs: {len(issues)}")
    print(f"[OK] output={out_path}")


if __name__ == "__main__":
    main()
