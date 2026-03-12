from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

from .data import PairSample


@dataclass
class ImageFeature:
    vector: np.ndarray
    named: dict[str, float]


@dataclass
class PairFeatureDataset:
    X: np.ndarray
    y: Optional[np.ndarray]
    groups: np.ndarray
    samples: list[PairSample]
    left_stats: list[dict[str, float]]
    right_stats: list[dict[str, float]]


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def _entropy_from_gray(gray: np.ndarray) -> float:
    hist, _ = np.histogram((gray * 255.0).clip(0, 255).astype(np.uint8), bins=256, range=(0, 255))
    p = hist.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _high_freq_ratio(gray: np.ndarray) -> float:
    h, w = gray.shape
    if h < 8 or w < 8:
        return 0.0

    f = np.fft.rfft2(gray)
    power = np.abs(f) ** 2

    yy = np.fft.fftfreq(h)[:, None]
    xx = np.fft.rfftfreq(w)[None, :]
    rr = np.sqrt(xx**2 + yy**2)
    cutoff = 0.18

    hf = power[rr >= cutoff].sum()
    total = power.sum() + 1e-12
    return float(hf / total)


def _compute_handcrafted(rgb: np.ndarray) -> dict[str, float]:
    gray = _rgb_to_gray(rgb)

    gray_mean = float(gray.mean())
    gray_std = float(gray.std())
    p05 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    contrast = p95 - p05

    under_ratio = float((gray < 0.05).mean())
    over_ratio = float((gray > 0.95).mean())

    gx = ndi.sobel(gray, axis=1, mode="reflect")
    gy = ndi.sobel(gray, axis=0, mode="reflect")
    grad_mag = np.sqrt(gx * gx + gy * gy)
    tenengrad = float(np.mean(grad_mag))

    lap = ndi.laplace(gray, mode="reflect")
    lap_var = float(np.var(lap))

    # Robust noise proxy (higher -> noisier / more grain)
    noise_sigma = float(np.median(np.abs(lap)) / 0.6745)

    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    sat = (maxc - minc) / (maxc + 1e-6)
    sat_mean = float(sat.mean())
    sat_std = float(sat.std())

    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = float(np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    entropy = _entropy_from_gray(gray)
    hf_ratio = _high_freq_ratio(gray)

    return {
        "gray_mean": gray_mean,
        "gray_std": gray_std,
        "gray_p05": p05,
        "gray_p95": p95,
        "gray_contrast": float(contrast),
        "under_ratio": under_ratio,
        "over_ratio": over_ratio,
        "tenengrad": tenengrad,
        "lap_var": lap_var,
        "noise_sigma": noise_sigma,
        "sat_mean": sat_mean,
        "sat_std": sat_std,
        "colorfulness": colorfulness,
        "entropy": entropy,
        "hf_ratio": hf_ratio,
    }


class ImageFeatureExtractor:
    HANDCRAFTED_NAMES = [
        "gray_mean",
        "gray_std",
        "gray_p05",
        "gray_p95",
        "gray_contrast",
        "under_ratio",
        "over_ratio",
        "tenengrad",
        "lap_var",
        "noise_sigma",
        "sat_mean",
        "sat_std",
        "colorfulness",
        "entropy",
        "hf_ratio",
    ]

    def __init__(
        self,
        use_handcrafted: bool = True,
        use_clip: bool = True,
        clip_model: str = "vit_base_patch32_clip_224.openai",
        device: str = "cpu",
        use_pyiqa: bool = False,
        pyiqa_metrics: Optional[list[str]] = None,
    ) -> None:
        self.use_handcrafted = bool(use_handcrafted)
        self.use_clip = bool(use_clip)
        self.clip_model_name = clip_model
        self.device = device
        self.use_pyiqa = bool(use_pyiqa)
        self.pyiqa_metrics = pyiqa_metrics or ["musiq", "clipiqa"]

        self._cache: dict[Path, ImageFeature] = {}

        self._clip_model = None
        self._clip_transform = None
        self._clip_dim = 0

        self._pyiqa_models: dict[str, object] = {}

        if self.use_clip:
            self._init_clip()
        if self.use_pyiqa:
            self._init_pyiqa()

        if not (self.use_handcrafted or self.use_clip or self.use_pyiqa):
            raise ValueError("At least one of handcrafted/clip/pyiqa must be enabled.")

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self.use_handcrafted:
            names.extend(self.HANDCRAFTED_NAMES)
        if self.use_clip:
            names.extend([f"clip_{i:04d}" for i in range(self._clip_dim)])
        if self.use_pyiqa:
            names.extend([f"pyiqa_{m}" for m in self.pyiqa_metrics])
        return names

    def _init_clip(self) -> None:
        try:
            import timm
            import torch
            from timm.data import create_transform, resolve_model_data_config
        except Exception as exc:
            raise RuntimeError(
                "CLIP feature extraction requires timm + torch. "
                "Install requirements and retry."
            ) from exc

        model = timm.create_model(self.clip_model_name, pretrained=True, num_classes=0)
        model.eval()
        model.to(self.device)

        cfg = resolve_model_data_config(model)
        transform = create_transform(**cfg, is_training=False)

        clip_dim = int(getattr(model, "num_features", 0))
        if clip_dim <= 0:
            clip_dim = 512

        self._clip_model = model
        self._clip_transform = transform
        self._clip_dim = clip_dim
        self._torch = torch

    def _init_pyiqa(self) -> None:
        try:
            import pyiqa
        except Exception as exc:
            raise RuntimeError(
                "pyiqa feature extraction enabled but pyiqa is not installed."
            ) from exc

        models: dict[str, object] = {}
        for metric in self.pyiqa_metrics:
            models[metric] = pyiqa.create_metric(metric, device=self.device)
        self._pyiqa_models = models

    def _extract_clip(self, image: Image.Image) -> np.ndarray:
        x = self._clip_transform(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            feat = self._clip_model(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        feat = feat.reshape(-1).float().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-12
        feat = feat / norm
        return feat

    def _extract_pyiqa(self, path: Path) -> dict[str, float]:
        scores: dict[str, float] = {}
        for metric in self.pyiqa_metrics:
            model = self._pyiqa_models[metric]
            try:
                score = model(str(path))
                if hasattr(score, "item"):
                    score = score.item()
                scores[f"pyiqa_{metric}"] = float(score)
            except Exception:
                scores[f"pyiqa_{metric}"] = float("nan")
        return scores

    def extract(self, path: Path) -> ImageFeature:
        path = Path(path).resolve()
        cached = self._cache.get(path)
        if cached is not None:
            return cached

        image = Image.open(path).convert("RGB")
        rgb = np.asarray(image, dtype=np.float32) / 255.0

        named: dict[str, float] = {}
        vec_parts: list[np.ndarray] = []

        if self.use_handcrafted:
            d = _compute_handcrafted(rgb)
            for name in self.HANDCRAFTED_NAMES:
                named[name] = float(d[name])
            vec_parts.append(np.array([named[name] for name in self.HANDCRAFTED_NAMES], dtype=np.float32))

        if self.use_clip:
            clip_feat = self._extract_clip(image)
            vec_parts.append(clip_feat)
            for i, v in enumerate(clip_feat.tolist()):
                named[f"clip_{i:04d}"] = float(v)

        if self.use_pyiqa:
            pyiqa_scores = self._extract_pyiqa(path)
            arr = []
            for metric in self.pyiqa_metrics:
                key = f"pyiqa_{metric}"
                val = float(pyiqa_scores.get(key, math.nan))
                named[key] = val
                arr.append(val)
            vec_parts.append(np.array(arr, dtype=np.float32))

        vector = np.concatenate(vec_parts, axis=0).astype(np.float32)
        feat = ImageFeature(vector=vector, named=named)
        self._cache[path] = feat
        return feat


def make_pair_feature(
    left: np.ndarray,
    right: np.ndarray,
    include_raw: bool = False,
) -> np.ndarray:
    diff = left - right
    abs_diff = np.abs(diff)
    parts = [diff, abs_diff]
    if include_raw:
        parts.extend([left, right])
    return np.concatenate(parts, axis=0).astype(np.float32)


def build_pair_dataset(
    samples: list[PairSample],
    extractor: ImageFeatureExtractor,
    include_raw: bool = False,
    show_progress: bool = True,
) -> PairFeatureDataset:
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    groups: list[str] = []
    left_stats: list[dict[str, float]] = []
    right_stats: list[dict[str, float]] = []

    iterator = tqdm(samples, desc="Extracting features", disable=not show_progress)
    for s in iterator:
        l_feat = extractor.extract(s.left_path)
        r_feat = extractor.extract(s.right_path)

        X_rows.append(make_pair_feature(l_feat.vector, r_feat.vector, include_raw=include_raw))
        groups.append(s.pair_id)

        if s.label is not None:
            y_rows.append(1 if s.label == "A" else 0)

        # Keep only handcrafted stats for textual reasoning.
        if extractor.use_handcrafted:
            left_stats.append({k: l_feat.named[k] for k in extractor.HANDCRAFTED_NAMES})
            right_stats.append({k: r_feat.named[k] for k in extractor.HANDCRAFTED_NAMES})
        else:
            left_stats.append({})
            right_stats.append({})

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.array(y_rows, dtype=np.int64) if y_rows else None
    g = np.array(groups)

    return PairFeatureDataset(
        X=X,
        y=y,
        groups=g,
        samples=samples,
        left_stats=left_stats,
        right_stats=right_stats,
    )
