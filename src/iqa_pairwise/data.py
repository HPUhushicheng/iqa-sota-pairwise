from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)


@dataclass(frozen=True)
class PairSample:
    source: str
    line_id: int
    pair_id: str
    left_name: str
    right_name: str
    left_path: Path
    right_path: Path
    label: Optional[str]


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    jsonl_path: Path
    image_root: Path


def parse_dataset_spec(spec: str) -> DatasetSpec:
    """
    Parse CLI spec: source:jsonl_path:image_root
    Example: train:1536/data.jsonl:1536/images
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --dataset '{spec}'. Expected format source:jsonl_path:image_root"
        )
    source, jsonl_path, image_root = parts
    return DatasetSpec(
        source=source.strip(),
        jsonl_path=Path(jsonl_path).expanduser().resolve(),
        image_root=Path(image_root).expanduser().resolve(),
    )


def _extract_images(record: dict[str, Any]) -> tuple[str, str]:
    if "images" in record and isinstance(record["images"], list) and len(record["images"]) >= 2:
        a, b = record["images"][0], record["images"][1]
        return str(a), str(b)

    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Record has neither 'images' nor valid 'messages'.")

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        images: list[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "image" in item:
                    images.append(str(item["image"]))
        if len(images) >= 2:
            return images[0], images[1]

    raise ValueError("Cannot find two images in user message content.")


def _normalize_label(text: str) -> Optional[str]:
    t = text.strip().upper()
    if t in {"A", "B"}:
        return t
    m = ANSWER_RE.search(t)
    if m:
        return m.group(1).upper()
    if t.endswith("A"):
        return "A"
    if t.endswith("B"):
        return "B"
    return None


def _extract_label_from_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return _normalize_label(content)

    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return _normalize_label(content["text"])
        return None

    if isinstance(content, list):
        text_chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_chunks.append(item)
            elif isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                text_chunks.append(item["text"])
        merged = "\n".join(text_chunks)
        return _normalize_label(merged)

    return None


def _extract_label(record: dict[str, Any]) -> Optional[str]:
    if "answer" in record and isinstance(record["answer"], str):
        lab = _normalize_label(record["answer"])
        if lab:
            return lab

    if "solution" in record and isinstance(record["solution"], str):
        lab = _normalize_label(record["solution"])
        if lab:
            return lab

    messages = record.get("messages")
    if not isinstance(messages, list):
        return None

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        lab = _extract_label_from_content(msg.get("content"))
        if lab:
            return lab

    return None


def _resolve_image_path(image_root: Path, image_name: str) -> Path:
    name = image_name.strip()
    p = Path(name)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((image_root / p).resolve())
        candidates.append((image_root / p.name).resolve())

    for cand in candidates:
        if cand.exists():
            return cand

    # Return most likely path for downstream error visibility.
    if p.is_absolute():
        return p
    return (image_root / p.name).resolve()


def load_samples(spec: DatasetSpec, require_label: bool = True) -> list[PairSample]:
    if not spec.jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {spec.jsonl_path}")
    if not spec.image_root.exists():
        raise FileNotFoundError(f"Image root not found: {spec.image_root}")

    samples: list[PairSample] = []
    with spec.jsonl_path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            left_name, right_name = _extract_images(record)
            label = _extract_label(record)
            if require_label and label not in {"A", "B"}:
                raise ValueError(
                    f"Missing/invalid label at {spec.jsonl_path}:{line_id}. label={label}"
                )

            left_path = _resolve_image_path(spec.image_root, left_name)
            right_path = _resolve_image_path(spec.image_root, right_name)
            if not left_path.exists() or not right_path.exists():
                raise FileNotFoundError(
                    f"Image missing at {spec.jsonl_path}:{line_id}. "
                    f"left={left_path} right={right_path}"
                )

            canonical = "|".join(sorted([Path(left_name).name, Path(right_name).name]))
            pair_id = f"{spec.source}:{canonical}"

            samples.append(
                PairSample(
                    source=spec.source,
                    line_id=line_id,
                    pair_id=pair_id,
                    left_name=left_name,
                    right_name=right_name,
                    left_path=left_path,
                    right_path=right_path,
                    label=label,
                )
            )

    return samples


def load_many(specs: Iterable[DatasetSpec], require_label: bool = True) -> list[PairSample]:
    all_samples: list[PairSample] = []
    for spec in specs:
        all_samples.extend(load_samples(spec=spec, require_label=require_label))
    return all_samples


def unique_image_paths(samples: Iterable[PairSample]) -> list[Path]:
    s: set[Path] = set()
    for row in samples:
        s.add(row.left_path)
        s.add(row.right_path)
    return sorted(s)
