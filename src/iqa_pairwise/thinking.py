from __future__ import annotations

from typing import Callable


def _exposure_score(stats: dict[str, float]) -> float:
    mean = stats.get("gray_mean", 0.5)
    under = stats.get("under_ratio", 0.0)
    over = stats.get("over_ratio", 0.0)
    # Higher is better: close to middle luminance and less clipping.
    return float((1.0 - abs(mean - 0.5) * 1.8) - 0.8 * under - 0.8 * over)


def _score_items(winner: dict[str, float], loser: dict[str, float]) -> list[tuple[float, str]]:
    items: list[tuple[float, str]] = []

    # Higher better
    d_sharp = (winner.get("tenengrad", 0.0) + 0.5 * winner.get("lap_var", 0.0)) - (
        loser.get("tenengrad", 0.0) + 0.5 * loser.get("lap_var", 0.0)
    )
    items.append((d_sharp, "细节锐度更高，边缘与纹理更清晰。"))

    d_hf = winner.get("hf_ratio", 0.0) - loser.get("hf_ratio", 0.0)
    items.append((d_hf, "高频纹理保留更充分，细节恢复更完整。"))

    d_contrast = winner.get("gray_contrast", 0.0) - loser.get("gray_contrast", 0.0)
    items.append((d_contrast, "局部对比度更好，层次更分明。"))

    d_exposure = _exposure_score(winner) - _exposure_score(loser)
    items.append((d_exposure, "曝光更均衡，高光和暗部细节更稳定。"))

    # Lower better metrics converted to positive-improvement form.
    d_noise = loser.get("noise_sigma", 0.0) - winner.get("noise_sigma", 0.0)
    items.append((d_noise, "噪点更少，平坦区域更干净。"))

    d_clip = (loser.get("under_ratio", 0.0) + loser.get("over_ratio", 0.0)) - (
        winner.get("under_ratio", 0.0) + winner.get("over_ratio", 0.0)
    )
    items.append((d_clip, "过曝/欠曝比例更低，信息保留更完整。"))

    return items


def build_thinking(
    left_stats: dict[str, float],
    right_stats: dict[str, float],
    answer: str,
    max_points: int = 4,
) -> str:
    if answer not in {"A", "B"}:
        answer = "A"

    winner = left_stats if answer == "A" else right_stats
    loser = right_stats if answer == "A" else left_stats

    if not winner:
        return (
            "<thinking>\n"
            "综合比较后，获胜图在细节清晰度、噪声控制与曝光平衡上更优。\n"
            "</thinking>\n"
            f"<answer>{answer}</answer>"
        )

    scored = sorted(_score_items(winner, loser), key=lambda x: abs(x[0]), reverse=True)
    chosen = [text for score, text in scored if abs(score) > 1e-8][:max_points]
    if not chosen:
        chosen = ["两图质量接近，但获胜图在综合观感上更稳定。"]

    lines = [f"- {line}" for line in chosen]
    lines.append(f"- 综合判断：Image {answer} 质量更好。")

    return "<thinking>\n" + "\n".join(lines) + "\n</thinking>\n" + f"<answer>{answer}</answer>"
