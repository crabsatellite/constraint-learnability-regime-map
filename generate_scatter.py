"""Generate the predictor scatter figure from saved experimental results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"

DISPLAY_ORDER = [
    "floor_count",
    "bbox_volume",
    "connected_components",
    "vertical_aspect",
    "z_symmetry_iou",
    "elongation",
    "surface_ratio",
    "layer_consistency",
    "footprint_convexity",
    "hollowness",
]

SHORT_LABELS = {
    "connected_components": "connected_comp",
    "layer_consistency": "layer_consist",
    "footprint_convexity": "fp_convexity",
}

NUMBER_OFFSETS = {
    1: (12, -8),
    2: (12, -8),
    3: (12, 0),
    4: (-14, 8),
    5: (14, -8),
    6: (14, 10),
    7: (0, 15),
    8: (-8, 15),
    9: (14, -10),
    10: (14, -8),
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_points() -> tuple[list[dict], dict]:
    predictor = read_json(OUT_DIR / "regime_predictor_results.json")
    bootstrap = read_json(OUT_DIR / "bootstrap_results.json")
    supplementary = read_json(OUT_DIR / "supplementary_results.json")

    feature_by_prop = {row["property"]: row for row in predictor["feature_table"]}
    points: list[dict] = []
    for idx, prop in enumerate(DISPLAY_ORDER, start=1):
        row = feature_by_prop[prop]
        if row["has_direct_cond"]:
            raise ValueError(f"{prop} is not an emergent property")
        eff = float(row["effective_frequency"])
        cv = float(row["training_cv"])
        ctrl = float(bootstrap["properties"][prop]["controllability_pct"])
        points.append(
            {
                "index": idx,
                "property": prop,
                "label": SHORT_LABELS.get(prop, prop),
                "effective_frequency": eff,
                "training_cv": cv,
                "composite": eff * min(cv, 1.0),
                "controllability_pct": ctrl,
                "regime": bootstrap["properties"][prop]["regime"],
            }
        )

    composite = supplementary["permutation_test"]["composite"]
    stats = {
        "rho": float(composite["rho"]),
        "permutation_p": float(composite["permutation_p"]),
        "sources": [
            "outputs/regime_predictor_results.json",
            "outputs/bootstrap_results.json",
            "outputs/supplementary_results.json",
        ],
    }
    return points, stats


def style_for_regime(regime: str) -> tuple[str, str]:
    if regime == "CONTROLLABLE":
        return "#2ca02c", "o"
    if regime == "APPROACHABLE":
        return "#ff7f0e", "s"
    return "#d62728", "^"


def plot(points: list[dict], stats: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for point in points:
        color, marker = style_for_regime(point["regime"])
        ax.scatter(
            point["composite"],
            point["controllability_pct"],
            c=color,
            marker=marker,
            s=50,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )

    arrow_style = dict(
        arrowstyle="-",
        color="#666666",
        linewidth=0.6,
        shrinkA=0,
        shrinkB=3,
    )
    for point in points:
        ox, oy = NUMBER_OFFSETS[point["index"]]
        ax.annotate(
            str(point["index"]),
            (point["composite"], point["controllability_pct"]),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            color="#333333",
            zorder=6,
            arrowprops=arrow_style,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
            ),
        )

    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(y=20, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(
        0.97,
        0.66,
        "Controllable",
        fontsize=6,
        color="#2ca02c",
        alpha=0.8,
        fontstyle="italic",
        transform=ax.transAxes,
        ha="right",
        va="center",
    )
    ax.text(
        0.97,
        0.19,
        "Approachable",
        fontsize=6,
        color="#ff7f0e",
        alpha=0.8,
        fontstyle="italic",
        transform=ax.transAxes,
        ha="right",
        va="center",
    )
    ax.text(
        0.97,
        0.06,
        "Unresponsive",
        fontsize=6,
        color="#d62728",
        alpha=0.8,
        fontstyle="italic",
        transform=ax.transAxes,
        ha="right",
        va="center",
    )

    ax.set_xlabel("Composite score (proxy corr. x min(CV, 1))", fontsize=8)
    ax.set_ylabel("Controllability (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.04, 0.85)
    ax.set_ylim(-35, 470)
    ax.text(
        0.50,
        0.97,
        rf"$\rho = {stats['rho']:+.3f}$, perm. $p = {stats['permutation_p']:.3f}$",
        transform=ax.transAxes,
        fontsize=7,
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
        zorder=10,
    )

    plt.tight_layout()
    legend_lines = []
    for start in (0, 5):
        legend_lines.append(
            "   ".join(
                f"{p['index']} {p['label']}" for p in points[start : start + 5]
            )
        )
    fig.text(
        0.50,
        -0.01,
        "\n".join(legend_lines),
        ha="center",
        va="top",
        fontsize=7,
        fontfamily="monospace",
        color="#444444",
    )
    return fig


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    points, stats = load_points()
    (FIG_DIR / "scatter_predictor_data.json").write_text(
        json.dumps({"points": points, "stats": stats}, indent=2),
        encoding="utf-8",
    )

    fig = plot(points, stats)
    fig.savefig(FIG_DIR / "scatter_predictor.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "scatter_predictor.png", bbox_inches="tight", dpi=300)
    print("[figure:generated] figures/scatter_predictor.pdf")
    print("[figure:generated] figures/scatter_predictor.png")


if __name__ == "__main__":
    main()
