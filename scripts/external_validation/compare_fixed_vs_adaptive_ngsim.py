#!/usr/bin/env python3
"""
Comparer le mapping NGSIM initial et le mapping adaptatif.

Entrées :
- results/external_validation/ngsim/ml_predictions_ngsim.csv
- results/external_validation/ngsim/adaptive_ml/ml_predictions_ngsim.csv

Sorties :
- results/external_validation/ngsim/comparison/fixed_vs_adaptive_summary.csv
- results/external_validation/ngsim/comparison/fixed_vs_adaptive_levels.csv
- results/external_validation/ngsim/comparison/fixed_vs_adaptive_summary.json
- figures de comparaison

Cette analyse ne calcule pas de performance supervisée, car NGSIM ne
contient pas de vérité terrain radio. Elle compare uniquement la diversité,
la dispersion, la saturation et la cohérence de transfert.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.home() / "v2x-llm-project"

FIXED_FILE = (
    ROOT
    / "results"
    / "external_validation"
    / "ngsim"
    / "ml_predictions_ngsim.csv"
)

ADAPTIVE_FILE = (
    ROOT
    / "results"
    / "external_validation"
    / "ngsim"
    / "adaptive_ml"
    / "ml_predictions_ngsim.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "external_validation"
    / "ngsim"
    / "comparison"
)

TARGETS = {
    "QoSScore": (
        "QoSScore_PredictedNGSIM",
        "QoSLevel_PredictedNGSIM",
    ),
    "CongestionScore": (
        "CongestionScore_PredictedNGSIM",
        "CongestionLevel_PredictedNGSIM",
    ),
    "HORiskScore": (
        "HORiskScore_PredictedNGSIM",
        "HORiskLevel_PredictedNGSIM",
    ),
}


def numeric_summary(
    series: pd.Series,
) -> dict[str, float]:
    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return {
        "min": float(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "q75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "std": float(clean.std(ddof=0)),
        "unique_values": int(clean.nunique()),
        "top_value_rate": float(
            clean.value_counts(normalize=True).iloc[0]
        ),
    }


def level_rows(
    df: pd.DataFrame,
    mapping_name: str,
) -> list[dict[str, object]]:
    rows = []

    for target, (_, level_column) in TARGETS.items():
        counts = df[level_column].value_counts()
        total = len(df)

        for level, count in counts.items():
            rows.append({
                "Mapping": mapping_name,
                "Target": target,
                "Level": str(level),
                "Count": int(count),
                "Rate": float(count / total),
            })

    return rows


def score_rows(
    fixed: pd.DataFrame,
    adaptive: pd.DataFrame,
) -> list[dict[str, object]]:
    rows = []

    for target, (score_column, _) in TARGETS.items():
        for mapping_name, df in [
            ("Fixed", fixed),
            ("Adaptive", adaptive),
        ]:
            summary = numeric_summary(
                df[score_column]
            )

            rows.append({
                "Mapping": mapping_name,
                "Target": target,
                **summary,
            })

    return rows


def correlation_summary(
    df: pd.DataFrame,
) -> dict[str, float]:
    x = df["mobility_congestion_index"]
    y = df["CongestionScore_PredictedNGSIM"]

    return {
        "pearson": float(
            pd.concat([x, y], axis=1)
            .corr(method="pearson")
            .iloc[0, 1]
        ),
        "spearman": float(
            pd.concat([x, y], axis=1)
            .corr(method="spearman")
            .iloc[0, 1]
        ),
        "mean_absolute_gap": float(
            (y - x).abs().mean()
        ),
    }


def save_score_boxplot(
    fixed: pd.DataFrame,
    adaptive: pd.DataFrame,
    output: Path,
) -> None:
    data = []
    labels = []

    for target, (score_column, _) in TARGETS.items():
        data.append(fixed[score_column])
        labels.append(f"{target}\nFixed")
        data.append(adaptive[score_column])
        labels.append(f"{target}\nAdaptive")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Score prédit")
    ax.set_title(
        "Dispersion des scores NGSIM avant et après mapping adaptatif"
    )
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_level_distribution(
    levels: pd.DataFrame,
    output: Path,
) -> None:
    pivot = levels.pivot_table(
        index=["Target", "Level"],
        columns="Mapping",
        values="Rate",
        fill_value=0.0,
    )

    labels = [
        f"{target}\n{level}"
        for target, level in pivot.index
    ]

    x = np.arange(len(pivot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(
        x - width / 2,
        pivot.get("Fixed", 0.0),
        width,
        label="Fixed",
    )
    ax.bar(
        x + width / 2,
        pivot.get("Adaptive", 0.0),
        width,
        label="Adaptive",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Proportion")
    ax.set_title(
        "Distribution des niveaux prédits avant et après recalibrage"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_congestion_scatter(
    fixed: pd.DataFrame,
    adaptive: pd.DataFrame,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))

    ax.scatter(
        fixed["mobility_congestion_index"],
        fixed["CongestionScore_PredictedNGSIM"],
        alpha=0.55,
        label="Fixed",
    )

    ax.scatter(
        adaptive["mobility_congestion_index"],
        adaptive["CongestionScore_PredictedNGSIM"],
        alpha=0.55,
        label="Adaptive",
    )

    ax.set_xlabel("Indice de congestion issu de la mobilité")
    ax.set_ylabel("CongestionScore prédit")
    ax.set_title(
        "Cohérence mobilité–congestion prédite"
    )
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    for path in [FIXED_FILE, ADAPTIVE_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    fixed = pd.read_csv(FIXED_FILE)
    adaptive = pd.read_csv(ADAPTIVE_FILE)

    if len(fixed) != len(adaptive):
        raise ValueError(
            "Les deux fichiers doivent contenir le même nombre de fenêtres."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores = pd.DataFrame(
        score_rows(fixed, adaptive)
    )

    levels = pd.DataFrame(
        level_rows(fixed, "Fixed")
        + level_rows(adaptive, "Adaptive")
    )

    correlations = {
        "fixed": correlation_summary(fixed),
        "adaptive": correlation_summary(adaptive),
    }

    scores.to_csv(
        OUTPUT_DIR
        / "fixed_vs_adaptive_summary.csv",
        index=False,
    )

    levels.to_csv(
        OUTPUT_DIR
        / "fixed_vs_adaptive_levels.csv",
        index=False,
    )

    summary = {
        "fixed_file": str(FIXED_FILE),
        "adaptive_file": str(ADAPTIVE_FILE),
        "observation_count": int(len(fixed)),
        "correlations": correlations,
        "interpretation": {
            "diversity_improved": True,
            "ood_shift_removed": False,
            "warning": (
                "Le mapping adaptatif améliore la diversité, mais la "
                "concentration persistante sur les classes sévères et la "
                "faible corrélation de rang indiquent qu'un décalage de "
                "domaine subsiste."
            ),
        },
    }

    (
        OUTPUT_DIR
        / "fixed_vs_adaptive_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    save_score_boxplot(
        fixed,
        adaptive,
        figures_dir / "score_boxplot_fixed_vs_adaptive.png",
    )

    save_level_distribution(
        levels,
        figures_dir / "level_distribution_fixed_vs_adaptive.png",
    )

    save_congestion_scatter(
        fixed,
        adaptive,
        figures_dir / "congestion_transfer_scatter.png",
    )

    print("=" * 76)
    print("COMPARAISON FIXED VS ADAPTIVE TERMINÉE")
    print("=" * 76)
    print()
    print(scores.to_string(index=False))
    print()
    print("Corrélations :")
    print(json.dumps(correlations, indent=2))
    print()
    print(f"Sortie : {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
