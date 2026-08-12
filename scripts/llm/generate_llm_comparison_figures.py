#!/usr/bin/env python3
"""
Génère les figures et tableaux comparatifs des modèles LLM V2X.

Entrées attendues :
- results/llm_evaluation/llm_model_comparison.csv
- results/llm_evaluation/<model_folder>/pilot_results.csv

Sorties :
- results/llm_evaluation/figures/
- results/llm_evaluation/llm_model_comparison.tex
- results/llm_evaluation/decision_distribution.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path.home() / "v2x-llm-project"
EVAL_DIR = PROJECT_DIR / "results" / "llm_evaluation"
COMPARISON_CSV = EVAL_DIR / "llm_model_comparison.csv"
FIGURE_DIR = EVAL_DIR / "figures"

MODEL_RESULT_DIRS = {
    "Llama 3.2 1B": EVAL_DIR / "llama3_2_1b_pilot_20",
    "Qwen 2.5 3B": EVAL_DIR / "qwen2_5_3b_pilot_20",
    "DeepSeek-R1 8B": EVAL_DIR / "deepseek-r1_8b_pilot_20",
}

DECISION_ORDER = [
    "KEEP_CURRENT_CONNECTION",
    "MONITOR_NETWORK",
    "REDUCE_TRAFFIC_LOAD",
    "PREPARE_HANDOVER",
    "HANDOVER_NOW",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Générer les figures comparatives des modèles LLM."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Résolution des figures PNG.",
    )
    return parser.parse_args()


def validate_inputs() -> None:
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError(
            f"Fichier comparatif introuvable : {COMPARISON_CSV}"
        )

    missing = []
    for model_name, folder in MODEL_RESULT_DIRS.items():
        path = folder / "pilot_results.csv"
        if not path.exists():
            missing.append(f"{model_name}: {path}")

    if missing:
        raise FileNotFoundError(
            "Fichiers pilot_results.csv manquants :\n- "
            + "\n- ".join(missing)
        )


def save_performance_figure(df: pd.DataFrame, dpi: int) -> Path:
    models = df["Model"].tolist()
    x = list(range(len(models)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        [value - width / 2 for value in x],
        df["Accuracy"],
        width=width,
        label="Accuracy",
    )
    ax.bar(
        [value + width / 2 for value in x],
        df["Macro F1"],
        width=width,
        label="Macro-F1",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Comparaison des performances des modèles LLM")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    output = FIGURE_DIR / "llm_accuracy_macro_f1.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output


def save_latency_figure(df: pd.DataFrame, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        df["Model"],
        df["Mean Time (s)"],
    )

    ax.set_ylabel("Temps moyen d'inférence (s)")
    ax.set_title("Comparaison du temps moyen d'inférence")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    output = FIGURE_DIR / "llm_inference_latency.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output


def save_tradeoff_figure(df: pd.DataFrame, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in df.iterrows():
        ax.scatter(
            row["Mean Time (s)"],
            row["Macro F1"],
            s=90,
        )
        ax.annotate(
            row["Model"],
            (
                row["Mean Time (s)"],
                row["Macro F1"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.set_xlabel("Temps moyen d'inférence (s)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Compromis performance–latence")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    output = FIGURE_DIR / "llm_macro_f1_latency_tradeoff.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output


def build_decision_distribution() -> pd.DataFrame:
    rows = []

    for model_name, folder in MODEL_RESULT_DIRS.items():
        path = folder / "pilot_results.csv"
        df = pd.read_csv(path)

        counts = (
            df["llm_decision"]
            .value_counts()
            .reindex(DECISION_ORDER, fill_value=0)
        )

        for decision, count in counts.items():
            rows.append(
                {
                    "Model": model_name,
                    "Decision": decision,
                    "Count": int(count),
                }
            )

    distribution = pd.DataFrame(rows)

    output = EVAL_DIR / "decision_distribution.csv"
    distribution.to_csv(output, index=False)

    return distribution


def save_decision_distribution_figure(
    distribution: pd.DataFrame,
    dpi: int,
) -> Path:
    pivot = (
        distribution
        .pivot(
            index="Model",
            columns="Decision",
            values="Count",
        )
        .reindex(columns=DECISION_ORDER, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)

    ax.set_xlabel("Modèle")
    ax.set_ylabel("Nombre de décisions")
    ax.set_title("Distribution des décisions produites par les LLM")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(
        title="Décision",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    output = FIGURE_DIR / "llm_decision_distribution.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output


def save_latex_table(df: pd.DataFrame) -> Path:
    formatted = df.copy()

    numeric_columns = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "JSON Valid Rate",
        "Mean Time (s)",
        "Median Time (s)",
    ]

    for column in numeric_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.3f}"
            )

    output = EVAL_DIR / "llm_model_comparison.tex"

    latex = formatted.to_latex(
        index=False,
        escape=True,
        caption=(
            "Comparaison des performances et du temps d'inférence "
            "des modèles LLM pour la prise de décision V2X."
        ),
        label="tab:llm_model_comparison",
        longtable=False,
    )

    output.write_text(latex, encoding="utf-8")

    return output


def main() -> int:
    args = parse_args()

    validate_inputs()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(COMPARISON_CSV)

    required_columns = {
        "Model",
        "Accuracy",
        "Macro F1",
        "Mean Time (s)",
    }

    missing = required_columns - set(comparison.columns)

    if missing:
        raise ValueError(
            f"Colonnes absentes dans {COMPARISON_CSV}: {sorted(missing)}"
        )

    generated_files = []

    generated_files.append(
        save_performance_figure(comparison, args.dpi)
    )
    generated_files.append(
        save_latency_figure(comparison, args.dpi)
    )
    generated_files.append(
        save_tradeoff_figure(comparison, args.dpi)
    )

    distribution = build_decision_distribution()

    generated_files.append(
        save_decision_distribution_figure(
            distribution,
            args.dpi,
        )
    )

    generated_files.append(
        save_latex_table(comparison)
    )

    print("=" * 72)
    print("GÉNÉRATION DES FIGURES LLM TERMINÉE")
    print("=" * 72)

    for path in generated_files:
        print(f"- {path}")

    print(
        f"- {EVAL_DIR / 'decision_distribution.csv'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
