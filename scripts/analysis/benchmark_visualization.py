#!/usr/bin/env python3
"""
Génération harmonisée des figures et tableaux du benchmark LLM V2X.

Ce script agrège les résultats déjà produits par :
- final_llm_benchmark.py
- benchmark_statistics.py
- benchmark_sensitivity.py

Il ne relance aucune inférence LLM.

Sorties principales :
- figures publication homogènes ;
- tableaux CSV consolidés ;
- tableaux LaTeX ;
- manifeste de reproductibilité.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.home() / "v2x-llm-project"

FINAL_DIR = ROOT / "results/llm_evaluation/final_benchmark_190"
STATS_DIR = ROOT / "results/llm_evaluation/benchmark_statistics_190"
SENS_DIR = ROOT / "results/llm_evaluation/benchmark_sensitivity_190"
DEFAULT_OUTPUT = ROOT / "results/llm_evaluation/benchmark_visualization_190"

LABELS = [
    "KEEP_CURRENT_CONNECTION",
    "MONITOR_NETWORK",
    "REDUCE_TRAFFIC_LOAD",
    "PREPARE_HANDOVER",
    "HANDOVER_NOW",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def save_accuracy_f1(comparison: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(comparison))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, comparison["Accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, comparison["MacroF1"], width, label="Macro-F1")

    ax.set_xticks(x)
    ax.set_xticklabels(comparison["Model"])
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des performances globales")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_latency(comparison: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(comparison["Model"], comparison["MeanInferenceSeconds"])
    ax.set_ylabel("Temps moyen d'inférence (s)")
    ax.set_title("Latence moyenne d'inférence")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff(comparison: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(
        comparison["MeanInferenceSeconds"],
        comparison["MacroF1"],
        s=100,
    )

    for _, row in comparison.iterrows():
        ax.annotate(
            row["Model"],
            (row["MeanInferenceSeconds"], row["MacroF1"]),
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.set_xlabel("Temps moyen d'inférence (s)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Compromis performance–latence")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_distribution(distribution: pd.DataFrame, output: Path) -> None:
    pivot = (
        distribution
        .pivot(index="Decision", columns="Model", values="Count")
        .reindex(LABELS)
        .fillna(0)
    )

    x = np.arange(len(pivot))
    width = 0.8 / len(pivot.columns)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, model in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[model], width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel("Nombre de décisions")
    ax.set_title("Distribution des décisions produites")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_per_class_f1(per_class: pd.DataFrame, output: Path) -> None:
    pivot = (
        per_class
        .pivot(index="Class", columns="Model", values="F1")
        .reindex(LABELS)
        .fillna(0)
    )

    x = np.arange(len(pivot))
    width = 0.8 / len(pivot.columns)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, model in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[model], width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score par classe de décision")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_bootstrap_ci(
    bootstrap: pd.DataFrame,
    metric: str,
    output: Path,
) -> None:
    subset = bootstrap[bootstrap["Metric"] == metric].copy()
    x = np.arange(len(subset))
    estimate = subset["Estimate"].to_numpy()
    lower = estimate - subset["CI95Low"].to_numpy()
    upper = subset["CI95High"].to_numpy() - estimate

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(
        x,
        estimate,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(subset["Model"])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} avec intervalle de confiance bootstrap à 95 %")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_profile_heatmap(profile_rankings: pd.DataFrame, output: Path) -> None:
    pivot = profile_rankings.pivot(
        index="Model",
        columns="Profile",
        values="Rank",
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Rang des modèles selon le profil multicritère")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, str(int(pivot.iloc[i, j])), ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Rang")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_sensitivity_curve(continuous: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, group in continuous.groupby("Model"):
        ordered = group.sort_values("LatencyWeight")
        ax.plot(
            ordered["LatencyWeight"],
            ordered["Score"],
            linewidth=2,
            label=model,
        )

    ax.set_xlabel("Poids attribué à la latence")
    ax.set_ylabel("Score multicritère")
    ax.set_title("Sensibilité du score multicritère à la latence")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dual_reference(dual: pd.DataFrame, output: Path) -> None:
    pivot = dual.pivot(
        index="Model",
        columns="Reference",
        values="Accuracy",
    )

    x = np.arange(len(pivot))
    width = 0.8 / len(pivot.columns)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for idx, ref in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[ref], width, label=ref)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy selon la politique de référence")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }.items():
        text = text.replace(old, new)
    return text


def write_master_table(
    comparison: pd.DataFrame,
    active_f1: pd.DataFrame,
    output: Path,
) -> None:
    merged = comparison.merge(active_f1, on="Model", how="left")

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Synthèse finale des performances des modèles de langage.}",
        r"\label{tab:llm_visualization_master}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Modèle} & \textbf{Accuracy} & "
            r"\textbf{Balanced Acc.} & \textbf{Macro-F1} & "
            r"\textbf{Macro-F1 actif} & \textbf{Temps moyen (s)} & "
            r"\textbf{P95 (s)} \\"
        ),
        r"\midrule",
    ]

    for _, row in merged.iterrows():
        lines.append(
            f"{latex_escape(row['Model'])} & "
            f"{row['Accuracy']:.4f} & "
            f"{row['BalancedAccuracy']:.4f} & "
            f"{row['MacroF1']:.4f} & "
            f"{row['ActiveMacroF1']:.4f} & "
            f"{row['MeanInferenceSeconds']:.2f} & "
            f"{row['P95InferenceSeconds']:.2f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]

    output.write_text("\n".join(lines), encoding="utf-8")


def write_figures_tex(output_dir: Path, output: Path) -> None:
    figures = [
        ("llm_accuracy_macro_f1.png", "Comparaison de l'accuracy et du Macro-F1.", "fig:llm_acc_f1_final", "0.82"),
        ("llm_latency.png", "Comparaison des temps moyens d'inférence.", "fig:llm_latency_final", "0.78"),
        ("llm_tradeoff.png", "Compromis entre Macro-F1 et latence.", "fig:llm_tradeoff_final", "0.78"),
        ("llm_decision_distribution.png", "Distribution des décisions produites par les modèles.", "fig:llm_distribution_final", "0.92"),
        ("llm_per_class_f1.png", "F1-score par classe de décision.", "fig:llm_class_f1_final", "0.92"),
        ("bootstrap_accuracy_ci.png", "Accuracy et intervalles de confiance bootstrap à 95\\,\\%.", "fig:llm_bootstrap_acc_final", "0.78"),
        ("bootstrap_active_macro_f1_ci.png", "Macro-F1 actif et intervalles de confiance bootstrap à 95\\,\\%.", "fig:llm_bootstrap_f1_final", "0.78"),
        ("profile_ranking_heatmap.png", "Rang des modèles selon les profils multicritères.", "fig:llm_profile_heatmap_final", "0.82"),
        ("continuous_latency_score.png", "Sensibilité du score au poids de la latence.", "fig:llm_latency_sensitivity_final", "0.86"),
        ("dual_reference_accuracy.png", "Accuracy selon la politique de référence.", "fig:llm_dual_ref_final", "0.82"),
    ]

    lines = []
    for filename, caption, label, width in figures:
        lines += [
            r"\begin{figure}[!htbp]",
            r"\centering",
            rf"\includegraphics[width={width}\textwidth]{{figures/llm_benchmark/{filename}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
            "",
        ]

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    figures = out / "figures"
    tables = out / "tables"

    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(require(FINAL_DIR / "llm_final_comparison.csv"))
    distribution = pd.read_csv(require(FINAL_DIR / "llm_decision_distribution.csv"))
    per_class = pd.read_csv(require(FINAL_DIR / "llm_per_class_metrics.csv"))
    dual = pd.read_csv(require(FINAL_DIR / "llm_dual_reference_metrics.csv"))
    bootstrap = pd.read_csv(require(STATS_DIR / "bootstrap_performance_intervals.csv"))
    profiles = pd.read_csv(require(SENS_DIR / "profile_rankings.csv"))
    continuous = pd.read_csv(require(SENS_DIR / "continuous_latency_sensitivity.csv"))

    active_f1 = (
        bootstrap[bootstrap["Metric"] == "ActiveMacroF1"]
        [["Model", "Estimate"]]
        .rename(columns={"Estimate": "ActiveMacroF1"})
    )

    save_accuracy_f1(comparison, figures / "llm_accuracy_macro_f1.png")
    save_latency(comparison, figures / "llm_latency.png")
    save_tradeoff(comparison, figures / "llm_tradeoff.png")
    save_distribution(distribution, figures / "llm_decision_distribution.png")
    save_per_class_f1(per_class, figures / "llm_per_class_f1.png")
    save_bootstrap_ci(bootstrap, "Accuracy", figures / "bootstrap_accuracy_ci.png")
    save_bootstrap_ci(
        bootstrap,
        "ActiveMacroF1",
        figures / "bootstrap_active_macro_f1_ci.png",
    )
    save_profile_heatmap(profiles, figures / "profile_ranking_heatmap.png")
    save_sensitivity_curve(continuous, figures / "continuous_latency_score.png")
    save_dual_reference(dual, figures / "dual_reference_accuracy.png")

    write_master_table(
        comparison,
        active_f1,
        tables / "llm_master_comparison.tex",
    )
    write_figures_tex(
        out,
        tables / "llm_benchmark_figures.tex",
    )

    merged = comparison.merge(active_f1, on="Model", how="left")
    merged.to_csv(out / "llm_visualization_master.csv", index=False)

    manifest = {
        "final_benchmark_dir": str(FINAL_DIR),
        "statistics_dir": str(STATS_DIR),
        "sensitivity_dir": str(SENS_DIR),
        "output_dir": str(out),
        "figure_count": 10,
    }
    (out / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("VISUALISATION FINALE DU BENCHMARK LLM TERMINÉE")
    print("=" * 76)
    print(f"Sortie  : {out}")
    print(f"Figures : {figures}")
    print(f"Tables  : {tables}")
    print()
    print(
        merged[
            [
                "Model",
                "Accuracy",
                "BalancedAccuracy",
                "MacroF1",
                "ActiveMacroF1",
                "MeanInferenceSeconds",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
