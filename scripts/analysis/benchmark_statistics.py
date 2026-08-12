#!/usr/bin/env python3
"""
Analyse statistique du benchmark LLM V2X.

Ce script exploite les résultats complets des trois LLM évalués sur
190 scénarios et produit :

- intervalles de confiance bootstrap à 95 % pour l'accuracy, le Macro-F1
  actif, le Weighted-F1 et les latences ;
- tests exacts de McNemar entre chaque paire de modèles ;
- tests de permutation appariés sur la différence d'accuracy ;
- tableaux CSV ;
- figures ;
- tableaux LaTeX ;
- résumé scientifique automatique.

La référence principale est, par défaut, la décision issue de la politique
appliquée aux états réseau prédits par le pipeline ML.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

ROOT = Path.home() / "v2x-llm-project"

MODEL_DIRS = {
    "Llama 3.2 1B": (
        ROOT
        / "results"
        / "llm_evaluation"
        / "llama3_2_1b_pilot_190"
    ),
    "Qwen 2.5 3B": (
        ROOT
        / "results"
        / "llm_evaluation"
        / "qwen2_5_3b_pilot_190"
    ),
    "DeepSeek-R1 8B": (
        ROOT
        / "results"
        / "llm_evaluation"
        / "deepseek-r1_8b_pilot_190"
    ),
}

LABELS = [
    "KEEP_CURRENT_CONNECTION",
    "MONITOR_NETWORK",
    "REDUCE_TRAFFIC_LOAD",
    "PREPARE_HANDOVER",
    "HANDOVER_NOW",
]

DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "llm_evaluation"
    / "benchmark_statistics_190"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        choices=[
            "predicted_state_policy_decision",
            "dataset_policy_decision",
        ],
        default="predicted_state_policy_decision",
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        default=190,
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def load_results(
    directory: Path,
    expected_count: int,
) -> pd.DataFrame:
    path = directory / "pilot_results.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "scenario",
        "status",
        "dataset_policy_decision",
        "predicted_state_policy_decision",
        "llm_decision",
        "elapsed_seconds",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path}: {missing}"
        )

    successful = df[
        df["status"].astype(str).str.upper() == "SUCCESS"
    ].copy()

    if len(successful) != expected_count:
        raise ValueError(
            f"{directory.name}: {len(successful)} succès, "
            f"{expected_count} attendus."
        )

    if successful["scenario"].nunique() != expected_count:
        raise ValueError(
            f"{directory.name}: scénarios dupliqués ou absents."
        )

    return successful.sort_values("scenario").reset_index(drop=True)


def active_labels(y_true: pd.Series) -> list[str]:
    present = set(y_true.astype(str))
    return [label for label in LABELS if label in present]


def metric_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(accuracy_score(y_true, y_pred))


def metric_macro_f1_active(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    labels = [
        label
        for label in LABELS
        if label in set(y_true.tolist())
    ]

    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def metric_weighted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        )
    )


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    n = len(y_true)
    values = np.empty(iterations, dtype=float)

    for index in range(iterations):
        sample_indices = rng.integers(
            0,
            n,
            size=n,
        )
        values[index] = metric(
            y_true[sample_indices],
            y_pred[sample_indices],
        )

    estimate = metric(y_true, y_pred)

    return (
        float(estimate),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
        float(np.std(values, ddof=1)),
    )


def bootstrap_latency(
    values: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> list[dict[str, float | str]]:
    statistics = {
        "MeanLatency": np.mean,
        "MedianLatency": np.median,
        "P95Latency": lambda x: np.quantile(x, 0.95),
    }

    rows = []

    for name, function in statistics.items():
        samples = np.empty(iterations, dtype=float)

        for index in range(iterations):
            resampled = rng.choice(
                values,
                size=len(values),
                replace=True,
            )
            samples[index] = float(function(resampled))

        rows.append(
            {
                "Metric": name,
                "Estimate": float(function(values)),
                "CI95Low": float(
                    np.percentile(samples, 2.5)
                ),
                "CI95High": float(
                    np.percentile(samples, 97.5)
                ),
                "BootstrapStd": float(
                    np.std(samples, ddof=1)
                ),
            }
        )

    return rows


def exact_binomial_two_sided(
    successes: int,
    trials: int,
    probability: float = 0.5,
) -> float:
    if trials == 0:
        return 1.0

    observed_probability = math.comb(
        trials,
        successes,
    ) * (
        probability ** successes
    ) * (
        (1.0 - probability) ** (trials - successes)
    )

    total = 0.0

    for k in range(trials + 1):
        probability_k = math.comb(
            trials,
            k,
        ) * (
            probability ** k
        ) * (
            (1.0 - probability) ** (trials - k)
        )

        if probability_k <= observed_probability + 1e-15:
            total += probability_k

    return float(min(1.0, total))


def mcnemar_exact(
    true_values: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, float | int]:
    correct_a = pred_a == true_values
    correct_b = pred_b == true_values

    b = int(
        np.sum(
            np.logical_and(
                correct_a,
                np.logical_not(correct_b),
            )
        )
    )

    c = int(
        np.sum(
            np.logical_and(
                np.logical_not(correct_a),
                correct_b,
            )
        )
    )

    discordant = b + c
    p_value = exact_binomial_two_sided(
        min(b, c),
        discordant,
    )

    if c == 0:
        odds_ratio = math.inf if b > 0 else 1.0
    else:
        odds_ratio = b / c

    return {
        "A_correct_B_wrong": b,
        "A_wrong_B_correct": c,
        "DiscordantPairs": discordant,
        "OddsRatio_A_over_B": odds_ratio,
        "PValue": p_value,
        "Significant_0_05": bool(p_value < 0.05),
    }


def paired_permutation_accuracy(
    true_values: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    score_a = (
        pred_a == true_values
    ).astype(float)

    score_b = (
        pred_b == true_values
    ).astype(float)

    differences = score_a - score_b
    observed = float(np.mean(differences))

    if np.allclose(differences, 0):
        return {
            "ObservedAccuracyDifference": observed,
            "PValue": 1.0,
        }

    extreme = 0

    for _ in range(iterations):
        signs = rng.choice(
            [-1.0, 1.0],
            size=len(differences),
        )
        permuted = float(
            np.mean(
                differences * signs
            )
        )

        if abs(permuted) >= abs(observed):
            extreme += 1

    p_value = (
        extreme + 1
    ) / (
        iterations + 1
    )

    return {
        "ObservedAccuracyDifference": observed,
        "PValue": float(p_value),
    }


def save_bootstrap_figure(
    bootstrap_df: pd.DataFrame,
    metric_name: str,
    output: Path,
) -> None:
    subset = bootstrap_df[
        bootstrap_df["Metric"] == metric_name
    ].copy()

    x = np.arange(len(subset))
    estimates = subset["Estimate"].to_numpy()
    lower = estimates - subset["CI95Low"].to_numpy()
    upper = subset["CI95High"].to_numpy() - estimates

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.errorbar(
        x,
        estimates,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(subset["Model"])
    ax.set_ylim(
        max(
            0.0,
            float(
                subset["CI95Low"].min()
            ) - 0.05,
        ),
        min(
            1.0,
            float(
                subset["CI95High"].max()
            ) + 0.05,
        ),
    )
    ax.set_ylabel(metric_name)
    ax.set_title(
        f"{metric_name} et intervalle de confiance bootstrap à 95 %"
    )
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_latency_figure(
    latency_df: pd.DataFrame,
    output: Path,
) -> None:
    subset = latency_df[
        latency_df["Metric"] == "MeanLatency"
    ].copy()

    x = np.arange(len(subset))
    estimates = subset["Estimate"].to_numpy()
    lower = estimates - subset["CI95Low"].to_numpy()
    upper = subset["CI95High"].to_numpy() - estimates

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.errorbar(
        x,
        estimates,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(subset["Model"])
    ax.set_ylabel("Temps moyen d'inférence (s)")
    ax.set_title(
        "Latence moyenne et intervalle de confiance bootstrap à 95 %"
    )
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
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


def write_bootstrap_latex(
    bootstrap_df: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Intervalles de confiance bootstrap à 95\,\% "
            r"des performances des LLM.}"
        ),
        r"\label{tab:llm_bootstrap_ci}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"\textbf{Modèle} & \textbf{Métrique} & "
            r"\textbf{Estimation} & \textbf{IC 95\,\% inf.} & "
            r"\textbf{IC 95\,\% sup.} & \textbf{Écart-type bootstrap} \\"
        ),
        r"\midrule",
    ]

    for _, row in bootstrap_df.iterrows():
        lines.append(
            f"{latex_escape(row['Model'])} & "
            f"{latex_escape(row['Metric'])} & "
            f"{row['Estimate']:.4f} & "
            f"{row['CI95Low']:.4f} & "
            f"{row['CI95High']:.4f} & "
            f"{row['BootstrapStd']:.4f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_mcnemar_latex(
    mcnemar_df: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Tests exacts de McNemar entre les modèles "
            r"de langage.}"
        ),
        r"\label{tab:llm_mcnemar}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        (
            r"\textbf{Modèle A} & \textbf{Modèle B} & "
            r"\textbf{$n_{10}$} & \textbf{$n_{01}$} & "
            r"\textbf{Paires discordantes} & \textbf{$p$-value} & "
            r"\textbf{Conclusion} \\"
        ),
        r"\midrule",
    ]

    for _, row in mcnemar_df.iterrows():
        conclusion = (
            "Différence significative"
            if row["Significant_0_05"]
            else "Non significative"
        )

        lines.append(
            f"{latex_escape(row['ModelA'])} & "
            f"{latex_escape(row['ModelB'])} & "
            f"{int(row['A_correct_B_wrong'])} & "
            f"{int(row['A_wrong_B_correct'])} & "
            f"{int(row['DiscordantPairs'])} & "
            f"{row['PValue']:.4f} & "
            f"{latex_escape(conclusion)} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_summary(
    bootstrap_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    reference: str,
    iterations: int,
) -> str:
    lines = [
        "ANALYSE STATISTIQUE DU BENCHMARK LLM V2X",
        "=" * 76,
        "",
        f"Référence principale : {reference}",
        f"Rééchantillonnages bootstrap : {iterations}",
        "",
        "INTERVALLES DE CONFIANCE",
        "-" * 76,
    ]

    for model in bootstrap_df["Model"].unique():
        lines.append(f"\n{model}")

        subset = bootstrap_df[
            bootstrap_df["Model"] == model
        ]

        for _, row in subset.iterrows():
            lines.append(
                f"  {row['Metric']} = {row['Estimate']:.4f} "
                f"[{row['CI95Low']:.4f}, {row['CI95High']:.4f}]"
            )

    lines.extend(
        [
            "",
            "TESTS EXACTS DE McNEMAR",
            "-" * 76,
        ]
    )

    for _, row in mcnemar_df.iterrows():
        conclusion = (
            "significative"
            if row["Significant_0_05"]
            else "non significative"
        )

        lines.append(
            f"{row['ModelA']} vs {row['ModelB']} | "
            f"p={row['PValue']:.6f} | différence {conclusion}"
        )

    lines.extend(
        [
            "",
            "TESTS DE PERMUTATION APPARIÉS",
            "-" * 76,
        ]
    )

    for _, row in permutation_df.iterrows():
        conclusion = (
            "significative"
            if row["Significant_0_05"]
            else "non significative"
        )

        lines.append(
            f"{row['ModelA']} vs {row['ModelB']} | "
            f"écart d'accuracy={row['ObservedAccuracyDifference']:.4f} | "
            f"p={row['PValue']:.6f} | différence {conclusion}"
        )

    lines.extend(
        [
            "",
            "PRÉCAUTIONS MÉTHODOLOGIQUES",
            "-" * 76,
            (
                "Les comparaisons sont appariées : chaque modèle est évalué "
                "sur les mêmes 190 scénarios."
            ),
            (
                "Le test exact de McNemar évalue les désaccords de correction "
                "entre deux modèles, et non la différence entre leurs scores "
                "multicritères."
            ),
            (
                "Le bootstrap quantifie l'incertitude d'échantillonnage, mais "
                "ne corrige ni les biais du dataset simulé ni l'effondrement "
                "vers une classe dominante."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(args.seed)

    model_results = {
        model: load_results(
            directory,
            args.scenario_count,
        )
        for model, directory in MODEL_DIRS.items()
    }

    scenario_reference = None

    for model, df in model_results.items():
        current = (
            df[
                [
                    "scenario",
                    args.reference,
                ]
            ]
            .copy()
            .rename(
                columns={
                    args.reference: "candidate_reference",
                }
            )
        )

        if scenario_reference is None:
            scenario_reference = current.rename(
                columns={
                    "candidate_reference": "reference",
                }
            )
        else:
            merged = scenario_reference.merge(
                current,
                on="scenario",
                how="inner",
                validate="one_to_one",
            )

            if len(merged) != args.scenario_count:
                raise ValueError(
                    "Les modèles ne couvrent pas exactement les mêmes scénarios."
                )

            if not (
                merged["reference"].astype(str)
                == merged["candidate_reference"].astype(str)
            ).all():
                raise ValueError(
                    f"Références incohérentes pour {model}."
                )

    bootstrap_rows = []
    latency_rows = []

    for model, df in model_results.items():
        y_true = df[
            args.reference
        ].astype(str).to_numpy()

        y_pred = df[
            "llm_decision"
        ].astype(str).to_numpy()

        metric_functions = {
            "Accuracy": metric_accuracy,
            "ActiveMacroF1": metric_macro_f1_active,
            "WeightedF1": metric_weighted_f1,
        }

        for metric_name, function in metric_functions.items():
            estimate, low, high, std = bootstrap_metric(
                y_true,
                y_pred,
                function,
                args.bootstrap_iterations,
                rng,
            )

            bootstrap_rows.append(
                {
                    "Model": model,
                    "Metric": metric_name,
                    "Estimate": estimate,
                    "CI95Low": low,
                    "CI95High": high,
                    "BootstrapStd": std,
                }
            )

        latency = df[
            "elapsed_seconds"
        ].astype(float).to_numpy()

        for row in bootstrap_latency(
            latency,
            args.bootstrap_iterations,
            rng,
        ):
            row["Model"] = model
            latency_rows.append(row)

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    latency_df = pd.DataFrame(
        latency_rows
    )

    mcnemar_rows = []
    permutation_rows = []

    for model_a, model_b in itertools.combinations(
        MODEL_DIRS.keys(),
        2,
    ):
        df_a = model_results[model_a][
            [
                "scenario",
                args.reference,
                "llm_decision",
            ]
        ].rename(
            columns={
                args.reference: "reference",
                "llm_decision": "prediction_a",
            }
        )

        df_b = model_results[model_b][
            [
                "scenario",
                "llm_decision",
            ]
        ].rename(
            columns={
                "llm_decision": "prediction_b",
            }
        )

        aligned = df_a.merge(
            df_b,
            on="scenario",
            how="inner",
        )

        y_true = aligned[
            "reference"
        ].astype(str).to_numpy()

        pred_a = aligned[
            "prediction_a"
        ].astype(str).to_numpy()

        pred_b = aligned[
            "prediction_b"
        ].astype(str).to_numpy()

        mcnemar_result = mcnemar_exact(
            y_true,
            pred_a,
            pred_b,
        )

        mcnemar_rows.append(
            {
                "ModelA": model_a,
                "ModelB": model_b,
                **mcnemar_result,
            }
        )

        permutation_result = paired_permutation_accuracy(
            y_true,
            pred_a,
            pred_b,
            args.permutation_iterations,
            rng,
        )

        permutation_rows.append(
            {
                "ModelA": model_a,
                "ModelB": model_b,
                **permutation_result,
                "Significant_0_05": bool(
                    permutation_result["PValue"] < 0.05
                ),
            }
        )

    mcnemar_df = pd.DataFrame(
        mcnemar_rows
    )

    permutation_df = pd.DataFrame(
        permutation_rows
    )

    bootstrap_df.to_csv(
        output_dir
        / "bootstrap_performance_intervals.csv",
        index=False,
    )

    latency_df.to_csv(
        output_dir
        / "bootstrap_latency_intervals.csv",
        index=False,
    )

    mcnemar_df.to_csv(
        output_dir
        / "mcnemar_exact_tests.csv",
        index=False,
    )

    permutation_df.to_csv(
        output_dir
        / "paired_permutation_tests.csv",
        index=False,
    )

    save_bootstrap_figure(
        bootstrap_df,
        "Accuracy",
        figure_dir
        / "bootstrap_accuracy_ci.png",
    )

    save_bootstrap_figure(
        bootstrap_df,
        "ActiveMacroF1",
        figure_dir
        / "bootstrap_active_macro_f1_ci.png",
    )

    save_bootstrap_figure(
        bootstrap_df,
        "WeightedF1",
        figure_dir
        / "bootstrap_weighted_f1_ci.png",
    )

    save_latency_figure(
        latency_df,
        figure_dir
        / "bootstrap_mean_latency_ci.png",
    )

    write_bootstrap_latex(
        bootstrap_df,
        table_dir
        / "llm_bootstrap_confidence_intervals.tex",
    )

    write_mcnemar_latex(
        mcnemar_df,
        table_dir
        / "llm_mcnemar_tests.tex",
    )

    summary = build_summary(
        bootstrap_df,
        mcnemar_df,
        permutation_df,
        args.reference,
        args.bootstrap_iterations,
    )

    summary_path = (
        output_dir
        / "statistics_summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    manifest = {
        "reference": args.reference,
        "scenario_count": args.scenario_count,
        "bootstrap_iterations": args.bootstrap_iterations,
        "permutation_iterations": args.permutation_iterations,
        "seed": args.seed,
        "models": {
            model: str(directory)
            for model, directory in MODEL_DIRS.items()
        },
        "active_classes": (
            active_labels(
                scenario_reference[
                    "reference"
                ]
            )
            if scenario_reference is not None
            else []
        ),
    }

    (
        output_dir
        / "statistics_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("ANALYSE STATISTIQUE LLM TERMINÉE")
    print("=" * 76)
    print()
    print("Intervalles bootstrap :")
    print(
        bootstrap_df.to_string(
            index=False
        )
    )
    print()
    print("Tests exacts de McNemar :")
    print(
        mcnemar_df[
            [
                "ModelA",
                "ModelB",
                "A_correct_B_wrong",
                "A_wrong_B_correct",
                "PValue",
                "Significant_0_05",
            ]
        ].to_string(
            index=False
        )
    )
    print()
    print("Tests de permutation :")
    print(
        permutation_df.to_string(
            index=False
        )
    )
    print()
    print(f"Sortie  : {output_dir}")
    print(f"Résumé  : {summary_path}")
    print(f"Figures : {figure_dir}")
    print(f"Tables  : {table_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
