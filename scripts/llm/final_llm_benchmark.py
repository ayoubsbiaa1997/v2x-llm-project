#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path.home() / "v2x-llm-project"
OUT_DEFAULT = ROOT / "results/llm_evaluation/final_benchmark_190"

LABELS = [
    "KEEP_CURRENT_CONNECTION",
    "MONITOR_NETWORK",
    "REDUCE_TRAFFIC_LOAD",
    "PREPARE_HANDOVER",
    "HANDOVER_NOW",
]

MODELS = {
    "Llama 3.2 1B": ROOT / "results/llm_evaluation/llama3_2_1b_pilot_190",
    "Qwen 2.5 3B": ROOT / "results/llm_evaluation/qwen2_5_3b_pilot_190",
    "DeepSeek-R1 8B": ROOT / "results/llm_evaluation/deepseek-r1_8b_pilot_190",
}

WEIGHTS = {
    "accuracy": 0.25,
    "macro_f1": 0.35,
    "macro_precision": 0.10,
    "macro_recall": 0.10,
    "json_valid_rate": 0.05,
    "latency_score": 0.15,
}


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
    parser.add_argument("--scenario-count", type=int, default=190)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Objet JSON attendu : {path}")
    return data


def normalize(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = values.astype(float)
    lo, hi = float(values.min()), float(values.max())
    if math.isclose(lo, hi):
        result = pd.Series(np.ones(len(values)), index=values.index)
    else:
        result = (values - lo) / (hi - lo)
    return result if higher_is_better else 1.0 - result


def successful_rows(path: Path, expected: int) -> pd.DataFrame:
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
        raise ValueError(f"Colonnes manquantes dans {path}: {missing}")

    ok = df[df["status"].astype(str).str.upper() == "SUCCESS"].copy()
    if len(ok) != expected:
        raise ValueError(
            f"{path.parent.name}: {len(ok)} succès au lieu de {expected}."
        )
    if ok["scenario"].nunique() != expected:
        raise ValueError(f"{path.parent.name}: scénarios dupliqués ou absents.")
    return ok


def compute_metrics(
    model: str,
    df: pd.DataFrame,
    reference: str,
    metrics_json: dict,
) -> dict:
    y_true = df[reference].astype(str)
    y_pred = df["llm_decision"].astype(str)

    return {
        "Model": model,
        "N": len(df),
        "JSONValidRate": float(metrics_json.get("valid_json_rate", 1.0)),
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "MacroPrecision": precision_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "MacroRecall": recall_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "MacroF1": f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "WeightedF1": f1_score(
            y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
        ),
        "MeanInferenceSeconds": float(df["elapsed_seconds"].mean()),
        "MedianInferenceSeconds": float(df["elapsed_seconds"].median()),
        "P95InferenceSeconds": float(df["elapsed_seconds"].quantile(0.95)),
    }


def per_class(model: str, df: pd.DataFrame, reference: str) -> pd.DataFrame:
    report = classification_report(
        df[reference].astype(str),
        df["llm_decision"].astype(str),
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(
        [
            {
                "Model": model,
                "Class": label,
                "Precision": report[label]["precision"],
                "Recall": report[label]["recall"],
                "F1": report[label]["f1-score"],
                "Support": int(report[label]["support"]),
            }
            for label in LABELS
        ]
    )


def decision_distribution(model: str, df: pd.DataFrame) -> pd.DataFrame:
    counts = df["llm_decision"].value_counts().reindex(LABELS, fill_value=0)
    return pd.DataFrame(
        {
            "Model": model,
            "Decision": counts.index,
            "Count": counts.values,
            "Rate": counts.values / len(df),
        }
    )


def dual_reference(model: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ref, title in [
        ("predicted_state_policy_decision", "Politique issue des états ML"),
        ("dataset_policy_decision", "Politique du dataset"),
    ]:
        y_true = df[ref].astype(str)
        y_pred = df["llm_decision"].astype(str)
        rows.append(
            {
                "Model": model,
                "Reference": title,
                "Accuracy": accuracy_score(y_true, y_pred),
                "MacroF1": f1_score(
                    y_true,
                    y_pred,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(rows)


def save_confusion(
    model: str,
    df: pd.DataFrame,
    reference: str,
    output: Path,
) -> None:
    cm = confusion_matrix(
        df[reference].astype(str),
        df["llm_decision"].astype(str),
        labels=LABELS,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Décision LLM",
        ylabel="Décision de référence",
        title=f"Matrice de confusion — {model}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    x = np.arange(len(df))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, df[metric], width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def distribution_plot(df: pd.DataFrame, output: Path) -> None:
    pivot = (
        df.pivot(index="Decision", columns="Model", values="Count")
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


def tradeoff_plot(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df["MeanInferenceSeconds"], df["MacroF1"], s=90)
    for _, row in df.iterrows():
        ax.annotate(
            row["Model"],
            (row["MeanInferenceSeconds"], row["MacroF1"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Temps moyen d'inférence (s)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Compromis performance–latence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def radar_plot(df: pd.DataFrame, output: Path) -> None:
    columns = [
        "AccuracyNorm",
        "MacroF1Norm",
        "MacroPrecisionNorm",
        "MacroRecallNorm",
        "JSONValidRateNorm",
        "LatencyScore",
    ]
    labels = [
        "Accuracy",
        "Macro-F1",
        "Précision",
        "Rappel",
        "JSON valide",
        "Latence inverse",
    ]
    angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.5, 8))
    ax = fig.add_subplot(111, polar=True)
    for _, row in df.iterrows():
        values = [float(row[col]) for col in columns]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=row["Model"])
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Profil multicritère normalisé", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))
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


def write_latex_tables(
    comparison: pd.DataFrame,
    ranking: pd.DataFrame,
    table_dir: Path,
) -> None:
    comparison_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Comparaison globale des modèles de langage sur 190 scénarios V2X.}",
        r"\label{tab:llm_final_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"\textbf{Modèle} & \textbf{Accuracy} & \textbf{Balanced Acc.} & \textbf{Précision macro} & \textbf{Rappel macro} & \textbf{Macro-F1} & \textbf{JSON valide} & \textbf{Temps moyen (s)} \\",
        r"\midrule",
    ]
    for _, row in comparison.iterrows():
        comparison_lines.append(
            f"{latex_escape(row['Model'])} & "
            f"{row['Accuracy']:.4f} & "
            f"{row['BalancedAccuracy']:.4f} & "
            f"{row['MacroPrecision']:.4f} & "
            f"{row['MacroRecall']:.4f} & "
            f"{row['MacroF1']:.4f} & "
            f"{row['JSONValidRate']:.4f} & "
            f"{row['MeanInferenceSeconds']:.2f} \\\\"
        )
    comparison_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    (table_dir / "llm_final_comparison.tex").write_text(
        "\n".join(comparison_lines) + "\n",
        encoding="utf-8",
    )

    rank_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Classement multicritère final des modèles de langage.}",
        r"\label{tab:llm_final_ranking}",
        r"\begin{tabular}{clrr}",
        r"\toprule",
        r"\textbf{Rang} & \textbf{Modèle} & \textbf{Score} & \textbf{Temps moyen (s)} \\",
        r"\midrule",
    ]
    for _, row in ranking.sort_values("Rank").iterrows():
        rank_lines.append(
            f"{int(row['Rank'])} & {latex_escape(row['Model'])} & "
            f"{row['MulticriteriaScore']:.4f} & "
            f"{row['MeanInferenceSeconds']:.2f} \\\\"
        )
    rank_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (table_dir / "llm_final_ranking.tex").write_text(
        "\n".join(rank_lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    figures = out / "figures"
    tables = out / "tables"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    rows, class_frames, dist_frames, ref_frames = [], [], [], []
    all_results: dict[str, pd.DataFrame] = {}

    for model, directory in MODELS.items():
        metrics_path = directory / "metrics.json"
        results_path = directory / "pilot_results.csv"
        if not metrics_path.exists() or not results_path.exists():
            raise FileNotFoundError(f"Résultats absents pour {model}: {directory}")

        results = successful_rows(results_path, args.scenario_count)
        metrics = read_json(metrics_path)
        all_results[model] = results

        rows.append(compute_metrics(model, results, args.reference, metrics))
        class_frames.append(per_class(model, results, args.reference))
        dist_frames.append(decision_distribution(model, results))
        ref_frames.append(dual_reference(model, results))

        slug = (
            model.lower()
            .replace(" ", "_")
            .replace(".", "_")
            .replace("-", "_")
        )
        save_confusion(
            model,
            results,
            args.reference,
            figures / f"confusion_matrix_{slug}.png",
        )

    comparison = pd.DataFrame(rows)
    per_class_df = pd.concat(class_frames, ignore_index=True)
    distribution_df = pd.concat(dist_frames, ignore_index=True)
    references_df = pd.concat(ref_frames, ignore_index=True)

    ranking = comparison.copy()
    ranking["AccuracyNorm"] = normalize(ranking["Accuracy"])
    ranking["MacroF1Norm"] = normalize(ranking["MacroF1"])
    ranking["MacroPrecisionNorm"] = normalize(ranking["MacroPrecision"])
    ranking["MacroRecallNorm"] = normalize(ranking["MacroRecall"])
    ranking["JSONValidRateNorm"] = normalize(ranking["JSONValidRate"])
    ranking["LatencyScore"] = normalize(
        ranking["MeanInferenceSeconds"], higher_is_better=False
    )
    ranking["MulticriteriaScore"] = (
        WEIGHTS["accuracy"] * ranking["AccuracyNorm"]
        + WEIGHTS["macro_f1"] * ranking["MacroF1Norm"]
        + WEIGHTS["macro_precision"] * ranking["MacroPrecisionNorm"]
        + WEIGHTS["macro_recall"] * ranking["MacroRecallNorm"]
        + WEIGHTS["json_valid_rate"] * ranking["JSONValidRateNorm"]
        + WEIGHTS["latency_score"] * ranking["LatencyScore"]
    )
    ranking["Rank"] = (
        ranking["MulticriteriaScore"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    ranking = ranking.sort_values("Rank").reset_index(drop=True)

    comparison.to_csv(out / "llm_final_comparison.csv", index=False)
    ranking.to_csv(out / "llm_multicriteria_ranking.csv", index=False)
    per_class_df.to_csv(out / "llm_per_class_metrics.csv", index=False)
    distribution_df.to_csv(out / "llm_decision_distribution.csv", index=False)
    references_df.to_csv(out / "llm_dual_reference_metrics.csv", index=False)

    grouped_bars(
        comparison,
        ["Accuracy", "MacroF1"],
        "Accuracy et Macro-F1 des LLM",
        "Score",
        figures / "llm_accuracy_macro_f1.png",
    )
    grouped_bars(
        comparison,
        ["MeanInferenceSeconds"],
        "Temps moyen d'inférence",
        "Secondes",
        figures / "llm_inference_latency.png",
    )
    grouped_bars(
        ranking,
        ["MulticriteriaScore"],
        "Classement multicritère",
        "Score normalisé",
        figures / "llm_multicriteria_score.png",
    )
    distribution_plot(
        distribution_df,
        figures / "llm_decision_distribution.png",
    )
    tradeoff_plot(
        comparison,
        figures / "llm_macro_f1_latency_tradeoff.png",
    )
    radar_plot(
        ranking,
        figures / "llm_multicriteria_radar.png",
    )

    f1_pivot = (
        per_class_df.pivot(index="Class", columns="Model", values="F1")
        .reindex(LABELS)
        .fillna(0)
        .reset_index()
        .rename(columns={"Class": "Model"})
    )
    grouped_bars(
        f1_pivot,
        [col for col in f1_pivot.columns if col != "Model"],
        "F1-score par classe de décision",
        "F1-score",
        figures / "llm_per_class_f1.png",
    )

    write_latex_tables(comparison, ranking, tables)

    winner = ranking.iloc[0]
    best_f1 = comparison.loc[comparison["MacroF1"].idxmax()]
    fastest = comparison.loc[comparison["MeanInferenceSeconds"].idxmin()]

    summary = [
        "BENCHMARK FINAL DES LLM V2X",
        "=" * 72,
        "",
        f"Référence principale : {args.reference}",
        f"Scénarios par modèle : {args.scenario_count}",
        "",
        comparison[
            ["Model", "Accuracy", "BalancedAccuracy", "MacroF1", "MeanInferenceSeconds"]
        ].to_string(index=False),
        "",
        f"Meilleur Macro-F1 : {best_f1['Model']} ({best_f1['MacroF1']:.4f})",
        f"Modèle le plus rapide : {fastest['Model']} ({fastest['MeanInferenceSeconds']:.2f} s)",
        f"Premier au classement multicritère : {winner['Model']} ({winner['MulticriteriaScore']:.4f})",
        "",
        "Poids du score multicritère :",
        json.dumps(WEIGHTS, indent=2, ensure_ascii=False),
        "",
        "Remarque : le classement dépend des poids choisis. Une analyse de sensibilité",
        "doit accompagner toute sélection définitive du modèle.",
    ]
    (out / "benchmark_summary.txt").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "reference": args.reference,
        "scenario_count": args.scenario_count,
        "models": {name: str(path) for name, path in MODELS.items()},
        "weights": WEIGHTS,
    }
    (out / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("BENCHMARK FINAL DES LLM TERMINÉ")
    print("=" * 72)
    print(comparison[
        ["Model", "Accuracy", "BalancedAccuracy", "MacroF1", "MeanInferenceSeconds"]
    ].to_string(index=False))
    print("\nClassement multicritère :")
    print(ranking[
        ["Rank", "Model", "MulticriteriaScore", "MacroF1", "MeanInferenceSeconds"]
    ].to_string(index=False))
    print(f"\nSortie : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
