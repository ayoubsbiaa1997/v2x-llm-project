#!/usr/bin/env python3
from __future__ import annotations

import json
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
BASE = ROOT / "results/external_validation/ngsim/llm_validation"
OUTPUT_DIR = BASE / "common_comparison"

LABELS = [
    "KEEP_CURRENT_CONNECTION",
    "MONITOR_NETWORK",
    "REDUCE_TRAFFIC_LOAD",
    "PREPARE_HANDOVER",
    "HANDOVER_NOW",
]

MODEL_FILES = {
    "Llama 3.2 1B": BASE / "llama3_2_1b/external_llm_results.csv",
    "Qwen 2.5 3B": BASE / "qwen2_5_3b/external_llm_results.csv",
    "DeepSeek-R1 8B": BASE / "deepseek-r1_8b/external_llm_results.csv",
}


def load_successful(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "Scenario",
        "reference_policy",
        "llm_decision",
        "inference_seconds",
        "status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes absentes dans {path}: {sorted(missing)}")
    return df[df["status"].eq("SUCCESS")][
        ["Scenario", "reference_policy", "llm_decision", "inference_seconds"]
    ].copy()


def active_labels(y_true: pd.Series) -> list[str]:
    present = set(y_true.astype(str))
    return [label for label in LABELS if label in present]


def compute_metrics(model: str, df: pd.DataFrame) -> dict[str, object]:
    y_true = df["reference_policy"].astype(str)
    y_pred = df["llm_decision"].astype(str)
    active = active_labels(y_true)
    return {
        "Model": model,
        "NCommon": int(len(df)),
        "Agreement": float(accuracy_score(y_true, y_pred)),
        "BalancedAgreement": float(balanced_accuracy_score(y_true, y_pred)),
        "MacroPrecision": float(precision_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )),
        "MacroRecall": float(recall_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )),
        "MacroF1": float(f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )),
        "ActiveMacroF1": float(f1_score(
            y_true, y_pred, labels=active, average="macro", zero_division=0
        )),
        "WeightedF1": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )),
        "MeanInferenceSeconds": float(df["inference_seconds"].mean()),
        "MedianInferenceSeconds": float(df["inference_seconds"].median()),
        "P95InferenceSeconds": float(df["inference_seconds"].quantile(0.95)),
        "PredictedClassCount": int(df["llm_decision"].nunique()),
    }


def save_confusion(model: str, df: pd.DataFrame, output: Path) -> None:
    cm = confusion_matrix(
        df["reference_policy"].astype(str),
        df["llm_decision"].astype(str),
        labels=LABELS,
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Décision LLM")
    ax.set_ylabel("Politique transférée")
    ax.set_title(f"Matrice de confusion — {model}")
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    frames = {model: load_successful(path) for model, path in MODEL_FILES.items()}
    common = set.intersection(*[set(df["Scenario"]) for df in frames.values()])
    if not common:
        raise ValueError("Aucun scénario commun réussi pour les trois modèles.")

    common_frames = {
        model: df[df["Scenario"].isin(common)].sort_values("Scenario").reset_index(drop=True)
        for model, df in frames.items()
    }

    references = [df["reference_policy"].tolist() for df in common_frames.values()]
    if not all(current == references[0] for current in references[1:]):
        raise ValueError("La politique de référence diffère entre les fichiers.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    tables_dir = OUTPUT_DIR / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"Scenario": sorted(common)}).to_csv(
        OUTPUT_DIR / "common_scenarios.csv", index=False
    )

    metric_rows = []
    per_class_rows = []
    distribution_rows = []

    for model, df in common_frames.items():
        metric_rows.append(compute_metrics(model, df))

        report = classification_report(
            df["reference_policy"].astype(str),
            df["llm_decision"].astype(str),
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        for label in LABELS:
            entry = report[label]
            per_class_rows.append({
                "Model": model,
                "Class": label,
                "Precision": float(entry["precision"]),
                "Recall": float(entry["recall"]),
                "F1": float(entry["f1-score"]),
                "Support": int(entry["support"]),
            })

        counts = df["llm_decision"].value_counts().reindex(LABELS, fill_value=0)
        for label in LABELS:
            distribution_rows.append({
                "Model": model,
                "Decision": label,
                "Count": int(counts[label]),
                "Rate": float(counts[label] / len(df)),
            })

        safe_name = model.lower().replace(" ", "_").replace(".", "_")
        save_confusion(
            model,
            df,
            figures_dir / f"confusion_matrix_{safe_name}.png",
        )

    metrics = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(per_class_rows)
    distributions = pd.DataFrame(distribution_rows)

    metrics.to_csv(OUTPUT_DIR / "common_model_metrics.csv", index=False)
    per_class.to_csv(OUTPUT_DIR / "common_per_class_metrics.csv", index=False)
    distributions.to_csv(
        OUTPUT_DIR / "common_decision_distribution.csv", index=False
    )

    # Pairwise agreement
    models = list(common_frames)
    pairwise_rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a_name, b_name = models[i], models[j]
            a = common_frames[a_name].set_index("Scenario")
            b = common_frames[b_name].set_index("Scenario")
            same = a["llm_decision"].eq(b["llm_decision"])
            a_correct = a["llm_decision"].eq(a["reference_policy"])
            b_correct = b["llm_decision"].eq(b["reference_policy"])
            pairwise_rows.append({
                "ModelA": a_name,
                "ModelB": b_name,
                "DecisionAgreement": float(same.mean()),
                "BothCorrect": int((a_correct & b_correct).sum()),
                "AOnlyCorrect": int((a_correct & ~b_correct).sum()),
                "BOnlyCorrect": int((~a_correct & b_correct).sum()),
                "BothWrong": int((~a_correct & ~b_correct).sum()),
            })

    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(OUTPUT_DIR / "common_pairwise_agreement.csv", index=False)

    # Performance figure
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, metrics["Agreement"], width, label="Agreement")
    ax.bar(x, metrics["BalancedAgreement"], width, label="Balanced Agreement")
    ax.bar(x + width, metrics["ActiveMacroF1"], width, label="Active Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["Model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des LLM sur les scénarios NGSIM communs")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "common_model_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Latency figure
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(metrics["Model"], metrics["MeanInferenceSeconds"])
    ax.set_ylabel("Temps moyen d'inférence (s)")
    ax.set_title("Latence moyenne sur le sous-ensemble commun")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "common_model_latency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Decision distribution figure
    pivot = distributions.pivot(index="Decision", columns="Model", values="Rate").reindex(LABELS)
    x = np.arange(len(pivot))
    width = 0.8 / len(pivot.columns)
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, model in enumerate(pivot.columns):
        offset = (index - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[model], width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=35, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Distribution des décisions sur les scénarios communs")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "common_decision_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # LaTeX tables
    metrics[
        [
            "Model",
            "NCommon",
            "Agreement",
            "BalancedAgreement",
            "ActiveMacroF1",
            "WeightedF1",
            "MeanInferenceSeconds",
        ]
    ].to_latex(
        tables_dir / "common_model_metrics.tex",
        index=False,
        escape=True,
        float_format="%.4f",
        caption="Comparaison des LLM sur le sous-ensemble commun NGSIM.",
        label="tab:ngsim_common_llm_metrics",
        position="!htbp",
    )

    pairwise.to_latex(
        tables_dir / "common_pairwise_agreement.tex",
        index=False,
        escape=True,
        float_format="%.4f",
        caption="Accord décisionnel entre les modèles sur les scénarios communs.",
        label="tab:ngsim_common_pairwise",
        position="!htbp",
    )

    reference_distribution = (
        next(iter(common_frames.values()))["reference_policy"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
    )

    summary = {
        "common_scenario_count": int(len(common)),
        "common_scenarios": sorted(common),
        "reference_distribution": {
            str(label): int(count)
            for label, count in reference_distribution.items()
        },
        "best_agreement_model": str(
            metrics.loc[metrics["Agreement"].idxmax(), "Model"]
        ),
        "best_active_macro_f1_model": str(
            metrics.loc[metrics["ActiveMacroF1"].idxmax(), "Model"]
        ),
        "fastest_model": str(
            metrics.loc[metrics["MeanInferenceSeconds"].idxmin(), "Model"]
        ),
        "methodological_note": (
            "Les résultats portent sur l'intersection des scénarios ayant "
            "produit une réponse exploitable pour les trois modèles. Ils "
            "mesurent l'accord avec une politique transférée issue des états "
            "ML, et non une performance radio externe supervisée."
        ),
    }

    (OUTPUT_DIR / "common_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("COMPARAISON COMMUNE DES LLM TERMINÉE")
    print("=" * 78)
    print(f"Scénarios communs : {len(common)}")
    print()
    print(metrics.to_string(index=False))
    print()
    print("Accord inter-modèles :")
    print(pairwise.to_string(index=False))
    print()
    print(f"Sortie : {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
