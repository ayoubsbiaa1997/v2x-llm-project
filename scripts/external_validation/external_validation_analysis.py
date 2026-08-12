#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.home() / "v2x-llm-project"
EXT = ROOT / "results/external_validation/ngsim"
LLM = EXT / "llm_validation"
COMMON = LLM / "common_comparison"
INTERNAL = ROOT / "results/llm_evaluation/final_benchmark_190"
OUT = EXT / "final_analysis"
FIG = OUT / "figures"
TAB = OUT / "tables"

LABELS = [
    "KEEP_CURRENT_CONNECTION",
    "MONITOR_NETWORK",
    "REDUCE_TRAFFIC_LOAD",
    "PREPARE_HANDOVER",
    "HANDOVER_NOW",
]

MODEL_FILES = {
    "Llama 3.2 1B": LLM / "llama3_2_1b/external_llm_results.csv",
    "Qwen 2.5 3B": LLM / "qwen2_5_3b/external_llm_results.csv",
    "DeepSeek-R1 8B": LLM / "deepseek-r1_8b/external_llm_results.csv",
}

STOPWORDS = {
    "the","and","to","a","of","is","in","for","with","that","this","on","as",
    "be","are","an","it","from","given","most","network","action","should",
    "current","scenario","due","based","because","would","can","while","which",
    "by","or","at","has","have",
}


def latex(df, path, caption, label):
    path.write_text(
        df.to_latex(
            index=False,
            escape=True,
            float_format="%.4f",
            caption=caption,
            label=label,
            position="!htbp",
        ),
        encoding="utf-8",
    )


def success_rates():
    rows = []
    for model, path in MODEL_FILES.items():
        df = pd.read_csv(path)
        n = len(df)
        s = int(df["status"].eq("SUCCESS").sum())
        rows.append({
            "Model": model,
            "Requested": n,
            "Successful": s,
            "ParseSuccessRate": s / n if n else 0.0,
            "Failed": n - s,
        })
    return pd.DataFrame(rows)


def reference_distribution():
    df = pd.read_csv(LLM / "transferred_reference_policy.csv")
    counts = df["reference_policy"].value_counts().reindex(LABELS, fill_value=0)
    return pd.DataFrame({
        "Decision": LABELS,
        "Count": [int(counts[x]) for x in LABELS],
        "Rate": [float(counts[x] / len(df)) for x in LABELS],
    })


def error_transitions():
    rows = []
    for model, path in MODEL_FILES.items():
        df = pd.read_csv(path)
        df = df[df["status"].eq("SUCCESS")]
        table = pd.crosstab(
            df["reference_policy"],
            df["llm_decision"],
        ).reindex(index=LABELS, columns=LABELS, fill_value=0)

        for ref in LABELS:
            support = int(table.loc[ref].sum())
            for pred in LABELS:
                count = int(table.loc[ref, pred])
                rows.append({
                    "Model": model,
                    "Reference": ref,
                    "Prediction": pred,
                    "Count": count,
                    "ConditionalRate": count / support if support else 0.0,
                })
    return pd.DataFrame(rows)


def qualitative():
    categories = {
        "QoSMentionRate": ["qos", "quality of service"],
        "CongestionMentionRate": ["congestion", "traffic load", "load"],
        "HandoverMentionRate": ["handover", "mobility", "cell"],
        "RiskMentionRate": ["risk", "failure"],
    }
    rows = []

    for model, path in MODEL_FILES.items():
        df = pd.read_csv(path)
        df = df[df["status"].eq("SUCCESS")].copy()
        explanations = df["llm_explanation"].fillna("").astype(str)

        tokens = []
        for text in explanations:
            words = re.findall(r"[A-Za-z][A-Za-z_-]{2,}", text.lower())
            tokens += [w for w in words if w not in STOPWORDS]

        row = {
            "Model": model,
            "SuccessfulExplanations": len(explanations),
            "AverageExplanationWords": float(
                explanations.map(lambda x: len(x.split())).mean()
            ),
            "TopTerms": ", ".join(
                f"{w}({c})" for w, c in Counter(tokens).most_common(10)
            ),
        }

        for name, terms in categories.items():
            row[name] = float(
                explanations.str.lower().map(
                    lambda x: any(term in x for term in terms)
                ).mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def internal_external(common):
    internal = pd.read_csv(INTERNAL / "llm_final_comparison.csv").rename(
        columns={
            "Accuracy": "InternalAgreement",
            "MacroF1": "InternalMacroF1",
            "MeanInferenceSeconds": "InternalMeanInferenceSeconds",
        }
    )

    external = common.rename(
        columns={
            "Agreement": "ExternalAgreement",
            "MacroF1": "ExternalMacroF1",
            "ActiveMacroF1": "ExternalActiveMacroF1",
            "MeanInferenceSeconds": "ExternalMeanInferenceSeconds",
        }
    )

    merged = internal[
        ["Model", "InternalAgreement", "InternalMacroF1",
         "InternalMeanInferenceSeconds"]
    ].merge(
        external[
            ["Model", "ExternalAgreement", "ExternalMacroF1",
             "ExternalActiveMacroF1", "ExternalMeanInferenceSeconds"]
        ],
        on="Model",
        how="inner",
    )

    merged["AgreementDelta"] = (
        merged["ExternalAgreement"] - merged["InternalAgreement"]
    )
    merged["MacroF1Delta"] = (
        merged["ExternalMacroF1"] - merged["InternalMacroF1"]
    )
    merged["LatencyRatioExternalInternal"] = (
        merged["ExternalMeanInferenceSeconds"]
        / merged["InternalMeanInferenceSeconds"]
    )
    return merged


def plot_reference(df):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(df["Decision"], df["Count"])
    ax.set_title("Distribution de la politique transférée — NGSIM")
    ax.set_ylabel("Nombre de scénarios")
    ax.tick_params(axis="x", rotation=35)
    for bar, count, rate in zip(bars, df["Count"], df["Rate"]):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{count}\n({rate:.1%})",
            ha="center",
            va="bottom",
        )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "reference_policy_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(common):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(
        common["MeanInferenceSeconds"],
        common["Agreement"],
        s=90,
    )
    for _, row in common.iterrows():
        ax.annotate(
            row["Model"],
            (row["MeanInferenceSeconds"], row["Agreement"]),
            xytext=(7, 7),
            textcoords="offset points",
        )
    ax.set_xlabel("Temps moyen d'inférence (s)")
    ax.set_ylabel("Agreement")
    ax.set_title("Compromis accord–latence sur NGSIM")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "external_agreement_latency_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_internal_external(df):
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x-width/2, df["InternalAgreement"], width, label="Interne")
    ax.bar(x+width/2, df["ExternalAgreement"], width, label="Externe")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy / Agreement")
    ax.set_title("Benchmark interne vs validation externe")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "internal_vs_external_agreement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_qualitative(df):
    metrics = [
        "QoSMentionRate",
        "CongestionMentionRate",
        "HandoverMentionRate",
        "RiskMentionRate",
    ]
    x = np.arange(len(metrics))
    width = 0.8 / len(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(df.iterrows()):
        offset = (i - (len(df)-1)/2) * width
        ax.bar(
            x + offset,
            [row[m] for m in metrics],
            width,
            label=row["Model"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["QoS", "Congestion", "Handover", "Risk"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Taux de mention")
    ax.set_title("Thèmes mobilisés dans les explications LLM")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "explanation_keyword_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmaps(errors):
    for model in MODEL_FILES:
        current = errors[errors["Model"].eq(model)]
        matrix = current.pivot(
            index="Reference",
            columns="Prediction",
            values="ConditionalRate",
        ).reindex(index=LABELS, columns=LABELS, fill_value=0.0)

        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1)
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=45, ha="right")
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Décision LLM")
        ax.set_ylabel("Politique transférée")
        ax.set_title(f"Transitions d'erreur normalisées — {model}")

        vals = matrix.to_numpy()
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center")

        safe = model.lower().replace(" ", "_").replace(".", "_")
        fig.tight_layout()
        fig.savefig(FIG / f"error_transition_{safe}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_report(common, success, reference, qual, comparison):
    best = common.loc[common["Agreement"].idxmax()]
    fastest = common.loc[common["MeanInferenceSeconds"].idxmin()]
    parse_best = success.loc[success["ParseSuccessRate"].idxmax()]

    lines = [
        "# Validation externe NGSIM — Synthèse finale",
        "",
        "## Protocole",
        "",
        "La validation externe repose sur NGSIM US-101. Les trajectoires réelles "
        "sont converties en variables compatibles avec le pipeline ML via le "
        "générateur adaptatif. Les scores ML alimentent ensuite une politique "
        "transférée puis les trois LLM.",
        "",
        "Cette évaluation mesure la transférabilité fonctionnelle et la cohérence "
        "décisionnelle. Elle ne constitue pas une validation radio supervisée, "
        "car NGSIM ne fournit pas de vérité terrain 5G.",
        "",
        "## Résultats communs",
        "",
        f"- Meilleur accord : {best['Model']} ({best['Agreement']:.3f}).",
        f"- Modèle le plus rapide : {fastest['Model']} "
        f"({fastest['MeanInferenceSeconds']:.2f} s).",
        f"- Meilleur taux de réponses exploitables : {parse_best['Model']} "
        f"({parse_best['ParseSuccessRate']:.1%}).",
        "",
        "## Politique transférée",
        "",
    ]

    for _, row in reference[reference["Count"] > 0].iterrows():
        lines.append(
            f"- {row['Decision']} : {int(row['Count'])} ({row['Rate']:.1%})"
        )

    lines += [
        "",
        "## Analyse qualitative des explications",
        "",
    ]

    for _, row in qual.iterrows():
        lines += [
            f"### {row['Model']}",
            "",
            f"- Longueur moyenne : {row['AverageExplanationWords']:.1f} mots.",
            f"- QoS : {row['QoSMentionRate']:.1%}.",
            f"- Congestion : {row['CongestionMentionRate']:.1%}.",
            f"- Handover : {row['HandoverMentionRate']:.1%}.",
            f"- Risque : {row['RiskMentionRate']:.1%}.",
            f"- Termes fréquents : {row['TopTerms']}.",
            "",
        ]

    lines += [
        "## Comparaison interne / externe",
        "",
        "Les différences entre les deux protocoles ne doivent pas être interprétées "
        "comme une simple baisse de performance, car le benchmark interne possède "
        "une référence différente de la validation externe. L'intérêt principal "
        "est la stabilité qualitative du compromis performance–latence et la "
        "capacité à appliquer le pipeline à une mobilité indépendante de NS-3.",
        "",
        "## Limites",
        "",
        "1. NGSIM ne contient pas de métriques radio 5G.",
        "2. Certains paramètres réseau sont reconstruits par mapping adaptatif.",
        "3. Le sous-ensemble commun est réduit par les réponses non exploitables "
        "de Llama et les timeouts de DeepSeek.",
        "4. La politique externe est transférée depuis les scores ML, pas annotée "
        "indépendamment par des experts humains.",
        "",
        "## Conclusion",
        "",
        "Le pipeline est transférable à des trajectoires de mobilité réelles. "
        "DeepSeek conserve le meilleur accord décisionnel, mais avec une forte "
        "latence. Qwen est très stable dans son format de sortie mais reste biaisé "
        "vers certaines actions. Llama est rapide mais moins robuste. Ces résultats "
        "soutiennent une architecture hybride où les modèles ML structurent l'état "
        "réseau et où le LLM agit comme couche de décision explicable.",
    ]

    (OUT / "external_validation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    for path in [
        COMMON / "common_model_metrics.csv",
        INTERNAL / "llm_final_comparison.csv",
        LLM / "transferred_reference_policy.csv",
        *MODEL_FILES.values(),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    common = pd.read_csv(COMMON / "common_model_metrics.csv")
    success = success_rates()
    reference = reference_distribution()
    errors = error_transitions()
    qual = qualitative()
    comparison = internal_external(common)

    success.to_csv(OUT / "parse_success_rates.csv", index=False)
    reference.to_csv(OUT / "reference_policy_distribution.csv", index=False)
    errors.to_csv(OUT / "error_transition_analysis.csv", index=False)
    qual.to_csv(OUT / "qualitative_explanation_analysis.csv", index=False)
    comparison.to_csv(OUT / "internal_external_comparison.csv", index=False)

    latex(
        common[
            ["Model","NCommon","Agreement","BalancedAgreement",
             "ActiveMacroF1","WeightedF1","MeanInferenceSeconds"]
        ],
        TAB / "external_common_metrics.tex",
        "Comparaison des LLM sur le sous-ensemble commun NGSIM.",
        "tab:external_common_metrics",
    )
    latex(
        success,
        TAB / "external_parse_success.tex",
        "Taux de réponses exploitables sur la validation externe.",
        "tab:external_parse_success",
    )
    latex(
        reference,
        TAB / "external_reference_distribution.tex",
        "Distribution de la politique transférée sur NGSIM.",
        "tab:external_reference_distribution",
    )
    latex(
        qual,
        TAB / "external_qualitative_explanations.tex",
        "Analyse qualitative des explications des LLM.",
        "tab:external_qualitative_explanations",
    )
    latex(
        comparison,
        TAB / "internal_external_comparison.tex",
        "Comparaison des benchmarks interne et externe.",
        "tab:internal_external_comparison",
    )

    plot_reference(reference)
    plot_tradeoff(common)
    plot_internal_external(comparison)
    plot_qualitative(qual)
    plot_error_heatmaps(errors)

    for name in [
        "common_model_performance.png",
        "common_model_latency.png",
        "common_decision_distribution.png",
    ]:
        src = COMMON / "figures" / name
        if src.exists():
            (FIG / name).write_bytes(src.read_bytes())

    write_report(common, success, reference, qual, comparison)

    summary = {
        "best_common_agreement_model": str(
            common.loc[common["Agreement"].idxmax(), "Model"]
        ),
        "fastest_common_model": str(
            common.loc[common["MeanInferenceSeconds"].idxmin(), "Model"]
        ),
        "best_parse_success_model": str(
            success.loc[success["ParseSuccessRate"].idxmax(), "Model"]
        ),
        "figures_dir": str(FIG),
        "tables_dir": str(TAB),
        "report": str(OUT / "external_validation_report.md"),
    }

    (OUT / "final_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("FINALISATION DE LA VALIDATION EXTERNE TERMINÉE")
    print("=" * 80)
    print()
    print(common.to_string(index=False))
    print()
    print(success.to_string(index=False))
    print()
    print(f"Figures : {FIG}")
    print(f"Tables  : {TAB}")
    print(f"Rapport : {OUT / 'external_validation_report.md'}")


if __name__ == "__main__":
    main()
