#!/usr/bin/env python3
"""
Validation scientifique du pipeline ML V2X à partir des prédictions OOF.

Entrées :
- results/network_state_v1/out_of_fold_predictions.csv
- results/network_state_v1/selected_features.json

Sorties :
- métriques de régression ;
- métriques de classification après discrétisation ;
- matrices de confusion ;
- graphes observé vs prédit ;
- graphes de résidus ;
- statistiques d'erreur ;
- tableaux LaTeX ;
- résumé méthodologique.

Aucun réentraînement n'est effectué.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

PROJECT_DIR = Path.home() / "v2x-llm-project"
OOF_FILE = PROJECT_DIR / "results/network_state_v1/out_of_fold_predictions.csv"
CONFIG_FILE = PROJECT_DIR / "results/network_state_v1/selected_features.json"

OUTPUT_DIR = PROJECT_DIR / "results/ml_validation_v2"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

TARGETS: dict[str, dict[str, Any]] = {
    "QoSScore": {
        "predicted": "QoSScore_PredictedOOF",
        "absolute_error": "QoSScore_AbsoluteError",
        "labels": ["GOOD", "ACCEPTABLE", "DEGRADED", "CRITICAL"],
    },
    "CongestionScore": {
        "predicted": "CongestionScore_PredictedOOF",
        "absolute_error": "CongestionScore_AbsoluteError",
        "labels": ["LOW", "MODERATE", "HIGH", "CRITICAL"],
    },
    "HORiskScore": {
        "predicted": "HORiskScore_PredictedOOF",
        "absolute_error": "HORiskScore_AbsoluteError",
        "labels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    },
}

THRESHOLDS = (0.25, 0.50, 0.75)


def score_to_class(score: float, labels: list[str]) -> str:
    if score <= THRESHOLDS[0]:
        return labels[0]
    if score <= THRESHOLDS[1]:
        return labels[1]
    if score <= THRESHOLDS[2]:
        return labels[2]
    return labels[3]


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-12
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def validate_inputs(df: pd.DataFrame) -> None:
    required = set()
    for target, config in TARGETS.items():
        required.update(
            [
                target,
                config["predicted"],
                config["absolute_error"],
            ]
        )

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Colonnes absentes : " + ", ".join(missing))

    if df.empty:
        raise ValueError("Le fichier OOF est vide.")

    if df[list(required)].isna().any().any():
        raise ValueError("Des valeurs manquantes existent dans les colonnes requises.")


def regression_row(
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int | str]:
    residuals = y_true - y_pred
    return {
        "Target": target,
        "N": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
        "ExplainedVariance": float(explained_variance_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "MAPE_percent": safe_mape(y_true, y_pred),
        "MeanResidual": float(np.mean(residuals)),
        "ResidualStd": float(np.std(residuals, ddof=1)),
        "MaxAbsoluteError": float(np.max(np.abs(residuals))),
    }


def classification_row(
    target: str,
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, float | int | str]:

    active_labels = [
        label
        for label in labels
        if label in set(y_true)
    ]

    return {
        "Target": target,
        "N": int(len(y_true)),
        "Accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "BalancedAccuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "MacroPrecision": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "MacroRecall": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "MacroF1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "ActiveClassCount": int(len(active_labels)),
        "ActiveMacroF1": float(
            f1_score(
                y_true,
                y_pred,
                labels=active_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "WeightedF1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def class_report(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> pd.DataFrame:
    rows = []
    for label in labels:
        true_binary = [value == label for value in y_true]
        pred_binary = [value == label for value in y_pred]
        rows.append(
            {
                "Class": label,
                "Support": int(sum(true_binary)),
                "Precision": float(
                    precision_score(true_binary, pred_binary, zero_division=0)
                ),
                "Recall": float(
                    recall_score(true_binary, pred_binary, zero_division=0)
                ),
                "F1": float(
                    f1_score(true_binary, pred_binary, zero_division=0)
                ),
            }
        )
    return pd.DataFrame(rows)


def save_observed_vs_predicted(
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    output = FIGURE_DIR / f"observed_vs_predicted_{target}.png"

    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))
    margin = max((upper - lower) * 0.05, 0.02)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, alpha=0.75)
    ax.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_xlabel("Score observé")
    ax.set_ylabel("Score prédit OOF")
    ax.set_title(f"Valeurs observées et prédites — {target}")
    ax.set_xlim(lower - margin, upper + margin)
    ax.set_ylim(lower - margin, upper + margin)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_residual_scatter(
    target: str,
    y_pred: np.ndarray,
    residuals: np.ndarray,
) -> Path:
    output = FIGURE_DIR / f"residuals_{target}.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.75)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Score prédit OOF")
    ax.set_ylabel("Résidu (observé - prédit)")
    ax.set_title(f"Analyse des résidus — {target}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_residual_histogram(
    target: str,
    residuals: np.ndarray,
) -> Path:
    output = FIGURE_DIR / f"residual_histogram_{target}.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=20)
    ax.set_xlabel("Résidu (observé - prédit)")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Distribution des résidus — {target}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_confusion(
    target: str,
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> Path:
    output = FIGURE_DIR / f"confusion_matrix_{target}.png"
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Classe prédite",
        ylabel="Classe observée",
        title=f"Matrice de confusion — {target}",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_error_distribution(df: pd.DataFrame) -> Path:
    output = FIGURE_DIR / "absolute_error_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for target, config in TARGETS.items():
        ax.hist(
            df[config["absolute_error"]],
            bins=20,
            alpha=0.45,
            label=target,
        )
    ax.set_xlabel("Erreur absolue")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des erreurs absolues OOF")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_error_boxplot(df: pd.DataFrame) -> Path:
    output = FIGURE_DIR / "absolute_error_boxplot.png"
    data = [
        df[config["absolute_error"]].to_numpy()
        for config in TARGETS.values()
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=list(TARGETS.keys()), showmeans=True)
    ax.set_ylabel("Erreur absolue")
    ax.set_title("Comparaison des erreurs absolues OOF")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def error_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, config in TARGETS.items():
        values = df[config["absolute_error"]].astype(float)
        rows.append(
            {
                "Target": target,
                "Mean": float(values.mean()),
                "Std": float(values.std(ddof=1)),
                "Min": float(values.min()),
                "Q1": float(values.quantile(0.25)),
                "Median": float(values.median()),
                "Q3": float(values.quantile(0.75)),
                "P90": float(values.quantile(0.90)),
                "P95": float(values.quantile(0.95)),
                "Max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def feature_summary(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for feature in config.get("selected_features", []):
        if feature not in df.columns:
            continue
        series = pd.to_numeric(df[feature], errors="coerce")
        rows.append(
            {
                "Feature": feature,
                "Count": int(series.count()),
                "Missing": int(series.isna().sum()),
                "Mean": float(series.mean()),
                "Std": float(series.std(ddof=1)),
                "Min": float(series.min()),
                "Q1": float(series.quantile(0.25)),
                "Median": float(series.median()),
                "Q3": float(series.quantile(0.75)),
                "Max": float(series.max()),
                "Unique": int(series.nunique()),
            }
        )
    return pd.DataFrame(rows)


def escape_latex(text: Any) -> str:
    result = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
    }
    for source, replacement in replacements.items():
        result = result.replace(source, replacement)
    return result


def write_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    output: Path,
    float_columns: list[str],
) -> None:
    columns = list(df.columns)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{escape_latex(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{'l' + 'r' * (len(columns) - 1)}}}",
        r"\toprule",
        " & ".join(escape_latex(column) for column in columns) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in float_columns and pd.notna(value):
                values.append(f"{float(value):.4f}")
            else:
                values.append(escape_latex(value))
        lines.append(" & ".join(values) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def build_summary(
    regression_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> str:
    lines = [
        "VALIDATION SCIENTIFIQUE DU PIPELINE ML V2X",
        "=" * 72,
        "",
        f"Observations OOF : {int(regression_df['N'].iloc[0])}",
        f"Variables d'entrée : {len(feature_df)}",
        "Validation fondée exclusivement sur les prédictions OOF.",
        "",
        "MÉTRIQUES DE RÉGRESSION",
        "-" * 72,
    ]

    for _, row in regression_df.iterrows():
        lines.append(
            f"{row['Target']} | MAE={row['MAE']:.6f} | "
            f"RMSE={row['RMSE']:.6f} | R²={row['R2']:.6f} | "
            f"MedAE={row['MedianAE']:.6f}"
        )

    lines.extend(["", "MÉTRIQUES APRÈS DISCRÉTISATION", "-" * 72])
    for _, row in classification_df.iterrows():
        lines.append(
            f"{row['Target']} | Accuracy={row['Accuracy']:.6f} | "
            f"BalancedAccuracy={row['BalancedAccuracy']:.6f} | "
            f"MacroF1={row['MacroF1']:.6f}"
        )

    lines.extend(["", "STATISTIQUES DES ERREURS", "-" * 72])
    for _, row in errors_df.iterrows():
        lines.append(
            f"{row['Target']} | Moyenne={row['Mean']:.6f} | "
            f"Médiane={row['Median']:.6f} | "
            f"P95={row['P95']:.6f} | Max={row['Max']:.6f}"
        )

    best_r2 = regression_df.loc[regression_df["R2"].idxmax()]
    worst_r2 = regression_df.loc[regression_df["R2"].idxmin()]
    best_f1 = classification_df.loc[classification_df["MacroF1"].idxmax()]

    lines.extend(
        [
            "",
            "INTERPRÉTATION",
            "-" * 72,
            (
                f"Meilleure cible de régression : {best_r2['Target']} "
                f"(R²={best_r2['R2']:.4f})."
            ),
            (
                f"Cible de régression la plus difficile : {worst_r2['Target']} "
                f"(R²={worst_r2['R2']:.4f})."
            ),
            (
                f"Meilleure discrimination qualitative : {best_f1['Target']} "
                f"(Macro-F1={best_f1['MacroF1']:.4f})."
            ),
            "",
            "PRÉCAUTIONS MÉTHODOLOGIQUES",
            "-" * 72,
            (
                "Les résultats reposent sur 190 observations simulées et sur "
                "des scores/pseudo-labels construits à partir de règles explicites."
            ),
            (
                "Les matrices de confusion évaluent la cohérence après "
                "discrétisation des scores continus et ne remplacent pas "
                "les métriques de régression."
            ),
            (
                "La performance de HORiskScore doit être interprétée avec "
                "prudence en raison de l'absence d'échecs réels de handover "
                "dans les observations valides."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if not OOF_FILE.exists():
        raise FileNotFoundError(f"Fichier OOF introuvable : {OOF_FILE}")
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration introuvable : {CONFIG_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(OOF_FILE)
    validate_inputs(df)
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    regression_rows = []
    classification_rows = []
    generated: list[Path] = []

    for target, target_config in TARGETS.items():
        y_true = df[target].astype(float).to_numpy()
        y_pred = df[target_config["predicted"]].astype(float).to_numpy()
        residuals = y_true - y_pred
        labels = target_config["labels"]

        regression_rows.append(regression_row(target, y_true, y_pred))

        true_classes = [score_to_class(value, labels) for value in y_true]
        pred_classes = [score_to_class(value, labels) for value in y_pred]

        classification_rows.append(
            classification_row(
                target,
                true_classes,
                pred_classes,
                labels,
            )
        )

        report = class_report(true_classes, pred_classes, labels)
        report_path = OUTPUT_DIR / f"classification_report_{target}.csv"
        report.to_csv(report_path, index=False)
        generated.append(report_path)

        generated.extend(
            [
                save_observed_vs_predicted(target, y_true, y_pred),
                save_residual_scatter(target, y_pred, residuals),
                save_residual_histogram(target, residuals),
                save_confusion(target, true_classes, pred_classes, labels),
            ]
        )

    regression_df = pd.DataFrame(regression_rows)
    classification_df = pd.DataFrame(classification_rows)
    errors_df = error_statistics(df)
    features_df = feature_summary(df, config)

    regression_path = OUTPUT_DIR / "regression_metrics.csv"
    classification_path = OUTPUT_DIR / "classification_metrics.csv"
    errors_path = OUTPUT_DIR / "prediction_error_statistics.csv"
    features_path = OUTPUT_DIR / "feature_summary.csv"

    regression_df.to_csv(regression_path, index=False)
    classification_df.to_csv(classification_path, index=False)
    errors_df.to_csv(errors_path, index=False)
    features_df.to_csv(features_path, index=False)

    generated.extend(
        [
            regression_path,
            classification_path,
            errors_path,
            features_path,
            save_error_distribution(df),
            save_error_boxplot(df),
        ]
    )

    regression_tex = TABLE_DIR / "regression_metrics.tex"
    classification_tex = TABLE_DIR / "classification_metrics.tex"

    write_latex_table(
        regression_df[
            ["Target", "MAE", "RMSE", "R2", "MedianAE", "MaxAbsoluteError"]
        ],
        "Performances OOF des modèles de régression pour les scores d'état réseau.",
        "tab:ml_regression_validation",
        regression_tex,
        ["MAE", "RMSE", "R2", "MedianAE", "MaxAbsoluteError"],
    )

    write_latex_table(
    classification_df[
        [
            "Target",
            "Accuracy",
            "BalancedAccuracy",
            "MacroPrecision",
            "MacroRecall",
            "MacroF1",
            "ActiveMacroF1",
        ]
    ],
    "Performances après discrétisation des scores prédits.",
    "tab:ml_classification_validation",
    classification_tex,
    [
        "Accuracy",
        "BalancedAccuracy",
        "MacroPrecision",
        "MacroRecall",
        "MacroF1",
        "ActiveMacroF1",
    ],
)

    generated.extend([regression_tex, classification_tex])

    summary_path = OUTPUT_DIR / "validation_summary.txt"
    summary_path.write_text(
        build_summary(
            regression_df,
            classification_df,
            errors_df,
            features_df,
        ),
        encoding="utf-8",
    )
    generated.append(summary_path)

    manifest = {
        "input_oof_file": str(OOF_FILE),
        "configuration_file": str(CONFIG_FILE),
        "observation_count": int(len(df)),
        "targets": list(TARGETS.keys()),
        "thresholds": list(THRESHOLDS),
        "generated_files": [str(path) for path in generated],
    }
    manifest_path = OUTPUT_DIR / "validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("VALIDATION ML TERMINÉE")
    print("=" * 72)
    print(f"Observations : {len(df)}")
    print(f"Sortie       : {OUTPUT_DIR}")
    print("\nMétriques de régression :")
    print(
        regression_df[
            ["Target", "MAE", "RMSE", "R2"]
        ].to_string(index=False)
    )
    print("\nMétriques après discrétisation :")
    print(
        classification_df[
    [
        "Target",
        "Accuracy",
        "BalancedAccuracy",
        "MacroPrecision",
        "MacroRecall",
        "MacroF1",
        "ActiveMacroF1",
    ]
].to_string(index=False)
    )
    print(f"\nRésumé  : {summary_path}")
    print(f"Figures : {FIGURE_DIR}")
    print(f"Tables  : {TABLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
