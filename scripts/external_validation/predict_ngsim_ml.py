#!/usr/bin/env python3
"""
Application des modèles ML V2X entraînés sur les fenêtres NGSIM.

Entrée :
    external_validation/ngsim/processed/ngsim_windows_ml_compatible.csv

Modèles :
    models/network_state_v1/QoSScore_HistGradientBoosting.joblib
    models/network_state_v1/CongestionScore_HistGradientBoosting.joblib
    models/network_state_v1/HORiskScore_ExtraTrees.joblib

Sorties :
    results/external_validation/ngsim/ml_predictions_ngsim.csv
    results/external_validation/ngsim/ml_prediction_summary.json

Cette étape mesure la transférabilité des modèles vers des données de
mobilité réelles transformées en variables compatibles. Elle ne constitue
pas une validation radio directe, car NGSIM ne contient pas de métriques
5G observées.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd


ROOT = Path.home() / "v2x-llm-project"

DEFAULT_INPUT = (
    ROOT
    / "external_validation"
    / "ngsim"
    / "processed"
    / "ngsim_windows_ml_compatible.csv"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "external_validation"
    / "ngsim"
)

FEATURES = [
    "gNbNum",
    "ueNumPerGnb",
    "totalUE",
    "bsDistance",
    "ueSpeed",
    "lambdaULL",
    "lambdaBe",
]

MODEL_PATHS = {
    "QoSScore": (
        ROOT
        / "models"
        / "network_state_v1"
        / "QoSScore_HistGradientBoosting.joblib"
    ),
    "CongestionScore": (
        ROOT
        / "models"
        / "network_state_v1"
        / "CongestionScore_HistGradientBoosting.joblib"
    ),
    "HORiskScore": (
        ROOT
        / "models"
        / "network_state_v1"
        / "HORiskScore_ExtraTrees.joblib"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Appliquer les modèles ML V2X aux fenêtres NGSIM."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--clip-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limiter les scores prédits à l'intervalle [0,1].",
    )
    return parser.parse_args()


def validate_inputs(
    input_file: Path,
    df: pd.DataFrame,
) -> None:
    if not input_file.exists():
        raise FileNotFoundError(input_file)

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Variables ML manquantes : "
            + ", ".join(missing_features)
        )

    if "Scenario" not in df.columns:
        raise ValueError(
            "La colonne Scenario est obligatoire."
        )

    for target, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle absent pour {target} : {model_path}"
            )


def qos_level(score: float) -> str:
    # QoSScore élevé = dégradation plus forte dans le pipeline courant.
    if score <= 0.25:
        return "GOOD"
    if score <= 0.50:
        return "ACCEPTABLE"
    if score <= 0.75:
        return "DEGRADED"
    return "CRITICAL"


def congestion_level(score: float) -> str:
    if score <= 0.25:
        return "LOW"
    if score <= 0.50:
        return "MODERATE"
    if score <= 0.75:
        return "HIGH"
    return "CRITICAL"


def ho_risk_level(score: float) -> str:
    if score <= 0.25:
        return "LOW"
    if score <= 0.50:
        return "MEDIUM"
    if score <= 0.75:
        return "HIGH"
    return "CRITICAL"


def summarize_numeric(
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
    }


def class_distribution(
    series: pd.Series,
) -> dict[str, dict[str, float | int]]:
    counts = series.value_counts(dropna=False)
    total = max(1, len(series))

    return {
        str(label): {
            "count": int(count),
            "rate": float(count / total),
        }
        for label, count in counts.items()
    }


def main() -> int:
    args = parse_args()

    input_file = args.input_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_file.exists():
        print(
            f"Fichier introuvable : {input_file}",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(input_file)
    validate_inputs(input_file, df)

    X = df[FEATURES].copy()

    for feature in FEATURES:
        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

    if X.isna().any().any():
        missing = {
            column: int(count)
            for column, count in X.isna().sum().items()
            if count > 0
        }
        raise ValueError(
            f"Valeurs manquantes dans les variables ML : {missing}"
        )

    output = df.copy()
    loaded_models = {}

    for target, model_path in MODEL_PATHS.items():
        model = joblib.load(model_path)
        loaded_models[target] = type(model).__name__

        predictions = np.asarray(
            model.predict(X),
            dtype=float,
        )

        if args.clip_scores:
            predictions = np.clip(
                predictions,
                0.0,
                1.0,
            )

        output[f"{target}_PredictedNGSIM"] = predictions

    output["QoSLevel_PredictedNGSIM"] = output[
        "QoSScore_PredictedNGSIM"
    ].map(qos_level)

    output["CongestionLevel_PredictedNGSIM"] = output[
        "CongestionScore_PredictedNGSIM"
    ].map(congestion_level)

    output["HORiskLevel_PredictedNGSIM"] = output[
        "HORiskScore_PredictedNGSIM"
    ].map(ho_risk_level)

    # Différence entre congestion issue de la mobilité et congestion prédite.
    if "mobility_congestion_index" in output.columns:
        output["CongestionTransferGap"] = (
            output["CongestionScore_PredictedNGSIM"]
            - output["mobility_congestion_index"]
        )
        output["CongestionTransferAbsoluteGap"] = (
            output["CongestionTransferGap"].abs()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "ml_predictions_ngsim.csv"
    )

    summary_path = (
        output_dir
        / "ml_prediction_summary.json"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    summary = {
        "input_file": str(input_file),
        "output_file": str(output_path),
        "observation_count": int(len(output)),
        "features": FEATURES,
        "models": {
            target: {
                "path": str(MODEL_PATHS[target]),
                "loaded_type": loaded_models[target],
            }
            for target in MODEL_PATHS
        },
        "score_statistics": {
            "QoSScore": summarize_numeric(
                output["QoSScore_PredictedNGSIM"]
            ),
            "CongestionScore": summarize_numeric(
                output["CongestionScore_PredictedNGSIM"]
            ),
            "HORiskScore": summarize_numeric(
                output["HORiskScore_PredictedNGSIM"]
            ),
        },
        "level_distributions": {
            "QoSLevel": class_distribution(
                output["QoSLevel_PredictedNGSIM"]
            ),
            "CongestionLevel": class_distribution(
                output["CongestionLevel_PredictedNGSIM"]
            ),
            "HORiskLevel": class_distribution(
                output["HORiskLevel_PredictedNGSIM"]
            ),
        },
        "methodological_warning": (
            "Les prédictions sont obtenues par transfert vers des "
            "variables construites à partir de NGSIM. Elles ne disposent "
            "pas de vérité terrain radio externe pour calculer MAE, RMSE "
            "ou R². L'analyse porte donc sur la plausibilité, la stabilité "
            "et la cohérence des distributions."
        ),
    }

    if "CongestionTransferAbsoluteGap" in output.columns:
        summary["mobility_vs_predicted_congestion"] = {
            "pearson_correlation": float(
                output[
                    [
                        "mobility_congestion_index",
                        "CongestionScore_PredictedNGSIM",
                    ]
                ]
                .corr(method="pearson")
                .iloc[0, 1]
            ),
            "spearman_correlation": float(
                output[
                    [
                        "mobility_congestion_index",
                        "CongestionScore_PredictedNGSIM",
                    ]
                ]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "mean_absolute_gap": float(
                output[
                    "CongestionTransferAbsoluteGap"
                ].mean()
            ),
        }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("PRÉDICTION ML SUR NGSIM TERMINÉE")
    print("=" * 76)
    print(f"Fenêtres      : {len(output)}")
    print(f"Entrée        : {input_file}")
    print(f"Prédictions   : {output_path}")
    print(f"Résumé        : {summary_path}")
    print()
    print("Distribution des niveaux prédits :")
    print()
    for column in [
        "QoSLevel_PredictedNGSIM",
        "CongestionLevel_PredictedNGSIM",
        "HORiskLevel_PredictedNGSIM",
    ]:
        print(f"{column}")
        print(output[column].value_counts().to_string())
        print()

    print("Aperçu :")
    print(
        output[
            [
                "Scenario",
                "QoSScore_PredictedNGSIM",
                "QoSLevel_PredictedNGSIM",
                "CongestionScore_PredictedNGSIM",
                "CongestionLevel_PredictedNGSIM",
                "HORiskScore_PredictedNGSIM",
                "HORiskLevel_PredictedNGSIM",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
