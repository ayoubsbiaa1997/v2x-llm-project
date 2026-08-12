#!/usr/bin/env python3
"""
Générateur adaptatif de scénarios radio à partir de fenêtres NGSIM.

Objectif
--------
Transformer les indicateurs de mobilité réelle issus de NGSIM en paramètres
compatibles avec le pipeline ML V2X :

    gNbNum
    ueNumPerGnb
    totalUE
    bsDistance
    ueSpeed
    lambdaULL
    lambdaBe

Important
---------
NGSIM ne fournit pas de mesures radio 5G. Les paramètres générés ici sont
des variables de scénario reconstruites à partir de la mobilité réelle.
Ils servent à étudier la transférabilité du pipeline ML-LLM, et non à
constituer une vérité terrain radio.

Entrée par défaut
-----------------
external_validation/ngsim/processed/ngsim_windows.csv

Sortie par défaut
-----------------
external_validation/ngsim/processed/ngsim_windows_adaptive_ml.csv
results/external_validation/ngsim/adaptive_scenario_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.home() / "v2x-llm-project"

DEFAULT_INPUT = (
    ROOT
    / "external_validation"
    / "ngsim"
    / "processed"
    / "ngsim_windows.csv"
)

DEFAULT_OUTPUT = (
    ROOT
    / "external_validation"
    / "ngsim"
    / "processed"
    / "ngsim_windows_adaptive_ml.csv"
)

DEFAULT_SUMMARY = (
    ROOT
    / "results"
    / "external_validation"
    / "ngsim"
    / "adaptive_scenario_summary.json"
)

REQUIRED_COLUMNS = [
    "Scenario",
    "vehicle_count",
    "vehicle_density_per_km",
    "mean_speed_mps",
    "speed_std_mps",
    "mean_space_headway_m",
    "mean_time_headway_s",
    "lane_change_rate_per_vehicle",
    "traffic_flow_veh_per_hour",
    "mobility_congestion_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Générer des paramètres radio adaptatifs à partir de NGSIM."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--summary-file",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    parser.add_argument(
        "--max-total-ue",
        type=int,
        default=60,
        help=(
            "Plafond totalUE aligné sur le domaine d'entraînement NS-3."
        ),
    )

    parser.add_argument(
        "--min-total-ue",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--speed-min-mps",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--speed-max-mps",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--lambda-ull-min",
        type=float,
        default=1000.0,
    )

    parser.add_argument(
        "--lambda-ull-max",
        type=float,
        default=4000.0,
    )

    parser.add_argument(
        "--lambda-be-min",
        type=float,
        default=1000.0,
    )

    parser.add_argument(
        "--lambda-be-max",
        type=float,
        default=4000.0,
    )

    return parser.parse_args()


def require_columns(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Colonnes requises absentes : "
            + ", ".join(missing)
        )


def minmax(
    series: pd.Series,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    low = float(values.quantile(lower_quantile))
    high = float(values.quantile(upper_quantile))

    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(
            f"Normalisation impossible pour {series.name}."
        )

    if math.isclose(low, high):
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
            dtype=float,
        )

    normalized = (values - low) / (high - low)

    return normalized.clip(
        lower=0.0,
        upper=1.0,
    )


def inverse_minmax(
    series: pd.Series,
) -> pd.Series:
    return 1.0 - minmax(series)


def choose_gnb_num(
    density_norm: pd.Series,
    flow_norm: pd.Series,
) -> pd.Series:
    infrastructure_pressure = (
        0.65 * density_norm
        + 0.35 * flow_norm
    )

    return pd.cut(
        infrastructure_pressure,
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=[2, 3, 4],
    ).astype(int)


def choose_bs_distance(
    density_norm: pd.Series,
    mobility_norm: pd.Series,
) -> pd.Series:
    coverage_pressure = (
        0.70 * density_norm
        + 0.30 * mobility_norm
    )

    return pd.cut(
        coverage_pressure,
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=[300.0, 200.0, 100.0],
    ).astype(float)


def discretize_to_grid(
    values: pd.Series,
    grid: list[float],
) -> pd.Series:
    array = np.asarray(grid, dtype=float)

    def nearest(value: float) -> float:
        index = int(
            np.abs(array - value).argmin()
        )
        return float(array[index])

    return values.map(nearest)


def build_adaptive_features(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = df.copy()

    density_norm = minmax(
        out["vehicle_density_per_km"]
    )
    flow_norm = minmax(
        out["traffic_flow_veh_per_hour"]
    )
    speed_norm = minmax(
        out["mean_speed_mps"]
    )
    speed_variability_norm = minmax(
        out["speed_std_mps"]
    )
    lane_change_norm = minmax(
        out["lane_change_rate_per_vehicle"]
    )
    headway_risk_norm = inverse_minmax(
        out["mean_space_headway_m"]
    )
    time_headway_risk_norm = inverse_minmax(
        out["mean_time_headway_s"]
    )
    congestion_norm = minmax(
        out["mobility_congestion_index"]
    )

    mobility_pressure = (
        0.35 * density_norm
        + 0.20 * flow_norm
        + 0.15 * speed_variability_norm
        + 0.15 * lane_change_norm
        + 0.10 * headway_risk_norm
        + 0.05 * time_headway_risk_norm
    ).clip(0.0, 1.0)

    radio_load_pressure = (
        0.35 * congestion_norm
        + 0.25 * density_norm
        + 0.15 * flow_norm
        + 0.10 * lane_change_norm
        + 0.10 * speed_variability_norm
        + 0.05 * headway_risk_norm
    ).clip(0.0, 1.0)

    out["density_norm"] = density_norm
    out["flow_norm"] = flow_norm
    out["speed_norm"] = speed_norm
    out["speed_variability_norm"] = speed_variability_norm
    out["lane_change_norm"] = lane_change_norm
    out["headway_risk_norm"] = headway_risk_norm
    out["time_headway_risk_norm"] = (
        time_headway_risk_norm
    )
    out["mobility_pressure"] = mobility_pressure
    out["radio_load_pressure"] = radio_load_pressure

    out["gNbNum"] = choose_gnb_num(
        density_norm,
        flow_norm,
    )

    out["bsDistance"] = choose_bs_distance(
        density_norm,
        mobility_pressure,
    )

    # Compression contrôlée vers le domaine NS-3 [5,60].
    total_ue_continuous = (
        args.min_total_ue
        + density_norm
        * (
            args.max_total_ue
            - args.min_total_ue
        )
    )

    out["totalUE"] = (
        total_ue_continuous
        .round()
        .clip(
            lower=args.min_total_ue,
            upper=args.max_total_ue,
        )
        .astype(int)
    )

    out["ueNumPerGnb"] = (
        np.ceil(
            out["totalUE"]
            / out["gNbNum"]
        )
        .astype(int)
    )

    out["ueSpeed"] = (
        pd.to_numeric(
            out["mean_speed_mps"],
            errors="coerce",
        )
        .clip(
            lower=args.speed_min_mps,
            upper=args.speed_max_mps,
        )
    )

    lambda_ull_continuous = (
        args.lambda_ull_min
        + radio_load_pressure
        * (
            args.lambda_ull_max
            - args.lambda_ull_min
        )
    )

    lambda_be_pressure = (
        0.35 * density_norm
        + 0.30 * flow_norm
        + 0.15 * congestion_norm
        + 0.10 * speed_variability_norm
        + 0.10 * headway_risk_norm
    ).clip(0.0, 1.0)

    lambda_be_continuous = (
        args.lambda_be_min
        + lambda_be_pressure
        * (
            args.lambda_be_max
            - args.lambda_be_min
        )
    )

    # Alignement sur la grille utilisée dans les simulations.
    load_grid = [
        1000.0,
        2000.0,
        4000.0,
    ]

    out["lambdaULL"] = discretize_to_grid(
        lambda_ull_continuous,
        load_grid,
    )

    out["lambdaBe"] = discretize_to_grid(
        lambda_be_continuous,
        load_grid,
    )

    out["adaptive_scenario_class"] = pd.cut(
        radio_load_pressure,
        bins=[
            -np.inf,
            0.25,
            0.50,
            0.75,
            np.inf,
        ],
        labels=[
            "LOW_LOAD",
            "MODERATE_LOAD",
            "HIGH_LOAD",
            "CRITICAL_LOAD",
        ],
    ).astype(str)

    ml_columns = [
        "Scenario",
        "gNbNum",
        "ueNumPerGnb",
        "totalUE",
        "bsDistance",
        "ueSpeed",
        "lambdaULL",
        "lambdaBe",
    ]

    context_columns = [
        "window_start_s",
        "window_end_s",
        "vehicle_count",
        "vehicle_density_per_km",
        "mean_speed_mps",
        "speed_std_mps",
        "mean_acceleration_mps2",
        "mean_space_headway_m",
        "mean_time_headway_s",
        "lane_change_count",
        "lane_change_rate_per_vehicle",
        "traffic_flow_veh_per_hour",
        "truck_ratio",
        "stopped_vehicle_ratio",
        "mobility_congestion_index",
        "density_norm",
        "flow_norm",
        "speed_norm",
        "speed_variability_norm",
        "lane_change_norm",
        "headway_risk_norm",
        "time_headway_risk_norm",
        "mobility_pressure",
        "radio_load_pressure",
        "adaptive_scenario_class",
    ]

    available_context = [
        column
        for column in context_columns
        if column in out.columns
    ]

    return out[
        ml_columns
        + available_context
    ].copy()


def distribution(
    series: pd.Series,
) -> dict[str, dict[str, float | int]]:
    counts = series.value_counts(
        dropna=False
    )

    total = max(
        1,
        len(series),
    )

    return {
        str(key): {
            "count": int(value),
            "rate": float(
                value / total
            ),
        }
        for key, value in counts.items()
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
    }


def main() -> int:
    args = parse_args()

    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    summary_file = args.summary_file.expanduser().resolve()

    if not input_file.exists():
        print(
            f"Fichier introuvable : {input_file}",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(input_file)
    require_columns(df)

    adaptive = build_adaptive_features(
        df,
        args,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    adaptive.to_csv(
        output_file,
        index=False,
    )

    summary = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "observation_count": int(
            len(adaptive)
        ),
        "ml_features": [
            "gNbNum",
            "ueNumPerGnb",
            "totalUE",
            "bsDistance",
            "ueSpeed",
            "lambdaULL",
            "lambdaBe",
        ],
        "parameter_distributions": {
            "gNbNum": distribution(
                adaptive["gNbNum"]
            ),
            "bsDistance": distribution(
                adaptive["bsDistance"]
            ),
            "lambdaULL": distribution(
                adaptive["lambdaULL"]
            ),
            "lambdaBe": distribution(
                adaptive["lambdaBe"]
            ),
            "adaptive_scenario_class": distribution(
                adaptive[
                    "adaptive_scenario_class"
                ]
            ),
        },
        "numeric_statistics": {
            "totalUE": numeric_summary(
                adaptive["totalUE"]
            ),
            "ueNumPerGnb": numeric_summary(
                adaptive["ueNumPerGnb"]
            ),
            "ueSpeed": numeric_summary(
                adaptive["ueSpeed"]
            ),
            "mobility_pressure": numeric_summary(
                adaptive["mobility_pressure"]
            ),
            "radio_load_pressure": numeric_summary(
                adaptive["radio_load_pressure"]
            ),
        },
        "mapping_rules": {
            "gNbNum": (
                "Fonction de la densité et du débit, discrétisée en 2, 3 ou 4."
            ),
            "bsDistance": (
                "Fonction de la densité et de la pression de mobilité, "
                "discrétisée en 300, 200 ou 100 m."
            ),
            "totalUE": (
                "Compression quantile de la densité NGSIM vers [5,60]."
            ),
            "ueSpeed": (
                "Vitesse moyenne NGSIM bornée au domaine [5,30] m/s."
            ),
            "lambdaULL": (
                "Fonction multicritère discrétisée sur {1000,2000,4000}."
            ),
            "lambdaBe": (
                "Fonction multicritère discrétisée sur {1000,2000,4000}."
            ),
        },
        "methodological_warning": (
            "Les paramètres radio sont reconstruits à partir de la mobilité "
            "réelle. Ils ne sont pas observés directement dans NGSIM."
        ),
    }

    summary_file.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("GÉNÉRATION ADAPTATIVE DES SCÉNARIOS TERMINÉE")
    print("=" * 76)
    print(f"Fenêtres       : {len(adaptive)}")
    print(f"Entrée         : {input_file}")
    print(f"Sortie         : {output_file}")
    print(f"Résumé         : {summary_file}")
    print()

    print("Distribution gNbNum :")
    print(
        adaptive["gNbNum"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()

    print("Distribution bsDistance :")
    print(
        adaptive["bsDistance"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()

    print("Distribution lambdaULL :")
    print(
        adaptive["lambdaULL"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()

    print("Distribution lambdaBe :")
    print(
        adaptive["lambdaBe"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()

    print("Aperçu ML adaptatif :")
    print(
        adaptive[
            [
                "Scenario",
                "gNbNum",
                "ueNumPerGnb",
                "totalUE",
                "bsDistance",
                "ueSpeed",
                "lambdaULL",
                "lambdaBe",
                "adaptive_scenario_class",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
