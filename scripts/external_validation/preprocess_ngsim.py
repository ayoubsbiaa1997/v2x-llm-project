#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "v2x-llm-project"
DEFAULT_INPUT = ROOT / "external_validation/ngsim/raw/us101.csv"
DEFAULT_OUTPUT_DIR = ROOT / "external_validation/ngsim/processed"
DEFAULT_SUMMARY = ROOT / "results/external_validation/ngsim/preprocess_summary.json"

NGSIM_COLUMNS = [
    "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time",
    "Local_X", "Local_Y", "Global_X", "Global_Y",
    "v_Length", "v_Width", "v_Class", "v_Vel", "v_Acc",
    "Lane_ID", "Preceding", "Following",
    "Space_Headway", "Time_Headway",
]

FT_TO_M = 0.3048


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--window-seconds", type=float, default=5.0)
    p.add_argument("--step-seconds", type=float, default=5.0)
    p.add_argument("--road-length-m", type=float, default=640.0)
    p.add_argument("--gnb-num", type=int, default=3)
    p.add_argument("--bs-distance", type=float, default=200.0)
    p.add_argument("--lambda-ull-reference", type=float, default=1000.0)
    p.add_argument("--lambda-be-reference", type=float, default=2000.0)
    p.add_argument("--max-total-ue", type=int, default=120)
    p.add_argument("--min-vehicles", type=int, default=5)
    return p.parse_args()


def load_ngsim(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=NGSIM_COLUMNS,
        engine="c",
    )

    for column in NGSIM_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "Vehicle_ID", "Frame_ID", "Global_Time",
            "Local_X", "Local_Y", "v_Vel", "v_Acc", "Lane_ID",
        ]
    ).copy()

    for column in [
        "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time",
        "v_Class", "Lane_ID", "Preceding", "Following",
    ]:
        df[column] = df[column].astype("int64")

    df["Local_X_m"] = df["Local_X"] * FT_TO_M
    df["Local_Y_m"] = df["Local_Y"] * FT_TO_M
    df["Speed_mps"] = df["v_Vel"] * FT_TO_M
    df["Acceleration_mps2"] = df["v_Acc"] * FT_TO_M
    df["Space_Headway_m"] = df["Space_Headway"] * FT_TO_M
    df["Relative_Time_s"] = (
        df["Global_Time"] - df["Global_Time"].min()
    ) / 1000.0

    return df


def safe_mean(series: pd.Series) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def safe_std(series: pd.Series) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.std()) if len(clean) > 1 else 0.0


def count_lane_changes(window: pd.DataFrame) -> tuple[int, float]:
    ordered = window.sort_values(["Vehicle_ID", "Global_Time"])
    flags = (
        ordered.groupby("Vehicle_ID")["Lane_ID"]
        .diff()
        .fillna(0)
        .ne(0)
    )
    count = int(flags.sum())
    vehicles = max(1, int(window["Vehicle_ID"].nunique()))
    return count, count / vehicles


def build_windows(
    df: pd.DataFrame,
    window_seconds: float,
    step_seconds: float,
    road_length_m: float,
    min_vehicles: int,
) -> pd.DataFrame:
    start = float(math.floor(df["Relative_Time_s"].min()))
    end = float(math.ceil(df["Relative_Time_s"].max()))
    rows = []
    window_id = 0
    current = start

    while current + window_seconds <= end + 1e-9:
        stop = current + window_seconds
        window = df[
            (df["Relative_Time_s"] >= current)
            & (df["Relative_Time_s"] < stop)
        ].copy()

        vehicle_count = int(window["Vehicle_ID"].nunique())

        if vehicle_count >= min_vehicles:
            lane_changes, lane_change_rate = count_lane_changes(window)

            mean_speed = safe_mean(window["Speed_mps"])
            speed_std = safe_std(window["Speed_mps"])
            mean_acc = safe_mean(window["Acceleration_mps2"])
            acc_std = safe_std(window["Acceleration_mps2"])

            valid_space = window.loc[
                window["Space_Headway_m"] > 0,
                "Space_Headway_m",
            ]
            valid_time = window.loc[
                window["Time_Headway"] > 0,
                "Time_Headway",
            ]

            mean_space = safe_mean(valid_space)
            mean_time = safe_mean(valid_time)

            density = vehicle_count / max(road_length_m / 1000.0, 1e-9)
            flow = vehicle_count / window_seconds * 3600.0
            truck_ratio = float(
                window.groupby("Vehicle_ID")["v_Class"]
                .first()
                .eq(3)
                .mean()
            )
            stopped_ratio = float(
                (
                    window.groupby("Vehicle_ID")["Speed_mps"]
                    .mean()
                    < 1.0
                ).mean()
            )
            speed_cv = speed_std / mean_speed if mean_speed > 1e-9 else 0.0

            congestion_index = float(
                np.clip(
                    0.40 * min(density / 180.0, 1.0)
                    + 0.30 * (1.0 - min(mean_speed / 30.0, 1.0))
                    + 0.15 * min(speed_cv / 0.8, 1.0)
                    + 0.15 * min(stopped_ratio, 1.0),
                    0.0,
                    1.0,
                )
            )

            rows.append({
                "Scenario": f"ngsim_us101_w{window_id:04d}",
                "window_id": window_id,
                "window_start_s": current,
                "window_end_s": stop,
                "window_duration_s": window_seconds,
                "observation_count": int(len(window)),
                "vehicle_count": vehicle_count,
                "lane_count": int(window["Lane_ID"].nunique()),
                "mean_speed_mps": mean_speed,
                "mean_speed_kmh": mean_speed * 3.6,
                "speed_std_mps": speed_std,
                "speed_cv": speed_cv,
                "mean_acceleration_mps2": mean_acc,
                "acceleration_std_mps2": acc_std,
                "mean_space_headway_m": mean_space,
                "mean_time_headway_s": mean_time,
                "lane_change_count": lane_changes,
                "lane_change_rate_per_vehicle": lane_change_rate,
                "vehicle_density_per_km": density,
                "traffic_flow_veh_per_hour": flow,
                "truck_ratio": truck_ratio,
                "stopped_vehicle_ratio": stopped_ratio,
                "mobility_congestion_index": congestion_index,
            })
            window_id += 1

        current += step_seconds

    return pd.DataFrame(rows)


def map_to_ml_features(
    windows: pd.DataFrame,
    gnb_num: int,
    bs_distance: float,
    lambda_ull_reference: float,
    lambda_be_reference: float,
    max_total_ue: int,
) -> pd.DataFrame:
    out = windows.copy()
    out["gNbNum"] = int(gnb_num)
    out["totalUE"] = (
        out["vehicle_count"]
        .clip(lower=1, upper=max_total_ue)
        .round()
        .astype(int)
    )
    out["ueNumPerGnb"] = np.ceil(
        out["totalUE"] / max(gnb_num, 1)
    ).astype(int)
    out["bsDistance"] = float(bs_distance)
    out["ueSpeed"] = out["mean_speed_mps"]

    congestion = out["mobility_congestion_index"]
    out["lambdaULL"] = (
        lambda_ull_reference * (0.75 + 0.50 * congestion)
    )
    out["lambdaBe"] = (
        lambda_be_reference * (0.75 + 0.75 * congestion)
    )

    ml_columns = [
        "Scenario", "gNbNum", "ueNumPerGnb", "totalUE",
        "bsDistance", "ueSpeed", "lambdaULL", "lambdaBe",
    ]
    context_columns = [
        "window_start_s", "window_end_s", "vehicle_count",
        "vehicle_density_per_km", "mean_speed_mps", "speed_std_mps",
        "mean_acceleration_mps2", "mean_space_headway_m",
        "mean_time_headway_s", "lane_change_count",
        "lane_change_rate_per_vehicle", "traffic_flow_veh_per_hour",
        "truck_ratio", "stopped_vehicle_ratio",
        "mobility_congestion_index",
    ]

    return out[ml_columns + context_columns].copy()


def main() -> int:
    args = parse_args()

    input_file = args.input_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    summary_file = args.summary_file.expanduser().resolve()

    if not input_file.exists():
        print(f"Fichier introuvable : {input_file}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    raw = load_ngsim(input_file)
    windows = build_windows(
        raw,
        args.window_seconds,
        args.step_seconds,
        args.road_length_m,
        args.min_vehicles,
    )

    if windows.empty:
        print("Aucune fenêtre valide produite.", file=sys.stderr)
        return 1

    ml = map_to_ml_features(
        windows,
        args.gnb_num,
        args.bs_distance,
        args.lambda_ull_reference,
        args.lambda_be_reference,
        args.max_total_ue,
    )

    windows_path = output_dir / "ngsim_windows.csv"
    ml_path = output_dir / "ngsim_windows_ml_compatible.csv"

    windows.to_csv(windows_path, index=False)
    ml.to_csv(ml_path, index=False)

    summary = {
        "input_file": str(input_file),
        "raw_observation_count": int(len(raw)),
        "unique_vehicle_count": int(raw["Vehicle_ID"].nunique()),
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "road_length_m": args.road_length_m,
        "window_count": int(len(windows)),
        "scenario_parameters": {
            "gNbNum": args.gnb_num,
            "bsDistance": args.bs_distance,
            "lambdaULL_reference": args.lambda_ull_reference,
            "lambdaBE_reference": args.lambda_be_reference,
            "max_totalUE": args.max_total_ue,
        },
        "ranges": {
            "vehicle_count": [
                int(windows["vehicle_count"].min()),
                int(windows["vehicle_count"].max()),
            ],
            "mean_speed_mps": [
                float(windows["mean_speed_mps"].min()),
                float(windows["mean_speed_mps"].max()),
            ],
            "mobility_congestion_index": [
                float(windows["mobility_congestion_index"].min()),
                float(windows["mobility_congestion_index"].max()),
            ],
            "lambdaULL": [
                float(ml["lambdaULL"].min()),
                float(ml["lambdaULL"].max()),
            ],
            "lambdaBe": [
                float(ml["lambdaBe"].min()),
                float(ml["lambdaBe"].max()),
            ],
        },
        "outputs": {
            "descriptive_windows": str(windows_path),
            "ml_compatible_windows": str(ml_path),
        },
        "methodological_warning": (
            "NGSIM ne contient pas de mesures radio. "
            "gNbNum, bsDistance, lambdaULL et lambdaBe sont des "
            "variables de scénario/proxies pour une analyse de transférabilité."
        ),
    }

    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("PRÉTRAITEMENT NGSIM TERMINÉ")
    print("=" * 76)
    print(f"Observations brutes : {len(raw)}")
    print(f"Véhicules uniques   : {raw['Vehicle_ID'].nunique()}")
    print(f"Fenêtres produites  : {len(windows)}")
    print(f"Fenêtres descriptives : {windows_path}")
    print(f"Entrées ML compatibles : {ml_path}")
    print(f"Résumé                : {summary_file}")
    print()
    print(
        ml[
            [
                "Scenario", "gNbNum", "ueNumPerGnb", "totalUE",
                "bsDistance", "ueSpeed", "lambdaULL", "lambdaBe",
            ]
        ].head(10).to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
