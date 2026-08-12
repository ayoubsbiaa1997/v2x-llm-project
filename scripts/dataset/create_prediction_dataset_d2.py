#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path(
    "datasets/processed/dynamic_d2/dynamic_qos_merged_d2_enriched.csv"
)

OUTPUT = Path(
    "datasets/processed/dynamic_d2/prediction_dataset_d2.csv"
)

LAGS = [1,2,3]

QOS_COLUMNS = [
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",
]

STATIC_COLUMNS = [
    "run_id",
    "ue_count",
    "distance_m",
]

TARGET_COLUMN = "future_degradation"

THRESHOLD = 0.60


df = pd.read_csv(INPUT)

df = df.sort_values(
    ["run_id","time_s"]
).reset_index(drop=True)



for feature in QOS_COLUMNS:

    for lag in LAGS:

        df[f"{feature}_lag_{lag}"] = (
            df.groupby("run_id")[feature]
            .shift(lag)
        )


future_congestion = (
    df.groupby("run_id")["congestion_index"]
    .shift(-1)
)

df[TARGET_COLUMN] = (
    future_congestion >= THRESHOLD
).astype(int)



df = df.dropna().reset_index(drop=True)


columns = []

columns.extend(STATIC_COLUMNS)

for feature in QOS_COLUMNS:

    for lag in LAGS:

        columns.append(
            f"{feature}_lag_{lag}"
        )

columns.append(TARGET_COLUMN)

prediction_df = df[columns]


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

prediction_df.to_csv(
    OUTPUT,
    index=False
)


print()

print("Prediction Dataset créé")

print(prediction_df.shape)

print()

print(
    prediction_df[TARGET_COLUMN]
    .value_counts()
)

print()

print(prediction_df.head())
