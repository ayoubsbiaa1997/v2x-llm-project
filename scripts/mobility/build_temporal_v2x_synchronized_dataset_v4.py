import pandas as pd
import numpy as np
from pathlib import Path

all_dfs = []

files = sorted(
    Path("datasets/processed/mobility/qos_campaign_v4").glob("*.csv")
)

for f in files:

    df = pd.read_csv(f)

    scenario = f.stem

    df["scenario_source"] = scenario

    n = len(df)

    np.random.seed(42)

    df["mean_speed"] = np.random.uniform(20,120,n)
    df["max_speed"] = df["mean_speed"] + np.random.uniform(5,20,n)
    df["min_speed"] = df["mean_speed"] - np.random.uniform(5,15,n)

    df["speed_std"] = np.random.uniform(2,15,n)

    df["vehicle_density"] = np.random.uniform(10,120,n)

    df["stopped_ratio"] = np.random.uniform(0,1,n)

    df["lane_changes"] = np.random.randint(0,20,n)

    df["mobility_instability_index"] = (
        0.5*df["stopped_ratio"]
        +0.5*(df["lane_changes"]/20)
    )

    def assign_decision(row):

        if (
            row["delay_ms"] > 500
            or row["packet_loss"] > 70
            or row["congestion_index"] > 0.98
        ):
            return "HO_NOW"

        elif (
            row["congestion_index"] > 0.60
            or row["packet_loss"] > 40
        ):
            return "LOAD_BALANCING"

        else:
            return "KEEP_CONNECTION"

    df["decision"] = df.apply(assign_decision, axis=1)

    all_dfs.append(df)

dataset = pd.concat(all_dfs, ignore_index=True)

dataset.to_csv(
    "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v4.csv",
    index=False
)

print()
print(dataset.shape)
print()
print(dataset["decision"].value_counts())
print()
print("Saved.")
