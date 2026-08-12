import pandas as pd
from pathlib import Path

input_file = Path("datasets/processed/mobility/mobility_profiles_windows_multi.csv")
output_file = Path("datasets/processed/mobility/temporal_sequences_multi.csv")

df = pd.read_csv(input_file)

features = [
    "mean_speed",
    "speed_std",
    "vehicle_density",
    "stopped_ratio",
    "lane_changes",
    "mobility_instability_index"
]

sequence_length = 5

rows = []

for scenario_source, group in df.groupby("scenario_source"):
    group = group.sort_values("window_start_s").reset_index(drop=True)

    for i in range(len(group) - sequence_length):
        sequence = group.iloc[i:i + sequence_length]
        target = group.iloc[i + sequence_length]

        row = {
            "sequence_id": f"{scenario_source}_SEQ_{i}",
            "scenario_source": scenario_source,
            "target_window": target["mobility_profile_id"],
            "target_vehicle_density": target["vehicle_density"],
            "target_mean_speed": target["mean_speed"],
            "target_stopped_ratio": target["stopped_ratio"],
            "target_mobility_instability_index": target["mobility_instability_index"]
        }

        for t in range(sequence_length):
            for f in features:
                row[f"{f}_t{t}"] = sequence.iloc[t][f]

        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(output_file, index=False)

print("Dataset temporel multi généré :", output_file)
print("Nombre de séquences :", len(out))
print(out.head())
