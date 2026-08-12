import pandas as pd

input_file = "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v4.csv"
output_file = "datasets/processed/mobility/temporal_sequences_v4.csv"

df = pd.read_csv(input_file)

features = [
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",
    "mean_speed",
    "speed_std",
    "vehicle_density",
    "stopped_ratio",
    "lane_changes",
    "mobility_instability_index"
]

sequence_length = 5
rows = []

for scenario, g in df.groupby("scenario_source"):
    g = g.reset_index(drop=True)

    for i in range(len(g) - sequence_length):
        row = {
            "sequence_id": f"{scenario}_SEQ_{i}",
            "scenario_source": scenario,
            "target_time_s": g.loc[i + sequence_length, "time_s"],
            "decision": g.loc[i + sequence_length, "decision"]
        }

        for t in range(sequence_length):
            for f in features:
                row[f"{f}_t{t}"] = g.loc[i + t, f]

        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(output_file, index=False)

print("Dataset séquentiel v4 généré :", output_file)
print(out.shape)
print(out["decision"].value_counts())
