import pandas as pd
from pathlib import Path

input_file = Path("datasets/processed/mobility/temporal_sequences_multi.csv")
output_file = Path("datasets/processed/mobility/temporal_sequences_decision_multi.csv")

df = pd.read_csv(input_file)

def generate_temporal_decision(row):
    density = row["target_vehicle_density"]
    speed = row["target_mean_speed"]
    stopped = row["target_stopped_ratio"]
    instability = row["target_mobility_instability_index"]

    if density >= 1200 or stopped >= 0.18 or instability >= 0.75:
        return "HO_NOW"

    elif density >= 400 or speed <= 18 or stopped >= 0.10 or instability >= 0.60:
        return "LOAD_BALANCING"

    else:
        return "KEEP_CONNECTION"

df["decision"] = df.apply(generate_temporal_decision, axis=1)

df.to_csv(output_file, index=False)

print("Dataset temporel décisionnel multi généré :", output_file)
print("Shape :", df.shape)
print("\nDistribution des décisions :")
print(df["decision"].value_counts())
