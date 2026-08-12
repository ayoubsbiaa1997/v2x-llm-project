import pandas as pd
from pathlib import Path

mob = pd.read_csv("datasets/processed/mobility/mobility_profiles_windows_multi.csv")

rows = []

def fuse_qos_with_mobility(qos_file, decision_label, random_state):
    qos = pd.read_csv(qos_file)
    mob_sample = mob.sample(len(qos), replace=True, random_state=random_state)

    df = pd.concat(
        [qos.reset_index(drop=True), mob_sample.reset_index(drop=True)],
        axis=1
    )

    df["qos_source"] = Path(qos_file).stem
    df["decision"] = decision_label

    return df

# KEEP_CONNECTION : scénarios stables
for i, f in enumerate(sorted(Path("datasets/processed/mobility").glob("qos_KEEP_*.csv"))):
    rows.append(fuse_qos_with_mobility(f, "KEEP_CONNECTION", 10 + i))

# LOAD_BALANCING : charge intermédiaire
rows.append(fuse_qos_with_mobility(
    "datasets/processed/mobility/qos_D200.csv",
    "LOAD_BALANCING",
    20
))

# HO_NOW : congestion forte
rows.append(fuse_qos_with_mobility(
    "datasets/processed/mobility/qos_D300.csv",
    "HO_NOW",
    30
))

dataset = pd.concat(rows, ignore_index=True)

out = "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v3.csv"
dataset.to_csv(out, index=False)

print("Dataset V3 généré :", out)
print(dataset.shape)
print(dataset["decision"].value_counts())
print(dataset[["qos_source", "decision"]].value_counts())
