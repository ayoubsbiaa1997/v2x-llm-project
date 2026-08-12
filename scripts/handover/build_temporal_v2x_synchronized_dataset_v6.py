import pandas as pd
from pathlib import Path

V4_FILE = "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v4.csv"
HO_DIR = Path("datasets/processed/handover_campaign_v3")
OUT_FILE = "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v6.csv"

df = pd.read_csv(V4_FILE)

# Colonnes handover agrégées
df["handover_start_count"] = 0
df["handover_success_count"] = 0
df["handover_failure_count"] = 0
df["failure_no_preamble_count"] = 0
df["failure_max_rach_count"] = 0
df["failure_joining_count"] = 0
df["failure_leaving_count"] = 0
df["handover_event_count"] = 0
df["mean_handover_delay_ms"] = 0.0

event_files = sorted(HO_DIR.glob("handover_events_*.csv"))

if not event_files:
    raise FileNotFoundError(f"Aucun fichier handover_events trouvé dans {HO_DIR}")

events_all = []

for f in event_files:
    ev = pd.read_csv(f)
    ev["source_file"] = f.name
    events_all.append(ev)

events = pd.concat(events_all, ignore_index=True)
events["time_s"] = events["time_s"].astype(float)

# Affectation de chaque événement à la fenêtre temporelle V4 la plus proche
for _, event in events.iterrows():
    idx = (df["time_s"] - event["time_s"]).abs().idxmin()

    df.loc[idx, "handover_event_count"] += 1

    if event["event"] == "START":
        df.loc[idx, "handover_start_count"] += 1

    elif event["event"] == "SUCCESS":
        df.loc[idx, "handover_success_count"] += 1

        if str(event["handover_delay_ms"]) != "NA":
            current_count = df.loc[idx, "handover_success_count"]
            current_mean = df.loc[idx, "mean_handover_delay_ms"]
            new_delay = float(event["handover_delay_ms"])

            df.loc[idx, "mean_handover_delay_ms"] = (
                ((current_count - 1) * current_mean) + new_delay
            ) / current_count

    elif event["event"] == "FAILURE":
        df.loc[idx, "handover_failure_count"] += 1

        failure_type = str(event["failure_type"])

        if failure_type == "NO_PREAMBLE":
            df.loc[idx, "failure_no_preamble_count"] += 1
        elif failure_type == "MAX_RACH":
            df.loc[idx, "failure_max_rach_count"] += 1
        elif failure_type == "JOINING":
            df.loc[idx, "failure_joining_count"] += 1
        elif failure_type == "LEAVING":
            df.loc[idx, "failure_leaving_count"] += 1

# Décision enrichie V5
def enriched_decision(row):
    if row["handover_failure_count"] > 0:
        return "HO_RISK"

    if row["handover_success_count"] > 0 and row["congestion_index"] > 0.6:
        return "HO_CONFIRMED"

    return row["decision"]

df["decision_v5"] = df.apply(enriched_decision, axis=1)

df.to_csv(OUT_FILE, index=False)

print("Dataset V5 généré :", OUT_FILE)
print("Shape :", df.shape)
print()
print(df["decision_v5"].value_counts())
print()
print(df[
    [
        "handover_start_count",
        "handover_success_count",
        "handover_failure_count",
        "failure_no_preamble_count",
        "handover_event_count",
        "mean_handover_delay_ms",
        "decision_v5"
    ]
].head(10))
