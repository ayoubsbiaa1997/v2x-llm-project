import pandas as pd
from pathlib import Path

HO_DIR = Path("datasets/processed/handover_campaign_v3")
OUT_FILE = "datasets/processed/handover_dataset_v6.csv"

rows = []

for f in sorted(HO_DIR.glob("handover_events_*.csv")):
    df = pd.read_csv(f)

    scenario = f.stem.replace("handover_events_", "")

    parts = scenario.split("_")
    ue = int(parts[2].replace("U", ""))
    speed = int(parts[3].replace("S", ""))
    distance = int(parts[4].replace("D", ""))

    start_count = (df["event"] == "START").sum()
    success_count = (df["event"] == "SUCCESS").sum()
    failure_count = (df["event"] == "FAILURE").sum()
    no_preamble_count = ((df["event"] == "FAILURE") & (df["failure_type"] == "NO_PREAMBLE")).sum()

    success_delays = pd.to_numeric(df.loc[df["event"] == "SUCCESS", "handover_delay_ms"], errors="coerce")

    rows.append({
        "scenario_id": scenario,
        "ue_num_per_gnb": ue,
        "ue_speed": speed,
        "bs_distance": distance,
        "handover_start_count": start_count,
        "handover_success_count": success_count,
        "handover_failure_count": failure_count,
        "failure_no_preamble_count": no_preamble_count,
        "handover_event_count": len(df),
        "mean_handover_delay_ms": success_delays.mean() if not success_delays.empty else 0.0,
        "max_handover_delay_ms": success_delays.max() if not success_delays.empty else 0.0,
    })

out = pd.DataFrame(rows)

def risk_label(row):
    if row["handover_failure_count"] > 0:
        return "HO_RISK"
    if row["handover_start_count"] > 0:
        return "HO_CONFIRMED"
    return "NO_HANDOVER"

out["handover_decision"] = out.apply(risk_label, axis=1)

out.to_csv(OUT_FILE, index=False)

print("Dataset handover V6 généré :", OUT_FILE)
print(out.shape)
print()
print(out["handover_decision"].value_counts())
print()
print(out.head())
