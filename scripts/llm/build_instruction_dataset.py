import pandas as pd
import json

df = pd.read_csv(
    "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v3.csv"
)

records = []

for _, row in df.iterrows():

    decision = row["decision"]

    context = (
        f"Throughput={row['throughput_mbps']:.2f} Mbps, "
        f"Delay={row['delay_ms']:.2f} ms, "
        f"PacketLoss={row['packet_loss']:.2f} %, "
        f"Density={row['vehicle_density']:.2f}, "
        f"MobilityInstability={row['mobility_instability_index']:.2f}"
    )

    if decision == "KEEP_CONNECTION":
        explanation = (
            "The network conditions are stable. "
            "Latency and packet loss remain acceptable, "
            "therefore the current connection should be maintained."
        )

    elif decision == "LOAD_BALANCING":
        explanation = (
            "Network congestion is increasing and traffic load "
            "should be redistributed to maintain service quality."
        )

    elif decision == "HO_NOW":
        explanation = (
            "The radio link quality is degrading. "
            "A handover is recommended immediately to preserve "
            "connectivity and QoS."
        )

    else:
        explanation = "No explanation available."

    sample = {
        "instruction":
            "Analyze the V2X network situation and explain the recommended decision.",
        "input": context,
        "output": explanation,
        "decision": decision
    }

    records.append(sample)

with open(
    "datasets/llm/instruction_dataset.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Instruction dataset generated")
print("Samples:", len(records))
