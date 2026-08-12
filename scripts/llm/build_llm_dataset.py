import pandas as pd
import json

df = pd.read_csv("datasets/processed/final_dataset.csv")

output_file = "datasets/llm/dataset_llm.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        sample = {
            "instruction": "Prendre une décision intelligente de handover dans un réseau V2X 5G/6G.",
            "input": (
                f"Nombre de véhicules: {row['vehicles']}\n"
                f"Distance gNB: {row['distance_gnb_m']} m\n"
                f"Throughput: {row['throughput_mbps']} Mbps\n"
                f"Délai: {row['delay_ms']} ms\n"
                f"Jitter: {row['jitter_ms']} ms\n"
                f"Packet Loss: {row['packet_loss']} %\n"
                f"PDR: {row['pdr']} %\n"
                f"Indice de congestion: {row['congestion_index']}\n"
                f"Handover attempts: {row['handover_attempts']}\n"
                f"Handover success rate: {row['handover_success_rate']} %\n"
                f"Délai moyen handover: {row['mean_handover_delay_ms']} ms"
            ),
            "output": (
                f"Décision: {row['decision']}\n"
                f"Explication: {row['explanation']}"
            )
        }

        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Dataset LLM généré : {output_file}")
print(f"Nombre d'exemples : {len(df)}")
