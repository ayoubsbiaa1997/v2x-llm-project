import re
import pandas as pd
from pathlib import Path

# =========================
# Dossier des logs bruts
# =========================

raw_dir = Path("datasets/raw/qos_logs_rich")
rows = []

# =========================
# Parcours des fichiers QoS
# =========================

for file in sorted(raw_dir.glob("qos_v*_l*_p*.txt")):
    name = file.stem

    vehicles = int(re.search(r"v(\d+)", name).group(1))
    traffic_lambda = int(re.search(r"l(\d+)", name).group(1))
    packet_size = int(re.search(r"p(\d+)", name).group(1))
    distance = int(re.search(r"d(\d+)", name).group(1))

    text = file.read_text(errors="ignore")

    throughput_match = re.search(
        r"Mean flow throughput:\s*([0-9.]+)",
        text
    )

    delay_match = re.search(
        r"Mean flow delay:\s*([0-9.]+)",
        text
    )

    throughput = float(throughput_match.group(1)) if throughput_match else 0.0
    delay = float(delay_match.group(1)) if delay_match else 0.0

    # =========================
    # Extraction robuste des flows
    # =========================

    flow_blocks = re.findall(
        r"Flow\s+\d+.*?Rx Packets:\s*\d+",
        text,
        re.DOTALL
    )

    tx = 0
    rx = 0
    jitters = []

    for block in flow_blocks:
        tx_match = re.search(r"Tx Packets:\s*(\d+)", block)
        rx_match = re.search(r"Rx Packets:\s*(\d+)", block)
        jitter_match = re.search(r"Mean jitter:\s*([0-9.]+)", block)

        if tx_match and rx_match:
            tx_val = int(tx_match.group(1))
            rx_val = int(rx_match.group(1))

            # Ignorer les flux totalement non reçus
            if rx_val == 0:
                continue

            tx += tx_val
            rx += rx_val

            if jitter_match:
                jitters.append(float(jitter_match.group(1)))

    jitter = sum(jitters) / len(jitters) if jitters else 0.0

    packet_loss = ((tx - rx) / tx) * 100 if tx > 0 else 0.0
    pdr = (rx / tx) * 100 if tx > 0 else 0.0

    # =========================
    # Congestion Index
    # =========================

    congestion_index = min(
        1.0,
        (0.35 * (delay / 30)) +
        (0.25 * (packet_loss / 20)) +
        (0.20 * (vehicles / 60)) +
        (0.20 * (traffic_lambda / 1000))
    )

    rows.append({
        "scenario_id": f"V{vehicles}_L{traffic_lambda}_P{packet_size}",
        "vehicles": vehicles,
        "distance_gnb_m": distance,
        "traffic_lambda": traffic_lambda,
        "packet_size": packet_size,
        "throughput_mbps": throughput,
        "delay_ms": delay,
        "jitter_ms": jitter,
        "packet_loss": packet_loss,
        "pdr": pdr,
        "congestion_index": congestion_index
    })

# =========================
# Sauvegarde du dataset QoS
# =========================

df = pd.DataFrame(rows)

df.to_csv("datasets/processed/qos_summary.csv", index=False)

print("qos_summary.csv enrichi généré.")
print("Nombre de scénarios :", len(df))
print(df.head())

print("\nStatistiques QoS :")
print(
    df[
        [
            "delay_ms",
            "packet_loss",
            "pdr",
            "congestion_index",
            "vehicles"
        ]
    ].describe()
)
