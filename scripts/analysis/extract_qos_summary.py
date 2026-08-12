import re
import pandas as pd
from pathlib import Path

# =========================
# Dossier des logs bruts
# =========================

raw_dir = Path("datasets/raw/qos_logs")

rows = []

# =========================
# Parcours des fichiers
# =========================

for file in sorted(raw_dir.glob("qos_v*.txt")):

    name = file.stem

    # Extraire nombre véhicules
    vehicles = int(re.search(r"v(\d+)", name).group(1))

    # Distance fixe pour le moment
    distance = 100

    # Lire le contenu
    text = file.read_text(errors="ignore")

    # =========================
    # Extraction throughput
    # =========================

    throughput_match = re.search(
        r"Mean flow throughput:\s*([0-9.]+)",
        text
    )

    delay_match = re.search(
        r"Mean flow delay:\s*([0-9.]+)",
        text
    )

    jitters = [
        float(x)
        for x in re.findall(r"Mean jitter:\s*([0-9.]+)\s*ms", text)
    ]

    tx_packets = [
        int(x)
        for x in re.findall(r"Tx Packets:\s*(\d+)", text)
    ]

    rx_packets = [
        int(x)
        for x in re.findall(r"Rx Packets:\s*(\d+)", text)
    ]

    # =========================
    # Calculs
    # =========================

    throughput = (
        float(throughput_match.group(1))
        if throughput_match else 0
    )

    delay = (
        float(delay_match.group(1))
        if delay_match else 0
    )

    jitter = (
        sum(jitters) / len(jitters)
        if jitters else 0
    )

    tx = sum(tx_packets)
    rx = sum(rx_packets)

    packet_loss = (
        ((tx - rx) / tx) * 100
        if tx > 0 else 0
    )

    pdr = (
        (rx / tx) * 100
        if tx > 0 else 0
    )

    # =========================
    # Congestion Index
    # =========================

    congestion_index = min(
        1.0,
        (0.4 * (delay / 30)) +
        (0.3 * (packet_loss / 20)) +
        (0.3 * (vehicles / 60))
    )

    # =========================
    # Ajouter ligne dataset
    # =========================

    rows.append({
        "scenario_id": f"V{vehicles}",
        "vehicles": vehicles,
        "distance_gnb_m": distance,
        "throughput_mbps": throughput,
        "delay_ms": delay,
        "jitter_ms": jitter,
        "packet_loss": packet_loss,
        "pdr": pdr,
        "congestion_index": congestion_index
    })

# =========================
# Génération CSV
# =========================

df = pd.DataFrame(rows)

output_file = "datasets/processed/qos_summary.csv"

df.to_csv(output_file, index=False)

print(f"Dataset généré : {output_file}")
print(df)
