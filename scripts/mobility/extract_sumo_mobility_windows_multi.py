import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# Configuration
# =========================

base_dir = Path("datasets/mobility/public_sumo/LuSTScenario/scenario")
output_file = Path("datasets/processed/mobility/mobility_profiles_windows_multi.csv")

output_file.parent.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 60  # 60 secondes = plus de fenêtres temporelles

xml_files = sorted(base_dir.glob("mobility_trace_*.xml"))

rows = []

# =========================
# Extraction multi-fichiers
# =========================

for xml_file in xml_files:
    scenario_name = xml_file.stem.replace("mobility_trace_", "")

    print(f"Traitement : {xml_file.name}")

    windows = {}
    current_time = 0.0
    current_vehicle_count = 0

    context = ET.iterparse(xml_file, events=("start", "end"))

    for event, elem in context:
        if event == "start" and elem.tag == "timestep":
            current_time = float(elem.attrib.get("time", 0))
            window_id = int(current_time // WINDOW_SIZE)

            if window_id not in windows:
                windows[window_id] = {
                    "speeds": [],
                    "vehicle_counts": [],
                    "lanes_by_vehicle": {}
                }

            current_vehicle_count = 0

        elif event == "end" and elem.tag == "vehicle":
            window_id = int(current_time // WINDOW_SIZE)

            vehicle_id = elem.attrib.get("id")
            speed = float(elem.attrib.get("speed", 0))
            lane = elem.attrib.get("lane", "")

            windows[window_id]["speeds"].append(speed)
            current_vehicle_count += 1

            if vehicle_id not in windows[window_id]["lanes_by_vehicle"]:
                windows[window_id]["lanes_by_vehicle"][vehicle_id] = []

            windows[window_id]["lanes_by_vehicle"][vehicle_id].append(lane)

            elem.clear()

        elif event == "end" and elem.tag == "timestep":
            window_id = int(current_time // WINDOW_SIZE)
            windows[window_id]["vehicle_counts"].append(current_vehicle_count)
            elem.clear()

    # =========================
    # Calcul des features
    # =========================

    for window_id, data in windows.items():
        speeds = np.array(data["speeds"])
        vehicle_counts = np.array(data["vehicle_counts"])

        if len(speeds) == 0:
            continue

        lane_changes = 0

        for _, lanes in data["lanes_by_vehicle"].items():
            for i in range(1, len(lanes)):
                if lanes[i] != lanes[i - 1]:
                    lane_changes += 1

        mean_speed = float(np.mean(speeds))
        max_speed = float(np.max(speeds))
        min_speed = float(np.min(speeds))
        speed_std = float(np.std(speeds))
        vehicle_density = float(np.mean(vehicle_counts))
        stopped_ratio = float(np.sum(speeds < 0.1) / len(speeds))
        mobility_instability_index = speed_std / mean_speed if mean_speed > 0 else 0

        rows.append({
            "scenario_source": scenario_name,
            "mobility_profile_id": f"{scenario_name}_W{window_id}",
            "window_start_s": window_id * WINDOW_SIZE,
            "window_end_s": (window_id + 1) * WINDOW_SIZE,
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "min_speed": min_speed,
            "speed_std": speed_std,
            "vehicle_density": vehicle_density,
            "stopped_ratio": stopped_ratio,
            "lane_changes": lane_changes,
            "mobility_instability_index": mobility_instability_index
        })

# =========================
# Sauvegarde
# =========================

df = pd.DataFrame(rows)
df = df.sort_values(["scenario_source", "window_start_s"])

df.to_csv(output_file, index=False)

print("\nFichier généré :", output_file)
print("Nombre total de fenêtres :", len(df))
print(df.head())
print("\nRésumé :")
print(df[[
    "mean_speed",
    "speed_std",
    "vehicle_density",
    "stopped_ratio",
    "lane_changes",
    "mobility_instability_index"
]].describe())
