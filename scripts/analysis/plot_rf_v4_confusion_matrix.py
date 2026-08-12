import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv(
    "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v4_balanced.csv"
)

features = [
    "throughput_mbps",
    "delay_ms",
    "jitter_ms",
    "packet_loss",
    "pdr",
    "congestion_index",
    "mean_speed",
    "max_speed",
    "min_speed",
    "speed_std",
    "vehicle_density",
    "stopped_ratio",
    "lane_changes",
    "mobility_instability_index"
]

X = df[features]
y = df["decision"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

disp = ConfusionMatrixDisplay.from_estimator(
    rf,
    X_test,
    y_test
)

plt.tight_layout()

plt.savefig(
    "rf_v4_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("Figure sauvegardée : rf_v4_confusion_matrix.png")
