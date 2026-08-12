import os
import matplotlib.pyplot as plt

os.makedirs("figures/temporal", exist_ok=True)

models = [
    "RF Temporel",
    "GRU",
    "LSTM"
]

accuracy = [
    93.15,
    86.76,
    88.58
]

plt.figure(figsize=(7,5))

bars = plt.bar(models, accuracy)

for bar, val in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        val + 0.3,
        f"{val:.2f}%",
        ha="center"
    )

plt.ylabel("Accuracy (%)")
plt.title("Comparaison des modèles temporels")
plt.ylim(80,100)

plt.tight_layout()

plt.savefig(
    "figures/temporal/temporal_models_comparison.png",
    dpi=300
)

print("Figure générée.")
