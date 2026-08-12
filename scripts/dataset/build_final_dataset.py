import pandas as pd

# =========================
# Lecture des fichiers
# =========================

qos = pd.read_csv("datasets/processed/qos_summary.csv")
handover = pd.read_csv("datasets/processed/handover_summary.csv")

# =========================
# Répéter les métriques handover
# pour chaque scénario QoS
# =========================

for col in handover.columns:
    qos[col] = handover.loc[0, col]

# =========================
# Création de labels décisionnels enrichis
# =========================

def generate_decision(row):
    congestion = row["congestion_index"]
    delay = row["delay_ms"]
    packet_loss = row["packet_loss"]
    pdr = row["pdr"]
    vehicles = row["vehicles"]
    distance = row["distance_gnb_m"]

    if (
        congestion >= 0.35
        or delay >= 0.85
        or packet_loss >= 0.4
        or pdr < 99.6
        or (vehicles >= 60 and distance >= 300)
    ):
        return "HO_NOW"

    elif (
        congestion >= 0.18
        or delay >= 0.55
        or packet_loss >= 0.1
        or pdr < 99.9
        or vehicles >= 40
        or distance >= 200
    ):
        return "LOAD_BALANCING"

    else:
        return "KEEP_CONNECTION"


def generate_explanation(row):
    decision = row["decision"]

    if decision == "HO_NOW":
        return (
            "Les indicateurs réseau montrent une forte dégradation "
            "des performances avec une congestion élevée, un délai "
            "important ou un taux de pertes critique. "
            "Un handover immédiat est recommandé afin de maintenir "
            "la continuité de service V2X."
        )

    elif decision == "LOAD_BALANCING":
        return (
            "Le réseau présente des signes de surcharge modérée "
            "ou une augmentation progressive du délai et de la congestion. "
            "Un mécanisme de rééquilibrage de charge ou une surveillance "
            "du handover est recommandé."
        )

    else:
        return (
            "Les métriques QoS restent stables avec une bonne qualité "
            "de transmission et un faible niveau de congestion. "
            "Aucune opération de handover n’est actuellement nécessaire."
        )


# =========================
# Application des labels
# =========================

qos["decision"] = qos.apply(generate_decision, axis=1)
qos["explanation"] = qos.apply(generate_explanation, axis=1)

# =========================
# Sauvegarde du dataset final
# =========================

qos.to_csv("datasets/processed/final_dataset.csv", index=False)

print("final_dataset.csv enrichi généré avec succès.")
print(qos)

print("\nDistribution des décisions :")
print(qos["decision"].value_counts())
