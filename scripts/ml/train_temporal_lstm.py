import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("figures/temporal", exist_ok=True)

df = pd.read_csv("datasets/processed/mobility/temporal_sequences_decision_multi.csv")

features = [
    "mean_speed",
    "speed_std",
    "vehicle_density",
    "stopped_ratio",
    "lane_changes",
    "mobility_instability_index"
]

sequence_length = 5

X_seq = []

for _, row in df.iterrows():
    seq = []
    for t in range(sequence_length):
        seq.append([row[f"{f}_t{t}"] for f in features])
    X_seq.append(seq)

X = np.array(X_seq, dtype=np.float32)

encoder = LabelEncoder()
y = encoder.fit_transform(df["decision"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

n_train, t_steps, n_features = X_train.shape
n_test = X_test.shape[0]

X_train_2d = X_train.reshape(-1, n_features)
X_test_2d = X_test.reshape(-1, n_features)

X_train = scaler.fit_transform(X_train_2d).reshape(n_train, t_steps, n_features)
X_test = scaler.transform(X_test_2d).reshape(n_test, t_steps, n_features)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = LSTMClassifier(
    input_size=n_features,
    hidden_size=32,
    num_classes=len(encoder.classes_)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 80

for epoch in range(epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()

with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1).numpy()

acc = accuracy_score(y_test, predictions)

print("\nAccuracy LSTM temporel :", acc)

print("\nClassification Report :")
print(classification_report(
    y_test,
    predictions,
    target_names=encoder.classes_
))

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion - LSTM temporel")
plt.tight_layout()
plt.savefig("figures/temporal/confusion_temporal_lstm.png", dpi=300)
plt.close()

print("\nFigure sauvegardée : figures/temporal/confusion_temporal_lstm.png")
