import pandas as pd

input_file = "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v3.csv"
output_file = "datasets/processed/mobility/temporal_v2x_synchronized_dataset_v4_balanced.csv"

df = pd.read_csv(input_file)

min_count = df["decision"].value_counts().min()

balanced = (
    df.groupby("decision", group_keys=False)
      .apply(lambda x: x.sample(min_count, random_state=42))
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

balanced.to_csv(output_file, index=False)

print("Dataset équilibré généré :", output_file)
print(balanced.shape)
print(balanced["decision"].value_counts())
