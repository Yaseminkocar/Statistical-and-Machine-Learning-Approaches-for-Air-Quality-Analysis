import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))


file_path = os.path.join(base_dir, "hi1_2025_09_eylul.xlsx")

df = pd.read_excel(file_path)

categorical_cols = df.select_dtypes(include=["object"]).columns

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

output_path = os.path.join(base_dir, "hi1_2025_09_eylul_encoded.xlsx")
df_encoded.to_excel(output_path, index=False)

print(" One-hot encoding successful!")
print("Encoded file saved to:", output_path)
print("New columns:", df_encoded.columns.tolist())
