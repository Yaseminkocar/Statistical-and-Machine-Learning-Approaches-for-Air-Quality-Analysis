import pandas as pd

df = pd.read_excel("/Users/yasemin/Desktop/hi1_2025_09_eylul.xlsx")

categorical_cols = df.select_dtypes(include=["object"]).columns

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

df_encoded.to_excel("hi1_2025_09_eylul_encoded.xlsx", index=False)

print("One-hot encoding successful")
print("Yeni sütunlar:", df_encoded.columns.tolist())
