import pandas as pd

# 1️⃣ Excel dosyasını oku
df = pd.read_excel("/Users/yasemin/Desktop/hi1_2025_09_eylul 2.xlsx")

# 2️⃣ Kategorik (harf veya metin içeren) sütunları seç
categorical_cols = df.select_dtypes(include=["object"]).columns

# 3️⃣ One-Hot Encoding uygula
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

df_encoded.to_excel("hi1_2025_09_eylul_encoded.xlsx", index=False)

print("One-hot encoding başarıyla tamamlandı!")
print("Yeni sütunlar:", df_encoded.columns.tolist())
