import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("hi1_2025_09_eylul.xlsx")

df["Temp_cat"] = pd.cut(df["Sicaklik(°C)"], bins=[0, 18, 24, 40], labels=["Low", "Moderate", "High"])
df["Humid_cat"] = pd.cut(df["Nem(%)"], bins=[0, 40, 70, 100], labels=["Dry", "Moderate", "Humid"])
df["PM25_cat"] = pd.cut(df["Pm2.5"], bins=[0, 15, 35, 200], labels=["Good", "Moderate", "Poor"])

# Histogram çizimi
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
df["Temp_cat"].value_counts().sort_index().plot(kind='bar', ax=axes[0], title='Temperature')
df["Humid_cat"].value_counts().sort_index().plot(kind='bar', ax=axes[1], title='Humidity')
df["PM25_cat"].value_counts().sort_index().plot(kind='bar', ax=axes[2], title='PM2.5')
plt.tight_layout()
plt.savefig("discretization_hist.pdf", dpi=300)
plt.show()
