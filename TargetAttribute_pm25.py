import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "hi1_2025_09_eylul.xlsx"
df = pd.read_excel(file_path)

pm = df["Pm2.5"]

bins = [0, 15, 35, np.inf]
labels = ["Good", "Moderate", "Poor"]

df["PM25_cat"] = pd.cut(pm, bins=bins, labels=labels, right=False)

print(df["PM25_cat"].value_counts())


counts = df["PM25_cat"].value_counts().reindex(labels)

plt.figure()
counts.plot(kind="bar")
plt.xlabel("PM2.5 category")
plt.ylabel("Number of records")
plt.title("Class distribution of PM25_cat")
plt.tight_layout()
plt.savefig("pm25_class_distribution.png", dpi=300)
plt.show()

