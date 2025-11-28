import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_path = "hi1_2025_09_eylul_encoded.xlsx"
df = pd.read_excel(file_path)


pm10 = df["Pm10"]

bins = [0, 55, 155, np.inf]
labels = ["Good", "Moderate", "Poor"]

df["PM10_cat"] = pd.cut(pm10, bins=bins, labels=labels, right=False)

print("\nPM10_cat class counts:")
print(df["PM10_cat"].value_counts())


counts = df["PM10_cat"].value_counts().reindex(labels)

plt.figure()
counts.plot(kind="bar")
plt.xlabel("PM10 category")
plt.ylabel("Number of records")
plt.title("Class distribution of PM10_cat")
plt.tight_layout()
plt.savefig("pm10_class_distribution.pdf", dpi=300)
plt.show()


output_file = "classification/hi1_2025_09_eylul_encoded_with_PM10cat.xlsx"
df.to_excel(output_file, index=False)

print(f"\nSaved updated dataset with PM10_cat to: {output_file}")
