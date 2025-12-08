import pandas as pd
import numpy as np

df = pd.read_excel("hi1_2025_09_eylul.xlsx")

df["Temp_cat"] = pd.cut(
    df["Sicaklik(°C)"],
    bins=[0, 18, 24, np.inf],
    labels=["Low", "Moderate", "High"],
    right=False
)

df["Humid_cat"] = pd.cut(
    df["Nem(%)"],
    bins=[0, 40, 70, np.inf],
    labels=["Dry", "Moderate", "Humid"],
    right=False
)

df["PM25_cat"] = pd.cut(
    df["Pm2.5"],
    bins=[0, 15, 35, np.inf],
    labels=["Good", "Moderate", "Poor"],
    right=False
)

df["PM10_cat"] = pd.cut(
    df["Pm10"],
    bins=[0, 55, 155, np.inf],
    labels=["Good", "Moderate", "Poor"],
    right=False
)

print(df[[
    "Sicaklik(°C)", "Temp_cat",
    "Nem(%)", "Humid_cat",
    "Pm2.5", "PM25_cat",
    "Pm10", "PM10_cat"
]].head())

df.to_excel("hi1_2025_09_eylul_discretized.xlsx", index=False)
print("Done, 'hi1_2025_09_eylul_discretized.xlsx' is ready to use.")
