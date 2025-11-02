import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_excel("hi1_2025_09_eylul.xlsx")

col_map = {
    "Pm2.5": ["Pm2.5", "PM2.5", "pm2_5"],
    "VOC": ["Voc", "VOC", "voc"],
    "CO2": ["CO2", "Co2", "co2"],
    "Sicaklik": ["Sıcaklık(°C)", "Sicaklik(°C)", "Sıcaklık (°C)", "Temperature"],
    "Nem": ["Nem(%)", "Nem (%)", "Humidity"],
    "CH2O": ["CH2O", "Ch2o", "ch2o"],
}

selected_cols = {}
for logical_name, candidates in col_map.items():
    for c in candidates:
        if c in df.columns:
            selected_cols[logical_name] = c
            break

cols_to_use = [
    selected_cols["Pm2.5"],
    selected_cols["VOC"],
    selected_cols["CO2"],
    selected_cols["CH2O"],
    selected_cols["Sicaklik"],
    selected_cols["Nem"]
]

df_before = df[cols_to_use].copy()

df_after = df_before.copy()

minmax_cols = []
zscore_cols = []

for key, real_col in selected_cols.items():
    if key in ["Pm2.5", "VOC", "CO2", "CH2O"]:
        minmax_cols.append(real_col)
    elif key in ["Sicaklik", "Nem"]:
        zscore_cols.append(real_col)

# Min–Max
if minmax_cols:
    mm = MinMaxScaler()
    df_after[minmax_cols] = mm.fit_transform(df_after[minmax_cols])

# Standardization
if zscore_cols:
    ss = StandardScaler()
    df_after[zscore_cols] = ss.fit_transform(df_after[zscore_cols])

# ---------------- FIGURE ----------------
n_features = len(cols_to_use)
fig, axes = plt.subplots(2, n_features, figsize=(4 * n_features, 6))

for i, col in enumerate(cols_to_use):
    ax1 = axes[0, i] if n_features > 1 else axes[0]
    ax2 = axes[1, i] if n_features > 1 else axes[1]

    ax1.hist(df_before[col].dropna(), bins=25)
    ax1.set_title(f"Before: {col}", fontsize=8)
    ax1.tick_params(axis='x', labelrotation=45)

    ax2.hist(df_after[col].dropna(), bins=25)
    ax2.set_title(f"After: {col}", fontsize=8)
    ax2.tick_params(axis='x', labelrotation=45)

axes[0, 0].set_ylabel("Count", fontsize=9)
axes[1, 0].set_ylabel("Count", fontsize=9)

plt.tight_layout()
plt.savefig("normalization_before_after.png", dpi=300)
