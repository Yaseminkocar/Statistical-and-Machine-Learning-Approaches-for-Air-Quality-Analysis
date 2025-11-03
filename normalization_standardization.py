import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_excel("hi1_2025_09_eylul.xlsx")

col_map = {
    "Pm2.5":     ["Pm2.5", "PM2.5", "pm2_5"],
    "VOC":       ["Voc", "VOC", "voc"],
    "CO2":       ["CO2", "Co2", "co2"],
    "CH2O":      ["CH2O", "Ch2o", "ch2o"],
    "Sicaklik":  ["Sıcaklık(°C)", "Sicaklik(°C)", "Sıcaklık (°C)", "Temperature"],
    "Nem":       ["Nem(%)", "Nem (%)", "Humidity"],
}

selected_cols = {}
for logical_name, candidates in col_map.items():
    for c in candidates:
        if c in df.columns:
            selected_cols[logical_name] = c
            break

norm_order = ["Pm2.5", "VOC", "CO2", "CH2O"]
std_order  = ["Sicaklik", "Nem"]

cols_norm = [selected_cols[k] for k in norm_order if k in selected_cols]
cols_std  = [selected_cols[k] for k in std_order  if k in selected_cols]

missing_norm = [k for k in norm_order if k not in selected_cols]
missing_std  = [k for k in std_order  if k not in selected_cols]
if missing_norm:
    print(f"[Uyarı] Normalization grubunda eksik bulunan mantıksal alanlar: {missing_norm}")
if missing_std:
    print(f"[Uyarı] Standardization grubunda eksik bulunan mantıksal alanlar: {missing_std}")

# --------- 1) NORMALIZATION (ilk 4 değişken) ---------
if cols_norm:
    df_before_norm = df[cols_norm].copy()
    # Min-Max sadece bu kolonlara uygulanır
    mm = MinMaxScaler()
    df_after_norm = pd.DataFrame(
        mm.fit_transform(df_before_norm.values),
        columns=cols_norm,
        index=df_before_norm.index
    )

    n_features = len(cols_norm)
    fig, axes = plt.subplots(2, n_features, figsize=(4 * n_features, 6))

    if n_features == 1:
        axes = axes.reshape(2, 1)

    for i, col in enumerate(cols_norm):
        ax1 = axes[0, i]
        ax2 = axes[1, i]

        ax1.hist(pd.to_numeric(df_before_norm[col], errors="coerce").dropna(), bins=25)
        ax1.set_title(f"Before: {col}", fontsize=9)
        ax1.tick_params(axis='x', labelrotation=45)

        ax2.hist(pd.to_numeric(df_after_norm[col], errors="coerce").dropna(), bins=25)
        ax2.set_title(f"After (Min–Max): {col}", fontsize=9)
        ax2.tick_params(axis='x', labelrotation=45)

    axes[0, 0].set_ylabel("Count", fontsize=9)
    axes[1, 0].set_ylabel("Count", fontsize=9)

    plt.tight_layout()
    plt.savefig("normalization_before_after.pdf", dpi=300)
    plt.close()
    print(" normalization_before_after.pdf oluşturuldu.")
else:
    print("[Bilgi] Normalization için uygun kolon bulunamadı; PDF üretilmedi.")

# --------- 2) STANDARDIZATION (son 2 değişken) ---------
if cols_std:
    df_before_std = df[cols_std].copy()
    # Z-score sadece bu kolonlara uygulanır
    ss = StandardScaler()
    df_after_std = pd.DataFrame(
        ss.fit_transform(df_before_std.values),
        columns=cols_std,
        index=df_before_std.index
    )

    n_features = len(cols_std)
    fig, axes = plt.subplots(2, n_features, figsize=(4 * n_features, 6))

    if n_features == 1:
        axes = axes.reshape(2, 1)

    for i, col in enumerate(cols_std):
        ax1 = axes[0, i]
        ax2 = axes[1, i]

        ax1.hist(pd.to_numeric(df_before_std[col], errors="coerce").dropna(), bins=25)
        ax1.set_title(f"Before: {col}", fontsize=9)
        ax1.tick_params(axis='x', labelrotation=45)

        ax2.hist(pd.to_numeric(df_after_std[col], errors="coerce").dropna(), bins=25)
        ax2.set_title(f"After (Z-score): {col}", fontsize=9)
        ax2.tick_params(axis='x', labelrotation=45)

    axes[0, 0].set_ylabel("Count", fontsize=9)
    axes[1, 0].set_ylabel("Count", fontsize=9)

    plt.tight_layout()
    plt.savefig("standardization_before_after.pdf", dpi=300)
    plt.close()
    print(" standardization_before_after.pdf oluşturuldu.")
else:
    print("[Bilgi] Standardization için uygun kolon bulunamadı; PDF üretilmedi.")
