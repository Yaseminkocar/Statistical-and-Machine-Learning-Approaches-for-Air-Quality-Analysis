import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

EXCEL_PATH = "hi1_2025_09_eylul.xlsx"
OUT_DIR = "."

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def load_data(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.astype(str).str.strip()
    time_col = pick_col(df, ["Kayıt Tarihi", "Kayit Tarihi", "Tarih", "Timestamp", "Date"])
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df

# -------- 1) CATEGORICAL MODES (TABLE) ---------
def make_categorical_modes(df, out_dir):
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    rows = []
    for c in obj_cols:
        mode_series = df[c].mode(dropna=True)
        mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan
        mode_cnt = int((df[c] == mode_val).sum()) if pd.notna(mode_val) else 0
        rows.append({
            "Attribute": c,
            "Mode": mode_val,
            "Mode Count": mode_cnt,
            "Unique": int(df[c].nunique(dropna=True))
        })
    out = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "categorical_modes.csv")
    out.to_csv(out_path, index=False)
    return out_path, out

# -------- 2) ORDINAL HISTOGRAM (PNG) -----------
def make_ordinal_hist(df, out_dir):
    ordinal_col = pick_col(df, ["His. Sıcaklık(°C)", "His. Sicaklik(°C)", "His. Sicaklik", "Nem(%)", "Nem"])
    if not ordinal_col or not pd.api.types.is_numeric_dtype(df[ordinal_col]):
        return None
    fig_path = os.path.join(out_dir, "figure_ordinal_hist.png")
    plt.figure(figsize=(6,4))
    plt.hist(pd.to_numeric(df[ordinal_col], errors="coerce").dropna(), bins=20)
    plt.title(f"Histogram of {ordinal_col}")
    plt.xlabel(ordinal_col); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(fig_path, dpi=180); plt.close()
    return fig_path

# -------- 3) BAR CHART (PNG) -------------------
def make_bar_chart(df, out_dir):
    bar_col = pick_col(df, ["Cihaz", "Aktif Mi"])
    if not bar_col:
        return None
    vc = df[bar_col].astype(str).value_counts().sort_values(ascending=False).head(15)
    fig_path = os.path.join(out_dir, "figure_bar_chart.png")
    plt.figure(figsize=(7,4))
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(f"Bar Chart of {bar_col}")
    plt.xlabel(bar_col); plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(fig_path, dpi=180); plt.close()
    return fig_path

# -------- 4) OUTLIER DETECTION (IQR) -----------
def make_outliers_iqr(df, out_dir):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    targets = [c for c in ["Pm10", "PM10", "Voc", "VOC"] if c in df.columns]
    if not targets:
        # pick two widest-range numeric columns as fallback
        ranges = df[num_cols].max(numeric_only=True) - df[num_cols].min(numeric_only=True)
        targets = ranges.sort_values(ascending=False).index.tolist()[:2]

    rows = []
    boxplots = []
    for c in targets:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        cnt = int(((s < lower) | (s > upper)).sum())
        rows.append({
            "Attribute": c,
            "Q1": float(q1), "Q3": float(q3),
            "IQR": float(iqr),
            "Lower Fence": float(lower), "Upper Fence": float(upper),
            "Outlier Count": cnt, "N": int(s.shape[0])
        })

        # boxplot
        pth = os.path.join(out_dir, f"figure_outliers_{c}.png")
        plt.figure(figsize=(6,4))
        plt.boxplot(s.values, vert=True, showmeans=True)
        plt.title(f"Outlier Detection (IQR) — {c}")
        plt.ylabel(c)
        plt.tight_layout(); plt.savefig(pth, dpi=180); plt.close()
        boxplots.append(pth)

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "outlier_summary_iqr.csv")
    out_df.to_csv(out_csv, index=False)
    return out_csv, out_df, boxplots

