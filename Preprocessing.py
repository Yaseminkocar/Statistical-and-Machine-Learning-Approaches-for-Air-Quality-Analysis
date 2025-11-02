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

# -------- 5) IMPUTATION PREVIEW (CSV) ----------
def make_imputation_preview(df, out_dir):
    # Sadece en az 1 geçerli değeri olan sayısal sütunları al
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if df[c].notna().any()]

    def safe_median_fill(s, fallback=0):
        if s.notna().any():
            return s.fillna(s.median())
        else:
            return s.fillna(fallback)

    X = df[num_cols].copy()
    X_imp = X.apply(lambda s: safe_median_fill(s, fallback=0))

    out_csv = os.path.join(out_dir, "imputed_numeric_preview.csv")
    X_imp.head(20).to_csv(out_csv, index=False)
    return out_csv


# -------- 6) PCA 2D (PNG) ----------------------
def make_pca_2d(df, out_dir):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"Kayıt No", "Kayit No", "Sıra No", "Sira No", "Kayıt Yapan", "Kayit Yapan"}
    pca_cols = [c for c in num_cols if c not in drop_like]
    X = df[pca_cols].copy()
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        return None, None

    X_filled = X.apply(lambda s: s.fillna(s.median()))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled.values)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)

    # color by PM2.5 if available
    color_col = pick_col(df, ["Pm2.5", "PM2.5", "pm2.5", "Pm25", "PM25"])
    cvals = None
    if color_col and color_col in df.columns:
        cvals = pd.to_numeric(df[color_col], errors="coerce")

    fig_path = os.path.join(out_dir, "figure_pca2d.png")
    plt.figure(figsize=(6,5))
    sc = plt.scatter(pcs[:,0], pcs[:,1], s=8, c=cvals if cvals is not None else None)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("PCA 2D Visualization")
    if cvals is not None and not np.isnan(cvals).all():
        plt.colorbar(sc, shrink=0.85, label=color_col)
    plt.tight_layout(); plt.savefig(fig_path, dpi=180); plt.close()

    return fig_path, pca.explained_variance_ratio_.tolist()

# -------------------- MAIN ----------------------
def main():
    ensure_dir(OUT_DIR)
    df = load_data(EXCEL_PATH)

    # 1) Categorical modes
    modes_csv, modes_df = make_categorical_modes(df, OUT_DIR)

    # 2) Ordinal histogram
    ordinal_hist_png = make_ordinal_hist(df, OUT_DIR)

    # 3) Bar chart
    bar_chart_png = make_bar_chart(df, OUT_DIR)

    # 4) Outliers (IQR)
    outlier_csv, outlier_df, boxplots = make_outliers_iqr(df, OUT_DIR)

    # 5) Imputation preview
    imp_csv = make_imputation_preview(df, OUT_DIR)

    # 6) PCA 2D
    pca_png, pca_var = make_pca_2d(df, OUT_DIR)

    # Summary printout
    print("=== CE477 Report Assets Generated ===")
    print(f"- Modes CSV: {modes_csv}")
    print(f"- Ordinal Histogram: {ordinal_hist_png}")
    print(f"- Bar Chart: {bar_chart_png}")
    print(f"- Outlier Summary CSV: {outlier_csv}")
    print(f"- Outlier Boxplots: {boxplots}")
    print(f"- Imputed Preview CSV: {imp_csv}")
    print(f"- PCA 2D PNG: {pca_png}")
    print(f"- PCA Variance Ratios (PC1, PC2): {pca_var}")

if __name__ == "__main__":
    main()