import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ks_2samp


df = pd.read_excel(r"C:\Users\90543\Downloads\CE477-Project-main\CE477-Project-main\hi1_2025_09_eylul.xlsx")

##this is to keep track of the augmented data. boolean. if augmented =1, not=0
df["Augmented"] = 0
print("✅ 'Augmented' column added to the original dataset.")
print(df.head())




#MISSING VALUE HANDLING
# Kayit Yapan tamamen bos dropla
df = df.drop(columns=["Kayıt Yapan"], errors="ignore")

# 3.timestamp column to datetime format
df["Kayıt Tarihi"] = pd.to_datetime(df["Kayıt Tarihi"], errors="coerce")

# bos var mi kontrol
print("Missing values before handling:")
print(df.isna().sum())

'''
# buna simdilik ihtiyac yok gibi
Handle missing values (if any appear in future data)
# Forward fill small gaps, then fill any remaining numeric NaNs with mean
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(method="ffill").fillna(df[numeric_cols].mean())

# For categorical columns, fill with most frequent value
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
'''


#OUTLIER DETECTION
#Voc: 96 outliers detected. others:0
def detect_outliers_iqr(dataframe, columns, save_dir="plots_outliers"):
    """
    IQR method.
    Saves boxplots for each column in a specified folder.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\nDetecting outliers using the IQR method...\n")

    for col in columns:
        if col not in dataframe.columns:
            print(f"⚠️  Column '{col}' not found in dataset, skipping.")
            continue

        # Calculate quartiles and IQR
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        num_outliers = ((dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)).sum()
        print(f"{col}: {num_outliers} outliers detected.")

        # Save boxplot
        plt.figure(figsize=(5, 3))
        plt.boxplot(dataframe[col], vert=True, patch_artist=True)
        plt.title(f"Outlier Detection for {col}")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{col}_boxplot.png", dpi=300)
        plt.close()

    print(f"\n✅ Outlier detection completed. Boxplots saved in '{save_dir}' folder.\n")

pollutant_cols = ["Pm1", "Pm2.5", "Pm10", "Voc", "Sicaklik(°C)", "Nem(%)"]

##detect_outliers_iqr(df, pollutant_cols)

# ================================
# 2.5 Remove Outliers (optional)
# ================================


#keeping outliers is recommended for VOC as they may be real values
def remove_outliers_iqr(dataframe, columns):
    #IQR method
    df_clean = dataframe.copy()

    for col in columns:
        if col not in df_clean.columns:
            print(f"⚠️ Column '{col}' not found, skipping.")
            continue

        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        after = len(df_clean)
        removed = before - after

        print(f"{col}: Removed {removed} outliers.")

    print(f"\n✅ Outlier removal complete. Final dataset length: {len(df_clean)} rows.\n")
    return df_clean


#if we want to remove outliers
pollutant_cols = ["Pm1", "Pm2.5", "Pm10", "Voc", "Sicaklik(°C)", "Nem(%)"]

# Create a cleaned version without outliers
df_no_outliers = remove_outliers_iqr(df, pollutant_cols)



def perform_pca(dataframe, title="PCA Visualization", save_path=None):
    """
    Performs PCA on numeric columns of the dataset and visualizes
    the first two principal components.

    Parameters:
        dataframe (pd.DataFrame): Input data.
        title (str): Title of the PCA plot.
        save_path (str): Optional path to save the plot (PNG).
    """
    # Select only numeric columns
    numeric_cols = dataframe.select_dtypes(include=["float64", "int64"]).columns

    # Check if there are enough numeric features
    if len(numeric_cols) < 2:
        print("⚠️ Not enough numeric columns for PCA.")
        return

    # Standardize (normalize) data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe[numeric_cols])

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Create DataFrame for PCA results
    df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

    # Explained variance ratio
    explained = pca.explained_variance_ratio_ * 100
    print(f"\n🔹 PCA completed: PC1 = {explained[0]:.2f}% , PC2 = {explained[1]:.2f}% of total variance.\n")

    # Plot PCA
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="PC1", y="PC2", data=df_pca, alpha=0.6)
    plt.title(title)
    plt.xlabel(f"Principal Component 1 ({explained[0]:.1f}% variance)")
    plt.ylabel(f"Principal Component 2 ({explained[1]:.1f}% variance)")
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ PCA plot saved: {save_path}")
    else:
        plt.show()



perform_pca(df, title="PCA with Outliers", save_path="pca_with_outliers.png")

# PCA excluding outliers
perform_pca(df_no_outliers, title="PCA without Outliers", save_path="pca_without_outliers.png")


'''
🔹 PCA completed: PC1 = 32.69% , PC2 = 25.03% of total variance.
✅ PCA plot saved: pca_with_outliers.png
🔹 PCA completed: PC1 = 35.84% , PC2 = 27.21% of total variance.
✅ PCA plot saved: pca_without_outliers.png


Moderate correlation between features
Since PC1 and PC2 together explain around 60% of the total variance (not, say, 90%), it suggests:
Your features (PM1, PM2.5, PM10, VOC, temperature, humidity) are not perfectly correlated,
Each variable contributes unique information — which is good in environmental data.
Likely relationships:
PM1, PM2.5, PM10 are strongly correlated (they measure similar particles).
VOC varies somewhat independently.
Temperature and humidity contribute separate environmental context.
'''



def augment_data(df, columns, num_copies=1, noise_level=0.02):
    """
    Augments numeric columns by adding small Gaussian noise.
    Uses existing 'Augmented' column to label synthetic data (0=real, 1=augmented).

    Parameters:
        df (pd.DataFrame): Original dataset, must contain column 'Augmented'.
        columns (list): List of numeric columns to augment.
        num_copies (int): How many synthetic copies to generate.
        noise_level (float): Noise standard deviation as a fraction of column std.

    Returns:
        pd.DataFrame: Combined dataset with real and augmented samples.
    """

    # Check that the 'Augmented' column exists
    if "Augmented" not in df.columns:
        raise ValueError("DataFrame must have an 'Augmented' column before augmentation.")

    # Copy to avoid modifying the original directly
    df = df.copy().reset_index(drop=True)

    # Verify columns exist in the dataset
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Ensure numeric columns are float for noise addition
    df[columns] = df[columns].astype(float)
    numeric = df[columns].astype(float).copy()
    df_aug = df.copy()

    for i in range(num_copies):
        noisy = numeric.copy()
        for col in columns:
            std = numeric[col].std()
            if pd.isna(std) or std == 0:
                std = numeric[col].abs().mean() * 1e-6 + 1e-6
            noise = np.random.normal(0, std * noise_level, size=len(numeric))
            noisy[col] = numeric[col].values + noise

        # Create the new (augmented) DataFrame
        df_noisy = df.copy()
        df_noisy.loc[:, columns] = noisy.values
        df_noisy["Augmented"] = 1  # mark as synthetic

        # Combine real + synthetic data
        df_aug = pd.concat([df_aug, df_noisy], ignore_index=True)

    print(f"✅ Data augmentation complete. Original rows: {len(df)}, total with augmented: {len(df_aug)}.")
    print("   - 'Augmented' column: 0 = real data, 1 = synthetic data.")
    return df_aug
df_augmented = augment_data(df, pollutant_cols, num_copies=1, noise_level=0.02)

# Augment the dataset without outliers
df_no_outliers_augmented = augment_data(df_no_outliers, pollutant_cols, num_copies=1, noise_level=0.02)

def plot_augmentation_comparison(original_df, augmented_df, column, save_dir="plots_augmentation"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    if column not in original_df.columns:
        print(f"⚠️ Column '{column}' not found in dataset, skipping plot.")
        return

    plt.figure(figsize=(6, 4))
    # Plot augmented first (so original is on top)
    plt.hist(augmented_df[column], bins=30, alpha=0.4, color="#ff7f0e", label="Augmented")
    plt.hist(original_df[column], bins=30, alpha=0.6, color="#1f77b4", label="Original")

    plt.legend()
    plt.title(f"{column} Distribution Before and After Augmentation")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{column}_augmentation_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved augmentation comparison plot for '{column}' → {save_path}")
for col in pollutant_cols:
    plot_augmentation_comparison(df, df_augmented, col)



print("NaN counts per column:")
print(df_augmented.isna().sum())
perform_pca(df_augmented, title="PCA with Outliers (Augmented)", save_path="pca_with_outliers_aug.png")
perform_pca(df_no_outliers_augmented, title="PCA without Outliers (Augmented)", save_path="pca_without_outliers_aug.png")




##testing if augmentation messed up the data set
## >0.05 good  <0.05 bad

def compare_distributions(original, augmented, columns):
    results = {}
    for col in columns:
        stat, p = ks_2samp(original[col], augmented[col])
        results[col] = {"KS_statistic": stat, "p_value": p}
    return pd.DataFrame(results).T

dist_results = compare_distributions(df, df_augmented, pollutant_cols)
print(dist_results)
