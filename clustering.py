import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (6, 5)

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, format="pdf", dpi=300)



df = pd.read_excel("hi1_2025_09_eylul.xlsx")

numeric_cols = [
    "Pm1", "Pm2.5", "Pm10",
    "CO2", "CH2O", "O3", "Voc",
    "Sicaklik(°C)", "Nem(%)", "His. Sıcaklık(°C)"
]

X = df[numeric_cols].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X_scaled)


(unique_k, counts_k) = np.unique(k_labels, return_counts=True)
for cid, size in zip(unique_k, counts_k):
    print(f"cluster {cid}: {size} observations")

centers_scaled = kmeans.cluster_centers_


centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_original, columns=numeric_cols)
pd.set_option("display.max_columns", None)
print(centers_df.round(2))


def plot_pca_clusters(X_pca, labels, title, filename):
    plt.figure()
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=10, cmap="viridis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(scatter, label="Cluster ID")
    save_plot(filename)
    plt.show()

plot_pca_clusters(X_pca, k_labels, "K-Means (k=3) Clusters in PCA Space", "kmeans_clusters.pdf")


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

Z_single = linkage(X_scaled, method="single", metric="euclidean")

plt.figure(figsize=(8, 4))
dendrogram(Z_single, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering Dendrogram (Single Linkage)")
plt.xlabel("observations (truncated)")
plt.ylabel("distance")
save_plot("dendrogram_single.pdf")
plt.show()

Z_complete = linkage(X_scaled, method="complete", metric="euclidean")

plt.figure(figsize=(8, 4))
dendrogram(Z_complete, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")
plt.xlabel("observations (truncated)")
plt.ylabel("distance")
save_plot("dendrogram_complete.pdf")
plt.show()

labels_single = fcluster(Z_single, t=3, criterion="maxclust")
labels_complete = fcluster(Z_complete, t=3, criterion="maxclust")


def print_cluster_sizes(name, labels):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{name}")
    for cid, size in zip(unique, counts):
        print(f"cluster {cid}: {size} observations")

print_cluster_sizes("Single linkage", labels_single)
print_cluster_sizes("Complete linkage", labels_complete)

plot_pca_clusters(X_pca, labels_complete, "Hierarchical (Complete, 3 clusters)", "hierarchical_complete_clusters.pdf")


eps = 1.0
min_samples = 30

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
db_labels = dbscan.fit_predict(X_scaled)

unique_db, counts_db = np.unique(db_labels, return_counts=True)
print("DBSCAN cluster size (label = -1 → noise):")
for cid, size in zip(unique_db, counts_db):
    print(f"Label {cid}: {size} observation")

cluster_centers_db = {}

for cid in unique_db:
    if cid == -1:
        continue  # noise
    mask = (db_labels == cid)
    center_scaled = X_scaled[mask].mean(axis=0)
    center_orig = scaler.inverse_transform(center_scaled.reshape(1, -1))[0]
    cluster_centers_db[int(cid)] = center_orig

db_centers_df = pd.DataFrame(cluster_centers_db).T
db_centers_df.columns = numeric_cols
print(db_centers_df.round(2))


plt.figure()
palette = ["grey", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
colors = [palette[label+1] if label != -1 else palette[0] for label in db_labels]

plt.scatter(X_pca[:,0], X_pca[:,1], c=colors, s=10)
plt.title(f"DBSCAN Clusters in PCA Space (eps={eps}, min_samples={min_samples})")
plt.xlabel("PC1")
plt.ylabel("PC2")

save_plot("dbscan_clusters.pdf")
plt.show()
