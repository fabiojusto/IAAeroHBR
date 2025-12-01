import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import matplotlib.cm as cm

# =========== CONFIGURAÇÃO ===========
BASE_DIR = Path(__file__).resolve().parent 
DATA_DIR = BASE_DIR / "Dataset"
GLOB_PATTERN = "*.csv"
OUTPUT_DIR = BASE_DIR / "SaidaKmeans"
MIN_CLUSTERS = 2
MAX_CLUSTERS = 11
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)

# =========== LEITURA DOS ARQUIVOS ===========
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, GLOB_PATTERN)))
if not csv_files:
    raise FileNotFoundError(f"Nenhum CSV encontrado em {DATA_DIR} com o padrão {GLOB_PATTERN}")

list_dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
    except Exception:
        df = pd.read_csv(f, encoding='latin1')
    filename = os.path.basename(f)
    df["__file__"] = filename
    list_dfs.append(df)

all_df = pd.concat(list_dfs, ignore_index=True)

# Apenas colunas numéricas
num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) == 0:
    raise ValueError("Nenhuma coluna numérica encontrada — verifique os CSVs")

X_raw = all_df[num_cols].values 
filenames = all_df["__file__"].values

# =========== ESCALONAMENTO ===========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# =========== PCA (uma única vez, em todas as linhas) ===========
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
all_df["pca1"] = X_pca[:, 0]
all_df["pca2"] = X_pca[:, 1]

# Médias em PCA por arquivo (instâncias)
file_groups = all_df.groupby("__file__")
file_pca_means = file_groups[["pca1", "pca2"]].mean()

# =========== LOOP K-MEANS ===========
for n_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
    print(f"\n=== K-Means com k = {n_clusters} ===")

    # K-Means nas LINHAS escaladas 
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(X_scaled)  # shape (N_linhas,)

    # Silhouette global nas linhas
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette médio (linhas): {silhouette_avg:.3f}")

    # Centróides no espaço escalado e projeção em PCA
    centers_scaled = kmeans.cluster_centers_
    centers_pca = pca.transform(centers_scaled)

    # =========== PLOT EM PCA  ===========
    fig, ax = plt.subplots(figsize=(10, 8))

    # Pontos (linhas) coloridos por cluster
    scatter = ax.scatter(
        all_df["pca1"],
        all_df["pca2"],
        c=cluster_labels,
        s=10,
        cmap="tab10",
        alpha=0.6
    )

    # Centróides (X) em preto
    ax.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        marker="X",
        s=200,
        edgecolor="k",
        facecolor="white"
    )

    # Nome dos clusters (cluster_1, cluster_2, ...)
    for i in range(n_clusters):
        txt = f"cluster_{i+1}"
        ax.annotate(
            txt,
            (centers_pca[i, 0], centers_pca[i, 1]),
            fontsize=9,
            fontweight="bold"
        )

    # Anotações dos nomes de arquivos (instâncias) com offset e seta
    for idx, (fname, row) in enumerate(file_pca_means.iterrows()):
        ax.scatter(
            row["pca1"],
            row["pca2"],
            marker="o",
            s=100,
            facecolors="none",
            edgecolors="k"
        )

        offset_x = 15 * ((-1) ** idx)
        offset_y = 10 * ((idx % 3) - 1)

        ax.annotate(
            fname,
            (row["pca1"], row["pca2"]),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
            arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", alpha=0.5)
        )

    ax.set_title(f"K-Means k={n_clusters} — Linhas (pontos), Centr. (X), Instâncias (labels)")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / f"kmeans_rows_k{n_clusters}.png", dpi=300)
    plt.close(fig)

    # =========== SILHOUETTE POR LINHA  ===========
    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
    y_lower = 10
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.title(f"Silhouette plot (linhas) para k = {n_clusters}\nScore médio = {silhouette_avg:.3f}")
    plt.xlabel("Coeficiente de Silhouette")
    plt.ylabel("Cluster")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / f"kmeans_rows_silhouette_k{n_clusters}.png", dpi=300)
    plt.close()

    # =========== TABELA AGRUPAMENTO ===========
    # clusters por linha
    resultado_linhas = pd.DataFrame({
        "Arquivo": filenames,
        "Cluster": cluster_labels
    })
    resultado_linhas.to_csv(
        OUTPUT_DIR / "tables" / f"kmeans_rows_assignments_k{n_clusters}.csv",
        index=False
    )

    # centróides em espaço original
    centers_orig = scaler.inverse_transform(centers_scaled)
    centroids_df = pd.DataFrame(centers_orig, columns=num_cols)
    centroids_df.index = [f"cluster_{i+1}" for i in range(n_clusters)]
    centroids_df.to_csv(
        OUTPUT_DIR / "tables" / f"kmeans_rows_centroids_k{n_clusters}.csv"
    )

print("Processo finalizado. Resultados em:", OUTPUT_DIR)
