# fuzzy_cmeans.py
"""
Pipeline Fuzzy C-Means com plots PCA + centróides + instâncias e
gráficos de silhueta fuzzy por k.

Requer: numpy, pandas, scikit-learn, matplotlib, scikit-fuzzy, openpyxl
pip install numpy pandas scikit-learn matplotlib scikit-fuzzy openpyxl
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from pathlib import Path

# =========== CONFIGURAÇÃO ===========
BASE_DIR = Path(__file__).resolve().parent 
DATA_DIR = BASE_DIR / "Dataset"
GLOB_PATTERN = "*.csv"
OUTPUT_DIR = BASE_DIR / "SaidaFCM20"
MIN_CLUSTERS = 2
MAX_CLUSTERS = 11
FCM_M = 2.0
FCM_ERROR = 0.00001
FCM_MAXITER = 1000
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)

# =========== LEITURA DOS ARQUIVOS ===========
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, GLOB_PATTERN)))
if not csv_files:
    raise FileNotFoundError(f"Nenhum CSV encontrado em {DATA_DIR} com o padrão {GLOB_PATTERN}")

list_dfs = []
file_row_counts = {}
for f in csv_files:
    try:
        df = pd.read_csv(f)
    except Exception:
        df = pd.read_csv(f, encoding='latin1')
    filename = os.path.basename(f)
    df['__file__'] = filename
    list_dfs.append(df)
    file_row_counts[filename] = len(df)

all_df = pd.concat(list_dfs, ignore_index=True)
num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) == 0:
    raise ValueError("Nenhuma coluna numérica encontrada — verifique os CSVs")

X_raw = all_df[num_cols].values
filenames = all_df['__file__'].values

# =========== ESCALONAMENTO/NORMALIZAÇÃO ===========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# =========== PCA ===========
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
all_df['pca1'] = X_pca[:, 0]
all_df['pca2'] = X_pca[:, 1]
file_groups = all_df.groupby('__file__')
file_pca_means = file_groups[['pca1', 'pca2']].mean()

# =========== FUNÇÕES DE AVALIACAO ===========
def save_df(df, name):
    csv_path = OUTPUT_DIR / 'tables' / f"{name}.csv"
    xlsx_path = OUTPUT_DIR / 'tables' / f"{name}.xlsx"
    df.to_csv(csv_path, index=True)
    try:
        df.to_excel(xlsx_path)
    except Exception:
        pass

def partition_coefficient(u):
    N = u.shape[1]
    return np.sum(u**2) / N

def partition_entropy(u, eps=0.000000000001):
    N = u.shape[1]
    u_clipped = np.clip(u, eps, 1.0)
    return - np.sum(u_clipped * np.log(u_clipped)) / N

def xie_beni(X, centers, u, m=2.0):
    N = X.shape[0]
    c = centers.shape[0]
    num = 0.0
    for j in range(c):
        d2 = np.sum((X - centers[j])**2, axis=1)
        num += np.sum((u[j]**m) * d2)
    if c > 1:
        cdists = np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2)
        np.fill_diagonal(cdists, np.inf)
        denom = np.min(cdists)
    else:
        denom = 1.0
    return (num / N) / denom

def fuzzy_silhouette_samples(X, centers, u, m=2.0):
    """
    Retorna s_i (silhueta) por amostra para o caso fuzzy aproximado.
    Método baseado em:
    - calcular a distância média a (intra-cluster) ponderada pelas pertinências elevadas a m
    - calcular b: menor distância média ponderada a clusters não atribuídos
    - s = (b - a) / max(a, b)
    """
    N = X.shape[0]
    c = centers.shape[0]
    um = u**m  # pertinências ^ m
    # distâncias NxC
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    # a_i: média ponderada das distâncias para cada cluster (usando um como peso por cluster)
    # Para cada j (cluster), o "peso" de cluster j é um[j, i] relativo — aqui somamos por j depois
    # Calculamos a_i como: sum_j (um[j,i] * dists[i,j]) / sum_j um[j,i]
    sum_um = np.sum(um, axis=0) #  soma por amostra 
    a = np.sum(um.T * dists, axis=1) / sum_um

    # assigned: cluster com máxima pertinência (hard assignment usado para definir "outros" clusters)
    assigned = np.argmax(u, axis=0)

    b = np.zeros(N)
    for i in range(N):
        # para cada cluster k != assigned[i], calcule média ponderada das distâncias a k
        # mas uma definição prática: b_i = min_k ≠ assigned[i] ( sum_j (um[j,i]*dists[i,j]) / sum_um )
        # no nosso caso usamos dists[i,k] diretamente (porque já temos dists por k)
        # uma alternativa é calcular a distância média ponderada para cada cluster separadamente.
        # Implementamos b_i como o mínimo dists[i,k] ponderado pelo perfil de pertença relativa a k.
        candidates = []
        for k in range(c):
            if k == assigned[i]:
                continue
            # distância média 'a_k(i)' considerando pertinência a cluster k: usar dists[i,k]
            # Para consistência com a definição de a_i (que mistura todos os clusters), 
            # aqui simplificamos para dists[i,k] (seguindo aproximações comuns em FSI).
            candidates.append(dists[i, k])
        b[i] = np.min(candidates) if candidates else a[i]

    # silhouette per-sample
    denom = np.maximum(a, b) #+ 0.000000000001
    s = (b - a) / denom
    return s

def fuzzy_silhouette_index(X, centers, u, m=2.0):
    s = fuzzy_silhouette_samples(X, centers, u, m=m)
    return np.nanmean(s)

# =========== EXECUÇÃO FCM ===========
results_summary = []
for n_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
    print(f"Executando FCM com {n_clusters} clusters...")
    data_for_fcm = X_scaled.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data_for_fcm, c=n_clusters, m=FCM_M,
        error=FCM_ERROR, maxiter=FCM_MAXITER, seed=RANDOM_STATE
    )

    centers_scaled = cntr  # shape (c, n_features)
    centers_orig = scaler.inverse_transform(centers_scaled)
    hard_labels = np.argmax(u, axis=0)  # por linha (amostra)

    # Salva centróides (espaço original)
    centroids_df = pd.DataFrame(centers_orig, columns=num_cols)
    centroids_df.index = [f"cluster_{i+1}" for i in range(n_clusters)]
    save_df(centroids_df, f"centroids_k{n_clusters}")

    # Salva pertinências por linha e por instância (arquivo)
    membership_df = pd.DataFrame(u.T, columns=[f"cluster_{i+1}" for i in range(n_clusters)])
    membership_df['__file__'] = filenames
    membership_df.to_csv(OUTPUT_DIR / 'tables' / f"memberships_rows_k{n_clusters}.csv", index=False)
    inst_membership = membership_df.groupby('__file__').mean()
    inst_membership.to_csv(OUTPUT_DIR / 'tables' / f"memberships_instances_k{n_clusters}.csv")

    # Centròides no espaço PCA
    centers_pca = pca.transform(centers_scaled)
    centroids_pca_df = pd.DataFrame(centers_pca, columns=['pca1', 'pca2'])
    centroids_pca_df.index = centroids_df.index
    centroids_pca_df.to_csv(OUTPUT_DIR / 'tables' / f"centroids_pca_k{n_clusters}.csv")

    # ======= PLOT PCA: pontos coloridos por cluster e instâncias =======
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(all_df['pca1'], all_df['pca2'], c=hard_labels, s=10, cmap='tab10', alpha=0.6)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='X', s=200, edgecolor='k')
    for i, txt in enumerate(centroids_pca_df.index):
        ax.annotate(txt, (centers_pca[i, 0], centers_pca[i, 1]), fontsize=9, fontweight='bold')

    # Anotações melhoradas dos nomes de arquivos (instâncias)
    for idx, (fname, row) in enumerate(file_pca_means.iterrows()):
        ax.scatter(row['pca1'], row['pca2'], marker='o', s=100, facecolors='none', edgecolors='k')
        # deslocamento alternado para evitar sobreposição
        offset_x = 15 * ((-1)**idx)
        offset_y = 10 * ((idx % 3) - 1)
        ax.annotate(fname, (row['pca1'], row['pca2']),
                    textcoords="offset points", xytext=(offset_x, offset_y), ha='center', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
                    arrowprops=dict(arrowstyle='-', lw=0.5, color='gray', alpha=0.5))

    ax.set_title(f"FCM k={n_clusters} — Linhas (pontos), Centr. (X), Instâncias (labels)")
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plots' / f"fcm_k{n_clusters}.png", dpi=300)
    plt.close(fig)

    # ======= PLOT SILHUETA FUZZY =======
    # calcula silhuetas por amostra
    s_values = fuzzy_silhouette_samples(X_scaled, centers_scaled, u, m=FCM_M)
    # média global (FSI aproximado)
    fsi = np.nanmean(s_values)

    # Preparar plot no estilo sklearn: barras empilhadas por cluster
    fig, ax = plt.subplots(figsize=(10, 8))
    y_lower = 10  # padding inicial
    y_tick_pos = []
    y_tick_labels = []
    cmap = plt.get_cmap('tab10')

    # trata clusters possivelmente vazios
    for i in range(n_clusters):
        # indices das amostras atribuídas "hard" ao cluster i
        ith_cluster_indices = np.where(np.argmax(u, axis=0) == i)[0]
        if ith_cluster_indices.size == 0:
            # cluster vazio -> pular (pode acontecer raramente)
            continue
        # valores de silhueta para essas amostras, ordenados
        ith_sil_values = s_values[ith_cluster_indices]
        ith_sil_values_sorted = np.sort(ith_sil_values)
        size_cluster = ith_sil_values_sorted.shape[0]
        y_upper = y_lower + size_cluster

        color = cmap(i % 10)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_sil_values_sorted,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # opção: desenhar linha média da silhueta deste cluster
        cluster_avg = np.mean(ith_sil_values_sorted)
        ax.plot([cluster_avg, cluster_avg], [y_lower, y_upper], linestyle='--', color='k', linewidth=0.8)

        # posição para ticks
        y_tick_pos.append(y_lower + 0.5 * size_cluster)
        y_tick_labels.append(f"cluster_{i+1} (n={size_cluster})")

        y_lower = y_upper + 10  # padding entre clusters

    ax.set_xlabel("Silhueta (fuzzy, aproximada)")
    ax.set_ylabel("Clusters agrupados por rótulo hard")
    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_labels)
    ax.set_title(f"Silhueta Fuzzy (aprox.) — k={n_clusters} — média global = {fsi:.4f}")
    # linha vertical média global
    ax.axvline(x=fsi, color="red", linestyle="--", linewidth=1)
    ax.set_xlim([-0.25, 1])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plots' / f"fcm_silhouette_k{n_clusters}.png", dpi=300)
    plt.close(fig)

    # ======= SALVA ÍNDICES E METADADOS =======
    pc = partition_coefficient(u)
    pe = partition_entropy(u)
    xb = xie_beni(X_scaled, centers_scaled, u, m=FCM_M)
    # fsi já calculado como média de s_values
    results_summary.append({'k': n_clusters, 'PC': pc, 'PE': pe, 'XB': xb, 'FSI_approx': fsi})

# salva sumário
results_df = pd.DataFrame(results_summary).set_index('k')
save_df(results_df, 'indices_summary')
print("Processo finalizado. Resultados em:", OUTPUT_DIR)
