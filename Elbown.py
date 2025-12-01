import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Caminho da pasta com os CSVs
BASE_DIR = Path(__file__).resolve().parent 
DATA_DIR = BASE_DIR / "Dataset"


# Leitura de todos os CSVs
arquivos = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

dados_lista = []
origem_lista = []

for arq in arquivos:
    caminho = os.path.join(DATA_DIR, arq)
    try:
        df = pd.read_csv(caminho)
        df = df.select_dtypes(include=[np.number]).dropna()
        if not df.empty:
            dados_lista.append(df)
            origem_lista.extend([arq] * len(df))
    except Exception as e:
        print(f"Erro ao ler {arq}: {e}")

# Concatena tudo em um único DataFrame
dados = pd.concat(dados_lista, ignore_index=True)
origens = pd.Series(origem_lista, name="Arquivo")

# Escalonamento dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(dados)

# ==============================
# 1️⃣ Método do Cotovelo
# ==============================
inertias = []
K_range = range(2, 19)  # testa de 2 até 18 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Gráfico do método do cotovelo
plt.figure(figsize=(8,6))
plt.plot(K_range, inertias, 'o-', color='blue')
plt.title("Método do Cotovelo (Elbow Method)")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inércia (Soma dos Quadrados)")
plt.grid(True)
plt.show()

# ==============================
# 2️⃣ Escolha do melhor K (cotovelo)
# ==============================
reducoes = np.diff(inertias) / inertias[:-1]
k_otimo = K_range[np.argmin(np.abs(reducoes < -0.1))] if any(reducoes < -0.1) else 3

print(f"\nNúmero ótimo de clusters sugerido pelo método do cotovelo: k = {k_otimo}")