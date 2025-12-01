# 📘 Análise de Similaridade Climática entre Marte e Regiões da Terra

### *Algoritmos K-Means e Fuzzy C-Means*

Este repositório reúne os arquivos, códigos e dados utilizados no estudo que investiga a similaridade climática entre Marte e diferentes regiões do planeta Terra, aplicando técnicas de aprendizado não supervisionado.

---

## 📄 Artigo Científico

O artigo no formato SBC encontra-se em:

**`Análise de Similaridade Climática entre Marte e Diferentes Regiões do Planeta Terra por Meio dos Algoritmos K-Means e Fuzzy C-Means.pdf`**


---

## 📂 Dataset

A pasta **`Dataset`** contém o conjunto de dados utilizados nos experimentos.
Cada arquivo CSV representa uma localidade da Terra e inclui variáveis climáticas equivalentes às disponíveis para Marte.

🔹 **Importante:**
Após baixar o diretório `Dataset`, os arquivos `.py` devem estar **no mesmo diretório** que a pasta `Dataset`
(**não** coloque os arquivos Python *dentro* da pasta `Dataset`).

---

## 📊 Resultados dos Agrupamentos

O repositório inclui três pastas com as saídas gráficas dos experimentos:

* **`SaidaKmeans/`** → gráficos e resultados do algoritmo **K-means**
* **`SaidaFCM15/`** → resultados do **Fuzzy C-means** com fator de ponderação **m = 1.5**
* **`SaidaFCM20/`** → resultados do **Fuzzy C-means** com fator de ponderação **m = 2.0**

Essas pastas contêm gráficos de clusterização e demais artefatos gerados nos experimentos.

---

## 🧠 Códigos Fonte

### 🔸 `Elbown.py`

Implementação do **método do cotovelo (Elbow Method)**
Utilizado para identificar o melhor número de clusters **k** no K-means.

### 🔸 `fuzzy_cmeans.py`

Código do algoritmo **Fuzzy C-means**, utilizando a biblioteca **scikit-fuzzy (skfuzzy)**.
Inclui cálculo das seguintes métricas:

* Partition Coefficient (PC)
* Partition Entropy (PE)
* Xie–Beni Index (XB)
* **Coeficiente de silhueta fuzzy**

### 🔸 `k-means.py`

Implementação do **K-means**, incluindo cálculo do **coeficiente de silhueta** para avaliação dos agrupamentos.

---

## ▶️ Como Executar

1. Baixe ou clone o repositório.
2. Garanta que o diretório contenha:

   ```
   .
   ├── Dataset/
   ├── Elbown.py
   ├── fuzzy_cmeans.py
   ├── k-means.py
   ```
3. Instale as dependências (exemplo):

   ```bash
   pip install -r requirements.txt
   ```
4. Execute cada script conforme desejado para gerar novamente os resultados.

