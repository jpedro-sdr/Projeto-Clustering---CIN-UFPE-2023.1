import numpy as np
import pandas as pd
from kMeans import kMeans_sklearn, kMeans_custom, elbow_method
from DBSCAN import apply_DBSCAN
from fuzzy_cMeans import apply_fuzzy_cMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn import metrics

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# Carregando os dados
wholesale_customers = fetch_ucirepo(id=292) 
X = wholesale_customers.data.features 

# Selecionando as colunas
selected_columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X_selected = X[selected_columns]

print(X_selected)

original_index = X_selected.index

# Redução de Dimensionalidade
pca = PCA(n_components=2)
X_selected = pca.fit_transform(X_selected)

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Aplicando o KMeans
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))

visualizer.fit(X_scaled)  
visualizer.poof()    

n_clusters = 4

kmeans_clusters_sklearn = kMeans_sklearn(X_scaled, n_clusters=4)
print(kmeans_clusters_sklearn)

silhouette_kmeans_sklearn = silhouette_score(X_scaled, kmeans_clusters_sklearn)
print(silhouette_kmeans_sklearn)

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

purity_kmeans_clusters_sklearn = purity_score(original_index, kmeans_clusters_sklearn)
print('Pureza para o cluster acima = ', purity_kmeans_clusters_sklearn)

# # Aplicando kMeans customizado
kmeans_clusters_custom = kMeans_custom(X_scaled, n_clusters=n_clusters)
print(kmeans_clusters_custom)

silhouette_kmeans_custom = silhouette_score(X_scaled, kmeans_clusters_custom)
print(silhouette_kmeans_custom)

purity_kmeans_clusters_custom = purity_score(original_index, kmeans_clusters_custom)
print('Pureza para o cluster acima = ', purity_kmeans_clusters_custom)

# # Aplicando DBSCAN
dbscan_clusters = apply_DBSCAN(X_scaled, eps=0.5, min_samples=5)
print(dbscan_clusters)

silhouette_dbscan = silhouette_score(X_scaled, dbscan_clusters)
print(silhouette_dbscan)

purity_dbscan_clusters = purity_score(original_index, dbscan_clusters)
print('Pureza para o cluster acima = ', purity_dbscan_clusters)

# # Aplicando Fuzzy cMeans
fuzzy_clusters = apply_fuzzy_cMeans(X_scaled, n_clusters=n_clusters, m=2)
print(fuzzy_clusters)

silhouette_fuzzy = silhouette_score(X_scaled, fuzzy_clusters)
print(silhouette_fuzzy)

purity_fuzzy_clusters = purity_score(original_index, fuzzy_clusters)
print('Pureza para o cluster acima = ', purity_fuzzy_clusters)


# Visualizando resultados graficamente

# Plotando kMeans usando sklearn
plt.figure(figsize=(8, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters_sklearn, cmap='viridis')
plt.title('kMeans (sklearn)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plotando kMeans customizado
plt.figure(figsize=(8, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters_custom, cmap='viridis')
plt.title('kMeans (customizado)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plotando DBSCAN
plt.figure(figsize=(8, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_clusters, cmap='viridis')
plt.title('DBSCAN')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plotando Fuzzy cMeans
plt.figure(figsize=(8, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=fuzzy_clusters, cmap='viridis')
plt.title('Fuzzy cMeans')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Lista de métodos de agrupamento
algorithms = ['kMeans (sklearn)', 'kMeans (customizado)', 'DBSCAN', 'Fuzzy cMeans']

# Lista de valores de silhueta correspondentes
silhouette_values = [silhouette_kmeans_sklearn, silhouette_kmeans_custom, silhouette_dbscan, silhouette_fuzzy]

# Criando o gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(algorithms, silhouette_values, color='skyblue')
plt.title('Valores de Silhueta para Cada Agrupamento')
plt.xlabel('Algoritmo de Agrupamento')
plt.ylabel('Valor de Silhueta')
plt.ylim(0, 1)  # Definindo o limite y para melhor visualização

# Adicionando os valores de silhueta nas barras
for i, value in enumerate(silhouette_values):
    plt.text(i, value + 0.01, round(value, 3), ha='center', va='bottom')

# Exibindo o gráfico
plt.show()

# Lista de métodos de agrupamento
algorithms = ['kMeans (sklearn)', 'kMeans (customizado)', 'DBSCAN', 'Fuzzy cMeans']

purity_values = [purity_kmeans_clusters_sklearn, purity_kmeans_clusters_custom, purity_dbscan_clusters, purity_fuzzy_clusters]

plt.figure(figsize=(10, 6))
plt.bar(algorithms, purity_values, color='orange')
plt.title('Valores de Pureza para Cada Agrupamento')
plt.xlabel('Algoritmo de Agrupamento')
plt.ylabel('Valor de Pureza')
plt.ylim(0, 0.01)  # Definindo o limite y para melhor visualização

for i, value in enumerate(purity_values):
    plt.text(i, value + 0.01, round(value, 3), ha='center', va='bottom')

plt.show()