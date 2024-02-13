import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

# Função de implementação do kMeans sem usar sklearn
def kMeans_custom(X, n_clusters, max_iters=100, tol=1e-4):
    # Inicialização aleatória dos centróides
    centroids = X[np.random.choice(X.shape[0], size=n_clusters, replace=False)]
    
    for _ in range(max_iters):
        # Atribuição de clusters
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Atualização dos centróides
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
        
        # Verificação da convergência
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
    
    return labels

# Fetch do conjunto de dados
breast_cancer_data = fetch_ucirepo(id=17)

# Extrair recursos e alvos
X = breast_cancer_data.data.features
y = breast_cancer_data.data.targets

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA para redução de dimensionalidade (opcional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Aplicar kMeans usando implementação personalizada
kmeans_custom_labels = kMeans_custom(X_scaled, n_clusters=2)

# Aplicar kMeans usando sklearn
kmeans_sklearn = KMeans(n_clusters=2, random_state=42)
kmeans_sklearn_labels = kmeans_sklearn.fit_predict(X_scaled)

# Aplicar DBSCAN usando sklearn
dbscan = DBSCAN(eps=1, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Imprimir resultados
print("Resultados kMeans (implementação personalizada):", kmeans_custom_labels)
print("Resultados kMeans (sklearn):", kmeans_sklearn_labels)
print("Resultados DBSCAN:", dbscan_labels)
