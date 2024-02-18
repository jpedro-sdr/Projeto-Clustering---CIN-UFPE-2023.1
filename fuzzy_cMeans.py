from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def apply_fuzzy_cMeans(X, n_clusters, m):
    # Substitua pelo seu código específico para fuzzy cMeans
    fuzzy_cmeans = KMedoids(n_clusters=n_clusters, metric='manhattan', method='alternate', random_state=42)
    clusters = fuzzy_cmeans.fit_predict(X)
    
    silhouette_avg = silhouette_score(X, clusters)
    return clusters


def elbow_method(X, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        # Substitua pelo seu código específico para fuzzy cMeans
        fuzzy_cmeans = KMedoids(n_clusters=i, metric='manhattan', method='alternate', random_state=42)
        fuzzy_cmeans.fit(X)
        # Adicione a inércia (soma dos quadrados das distâncias dos pontos para o centróide mais próximo)
        distortions.append(fuzzy_cmeans.inertia_)
    return distortions
