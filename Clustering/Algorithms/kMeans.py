import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def kMeans_sklearn(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    kmeans.cluster_centers_
    return clusters

def kMeans_custom(X, n_clusters, n_iterations=100):
    centroids = X[np.random.choice(len(X), n_clusters, replace=False)]
    
    for _ in range(n_iterations):
        # Atribuição dos pontos aos clusters
        clusters = pairwise_distances_argmin_min(X, centroids, axis=1)[0]
        
        # Atualização dos centróides
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(n_clusters)])
        
        # Verifica convergência
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return clusters

def elbow_method(X, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        pipeline = make_pipeline(StandardScaler(), kmeans)
        pipeline.fit(X)
        distortions.append(kmeans.inertia_)

    # Plotando o método Elbow
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()