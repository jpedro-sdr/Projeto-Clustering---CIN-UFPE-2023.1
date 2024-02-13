import numpy as np

def kMeans_custom(X, n_clusters, max_iters=100):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    for _ in range(max_iters):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return labels, centroids
