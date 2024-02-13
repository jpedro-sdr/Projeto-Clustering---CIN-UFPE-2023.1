from sklearn.cluster import KMeans

def kMeans_sklearn(X, n_clusters):
    kMeans_sklearn = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kMeans_sklearn.fit_predict(X)
    centroids = kMeans_sklearn.cluster_centers_
    
    return labels, centroids
