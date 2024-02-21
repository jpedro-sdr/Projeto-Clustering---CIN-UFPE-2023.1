from sklearn.cluster import DBSCAN

def apply_DBSCAN(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)
    return clusters
