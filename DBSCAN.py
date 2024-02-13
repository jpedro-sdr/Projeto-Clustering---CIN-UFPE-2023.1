from sklearn.cluster import DBSCAN

def dbscan(X, eps=0.3, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    return labels
