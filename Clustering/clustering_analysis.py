from sklearn.metrics import silhouette_score
from Clustering.clustering_utils import kMeans_sklearn, kMeans_custom, apply_DBSCAN, apply_fuzzy_cMeans, purity_score

def run_kmeans_sklearn(X_scaled, y_true, n_clusters):

    kmeans_clusters = kMeans_sklearn(X_scaled, n_clusters=n_clusters)
    silhouette = silhouette_score(X_scaled, kmeans_clusters)
    purity = purity_score(y_true, kmeans_clusters)
    return kmeans_clusters, silhouette, purity

def run_kmeans_custom(X_scaled, y_true, n_clusters):
    kmeans_clusters = kMeans_custom(X_scaled, n_clusters)
    silhouette = silhouette_score(X_scaled, kmeans_clusters)
    purity = purity_score(y_true, kmeans_clusters)
    return kmeans_clusters, silhouette, purity

def run_dbscan(X_scaled , y_true, eps=0.5, min_samples=5):
    dbscan_clusters = apply_DBSCAN(X_scaled, eps=eps, min_samples=min_samples)
    silhouette = silhouette_score(X_scaled, dbscan_clusters)
    purity = purity_score(y_true, dbscan_clusters)
    return dbscan_clusters, silhouette, purity

def run_fuzzy_cmeans(X_scaled , y_true, n_clusters, m=2):
    fuzzy_clusters = apply_fuzzy_cMeans(X_scaled, n_clusters, m=m)
    silhouette = silhouette_score(X_scaled, fuzzy_clusters)
    purity = purity_score(y_true, fuzzy_clusters)
    return fuzzy_clusters, silhouette, purity
