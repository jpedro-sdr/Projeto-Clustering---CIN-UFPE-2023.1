import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ucimlrepo import fetch_ucirepo
from Clustering.clustering_utils import kMeans_sklearn, kMeans_custom, apply_DBSCAN, apply_fuzzy_cMeans, purity_score

def run_kmeans_sklearn(X_scaled, original_index, n_clusters=4):
    kmeans_clusters = kMeans_sklearn(X_scaled, n_clusters=n_clusters)
    silhouette = silhouette_score(X_scaled, kmeans_clusters)
    purity = purity_score(original_index, kmeans_clusters)
    return kmeans_clusters, silhouette, purity

def run_kmeans_custom(X_scaled, original_index, n_clusters=4):
    kmeans_clusters = kMeans_custom(X_scaled, n_clusters=n_clusters)
    silhouette = silhouette_score(X_scaled, kmeans_clusters)
    purity = purity_score(original_index, kmeans_clusters)
    return kmeans_clusters, silhouette, purity

def run_dbscan(X_scaled, original_index, eps=0.5, min_samples=5):
    dbscan_clusters = apply_DBSCAN(X_scaled, eps=eps, min_samples=min_samples)
    silhouette = silhouette_score(X_scaled, dbscan_clusters)
    purity = purity_score(original_index, dbscan_clusters)
    return dbscan_clusters, silhouette, purity

def run_fuzzy_cmeans(X_scaled, original_index, n_clusters=4, m=2):
    fuzzy_clusters = apply_fuzzy_cMeans(X_scaled, n_clusters=n_clusters, m=m)
    silhouette = silhouette_score(X_scaled, fuzzy_clusters)
    purity = purity_score(original_index, fuzzy_clusters)
    return fuzzy_clusters, silhouette, purity