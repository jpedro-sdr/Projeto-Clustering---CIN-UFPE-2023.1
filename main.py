from Graficos.purityGraph import plotPurityGraph
from Graficos.silhouetteGraph import plotSilhouetteGraph
from Graficos.clusteringGraphs import plotClusteringGraphs
from data_processing import process_data
from Clustering.clustering_analysis import run_kmeans_sklearn, run_kmeans_custom, run_dbscan, run_fuzzy_cmeans

n_clusters = 4

# Processamento dos dados
X_scaled, y_true, y_pred = process_data(n_clusters)

# Execução do kMeans (sklearn)
kmeans_clusters_sklearn, \
silhouette_kmeans_sklearn, \
purity_kmeans_sklearn = run_kmeans_sklearn(X_scaled, y_true, n_clusters)

print("kMeans (sklearn) Result:")
print("Clusters:", kmeans_clusters_sklearn)
print("Silhouette:", silhouette_kmeans_sklearn)
print("Purity:", purity_kmeans_sklearn)

# Execução do kMeans customizado
kmeans_clusters_custom, \
silhouette_kmeans_custom, \
purity_kmeans_custom = run_kmeans_custom(X_scaled, y_true, n_clusters)

print("\nkMeans (customizado) Result:")
print("Clusters:", kmeans_clusters_custom)
print("Silhouette:", silhouette_kmeans_custom)
print("Purity:", purity_kmeans_custom)

# Execução do DBSCAN
dbscan_clusters, \
silhouette_dbscan, \
purity_dbscan = run_dbscan(X_scaled, y_true, eps=0.5, min_samples=5)

print("\nDBSCAN Result:")
print("Clusters:", dbscan_clusters)
print("Silhouette:", silhouette_dbscan)
print("Purity:", purity_dbscan)

# Execução do Fuzzy cMeans
fuzzy_clusters, \
silhouette_fuzzy, \
purity_fuzzy = run_fuzzy_cmeans(X_scaled, y_true, n_clusters, m=2)

print("\nFuzzy cMeans Result:")
print("Clusters:", fuzzy_clusters)
print("Silhouette:", silhouette_fuzzy)
print("Purity:", purity_fuzzy)

# Visualizando resultados graficamente
plotClusteringGraphs(X_scaled,kmeans_clusters_sklearn,kmeans_clusters_custom,dbscan_clusters,fuzzy_clusters)
plotSilhouetteGraph(silhouette_kmeans_sklearn, silhouette_kmeans_custom, silhouette_dbscan, silhouette_fuzzy)
plotPurityGraph(purity_kmeans_sklearn, purity_kmeans_custom, purity_dbscan, purity_fuzzy)
