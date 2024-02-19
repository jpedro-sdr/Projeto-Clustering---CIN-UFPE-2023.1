import matplotlib.pyplot as plt

def plotClusteringGraphs(X_scaled,kmeans_clusters_sklearn,kmeans_clusters_custom,dbscan_clusters,fuzzy_clusters):
    # Plotando kMeans usando sklearn
    plt.figure(figsize=(8, 8))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters_sklearn, cmap='viridis')
    plt.title('kMeans (sklearn)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Plotando kMeans customizado
    plt.figure(figsize=(8, 8))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters_custom, cmap='viridis')
    plt.title('kMeans (customizado)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Plotando DBSCAN
    plt.figure(figsize=(8, 8))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_clusters, cmap='viridis')
    plt.title('DBSCAN')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Plotando Fuzzy cMeans
    plt.figure(figsize=(8, 8))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=fuzzy_clusters, cmap='viridis')
    plt.title('Fuzzy cMeans')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()