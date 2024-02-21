import matplotlib.pyplot as plt

def plotClusteringGraphs(X_scaled, kmeans_clusters_sklearn, kmeans_clusters_custom, dbscan_clusters, fuzzy_clusters):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plotando kMeans usando sklearn
    axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters_sklearn, cmap='viridis')
    axes[0, 0].set_title('kMeans (sklearn)')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Fresh')

    # Plotando kMeans customizado
    axes[0, 1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters_custom, cmap='viridis')
    axes[0, 1].set_title('kMeans (customizado)')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Fresh')

    # Plotando DBSCAN
    axes[1, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_clusters, cmap='viridis')
    axes[1, 0].set_title('DBSCAN')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Fresh')

    # Plotando Fuzzy cMeans
    axes[1, 1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=fuzzy_clusters, cmap='viridis')
    axes[1, 1].set_title('Fuzzy cMeans')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Fresh')

    plt.tight_layout()
    plt.show()

