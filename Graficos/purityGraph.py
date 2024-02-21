import matplotlib.pyplot as plt

def plotPurityGraph(purity_kmeans_clusters_sklearn, purity_kmeans_clusters_custom, purity_dbscan_clusters, purity_fuzzy_clusters):

    algorithms = ['kMeans (sklearn)', 'kMeans (customizado)', 'DBSCAN', 'Fuzzy cMeans']

    purity_values = [purity_kmeans_clusters_sklearn, purity_kmeans_clusters_custom, purity_dbscan_clusters, purity_fuzzy_clusters]

    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, purity_values, color='orange')
    plt.title('Valores de Pureza para Cada Agrupamento')
    plt.xlabel('Algoritmo de Agrupamento')
    plt.ylabel('Valor de Pureza')
    plt.ylim(0, 1) 

    for i, value in enumerate(purity_values):
        plt.text(i, value + 0.01, round(value, 3), ha='center', va='bottom')

    plt.show()