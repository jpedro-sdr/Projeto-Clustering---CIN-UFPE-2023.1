import matplotlib.pyplot as plt

def plotSilhouetteGraph(silhouette_kmeans_sklearn, silhouette_kmeans_custom, silhouette_dbscan, silhouette_fuzzy):

    # Lista de métodos de agrupamento
    algorithms = ['kMeans (sklearn)', 'kMeans (customizado)', 'DBSCAN', 'Fuzzy cMeans']

    # Lista de valores de silhueta correspondentes
    silhouette_values = [silhouette_kmeans_sklearn, silhouette_kmeans_custom, silhouette_dbscan, silhouette_fuzzy]

    # Criando o gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, silhouette_values, color='skyblue')
    plt.title('Valores de Silhueta para Cada Agrupamento')
    plt.xlabel('Algoritmo de Agrupamento')
    plt.ylabel('Valor de Silhueta')
    plt.ylim(0, 1)  # Definindo o limite y para melhor visualização

    # Adicionando os valores de silhueta nas barras
    for i, value in enumerate(silhouette_values):
        plt.text(i, value + 0.01, round(value, 3), ha='center', va='bottom')

    # Exibindo o gráfico
    plt.show()
