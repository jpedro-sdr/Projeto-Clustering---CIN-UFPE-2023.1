from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def elbow_method(X, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        pipeline = make_pipeline(StandardScaler(), kmeans)
        pipeline.fit(X)
        distortions.append(kmeans.inertia_)  # Inertia is the sum of squared distances to the nearest cluster center

    # Plotando o método Elbow
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

