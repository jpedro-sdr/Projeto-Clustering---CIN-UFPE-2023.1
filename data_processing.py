from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from ucimlrepo import fetch_ucirepo
from yellowbrick.cluster import KElbowVisualizer

def process_data(n_clusters):

    # Carrega o conjunto de dados
    wholesale_customers = fetch_ucirepo(id=292)
    X = wholesale_customers.data.features
    y_true = wholesale_customers.data.targets

    # Redução de Dimensionalidade
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)

    # Aplicando o KMeans
    model = KMeans(n_clusters, random_state=42)
    y_pred = model.fit_predict(X_scaled)

    # Aplicando o KMeans
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,10))

    visualizer.fit(X_scaled)  
    visualizer.poof()    
    return X_scaled, y_true, y_pred

