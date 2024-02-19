
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from ucimlrepo import fetch_ucirepo

def process_data(id=292):
    """
    Processa os dados do conjunto de dados de clientes atacadistas.

    Parâmetros:
    - id (int): O ID do conjunto de dados a ser carregado. Padrão é 292.

    Retorna:
    - original_index (pd.Index): O índice original dos dados.
    - X_scaled (np.ndarray): Os dados normalizados e reduzidos em dimensionalidade.
    """
        
    wholesale_customers = fetch_ucirepo(id=id) 
    X = wholesale_customers.data.features 

    # Selecionando as colunas
    selected_columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    X_selected = X[selected_columns]

    original_index = X_selected.index

    # Redução de Dimensionalidade
    pca = PCA(n_components=2)
    X_selected_pca = pca.fit_transform(X_selected)

    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected_pca)

    # Aplicando o KMeans
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,10))

    visualizer.fit(X_scaled)  
    visualizer.poof()    

    return original_index, X_scaled
