from sklearn import metrics
import numpy as np

def purity_score(y_true, y_pred):
    # Matriz de contingência
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
