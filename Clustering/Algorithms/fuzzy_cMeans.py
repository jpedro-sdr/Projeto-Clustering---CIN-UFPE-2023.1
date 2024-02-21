import numpy as np
import skfuzzy as fuzz

def apply_fuzzy_cMeans(X, n_clusters, m):
    X_transposed = np.transpose(X)

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_transposed, n_clusters, m, error=0.005, maxiter=1000, init=None)

    clusters = np.argmax(u, axis=0)

    return clusters
