import numpy as np
import numpy.linalg as LA
from copy import copy, deepcopy
from util import visualize, gen_gif

def init_centroids(X, n_clusters, init):
    '''
    X: (# data, # features)
    '''
    n_data, n_features = X.shape # n: # data points, p: # features
    centroids = np.zeros((n_clusters, n_features))

    if init == 'kmeans++':
        # 1. Choose one center uniformly at random among the data points.
        centroids[0] = X[np.random.randint(0, n_data),:]
        
        # 4. Repeat Steps 2 and 3 until k centers have been chosen.
        for k in range(1, n_clusters):

            # 2. For each data point x not chosen yet, compute D(x), 
            #    the distance between x and the nearest center that has already been chosen.
            D = np.min(LA.norm(X-centroids[:k][:, np.newaxis], axis=2), axis=0)

            # 3. Choose one new data point at random as a new center, 
            #    using a weighted probability distribution where a point x 
            #    is chosen with probability proportional to D(x)^2.
            prob_distr = (D**2)/(D**2).sum()
            centroids[k] = X[np.random.choice(np.arange(n_data), p=prob_distr),:]
        # Now that the initial centers have been chosen, proceed using standard k-means clustering.

    elif init == 'random':
        X_mean = X.mean(axis=0)
        X_std  = X.std(axis=0)
        centroids = np.random.normal(loc=X_mean, scale=X_std, size=(n_clusters, n_features))
        
    return centroids

def KMeans(X, n_clusters, init, path):
    '''
    X: (# data, # features)
    n_clusters: k
    init: the method to initialize k means
    '''
    max_iter = 100
    tol = 1e-6

    n_data, n_features = X.shape
    labels = np.zeros((n_data,)) # class: ()
    centroids = init_centroids(X, n_clusters, init) # (n_data,n_features)
    frames = []

    for _iter in range(max_iter):
        # E step: Calculate distances and determine the nearest cluster that x belongs to 
        labels_new = np.argmin(LA.norm(X-centroids[:,np.newaxis], axis=2), axis=0)
        frames.append(visualize(labels_new, "{}_{}".format(path, _iter)))
        # M step: update centroids
        centroids_new = np.empty((n_clusters,n_features))
        for k in range(n_clusters):
            cluster_k_members = np.where(labels_new == k)[0]
            centroids_new[k] = X[cluster_k_members,:].sum(axis=0)/len(cluster_k_members)

        if LA.norm(centroids_new-centroids) < tol:
            gen_gif(path, frames)
            break
        labels = labels_new
        centroids = centroids_new

    return labels