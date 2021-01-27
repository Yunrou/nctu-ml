import numpy as np
import numpy.linalg as LA
from util import visualize, gen_gif

def initR(G, n_clusters, init):
    
    n_data = G.shape[0]
    R = np.zeros((n_data, n_clusters))
    if init == 'kmeans++':

        data_indices = np.arange(n_data) # record remaining data indices
        centroids = np.zeros((n_clusters,), dtype=np.ulonglong) # record data indices that are chosen to be centroids

        # compute distances from phi_i to phi_j
        all_D = G.diagonal()[:,np.newaxis] -2*G + G.diagonal()[np.newaxis,:]
        # 1. Choose one center uniformly at random among the data points.
        centroids[0] = np.random.randint(0, n_data)
        
        # 4. Repeat Steps 2 and 3 until k centers have been chosen.
        for k in range(1, n_clusters):

            # 2. For each data point x not chosen yet, compute D(x), 
            #    the distance between x and the nearest center that has already been chosen.
            D = np.min(all_D[centroids[:k],:], axis=0)

            # 3. Choose one new data point at random as a new center, 
            #    using a weighted probability distribution where a point x 
            #    is chosen with probability proportional to D(x)^2.
            prob_distr = (D**2)/(D**2).sum()
            centroids[k] = np.random.choice(np.arange(n_data), p=prob_distr)

        R[np.arange(n_data), np.argmin(all_D[centroids,:], axis=0)] = 1

    elif init == 'random':
        R[np.arange(n_data), np.random.randint(n_clusters,size=n_data)] = 1

    return R

def kernel_trick(G, R):
    '''
    Parameters
    ----------
    G: Gram matrix (10000,10000)
    R: label matrix (10000, n_clusters) 
       Rik = 1 if xi belongs to kth cluster

    Returns
    -------
    D: Distance matrix (10000, n_clusters)
    '''
    n_data = G.shape[0]
    n_clusters = R.shape[1]
    
    Rk = R.sum(axis=0)
    
    D = np.matmul(G*np.eye(n_data), np.ones((n_data, n_clusters))) - 2*(np.matmul(G, R)/Rk) + \
        (np.matmul(np.ones((n_data, n_clusters)), np.matmul(np.matmul(R.T, G), R)*np.eye(n_clusters)))/(Rk**2)

    return D
    
    
def KernelKMeans(G, n_clusters, init, path):
    '''
    Parameters
    ----------
    G: Gram matrix (10000,10000)
    n_clusters
    init: the method to initialize k means

    Returns
    -------
    ndarray, shape=(10000,), value=kth cluster
    '''
    max_iter = 100
    tol = 1e-6

    n_data = G.shape[0] 
    D = np.full((n_data, n_clusters), np.inf)
    R = initR(G, n_clusters, init)
    frames = []

    for _iter in range(max_iter):
        frames.append(visualize(R.argmax(axis=1), "{}_{}".format(path, _iter)))
        # E step: calculate distance b/w points and centroids using kernel trick
        D_new = kernel_trick(G, R)
        # M step: update R (Class matrix), reassign each points to its nearest centroid
        R_new = np.zeros(R.shape)
        R_new[np.arange(n_data), np.argmin(D_new, axis=1)] = 1
        
        # Converge check
        if LA.norm(D_new-D) < tol: 
            frames.append(visualize(R.argmax(axis=1), "{}_{}".format(path, _iter+1)))
            gen_gif(path, frames)
            break
        D = D_new
        R = R_new

    return R.argmax(axis=1)