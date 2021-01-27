import numpy as np
import numpy.linalg as LA
from kmeans import KMeans
from util import plot_eigenspace

def SpectralClustering(S, n_clusters, method, init, path):
    '''
    S: Similarity matrix (n,n)
    n_cluster: k
    '''
    # 1. Construct a similarity graph, Let W be its weighted adjacency matrix
    W = S
    # 2. Compute the unnormalized Laplacian
    # Degree matrix
    D = W.sum(axis=1) * np.eye(W.shape[0])
    # Unnormalized Graph Laplacian
    L = D - W
    # 3. Compute the first k eigenvectors u1~uk of L, and
    #    let U = (n,k) be the matrix containing the vectors u1~uk as columns
    # 4. For i = 1~n, let yi = (k,) be the vector corresponding to the i-th row of U
    U = np.zeros((10000,3))

    if method == 'normalized':
        D_inv_sqrt = np.diag(1/np.diag(np.sqrt(D)))
        Lsym = D_inv_sqrt @ L @ D_inv_sqrt
        eigenvalues, eigenvectors = LA.eigh(Lsym)

        for i, ev in enumerate(eigenvalues):
            if ev < 1e-10: continue
            U = eigenvectors[:,np.arange(i, i+n_clusters)]
            break
        # 5. Form the matrix T = (n.k) from U by normalizing the rows to norm 1
        T = U/np.sqrt(np.sum(np.square(U),axis=1)).reshape(-1,1)
        # 6. Cluster the points (yi)i=1~n in (k,) with the kmeans algorithm into clusters C1~Ck
        labels = KMeans(T, n_clusters, init, path)
        plot_eigenspace(T, labels, "{}_{}_{}_{}.png".format(path.split('/')[1], method, n_clusters, init))
    else:
        eigenvalues, eigenvectors = LA.eigh(L)
        for i, ev in enumerate(eigenvalues):
            if ev < 1e-10: continue
            U = eigenvectors[:,np.arange(i, i+n_clusters)]
            break 
        # 5. Cluster the points (yi)i=1~n in (k,) with the kmeans algorithm into clusters C1~Ck
        labels = KMeans(U, n_clusters, init, path)
        plot_eigenspace(U, labels, "{}_{}_{}_{}.png".format(path.split('/')[1], method, n_clusters, init))

    return labels