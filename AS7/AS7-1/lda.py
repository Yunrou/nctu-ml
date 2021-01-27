import numpy as np
import numpy.linalg as LA

def LDA(X, y, D_):

    N, P = X.shape # N: n_data, P: n_dim
    K = len(np.unique(y)) # K: n_class
    
    # To indicate each data belongs to which label
    R = np.zeros((N,K)) # (N,K)
    R[np.arange(N), y] = 1
    Nk = R.sum(axis=0) # (K,)
    # Calculate mean of each class and total class
    mk = np.matmul(R.T, X) / Nk[:,np.newaxis] # (K,P)
    m  = X.mean(axis=0) # shape=(P,)

    # Calculate SW
    xn_mk = X - np.matmul(R, mk) # (N,P)
    SW = np.matmul(xn_mk.T, xn_mk) # (P,P)

    # Calculate SB
    mk_m = mk - m # (K,P)
    SB = np.matmul(Nk * mk_m.T, mk_m) # (P,P)

    # Optimizw w by eigendecomposition of SW^{-1}SB
    S = np.matmul(LA.pinv(SW), SB)
    eigenvalues, eigenvectors = LA.eigh(S)

    W = np.flip(eigenvectors[:, -D_:], axis=1) # (P,D_)

    X_proj = np.matmul(X, W) # (N,D_)

    return X_proj, W

def kernelLDA(X, y, D_, kernel):
    N, P = X.shape # N: n_data, P: n_dim
    K = len(np.unique(y)) # K: n_class
    
    Kernel = kernel(X, X)

    # To indicate each data belongs to which label
    R = np.zeros((N,K)) # (N,K)
    R[np.arange(N), y] = 1
    Nk = R.sum(axis=0) # (K,)

    # Calsulate mean of each class and total class
    mk = np.matmul(R.T, Kernel) / Nk[:,np.newaxis] # (K,P)
    m  = Kernel.mean(axis=0) # shape=(P,)

    # Calculate SW
    xn_mk = Kernel - np.matmul(R, mk) # (N,P)
    SW = np.matmul(xn_mk.T, xn_mk) # (P,P)

    # Calculate SB
    mk_m = mk - m # (K,P)
    SB = np.matmul(Nk * mk_m.T, mk_m) # (P,P)

    # Optimizw w by eigendecomposition of SW^{-1}SB
    S = np.matmul(LA.pinv(SW),SB)
    eigenvalues, eigenvectors = LA.eigh(S)

    W = np.flip(eigenvectors[:, -D_:], axis=1) # (P,D_)

    Kernel_proj = np.matmul(Kernel, W) # (N,D_)

    return Kernel_proj, W