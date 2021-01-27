import numpy as np
import numpy.linalg as LA

def PCA(X, M):
    # Compute the sample mean and translate the data, 
    # so that it’s centered around origin.
    X_centered = X - X.mean(axis=0) # (N,P)
    # Compute the covariance matrix
    S = np.cov(X_centered.T) # (P,P)
    # Eigendecomposition
    eigenvalues, eigenvectors = LA.eigh(S)
    # Principal components are first m largest eigenvectors
    PCs = np.flip(eigenvectors[:, -M:], axis=1) # (P,M)
    # Project the centered data on to the space spanned by PCs
    X_proj = np.matmul(X_centered, PCs) # (N,P)

    return X_proj, PCs, X.mean(axis=0)

def kernelPCA(X, M, kernel):
    K = kernel(X, X)
    N = K.shape[0]
    oneN = np.ones((N,N))/N
    K_centered = K - np.matmul(oneN, K) - np.matmul(K, oneN) + \
                 np.matmul(oneN, np.matmul(K, oneN)) 
    eigenvalues, eigenvectors = LA.eigh(K_centered/N)
    PCs = np.flip(eigenvectors[:, -M:], axis=1)
    K_proj = np.matmul(K_centered, PCs)

    return K_proj, PCs, K.mean(axis=0)