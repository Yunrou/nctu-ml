import os
import numpy as np

from util import loadData, show_eigenfaces, show_reconstructions
from pca import PCA
from lda import LDA

train_database = './Yale_Face_Database/Training'
test_database = './Yale_Face_Database/Testing'

def main():
    
    # Load data
    X_train, y_train = loadData(train_database)
    X_test, y_test   = loadData(test_database)

    # PCA
    X_proj, PCs, X_mean = PCA(X_train, 25)
    show_eigenfaces(PCs, 25)
    show_reconstructions(X_train, np.matmul(X_proj, PCs.T)+X_mean, 10)
    # LDA
    X_proj, W = LDA(X_train-X_train.mean(axis=0), y_train, 25)
    show_eigenfaces(W, 25)
    show_reconstructions(X_train, np.matmul(X_proj, W.T)+X_train.mean(axis=0), 10)

if __name__ == '__main__':
    main()