import os
import numpy as np

from util import loadData, KNN, performance
from util import linearKernel, rbfKernel, linear_plus_rbfKernel
from pca import kernelPCA
from lda import kernelLDA

train_database = '../Yale_Face_Database/Training'
test_database = '../Yale_Face_Database/Testing'

def main():
    
    # Load data
    X_train, y_train = loadData(train_database)
    X_test, y_test   = loadData(test_database)

    # Test kernel
    K_test = rbfKernel(X_test, X_train)

    # PCA
    K_train_proj, PCs, K_train_mean = kernelPCA(X_train, 25, rbfKernel)
    # Project test kernel into principal components of kernel PCA
    K_test_proj = np.matmul(K_test-K_train_mean, PCs)

    prediction = KNN(K_train_proj, y_train, K_test_proj, k=7)
    acc = performance(y_test, prediction)
    print("ACC of kernel PCA = {:.2f}%".format(acc*100))

    # LDA
    K_train_proj, W = kernelLDA(X_train, y_train, 25, rbfKernel)
    # Project test kernel by W
    K_test_proj = np.matmul(K_test, W)
    prediction = KNN(K_train_proj, y_train, K_test_proj, k=7)
    acc = performance(y_test, prediction)
    print("ACC of kernel LDA = {:.2f}%".format(acc*100))

if __name__ == '__main__':
    main()