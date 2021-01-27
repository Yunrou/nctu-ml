import os
import numpy as np

from util import loadData, KNN, performance
from pca import PCA
from lda import LDA

train_database = '../Yale_Face_Database/Training'
test_database = '../Yale_Face_Database/Testing'

def main():
    
    # Load data
    X_train, y_train = loadData(train_database)
    X_test, y_test   = loadData(test_database)

    # PCA
    X_train_proj, PCs, X_train_mean = PCA(X_train, 25)
    X_test_proj = np.matmul(X_test - X_train_mean, PCs)
    prediction = KNN(X_train_proj, y_train, X_test_proj, k=5)
    acc = performance(y_test, prediction)
    print("ACC of PCA = {:.2f}%".format(acc*100))

    # LDA
    X_train_proj, W = LDA(X_train, y_train, 25)
    X_test_proj = np.matmul(X_test, W)
    prediction = KNN(X_train_proj, y_train, X_test_proj, k=5)
    acc = performance(y_test, prediction)
    print("ACC of LDA = {:.2f}%".format(acc*100))

if __name__ == '__main__':
    main()