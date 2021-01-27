import os
import numpy as np
import numpy.linalg as LA
from scipy.spatial.distance import cdist
from PIL import Image
import matplotlib.pyplot as plt

H, W = 77, 65

def linearKernel(X, X_):
    return np.matmul(X, X_.T)

def rbfKernel(X, X_, gamma=1e-4):
    return np.exp(-gamma*cdist(X, X_,'euclidean'))

def linear_plus_rbfKernel(X, X_, gamma=1e-4):
    return np.matmul(X, X_.T)+np.exp(-gamma*cdist(X, X_,'euclidean'))

def loadData(_dir:str):
    image_files = os.listdir(_dir)
    N = len(image_files)
    data = np.zeros((N, H*W))
    labels = np.zeros((N,), dtype=np.uint8)

    for i, file in enumerate(image_files):
        img = Image.open(os.path.join(_dir, file)).resize((W,H), Image.ANTIALIAS)
        data[i,:] = np.asarray(img.getdata()).flatten() # H*W
        labels[i] = int(file.split('.')[0][7:9])-1

    return data, labels

def show_eigenfaces(X, n_images):
    n = int(np.sqrt(n_images))
    fig, axes = plt.subplots(n, n, figsize=(8, 8))

    for i in range(n_images):
        ax = axes[int(i/n),int(i%n)]
        ax.imshow(X[:,i].reshape(H,W), cmap='gray')
        ax.axis('off')

    fig.patch.set_visible(False)
    fig.tight_layout()
    plt.show()

def show_reconstructions(X, X_recover, n_faces):
    fig, axes = plt.subplots(2, n_faces, figsize=(10, 4))

    idx = np.random.choice(X_recover.shape[0], n_faces)
    for i, face_idx in enumerate(idx):
        ax1 = axes[0,int(i%n_faces)]
        ax2 = axes[1,int(i%n_faces)]
        ax1.imshow(X[face_idx,:].reshape(H,W), cmap='gray')
        ax2.imshow(X_recover[face_idx,:].reshape(H,W), cmap='gray')
        ax1.axis('off')
        ax2.axis('off')

    fig.patch.set_visible(False)
    fig.tight_layout()
    plt.show()

def KNN(X_train, y_train, X_test, k):
    max_vote = lambda x: np.argmax(np.bincount(x))

    prediction = np.zeros((X_test.shape[0],))
    for i, xn in enumerate(X_test):
        # Compute the distances b/w a test data xn and training dataset
        D = LA.norm(X_train-xn, axis=1) 
        # Pick the k training data which are closest to xn, 
        # and derive their corresponding labels
        k_candidates = y_train[np.argsort(D)[:k]]
        # do maximum voting to determine the final prediction
        prediction[i] = max_vote(k_candidates)

    return prediction

def performance(y_test, prediction):
    return np.sum(y_test == prediction)/y_test.shape[0]