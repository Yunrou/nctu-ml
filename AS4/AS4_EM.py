import struct
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

train_images_file = "data/train-images-idx3-ubyte"
train_labels_file = "data/train-labels-idx1-ubyte"
test_images_file = "data/t10k-images-idx3-ubyte"
test_labels_file = "data/t10k-labels-idx1-ubyte"
magic_image = 2051
magic_label = 2049

N = 60000
D = 784 # pixels in an image(dimension)
K = 10 # 10 classes, from 0 to 9
epsilon = np.finfo(float).eps

def loadfile(filename):
    data = []
    with open(filename, 'rb') as f:
        magic, n = struct.unpack(">II", f.read(8)) # >: big endian, I: unsigned int 32, 8 bytes = 64 bits
        
        nrows, ncols = (1, 1)
        if magic == magic_image:
            nrows, ncols = struct.unpack(">II", f.read(8))
        
        data = np.fromfile(f, dtype=np.uint8)
        if magic == magic_image:
            data = data.reshape((n, nrows*ncols))
        else:
            data = data.reshape((n,))
    f.close()
    return data

def binning(images):
    # bin
    binned_images = np.zeros(images.shape, dtype=np.uint8)
    it = np.nditer([images, binned_images],
                   ['external_loop', 'buffered'],
                   [['readonly'], ['writeonly']])
    with it:
        for instance, instance_new in it:
            instance_new[...] = instance >> 7

    return binned_images

def E(pi, mu, data):

    P = np.zeros((N, K))
    # E Step 
    # Probability of P(x_n|mu_k)
    for n in range(N):
        for k in range(K):
            P[n, k] = np.prod((mu[k]*data[n]) + ((1-mu[k])*(1-data[n])))

    # Weight wk = pi_k P(x_n|mu_k)/sum(pi_j P(x_n|mu_k)), w.shape = (N,K)
    P = P*pi # multiply along axis=1, pi.shape = (K)
    sum_P = P.sum(axis=1)# sum by j = 1~K, sum_P.shape = (N)

    try:
        P = P/sum_P[:, np.newaxis] # divide along axis=0
    except ZeroDivisionError:
        print("Division by zero occurred in calculating w!")
    return P

def M(w, data):
    # M Step
    Nk = w.sum(axis=0)

    sum_wx = w.T.dot(data)

    new_pi = Nk/N
    new_mu = np.zeros((K,D))
    try:
        new_mu = sum_wx/Nk[:,np.newaxis]
    except ZeroDivisionError:
        print("Division by zero occurred in M step!")
        return new_pi, new_mu

    return new_pi, new_mu

def loglikelohood(pi, mu, data, w):
    ll = 0
    for n in range(N):
        sumK = 0
        for k in range(K):
            try:
                logP = (mu[k]*data[n])+((1-mu[k])*(1-data[n]))
                logP = np.log(logP.clip(min=epsilon)) # set lower bound
            except:
                print("Problem computing log(probability)")
            if pi[k]<=epsilon: continue
            sumK += w[n, k] * (np.log(pi[k])+np.sum(logP)) #
        ll += sumK
    return ll

def plot_images(mu, k2number):
    images = ['' for x in range(10)]
    for k, image in enumerate(mu):
        s = "class: "+str(k2number[int(k)]) + "\n"
        for row in image.reshape(28,28).tolist():
            for pixel in row:
                if pixel >= 0.5: s += "1 "
                else: s += "0 "
            s += "\n"
        images[k2number[k]] = s

    for image in images:
        print(image)

def assign_labels(pi, mu, data, k2number):
    w = E(pi, mu, data)
    assigned_ks = w.argmax(axis=1) # labels.shape = (N)
    vfunc = np.vectorize(lambda k: k2number[int(k)])
    assigned_labels = vfunc(assigned_ks)
    return assigned_labels

def plot_confusion_matrices(labels, assigned_labels):
    for i in range(10):
        k = str(i)
        print("------------------------------------------------------------\n")
        print("Confusion Matrix {}:".format(i))
        tn, fp, fn, tp = confusion_matrix((labels != i).astype(int), 
                                          (assigned_labels != i).astype(int)).ravel()
        cm = pd.DataFrame([[tn, fp], [fn, tp]], 
                           index=['Is number '+k, "Isn't number "+k], 
                           columns=['Predict number '+k, 'Predict not number '+k])
        print(cm)
        print("\nSensitivity (Successfully predict number {}): {:7.5f}".format(i, tp/(fn+tp)))
        print("Specificity (Successfully predict not number {}): {:7.5f}".format(i, tn/(fp+tn)))
        print("")

def assignk2number(pi, labels):
    order_k = pi.argsort() # min to max : k
    unique, counts = np.unique(labels, return_counts=True)
    order_number = counts.argsort() # min to max : label/number
    
    k2number = np.zeros((K,), dtype=int)

    for (k, number) in zip(order_k, order_number):
        k2number[int(k)] = int(number)

    return k2number

def main():
    data = binning(loadfile(train_images_file))[:N]
    labels = loadfile(train_labels_file)[:N]
    test_images = binning(loadfile(test_images_file))
    test_labels = loadfile(test_labels_file)

    # Initialization
    
    pi, new_pi = np.random.uniform(.3,.7,K), np.zeros((K,)) # K 
    pi /= pi.sum()
    mu, new_mu = np.random.randn(K,D), np.random.randn(K,D) # K x D
    ll, delta_ll  = np.inf, np.inf
    delta_pi, delta_mu = np.inf, np.inf

    w = np.zeros((N,K))
    tol, iterations = 1e-4, 0
    k2number = np.zeros((K,), dtype=int)

    for i in range(100):
        w = E(pi, mu, data)
        new_pi, new_mu = M(w, data)
        new_ll = loglikelohood(pi, mu, data, w)
        delta_pi, delta_mu = LA.norm(new_pi-pi), LA.norm(new_mu-mu)
        pi, mu = new_pi, new_mu
        
        delta_ll = abs(new_ll - ll)
        ll = new_ll
        
        if i in (0,1):
            k2number = assignk2number(pi, labels)

            plot_images(mu, k2number)
            print("No. of Iteration: {}".format(i+1))
            print("\n------------------------------------------------------------")
            print("------------------------------------------------------------\n")

        if delta_pi < tol and delta_mu < tol and delta_ll < tol: 
            iterations = i+1
            break
    
    k2number = assignk2number(pi, labels)
    plot_images(mu, k2number)
    print("No. of Iteration: {}".format(iterations))
    print("\n------------------------------------------------------------")
    print("------------------------------------------------------------\n")
    assigned_labels = assign_labels(pi, mu, data, k2number)
    plot_confusion_matrices(labels, assigned_labels)
    print("Total iterations to converge:", iterations)
    print("Total error rate:", np.sum(labels == assigned_labels)/N)

if __name__ == '__main__':
    main()