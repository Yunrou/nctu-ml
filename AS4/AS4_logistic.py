import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

def N(mean, variance):
    # Generating value from univariate Gaussian distribution by Box-Muller Method
    # two_pi: 6.283185307179586
    epsilon = np.finfo(float).resolution

    u1, u2 = 0, 0
    while u1 <= epsilon:
        u1 = np.random.rand()
        u2 = np.random.rand()

    z = np.sqrt(-2*np.log(u1)) * np.cos(6.283185307179586*u2)

    return z * np.sqrt(variance) + mean

def plot(ax, data, labels, title):
    ax.set_title(title)
    f = lambda x: 'firebrick' if x < 0.5 else 'steelblue'
    ax.scatter(data.T[0], data.T[1], color=list(map(f, labels)), marker='o')
    
def print_result(method, w, predictions, labels):
    print(method + ':\n')
    print('w:')
    print("{:15.10f}".format(np.round(w.flatten()[0], 10)))
    print("{:15.10f}".format(np.round(w.flatten()[1], 10)))
    print("{:15.10f}".format(np.round(w.flatten()[2], 10)))
    print("\nConfusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(labels.flatten(), predictions.flatten()).ravel()
    cm = pd.DataFrame([[tn, fp], [fn, tp]], 
                       index=['Is cluster 1', 'Is cluster 2'], columns=['Predict cluster 1', 'Predict cluster 2'])
    print(cm)
    print("\nSensitivity (Successfully predict cluster 1): {:7.5f}".format(tp/(fn+tp)))
    print("Specificity (Successfully predict cluster 2): {:7.5f}".format(tn/(fp+tn)))

def logit(a):
    return np.divide(1, (1+np.exp(-a)))

def main():
    while True:
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(8, 6))
        n, m1, m2, v1, v2 = list(map(int, (input("N, mx1=my1, mx2=my2, vx1=vy1, vx2=vy2: ")).split(', ')))

        data = np.concatenate((np.array([[N(m1, v1), N(m1, v1)] for x in range(n)]), 
                            np.array([[N(m2, v2), N(m2, v2)] for x in range(n)])), axis=0)
        labels = np.concatenate((np.zeros((50,1)), np.ones((50,1))), axis=0) # D1: label=0; D2: label=1
        Phi = np.array([[1, d[0], d[1]] for d in data]) #  shape=(n, 3)
        # Gradient descent
        w, wn = np.random.rand(3, 1), np.zeros((3,1)) # [w0, w1, w2].T
        delta_w, tol, eta, In = np.inf, 1e-4, 0.1, np.identity(2*n, dtype=np.float64)

        iteration = 0
        while delta_w > tol:
            gradient = Phi.T.dot(logit(Phi.dot(w))-labels) # shape=(3,1)
            wn = w - eta*gradient
            delta_w = np.linalg.norm(wn-w)
            w = wn
            iteration += 1
            if iteration > 10000: break
        plot(ax1, data, labels, "Ground truth")
        plot(ax2, data, logit(Phi.dot(w)), "Gradient descent")
        print_result("Gradient descent", w, np.round(logit(Phi.dot(w))), labels)

        # Newton's method
        for i in range(100):
            w, wn = np.random.rand(3, 1), np.zeros((3,1)) # [w0, w1, w2].T
            delta_w = np.inf
            iteration = 0
            converge = True
            while delta_w > tol:
                if np.any(Phi.dot(w) < -700): # if overflow, use another w0
                    converge = False
                    break
                D = logit(Phi.dot(w)) * (1-logit(Phi.dot(w)))*In
                H = Phi.T @ D @ Phi
                gradient = Phi.T.dot(logit(Phi.dot(w))-labels)
                if np.linalg.det(H) == 0:
                    wn = w - eta*gradient
                else:
                    wn = w - inv(H)@gradient
                delta_w = np.linalg.norm(wn-w)
                w = wn
                iteration += 1
                if iteration > 1000: break

            if converge: break
        plot(ax3, data, logit(Phi.dot(w)), "Newton's method")
        print("----------------------------------------")
        print_result("Newton's method", w, np.round(logit(Phi.dot(w))), labels)
        plt.show()   

if __name__ == '__main__':
    main()