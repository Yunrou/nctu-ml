import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N = 34

def kernel(X1, X2, a, l, sigma):
    '''
    X1: m points
    X2: n points
    K : (m,n)
    '''
    rqdist2 = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
    return sigma**2 * (1 + rqdist2/(2*a*l**2))**(-a)

def nll(X_train, Y_train, noise):
    Y_train = Y_train.ravel()
    def nll_naive(theta):
        C = kernel(X_train, X_train, a=theta[0], l=theta[1], sigma=theta[2]) + \
            noise*np.identity(N)
        return 0.5*np.log(det(C)) + 0.5*N*np.log(2*np.pi) + 0.5*Y_train.T @ inv(C) @ Y_train
    return nll_naive

def posterior(X_test, X_train, Y_train, noise, a, l, sigma):
    C_N     = kernel(X_train, X_train, a, l, sigma) + noise * np.identity(N)
    C_N_inv = inv(C_N)
    K       = kernel(X_train, X_test, a, l, sigma) # (N,N_test)
    c       = kernel(X_test, X_test, a, l, sigma) + noise * np.identity(X_test.shape[0]) # (N_test, N_test)
    
    mu_new = K.T @ C_N_inv @ Y_train # (N_test,1)
    cov_new = c - K.T @ C_N_inv @ K # (N_test, N_test)

    return mu_new, cov_new

def plot(X_train, Y_train, noise, a, l, sigma):

    fig, (ax) = plt.subplots(1, 1, figsize=(8, 6))
    
    # ax.set_xlim(-100, 100)
    ax.set_ylim(-6, 6)
    
    # Show all training data points
    ax.scatter(X_train, Y_train, color='steelblue', marker='.')
    X = np.linspace(-60,60,1000)
    mu, cov = posterior(X.reshape(-1,1), X_train, Y_train, noise, a, l, sigma)
    mu, var = mu.ravel(), cov.diagonal()

    # Draw a line to represent the mean of f in range
    ax.plot(X, mu, color='steelblue')
    # Mark 95% confidence interval of f
    
    confidence = 1.96*(np.sqrt(var)/np.sqrt(34))
    ax.fill_between(X, mu-confidence, mu+confidence, color='steelblue', alpha=0.5)
    plt.show()

def main():
    # Load data 34x2
    with open("input.data") as f:
        f.seek(0)
        data_str = f.read().split('\n')
        data = np.zeros((N,2))
        for i, instance in enumerate(data_str):
            if not instance: break
            x, y = instance.split(' ')
            data[i] = [float(x), float(y)]
    X_train, Y_train = data[:,0].reshape(N,1), data[:,1].reshape(N,1)
    noise = 0.2 # 1/beta
    # Optimize parameters of kernel function
    theta = minimize(nll(X_train, Y_train, noise), x0=[1,1,1], method='L-BFGS-B',
                     bounds=((1e-5, None),(1e-5, None),(1e-5, None)))

    a, l, sigma = theta.x
    print("theta (a, l, sigma) =", np.round(theta.x, 4))
    print("negative log-likelihood =", np.round(nll(X_train, Y_train, noise)(theta.x), 4))
    # plot prediction
    plot(X_train, Y_train, noise, a, l, sigma)

if __name__ == '__main__':
    main()