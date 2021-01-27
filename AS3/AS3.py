import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt

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

def basis_functions(x, n):
    x.reshape((len(x), 1))
    Phi = np.zeros(shape=(len(x),n), dtype=np.double)
    for i in range(len(x)):
        for j in range(n):
            Phi[i, j] = x[i]**j
    return Phi

def poly_basis_linear(n, variance, w):
    x = -1 + 2 * np.random.rand(1,)
    Phi = basis_functions(x, n)
    return (x[0], np.dot(w.T, Phi[0])[0] + N(0, variance))

def posterior(phi, t, a, L, m):
    L_new = a**(-1) * phi.dot(phi.T) + L
    S_new = inv(L_new)
    m_new = S_new.dot(a**(-1) * phi * t + L.dot(m))
    return m_new, S_new, L_new

def posterior_predictive(phi, m_w, S_w, a):
    t     = phi.T.dot(m_w)[0][0]
    t_var = a + phi.T.dot(S_w).dot(phi)[0][0]
    return t, t_var

def print_posterier(mean, variance):
    print("Posterier mean:")
    for value in mean:
        print("{:.10f}".format(value[0]))

    print("\nPosterier variance:")
    for row in variance:
        for i, value in enumerate(row):
            if i == 0:
                print("{:.10f}".format(value), end='')
                continue
            print(", {:.10f}".format(value), end='')
        print("")
def plot_mean(ax, n, m):
    
    x = np.linspace(-2, 2)
    f = basis_functions(x, n).dot(m)
    ax.plot(x, f, color='black')

def plot_variance(ax, n, m, a, S):

    x = np.linspace(-2, 2)
    Phi = basis_functions(x, n)

    f1 = Phi.dot(m) + (a + Phi.dot(S).dot(Phi.T)).diagonal().reshape(len(x), 1)
    f2 = Phi.dot(m) - (a + Phi.dot(S).dot(Phi.T)).diagonal().reshape(len(x), 1)
    
    ax.plot(x, f1, color='firebrick')
    ax.plot(x, f2, color='firebrick')

def plot_point(ax, x, y):
    ax.scatter(x, y, color='steelblue', marker='.')    

def plot(ax, n, m, a, S, X, t, title):

    ax.set_title(title)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-20, 20)
    if len(X): 
        plot_point(ax, X, t)
    plot_variance(ax, n, m, a, S)
    plot_mean(ax, n, m)

def main():
    # 1 Random Data Generator
    # b. polynomial basis linear model data generator

    # 2 Sequential Estimator
    print("2 Sequential Estimator")
    # Initialize
    mu, sigma = N(3, 5), 0
    data = np.array([mu,])
    tol = 1e-6
    delta_mu, delta_sigma = np.inf, np.inf

    while delta_mu > tol or delta_sigma > tol:
        # until converge
        
        # a. univariate gaussian data generator
        x_new     = N(3, 5)
        data = np.append(data, x_new)
        print("Add data point:", x_new)

        mu_new    = mu + (x_new - mu)/len(data)
        sigma_new = sigma + (x_new - mu) * (x_new - mu_new) # Welford's algorithm
        print("Mean =", mu_new, " Variance =", sigma_new/(len(data)-1))

        delta_mu, delta_sigma = abs(mu_new - mu), abs(sigma_new - sigma)
        mu, sigma = mu_new, sigma_new

    # 3 Bayesian Linear Regression
    print("\n3 Bayesian Linear Regression")
    while True:
        # Input 
        cmd = input("b, n, a, w: ")
        cmd = cmd.split(', ')
        b, n, a = list(map(int, cmd[:3])) # prior ~ N(0, b^{-1}I), likelihood ~ N(0, a) 
        w = cmd[3:]
        w[0], w[-1] = w[0][1:], w[-1][:-1]
        w = np.array(list(map(int, w))).reshape(n, 1)

        m, L = np.zeros(shape=(n, 1)), b * np.identity(n)
        X, t, delta_m, delta_var = [], [], np.inf, np.inf
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        
        plot(ax1, n, w, a, np.zeros((n,n)), X, t, "Ground truth")
        
        for i in range(1000):
            # Converge
            if delta_m < tol and delta_var < tol:
                break
            # b. Polynomial basis linear model data generator ~N(0, a)
            x_new, t_new = poly_basis_linear(n, a, w)
            print("----------------------------------------------")
            print("Add data point ({:.5f}, {:.5f}):\n".format(x_new, t_new)) 
            X, t = np.append(X, x_new), np.append(t, t_new)
            phi    = basis_functions(np.array([x_new]), n).T # phi.shape: 4x1
            
            # Mean and covariance matrix of posterior
            m_w, S_w, L_w = posterior(phi, t_new, a, L, m)
            print_posterier(m_w, S_w)
            delta_m, delta_var = LA.norm(m-m_w, 2), LA.norm(S_w) 
            m = m_w

            # Mean and variances of posterior predictive
            predict_t, predict_t_var = posterior_predictive(phi, m_w, S_w, a)
            print("\nPredictive distribution ~ N({:.5f}, {:.5f})".format(predict_t, predict_t_var))
            
            if i == 9:
                plot(ax3, n, m_w, a, S_w, X, t, "After 10 incomes")
            elif i == 49:
                plot(ax4, n, m_w, a, S_w, X, t, "After 50 incomes")
            
        print("----------------------------------------------")

        plot(ax2, n, m_w, a, S_w, X, t, "Predict result")
        plt.show()

if __name__ == '__main__':
    main()