'''
Regularized Linear Model Regression & Visualization
===================================================
Methods: LSE & Newton's method

1. LSE method
   Objective: Minimize: f(x) = norm2(t - Aw) + lambda^2*I
   Using LUP decomposition and LUP solver to evaluate inverse of a matrix, 
   and compute w = (ATA+_lambdaI)^{-1} . AT . y

2. Newton's method
   Objective: Minimize: f(x) = norm2(t - Aw)
   w' = w + f'(w)/f''(w) = w + H^{-1} . f'(w)
   where f'(w) = ATAw - ATy and H = ATA
'''
import numpy as np
import random
from copy import copy
import matplotlib.pyplot as plt

def construct_A(x, n):
    A = np.zeros(shape=(len(x),n), dtype=np.double)
    for i in range(len(x)):
        for j in range(n):
            A[i, j] = x[i]**j
    return A

def IdentityMatrix(n):
    I = np.zeros((n,n), dtype=np.double)
    for i in range(n): I[i, i] = 1
    return I

def dot(A, B):
    m = A.shape[0]
    n = B.shape[1]
    p = A.shape[1]
    if p != B.shape[0]:
        print("error: dimension incorrect!")
        return
    
    C = np.zeros((m, n), dtype=np.double)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i, j] += A[i, k]*B[k, j]
    return C

def transpose(A):
    B = np.zeros(A.shape[::-1], dtype=A.dtype)
    for i in range(A.shape[1]):
        for j in range(A.shape[0]):
            B[i, j] = A[j, i]
    return B

def LUP_decomposition(A):
    B = copy(A)
    n = B.shape[0] # A.rows
    
    # Initialize U, L, P
    L, U, P = IdentityMatrix(n), np.zeros((n,n), np.double), np.zeros((n,n), np.int)
    for i in range(n): L[i, i] = 1
    p = np.zeros(shape=(n,), dtype=np.int) # permutation matrix
    for i in range(n): p[i] = i

    for k in range(n): # for each row
        pivot = 0
        k_prime = k
        for i in range(k, n): # find pivot in each col to prevent dividing a small value
            if abs(B[i, k]) > pivot:
                pivot = abs(A[i, k])
                k_prime = i
                
        if pivot == 0:
            print("error: singula matrix!")
            return

        # Exchange rows for pivoting
        p[k], p[k_prime] = p[k_prime], p[k] 
        for i in range(n): B[k, i], B[k_prime, i] = B[k_prime, i], B[k, i]
            
        for i in range(k+1, n): # for the rest rows
            B[i, k] = B[i, k]/B[k, k]
            for j in range(k+1, n):
                B[i, j] = B[i, j] - B[i, k]*B[k, j]
    
    # Assign values to L, U
    for i in range(n):
        for j in range(i, n):
            U[i, j] = B[i, j]
            if i != j: L[j, i] = B[j, i]
                
    # Assign values to P
    for i in range(n): P[i, p[i]] = 1
    return L, U, P

def LUP_solve(L, U, P, B):
    # LU = PA -> PAx = Pb -> LUx=Pb
    n, m = B.shape
    # Ly = Pb: Forward substitution to obtain y
    PB = dot(P, B)
    X, Y = np.zeros((n, m), dtype=np.double), np.zeros((n, m), dtype=np.double)
    
    for col in range(m):
        for i in range(n):
            sum_of_rest = 0
            for j in range(0, i): sum_of_rest += L[i, j]*Y[j, col]
            Y[i, col] = PB[i, col] - sum_of_rest
        # Ux = y: Backward substitution to obtain x
        for i in range(n-1, -1, -1):
            sum_of_rest = 0
            for j in range(i+1, n): sum_of_rest += U[i, j]*X[j, col]
            X[i, col] = (Y[i, col] - sum_of_rest)/U[i, i]
    return X

def inverse(A):
    n = A.shape[0]
    I = IdentityMatrix(n)
    L, U, P = LUP_decomposition(A)
    return LUP_solve(L, U, P, I)

def norm2(A):
    return dot(transpose(A), A)

def LSE(x, y, n, _lambda):
    # Minimize: f(x) = norm2(t - Aw) + lambda^2*I
    # w = (ATA+_lambdaI)^{-1}ATy
    A = construct_A(x, n)
    I = IdentityMatrix(A.shape[1])

    ATA_lambdaI = dot(transpose(A), A) + _lambda*I
    w = dot(inverse(ATA_lambdaI), dot(transpose(A), y))
    e = total_error(A, w, y)
    return w, e

def Newton(x, y, n):
    # Minimize: f(x) = norm2(t - Aw)
    # first derivative: ATAw - ATy
    # Hessian matrix: ATA
    # Since objective function is quadratic form, we can only update once.
    A = construct_A(x, n) # nxm
    AT = transpose(A)
    w = np.zeros((A.shape[1],1), dtype=np.double)
    for wi in w.tolist(): wi = random.gauss(mu=0, sigma=1)

    first_derivative = dot(dot(AT, A), w) - dot(AT, y)
    H = dot(AT, A)

    w = w - dot(inverse(H), first_derivative)
    e = total_error(A, w, y)

    return w, e

def print_fitting_line(w):
    s = 'Fitting line: '
    first = True
    newline = False
    for i in range(w.shape[0]-1, -1, -1):
        coefficient = np.round(w[i][0], 12)
        if first: 
            s += str(coefficient) 
            first = False
        elif coefficient >= 0: 
            s += ' + ' + str(coefficient)
        else:
            s += ' - ' + str(abs(coefficient))

        if i == 0: 
            continue
        s += 'X^' + str(i)
        if newline: s += '\n             '
        newline = not newline
    print(s)

def print_error(e):
    print('Total error: ' + str(e))

def total_error(A, w, t):
    return norm2(t - dot(A, w))[0][0]

def f(x, w):
    n, t = len(w), 0
    for i in range(n):
        t += w[i, 0]*x**i
    return t

def plot(x, y, w_LSE, w_newton):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("LSE vs. Newton's Method")

    # Scatter plot
    ax1.plot(x, y, 'o')
    ax2.plot(x, y, 'o')

    # Line plot
    x1 = np.linspace(min(x), max(x))
    y1 = np.linspace(min(y), max(y))

    ax1.plot(x1, f(x1, w_LSE), color='grey')
    ax2.plot(x1, f(x1, w_newton), color='grey')

    plt.show()

def main():

    while True:
        filename = input("/[Path]/[filename] or filename: ")
        n = int(input("Number of polynomial base n: ").split(" ")[0])
        _lambda = float(input("Lambda for LSE case: ").split(" ")[0])

        # Read data
        readdata = ''
        with open(filename, 'r') as f:
            readdata = f.read()
        f.closed
        # Construct vector x and y
        lines = readdata.split('\n')
        N = len(lines)
        x, y = np.zeros(shape=(N,1), dtype=np.double), np.zeros(shape=(N,1), dtype=np.double)

        for i in range(N):
            x[i], y[i] = tuple(lines[i].split(',')[:2])

        print("\nLSE:")
        w_LSE, e_LSE = LSE(x, y, n, _lambda)
        print_fitting_line(w_LSE)
        print_error(e_LSE)

        print("\nNewton's Method:")
        w_newton, e_newton = Newton(x, y, n)
        print_fitting_line(w_newton)
        print_error(e_newton)
        plot(x, y, w_LSE, w_newton)
        print("===============")

if __name__ == '__main__':
    main()