import numpy as np
from math import factorial

def comb(n, k):
    return factorial(n) / factorial(k) / factorial(n - k)

# Binomial Likelihood
def BinomialLikelihood(n, x):
    p = x/n
    return comb(n, x) * np.power(p, x) * np.power(1-p, n-x)

# Beta Posterior
def BetaPosterior(N, m, a, b):
    return m+a, N-m+b

def main():

    filename = input("filename: ")
    # Read data
    readdata = ''
    with open(filename, 'r') as f:
        readdata = f.read()
    f.closed
    # Handle data
    cases_string = readdata.split("\n")
    cases = np.array([list(map(lambda x: int(x), c)) for c in cases_string])

    while True:
        a, b = np.array(input("Initial a b: ").split(" "), dtype=np.int64)

        i = 1
        for c in cases:
            print("\x1b[93mcase", i, "\x1b[0m:", cases_string[i-1])
            n = len(c)
            x = np.count_nonzero(c)
            likelihood = BinomialLikelihood(n, x)
            print("\x1b[92mLikelihood:\x1b[0m", likelihood)
            print("\x1b[92mBeta prior:\x1b[0m     a =", a, "b =", b)
            a, b = BetaPosterior(n, x, a, b)
            print("\x1b[92mBeta posterior:\x1b[0m a =", a, "b =", b)
            i += 1
        print("==============")

if __name__ == '__main__':
    main()