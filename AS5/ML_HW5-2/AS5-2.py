import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import libsvm.svm as svm
import libsvm.svmutil as svmutil

X_train = np.genfromtxt('X_train.csv', delimiter=',') # shape = (5000, 784)
X_test  = np.genfromtxt('X_test.csv', delimiter=',') # shape = (2500, 784)
Y_train = np.genfromtxt('Y_train.csv', delimiter=',') # shape = (5000, )
Y_test  = np.genfromtxt('Y_test.csv', delimiter=',') # shape = (2500, )

def grid_search(kernel, opt, Y_train, X_train):
    '''
    Parameters
    ----------
    C:      [1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3]
    gamma:  [1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3]
    degree: [2, 3, 4]

    Options
    -------
    -v 3: set cross validation to 3-fold
    -c:   set cost
    -g:   set gamma
    -d:   set degree
    '''
    param = {'C': np.logspace(-3, 3, 7), 
             'gamma': np.logspace(-3, 3, 7), 
             'degree': range(2, 5)}
    opt += ' -v 3'
    
    if kernel == 'Linear':
        df = pd.DataFrame(np.zeros((7,2)), columns=['C', 'ACC'])
        for i, C in enumerate(param['C']):
            acc = svmutil.svm_train(Y_train, X_train, '{} -c {}'.format(opt, C))
            df.iloc[i] = [C, acc] 
        return df
    
    elif kernel == 'Polynomial':
        i = 0
        df = pd.DataFrame(np.zeros((7*7*3,4)), columns=['C', 'gamma', 'degree', 'ACC'])
        for C in param['C']:
            for gamma in param['gamma']:
                for degree in param['degree']:
                    acc = svmutil.svm_train(Y_train, X_train, '{} -c {} -g {} -d {}'.format(opt, C, gamma, degree))
                    df.iloc[i] = [C, gamma, degree, acc]
                    i += 1
        return df
    
    elif kernel == 'RBF':
        i = 0
        df = pd.DataFrame(np.zeros((7*7,3)), columns=['C', 'gamma', 'ACC'])
        for C in param['C']:
            for gamma in param['gamma']:
                acc = svmutil.svm_train(Y_train, X_train, '{} -c {} -g {}'.format(opt, C, gamma))
                df.iloc[i] = [C, gamma, acc]
                i += 1
        return df

def plot(ax, df, title):
        g = sns.heatmap(df.set_index(['C', 'gamma']).ACC.unstack(1), annot=True, cmap="YlGnBu", ax=ax)
        ax.invert_yaxis()
        ax.set_title(title)

def customized_kernel(X, X_, gamma):
    '''
    Combine linear and RBF kernel
    '''
    kernel_linear = X @ X_.T 
    dist          = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X_**2, 1) - 2 * X @ X_.T
    kernel_rbf    = np.exp(-gamma*dist)
    kernel = np.hstack((np.arange(1,len(X)+1).reshape(-1,1), kernel_linear+kernel_rbf))
    return kernel

def main():

    print("======= Part 1 =======")
    '''
    -s 0: C-SVC
    -t 0: Linear kernel w/ default C=1
    -t 1: Polynomial kernel w/ default C=1, degree=3 and default gamma=1/k
    -t 2: RBF kernel w/ default C=1 and gamma=1/k
    '''
    kernels = {'Linear': 0, 'Polynomial': 1, 'RBF': 2}
    options = np.array(['-q -s 0 -t 0', '-q -s 0 -t 1', '-q -s 0 -t 2']) 

    for kernel_type, idx in kernels.items():
        model =  svmutil.svm_train(Y_train, X_train, options[idx])
        p_label, p_acc, p_val = svmutil.svm_predict(Y_test, X_test, model,'-q')
        print('{}: Accuracy = {:.2f}%'.format(kernel_type, p_acc[0]))

    print("======= Part 2 =======")
    dfs = []
    for kernel_type, idx in kernels.items():
        df = grid_search(kernel_type, options[idx], Y_train, X_train)
        best_param, best_acc = tuple(df.loc[df['ACC'].argmax()][:-1]), df.loc[df['ACC'].argmax()][-1]
        print('{}: {}, Cross Validation Accuracy = {:.2f}%'.format(kernel_type, best_param, best_acc))
        dfs.append(df)

    # plot heatmaps
    fig, axes = plt.subplots(4, 1, figsize=(9, 32))    
    for i in range(3):
        df = dfs[1].sort_values(by=['degree']).iloc[i*49:(i+1)*49]
        plot(axes[i], df, "Polynomial degree {}".format(i+2))
    plot(axes[3], dfs[2], "RBF")
    plt.show()

    print("======= Part 3 =======")
    K_train = customized_kernel(X_train, X_train, 1e-2)
    K_test = customized_kernel(X_test, X_train, 1e-2)
    model = svmutil.svm_train(Y_train, K_train, '-q -t 4')
    p_label, p_acc, p_val = svmutil.svm_predict(Y_test, K_test, model,'-q')
    print('Linear + RBF Kernel:  Accuracy = {:.2f}%'.format(p_acc[0]))

if __name__ == '__main__':
    main()