import numpy as np
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt
import imageio
import os

HEIGHT, WIDTH = 100, 100
colormap = {0: (0.41960784, 0.43137255, 0.81176471), 
            1: (0.54901961, 0.63529412, 0.32156863), 
            2: (0.90588235, 0.72941176, 0.32156863), 
            3: (0.83921569, 0.38039216, 0.41960784)}

def kernel(x, gamma_s, gamma_c):
    s, c = x[:, :2], x[:, 2:]
    return np.exp(-gamma_s*cdist(s, s,'sqeuclidean')-gamma_c*cdist(c, c, 'sqeuclidean'))

def plot_eigenspace(eigenvectors, labels, filename):
    '''
    eigenvectors: (n_data, n_coordinates)
    labels: clusters that the data points belong to
    '''
    print(filename)
    n_coord = eigenvectors.shape[1]
    fig, axes = plt.subplots(n_coord, 1, figsize=(10, 5*n_coord))
    for i in range(n_coord): 
        plot(axes[i], eigenvectors.T[i, labels.argsort()], np.sort(labels), "Eigenvector "+str(i+1))
    fig.savefig(os.path.join('media','eigenspace',filename), 
                 format="png", dpi=300, bbox_inches="tight")


def plot(ax, eigenvector, labels, title):
    ax.set_title(title)
    f = lambda x: colormap[x]
    x = np.arange(labels.shape[0])
    ax.scatter(x, eigenvector, color=list(map(f, labels)), marker='.')

def imread(filename):
    img = cv2.imread(filename) # B, G, R
    data = np.zeros((HEIGHT*WIDTH,5)) # attribute: r, c, b, g, r
    for i in range(HEIGHT):
        for j in range(WIDTH):
            b, g, r = img[i, j]
            data[i*HEIGHT+j] = [i, j, b, g, r]
    data = ((data - data.min(axis=0))/(data.max(axis=0)-data.min(axis=0)))*100
    return data

def visualize(X, filename):
    length = int(np.sqrt(X.shape[0]))
    r = [107,140,231,214]
    g = [110,162,186, 97]
    b = [207, 82, 82,107]

    img = np.zeros([length,length,3], dtype=np.uint8)

    vfunc_r = np.vectorize(lambda k: r[int(k)])
    vfunc_g = np.vectorize(lambda k: g[int(k)])
    vfunc_b = np.vectorize(lambda k: b[int(k)])

    img[:,:,0] = vfunc_r(X).reshape(length, -1)
    img[:,:,1] = vfunc_g(X).reshape(length, -1)
    img[:,:,2] = vfunc_b(X).reshape(length, -1)
    cv2.imwrite(filename + '.png', cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    return img

def gen_gif(path, frames):
    folders = path.split('/')
    filename = os.path.join('media', folders[1]+'_videos', folders[2])
    imageio.mimsave(filename+'.gif', frames)