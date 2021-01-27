import numpy as np
import matplotlib.pyplot as plt

def visualizeEmbedding(Y, labels, method, perplexity, iter):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.set_title("{} with Perp={}".format(method, int(perplexity)))
    ax.scatter(Y[:, 0], Y[:, 1], 20, labels)
    ax.axis('off')
    fig.savefig("./media/embedding/{}_{}_{}.png".format(method, int(perplexity), iter),
                format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def visualizeAffinities(P, Q, labels, method, perplexity):
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    
    idx = labels.argsort()
    P_sorted = P[:,idx][idx]
    Q_sorted = Q[:,idx][idx]

    ax[0].set_title('P (H-dim)')
    ax[0].imshow(np.log(P_sorted), cmap='gray')
    ax[0].axis('off')

    ax[1].set_title('Q (H-dim)')
    ax[1].imshow(np.log(Q_sorted), cmap='gray')
    ax[1].axis('off')

    fig.patch.set_visible(False)
    fig.tight_layout()
    fig.savefig("./media/heatmap_{}_{}.png".format(method, int(perplexity)))