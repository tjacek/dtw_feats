import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import random

def plot_embedding(X,y,title=None):
    n_points=X.shape[0]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    cats=np.unique(y)
    n_cats= cats.shape[0]
    random.shuffle(cats)
    plt.figure()
    ax = plt.subplot(111)

#    plt.cm.get_cmap('Vega20c')
    for i in range(n_points):
        color_i= float(cats[int(y[i])-1]) / float(n_cats)
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                   color=plt.cm.tab20( color_i),
                   fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

def show_distance(D):
    plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
    plt.colorbar();
    plt.show()
