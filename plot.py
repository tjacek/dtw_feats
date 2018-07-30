import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import random
from sets import Set

def plot_embedding(X,y,title=None,highlist=None):
    n_points=X.shape[0]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    cats=np.unique(y)
    n_cats= cats.shape[0]
    if(not highlist is None):
        highlist=Set(highlist)
        for i in range(n_cats):
            cat_i= int(cats[i])
            if(not cat_i in highlist):
                cats[i]=0
    else:
        random.shuffle(cats)
    plt.figure()
    ax = plt.subplot(111)

    def color_helper(i):
        return float(cats[int(y[i])-1]) / float(n_cats)

    for i in range(n_points):
        color_i= color_helper(i)
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

def show_histogram(hist,title='hist',cumsum=True):
    if(type(hist)==list):
        hist=np.array(hist)
    if(cumsum):
        hist=np.cumsum(hist)
    fig = plt.figure()
    x=range(hist.shape[0])
    plt.bar(x,hist)
    fig.suptitle(title)
    plt.show()
