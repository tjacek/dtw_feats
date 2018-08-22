import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import random
import pandas as pd,seaborn as sn
from sets import Set

class CatColors(object):
    def __init__(self,cats):
        self.cats=cats
        self.n_cats= cats.shape[0]

    def __call__(self,i,y_i):    
        num=float(self.cats[int(y_i)-1])
        div=float(self.n_cats)
        return  num/div

class PersonColors(object):
    def __init__(self, persons):
        self.persons = persons
    
    def __call__(self,i,y_i):
        return float(self.persons[i]%2)

def make_cat_colors(y,highlist=None):
    cats=np.unique(y)
    if(highlist):
        highlist=Set(highlist)
        for i in range(n_cats):
            cat_i= int(cats[i])
            if(not cat_i in highlist):
                cats[i]=0
    else:
        random.shuffle(cats)
    return CatColors(cats)

def plot_embedding(X,y,title=None,color_helper=None):
    n_points=X.shape[0]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
   
    color_helper=color_helper if(color_helper) else lambda i:0
    plt.figure()
    ax = plt.subplot(111)

    for i in range(n_points):
        color_i= color_helper(i,y[i])
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                   color=plt.cm.tab20( color_i),
                   fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

def heat_map(conf_matrix):
    dim=conf_matrix.shape
    df_cm = pd.DataFrame(conf_matrix, range(dim[0]),range(dim[1]))
    sn.set(font_scale=1.0)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 8}, fmt='g')
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