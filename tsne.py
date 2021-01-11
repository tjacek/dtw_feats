import os,numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import exp,feats,files

def show_all(in_path,out_path):
    files.make_dir(out_path)
    common,binary=exp.find_path(in_path)
    paths=common+binary
    for in_i in paths:
        out_i="%s/%s" % (out_path,in_i.split('/')[-2])
        make_plot(in_i,out_i)

def make_plot(in_path,out_path):
    if(os.path.isdir(in_path)):
        plot_i=tsne_plot(in_path,show=False)
        plot_i.savefig(out_path)

def tsne_plot(in_path,show=True,color_helper="cat",names=False):
    feat_dataset= feats.read(in_path)[0].split()[1]
    tsne=manifold.TSNE(n_components=2,perplexity=30)
    X,y,names=feat_dataset.as_dataset()
    X=tsne.fit_transform(X)
    color_helper=lambda i,y_i:y_i 
    return plot_embedding(X,y,title="tsne",color_helper=color_helper,show=show,names=names)

def plot_embedding(X,y,title="plot",color_helper=None,show=True,names=None):
    n_points=X.shape[0]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
   
    color_helper=color_helper if(color_helper) else lambda i,y_i:0

    plt.figure()
    ax = plt.subplot(111)

    rep= names if(names) else y
    for i in range(n_points):
        color_i= color_helper(i,y[i])
        plt.text(X[i, 0], X[i, 1],str(rep[i]),
                   color=plt.cm.tab20( color_i),
                   fontdict={'weight': 'bold', 'size': 9})
    print(x_min,x_max)
    if title is not None:
        plt.title(title)
    if(show):
        plt.show()
    return plt

show_all('../ICSS_exp/3DHOI/',"plots")