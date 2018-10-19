import time
import numpy as np
import feats,feats.preproc,feats.tools,feats.action_imgs
import metric,graph,pairs,utils
import ensemble,learn
import seqs.io,plot,plot.ts
from sklearn.manifold import TSNE

def make_dtw_pairs(in_path,out_path,single=False):
    ens=learn.ensemble_pairs()
    ens(in_path,out_path,single)

def preproc_feats(in_path,out_path):
    feat_preproc=feats.tools.smooth_feats()
    feat_preproc.apply(in_path,out_path)

def show_dtw_feats(pairs_path='mhad/text_pairs/nn1',title="full" ):
    dtw_pairs=pairs.from_txt(pairs_path)
    X,y,persons=dtw_pairs.as_matrix()
    color_helper=plot.make_cat_colors(y)
    tsne_embd(X,y,title,color_helper)

def tsne_embd(X,y,title='tsne',color_helper=None):
    embd=TSNE(n_components=2,perplexity=30).fit_transform(X)
    plot.plot_embedding(embd,y,title=title,color_helper=color_helper)

#feats.action_imgs.TimelessActionImgs(hough="ellipse")('../mhad/max_z','../mhad/imgs')
#make_stat_feats(in_path='../mhad/smooth',out_path='../mhad/distance')
preproc_feats(in_path='../mra/all',out_path='../mra/smooth.txt')
#make_dtw_pairs('../mhad/four/seqs')
#plot.ts.plot_stats(in_path='../mhad/distance',out_path='../mhad/d_plot')
#make_dtw_pairs(in_path='../mhad/distance',out_path='../mhad/pairs.txt')