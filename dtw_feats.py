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

def make_stat_feats(in_path,out_path,single=False):
    stats_feats=feats.tools.extrem_feats()
    stats_feats.apply(in_path,out_path)

def preproc_feats(in_path,out_path):
    preproc_fun=feats.preproc.fourier_magnitude
    feat_preproc=feats.preproc.FeaturePreproc(preproc_fun)
    feat_preproc.transform(in_path,out_path)

def show_dtw_feats(pairs_path='mhad/text_pairs/nn1',title="full" ):
    dtw_pairs=pairs.from_txt(pairs_path)
    X,y,persons=dtw_pairs.as_matrix()
    color_helper=plot.make_cat_colors(y)
    tsne_embd(X,y,title,color_helper)

def tsne_embd(X,y,title='tsne',color_helper=None):
    embd=TSNE(n_components=2,perplexity=30).fit_transform(X)
    plot.plot_embedding(embd,y,title=title,color_helper=color_helper)

#feats.action_imgs.TimelessActionImgs(hough="ellipse")('../mhad/max_z','../mhad/imgs')
make_stat_feats(in_path='../mhad/max_z',out_path='../mhad/basic/extr.txt')
#preproc_feats(in_path='mhad/seqs/raw/max_z',out_path='mhad/seqs/fourrier/max_z')
#make_dtw_pairs('../mhad/four/seqs')
#plot.ts.plot_stats(in_path='mhad/seqs/fourrier/max_z',out_path='mhad/seqs/fourrier')