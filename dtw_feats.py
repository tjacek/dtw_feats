import time
import numpy as np
import feats.tools
import metric,graph,pairs,utils
import feats,feats.preproc,plot,ensemble
import seqs.io,seqs.concat,plot.ts
from sklearn.manifold import TSNE

def make_dtw_pairs(in_path):
    name=in_path.split('/')[-1]
    out_path=in_path.replace(name,'pairs.txt')
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=True)
    actions=read_actions(in_path)
    dtw_pairs=pairs.make_pairwise_distance(actions)
    dtw_pairs.save(out_path)

def make_stat_feats(in_path,out_path,single=False):
    stats_feats=feats.tools.stats_feats()
    ens=ensemble.global_feats(stats_feats)
    if(single):
        ens(in_path,out_path)
    else:
        ens.single_call(in_path,out_path)

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

make_stat_feats(in_path='../../mhad/seqs',out_path='../../mhad/stats')
#preproc_feats(in_path='mhad/seqs/raw/max_z',out_path='mhad/seqs/fourrier/max_z')
#make_dtw_pairs('../mhad/four/seqs')
#plot.ts.plot_stats(in_path='mhad/seqs/fourrier/max_z',out_path='mhad/seqs/fourrier')