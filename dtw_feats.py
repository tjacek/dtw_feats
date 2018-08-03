import time
import numpy as np
import metric,graph,pairs,utils,plot,feats
import seqs.io,seqs.concat
from sklearn.manifold import TSNE

def concat_actions(in_path1='seqs/max_z',in_path2='seqs/all',out_path='seqs/full'):
    read_actions=seqs.io.ActionReader(as_dict=True)
    actions1=read_actions(in_path1)
    actions2=read_actions(in_path2)
    unified_actions=actions.concat.concat_actions(actions1,actions2)
    save_actions=actions.io.ActionWriter()
    save_actions(unified_actions,out_path) 

def show_dtw_feats(pairs_path='mra/pairs/corl_pairs',title="full" ):
    dtw_pairs=utils.read_object(pairs_path)
    X,y=pairs.as_matrix(dtw_pairs)
    color_helper=plot.cat_colors(y)
    tsne_embd(X,y,title,color_helper)

def show_global_feats(action_path='seqs/all',title="global features"):
    read_actions=seqs.io.ActionReader(as_dict=False)
    actions=read_actions(action_path)
    global_extractor=feats.GlobalFeatures()
    y=[ action_i.cat for action_i in seqs]
    X=[ global_extractor(action_i)  for action_i in actions]
    X=np.array(X)
    tsne_embd(X,y,title)

def tsne_embd(X,y,title='tsne',color_helper=None):
    embd=TSNE(n_components=2,perplexity=30).fit_transform(X)
    plot.plot_embedding(embd,y,title=title,color_helper=color_helper)

#seqs.concat.simple_concat(in_path1="data/proj",
#                  in_path2="data/time",
#                  out_path="data/MSR",img_seq=True)
#concat_actions()
#compute_pairs()
show_dtw_feats()
#show_global_feats()