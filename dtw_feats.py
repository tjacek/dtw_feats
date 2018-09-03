import time
import numpy as np
import feats.tools
import metric,graph,pairs,utils,plot,feats
import seqs.io,seqs.concat
from sklearn.manifold import TSNE

def make_dtw_pairs(in_path):
    name=in_path.split('/')[-1]
    out_path=in_path.replace(name,'pairs.txt')
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=True)
    actions=read_actions(in_path)
    dtw_pairs=pairs.make_pairwise_distance(actions)
    dtw_pairs.save(out_path)

def make_stat_feats(in_path):
    name=in_path.split('/')[-1]
    out_path=in_path.replace(name,'dataset.txt')
    global_feats=feats.tools.stats_feats()
    global_feats.apply(in_path,out_path)

def concat_actions(in_path1='seqs/max_z',in_path2='seqs/all',out_path='seqs/full'):
    read_actions=seqs.io.ActionReader(as_dict=True)
    actions1=read_actions(in_path1)
    actions2=read_actions(in_path2)
    unified_actions=actions.concat.concat_actions(actions1,actions2)
    save_actions=actions.io.ActionWriter()
    save_actions(unified_actions,out_path) 

def show_dtw_feats(pairs_path='mhad/text_pairs/nn1',title="full" ):
    dtw_pairs=pairs.from_txt(pairs_path)
    X,y,persons=dtw_pairs.as_matrix()
    color_helper=plot.make_cat_colors(y)
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

#concat_actions()
#compute_pairs()
make_dtw_pairs('../mhad/four/seqs')
#show_global_feats()