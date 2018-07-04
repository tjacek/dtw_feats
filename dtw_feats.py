import time
import numpy as np
import metric,graph,utils,plot,feats
import actions.io,actions.concat
from sklearn.manifold import TSNE

def concat_actions(in_path1='seqs/max_z',in_path2='seqs/all',out_path='seqs/full'):
    read_actions=actions.io.ActionReader(as_dict=True)
    actions1=read_actions(in_path1)
    actions2=read_actions(in_path2)
    unified_actions=actions.concat.concat_actions(actions1,actions2)
    save_actions=actions.io.ActionWriter()
    save_actions(unified_actions,out_path) 

def compute_pairs(in_path='seqs/full',out_path='pairs/full_pairs'):
    read_actions=actions.io.ActionReader()
    print(in_path)
    seqs=read_actions(in_path)
    t0=time.time()
    pairs=graph.make_pairwise_distance(seqs)
    print("pairs computation %d" % (time.time()-t0))
    utils.save_object(pairs.raw_pairs,out_path)

def show_dtw_feats(action_path='seqs/full',pairs_path='pairs/full_pairs',title="full" ):
    read_actions=actions.io.ActionReader(as_dict=True)
    seqs=read_actions(action_path)
    raw_pairs=utils.read_object(pairs_path)
    dtw_pairs=graph.PairwiseDistance(raw_pairs)
    action_names=seqs.keys()
    X=dtw_pairs.as_matrix(action_names)
    y=[ seqs[name_i].cat for name_i in action_names]
    tsne_embd(X,y,title)

def show_global_feats(action_path='seqs/all',title="global features"):
    read_actions=actions.io.ActionReader(as_dict=False)
    seqs=read_actions(action_path)
    global_extractor=feats.GlobalFeatures()
    y=[ action_i.cat for action_i in seqs]
    X=[ global_extractor(action_i)  for action_i in seqs]
    X=np.array(X)
    tsne_embd(X,y,title)

def tsne_embd(X,y,title):
    embd=TSNE(n_components=2,perplexity=30).fit_transform(X)
    plot.plot_embedding(embd,y,title=title,highlist=[12,15])

#concat_actions()
#compute_pairs()
show_dtw_feats()
#show_global_feats()