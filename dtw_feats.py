import time
import numpy as np
import actions.io,metric,graph,utils,plot,feats
from sklearn.manifold import TSNE

def compute_pairs(in_path='seqs/all',out_path='pairs/all_pairs'):
    read_actions=actions.io.ActionReader()
    print(in_path)
    seqs=read_actions(in_path)
    t0=time.time()
    pairs=graph.make_pairwise_distance(seqs)
    print("pairs computation %d" % (time.time()-t0))
    utils.save_object(pairs.raw_pairs,out_path)

def show_dtw_feats(action_path='seqs/all',pairs_path='pairs/all_pairs',title="all" ):
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
    plot.plot_embedding(embd,y,title=title)

#compute_pairs()
#show_dtw_feats()
show_global_feats()