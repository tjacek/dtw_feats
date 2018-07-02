import time
import actions.io,metric,graph,utils,plot
import sklearn.manifold
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

def compute_pairs(in_path='seqs/max_z',out_path='pairs/max_z_pairs'):
    read_actions=actions.io.ActionReader()
    print(in_path)
    seqs=read_actions(in_path)
    t0=time.time()
    pairs=graph.make_pairwise_distance(seqs)
    print("pairs computadion %d" % (time.time()-t0))
    utils.save_object(pairs.raw_pairs,out_path)

def show_dtw_feats(action_path='seqs/max_z',pairs_path='pairs/max_z_pairs' ):
    read_actions=actions.io.ActionReader(as_dict=True)
    seqs=read_actions(action_path)
    raw_pairs=utils.read_object(pairs_path)
    dtw_pairs=graph.PairwiseDistance(raw_pairs)
    action_names=seqs.keys()
    X=dtw_pairs.as_matrix(action_names)
    y=[ seqs[name_i].cat for name_i in action_names]
    embd=TSNE(n_components=2,perplexity=30).fit_transform(X)
    plot.plot_embedding(embd,y)

#compute_pairs()
show_dtw_feats()