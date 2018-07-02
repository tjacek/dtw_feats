import time
import actions.io,metric,graph,utils

def compute_pairs(in_path='seqs/skew',out_path='pairs/skew_pairs'):
    read_actions=actions.io.ActionReader()
    print(in_path)
    seqs=read_actions(in_path)
    t0=time.time()
    pairs=graph.make_pairwise_distance(seqs)
    print("pairs computadion %d" % (time.time()-t0))
    utils.save_object(pairs.raw_pairs,out_path)

compute_pairs()