import numpy as np
import time,utils,seqs.io
from metric import dtw_metric

def compute_pairs(in_path='mra/seqs/corl',out_path='mra/corl_pairs'):
    read_actions=seqs.io.ActionReader(as_dict=True)
    print(in_path)
    actions=read_actions(in_path)
    t0=time.time()
    pairs=make_pairwise_distance(actions)
    print("pairs computation %d" % (time.time()-t0))
    utils.save_object(pairs,out_path)

def make_pairwise_distance(actions,fraction=1000):
    pairs_dict={ name_i:{name_i:0.0}
                    for name_i in actions.keys()}
    names=actions.keys()
    n_names=len(actions.keys())
    for i in range(1,n_names):
        print(i)
        for j in range(0,i):	
            action_i=actions[names[i]]#pair_i[0]]
            action_j=actions[names[j]]#pair_i[1]]
            distance=dtw_metric(action_i.img_seq,action_j.img_seq)
            pairs_dict[action_i.name][action_j.name]=distance
            pairs_dict[action_j.name][action_i.name]=distance
    return pairs_dict

def all_pairs(names):
    pairs=[]
    n_names=len(names)
    for i in range(1,n_names):
        for j in range(0,i):
            pairs.append((names[i],names[j])) 
    return pairs

def as_matrix(actions,pairs_dict):
    names=actions.keys()
    names.sort()
    distance=[[ pairs_dict[name_i][name_j]
                        for name_i in names]
                            for name_j in names]
    X=np.array(distance)
    y=[actions[name_i].cat for name_i in names]
    return X,y

if __name__ == "__main__":
    compute_pairs()