import numpy as np
import time,utils,seqs.io
from metric import dtw_metric
import dataset.instances

def compute_pairs(in_path='mhad/skew',out_path='mhad/skew_pairs'):
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

def as_instances(pairs):
    names=pairs.keys()
    insts=[dataset.instances.empty_instance(name_i)
                    for name_i in names]
    train,test=dataset.instances.split_instances(insts)
    train_names=[inst_i.name for inst_i in train]
    train_names.sort()
    def feat_helper(inst_i):
        name_i=inst_i.name
        return [pairs[name_i][name_j]
                    for name_j in train_names]
    for inst_i in insts:
        inst_i.data=feat_helper(inst_i)
    return insts

if __name__ == "__main__":
    dtw_pairs=utils.read_object('mra/pairs/max_z_pairs')
    insts=as_instances(dtw_pairs)
    dataset.instances.to_txt('mra/simple/max_z_feats.txt',insts)
#    compute_pairs()