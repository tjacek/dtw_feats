import numpy as np
import time,feats,utils,seqs.io
from metric import dtw_metric
import dataset.instances

def compute_pairs(in_path='mhad/skew',out_path='mhad/skew_pairs'):
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=True)
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

def as_matrix(pairs_dict):
    insts=get_descs(pairs_dict,as_dict=False)
    distance=[ distance_vector(inst_i.name,pairs_dict) 
                for inst_i in insts]
    X=np.array(distance)
    y=[inst_i.cat for inst_i in insts]
    persons=[inst_i.person for inst_i in insts]
    return X,y,persons

def as_instances(pairs):
    insts=get_descs(pairs)
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

def get_descs(names):
    if(type(names)==dict):
        names=list(names.keys())
    insts=[dataset.instances.empty_instance(name_i)
            for name_i in names]
    return dataset.instances.InstsGroup(insts)

def distance_vector(name_i,pairs):
    sub_dict=pairs[name_i]
    names=pairs.keys()
    names.sort()
    return [sub_dict[key_j]  for key_j in names]

def make_dtw_feats(in_path='mra/pairs/corl_pairs',
                   out_path='mra/simple/corl_feats.txt'):
    dtw_pairs=utils.read_object(in_path)
    insts=as_instances(dtw_pairs)
    dataset.instances.to_txt(out_path,insts)

def make_stats_feat(in_path='mra/seqs/all',out_path='mra/simple/basic.txt'):
    read_action=seqs.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_action(in_path)
    feat_extractor=feats.GlobalFeatures()
    insts=[feat_extractor(action_i) for action_i in actions]
    dataset.instances.to_txt(out_path,insts)

if __name__ == "__main__":
#    compute_pairs(in_path='mhad/seqs/corl',out_path='mhad/pairs/corls')
    make_dtw_feats(in_path='mhad/pairs/corls',out_path='mhad/simple/corls.txt')
#    make_stats_feat(in_path='mhad/seqs/all',out_path='mhad/simple/basic.txt')