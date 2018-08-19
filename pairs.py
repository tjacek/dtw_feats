import numpy as np
import time,feats,utils,seqs.io,ensemble
from metric import dtw_metric
import dataset.instances

def compute_pairs(in_path='mhad/skew',out_path='mhad/skew_pairs'):
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=True)
    print(in_path)
    actions=read_actions(in_path)
    t0=time.time()
    pairs=make_pairwise_distance(actions)
    prin("pairs computation %d" % (time.time()-t0))
    utils.save_object(pairs,out_path)

def make_pairwise_distance(actions):
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

def get_pairs_ensemble():
    def pair_helper(in_path):
        dtw_pairs=utils.read_object(in_path)
        return as_instances(dtw_pairs)
    out_fun=lambda out_path,insts:insts.to_txt(out_path)
    return  ensemble.EnsembleFun(pair_helper,out_fun)

def as_txt(dtw_distance):
    dtw_tuples=as_tuples(dtw_distance)
    return [",".join(tuple_i) for tuple_i in dtw_tuples]

def as_tuples(dtw_distance):
    names=dtw_distance.keys()
    return [ (name_i,name_j,str(dtw_distance[name_i][name_j]))
                for name_i in names
                    for name_j in names]

def as_matrix(pairs_dict):
    insts=get_descs(pairs_dict,as_dict=False)
    distance=[ distance_vector(inst_i.name,pairs_dict) 
                for inst_i in insts]
    X=np.array(distance)
    y,persons=insts.cats(),insts.persons()
    return X,y,persons

def as_instances(pairs):
    insts=get_descs(pairs)
    train,test=insts.split()
    train_names= train.names()
    train_names.sort()
    def feat_helper(inst_i):
        name_i=inst_i.name
        return [pairs[name_i][name_j]
                    for name_j in train_names]
    for inst_i in insts.raw():
        inst_i.data=feat_helper(inst_i)
    return insts

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
    in_fun=utils.read_decorate(as_txt)
    ens=ensemble.EnsembleFun(in_fun,utils.save_string)
    ens("mhad/_deep_pairs","mhad/text_pairs")