import numpy as np
import scipy.io
import seqs,seqs.io,seqs.select 
import plot.ts
import scipy.io

def select_actions(in_path,out_path,cat_i=0,feat_j=0):
    actions=plot.ts.get_actions(in_path)
    actions=actions[cat_i+1]
    feats={ action_k.name:action_k.as_features()[feat_j] 
                for action_k in actions }
    scipy.io.savemat(out_path,feats)

def inspect_dataset(in_path,img_seq=False):
    actions=get_actions(in_path,img_seq=img_seq,select=1)
    print(len(actions))
    show_stats(action_sizes(actions))
    show_stats(cat_sizes(actions))

def get_actions(in_path,img_seq=False,select=False):
    read_actions=seqs.io.build_action_reader(img_seq=img_seq,as_dict=False)
    actions=read_actions(in_path)
    if(select):
        actions=seqs.select.select(actions,selector=select)
    return actions

def action_sizes(actions):
    return [len(action_i) for action_i in actions]

def cat_sizes(actions):
    actions_by_cat=seqs.by_cat(actions).values()
    return [ len(cat_i) for cat_i in actions_by_cat]

def show_stats(array):
    print("total %d" % np.sum(array))
    print("median %d" % np.median(array))
    print("mean %d" % np.mean(array))

select_actions(in_path='mra/imgs/skew',out_path='../test')
#inspect_dataset(in_path='mra/frames/proj',img_seq=True)