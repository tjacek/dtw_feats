import numpy as np
import seqs,seqs.io,seqs.select 

def inspect_dataset(in_path,img_seq=False):
    actions=get_actions(in_path,img_seq=img_seq,select=None)
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
    print("median %d" % np.median(array))
    print("mean %d" % np.mean(array))

inspect_dataset(in_path='mra/seqs/corl',img_seq=False)